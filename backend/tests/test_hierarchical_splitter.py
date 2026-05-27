"""Unit tests for :mod:`src.hybridrag.ingestion.chunking.hierarchical`.

Self-contained: no on-disk fixtures, no network. Each test builds its own
markdown input.
"""
from __future__ import annotations

import pytest

from src.hybridrag.ingestion.chunking.hierarchical import (
    HierarchicalChunk,
    HierarchicalSplitter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_splitter(**overrides) -> HierarchicalSplitter:
    """A small-size splitter so unit-test bodies stay tiny but still split."""
    defaults = dict(parent_size=120, child_size=40, parent_overlap=10, child_overlap=5)
    defaults.update(overrides)
    return HierarchicalSplitter(**defaults)


def _long_paragraph(seed: str, repeats: int = 30) -> str:
    """Build a paragraph long enough to force a child-level split."""
    return " ".join([seed] * repeats)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_simple_two_section() -> None:
    """Two H2 sections → ≥2 parents, more children than parents, every chunk
    has a non-empty header_path and valid parent_id linkage."""
    md = (
        "# Tuyển sinh\n"
        "## Cơ sở 1\n"
        f"{_long_paragraph('Địa chỉ cơ sở 1 nằm tại Hưng Yên.', 20)}\n"
        f"{_long_paragraph('Thông tin liên hệ qua website hoặc fanpage.', 20)}\n"
        "## Cơ sở 2\n"
        f"{_long_paragraph('Cơ sở 2 đào tạo các ngành kinh tế.', 20)}\n"
        f"{_long_paragraph('Học phí và lịch học khác nhau theo kỳ.', 20)}\n"
    )

    splitter = _make_splitter()
    parents, children = splitter.split(
        md, file_id="f-1", key="docs/two_sections.md"
    )

    assert len(parents) >= 2, "Expected at least one parent per H2 section"
    assert len(children) > len(parents), (
        "Children should outnumber parents when sections are long"
    )

    parent_ids = {p.chunk_id for p in parents}
    for c in children:
        assert c.parent_id in parent_ids, (
            f"Child {c.chunk_id} references missing parent {c.parent_id}"
        )

    for chunk in (*parents, *children):
        assert chunk.header_path, "header_path must be non-empty"
        assert chunk.section, "section must be non-empty"

    # At least one parent per H2 label should appear.
    labels = {p.section for p in parents}
    assert "Cơ sở 1" in labels
    assert "Cơ sở 2" in labels


def test_idempotent_ids() -> None:
    """Splitting the same text twice yields identical chunk_id and parent_id
    sequences — required for re-ingest idempotency."""
    md = (
        "# Doc\n"
        "## Sec A\n"
        f"{_long_paragraph('alpha text content here', 15)}\n"
        "## Sec B\n"
        f"{_long_paragraph('beta text content here', 15)}\n"
    )

    splitter = _make_splitter()
    parents_a, children_a = splitter.split(md, file_id="file-X", key="k")
    parents_b, children_b = splitter.split(md, file_id="file-X", key="k")

    assert [p.chunk_id for p in parents_a] == [p.chunk_id for p in parents_b]
    assert [c.chunk_id for c in children_a] == [c.chunk_id for c in children_b]
    assert [c.parent_id for c in children_a] == [c.parent_id for c in children_b]


def test_table_section_flag() -> None:
    """A section containing a markdown table → is_table=True on parents AND
    its children. Sibling non-table sections stay is_table=False."""
    md = (
        "# Điểm chuẩn\n"
        "## Bảng điểm 2024\n"
        "| STT | Ngành | Điểm |\n"
        "| --- | --- | --- |\n"
        "| 1 | CNTT | 17 |\n"
        "| 2 | Kỹ thuật ô tô | 17 |\n"
        "| 3 | Cơ điện tử | 16 |\n"
        "## Ghi chú\n"
        f"{_long_paragraph('Đây là phần ghi chú không có bảng nào.', 10)}\n"
    )

    splitter = _make_splitter()
    parents, children = splitter.split(md, file_id="f-tbl", key="k")

    table_parents = [p for p in parents if p.section == "Bảng điểm 2024"]
    note_parents = [p for p in parents if p.section == "Ghi chú"]
    assert table_parents, "Expected at least one parent for the table section"
    assert note_parents, "Expected at least one parent for the note section"

    for p in table_parents:
        assert p.is_table is True
    for p in note_parents:
        assert p.is_table is False

    table_parent_ids = {p.chunk_id for p in table_parents}
    note_parent_ids = {p.chunk_id for p in note_parents}
    for c in children:
        if c.parent_id in table_parent_ids:
            assert c.is_table is True, "Children of table section must inherit is_table"
        elif c.parent_id in note_parent_ids:
            assert c.is_table is False


def test_header_prefix_present() -> None:
    """Every chunk's content starts with ``header_path joined by ' > '`` + \\n\\n."""
    md = (
        "# Root\n"
        "## Sub A\n"
        "### Leaf\n"
        f"{_long_paragraph('payload payload', 10)}\n"
        "## Sub B\n"
        f"{_long_paragraph('payload again', 10)}\n"
    )

    splitter = _make_splitter()
    parents, children = splitter.split(md, file_id="f-prefix", key="k")

    assert parents and children
    for chunk in (*parents, *children):
        expected = " > ".join(chunk.header_path) + "\n\n"
        assert chunk.content.startswith(expected), (
            f"Missing header prefix {expected!r} on chunk "
            f"{chunk.chunk_id} (got {chunk.content[:80]!r})"
        )


def test_doc_meta_propagation() -> None:
    """doc_meta keys land in every chunk's extra dict."""
    md = (
        "# T\n"
        "## S1\n"
        f"{_long_paragraph('content one', 10)}\n"
        "## S2\n"
        f"{_long_paragraph('content two', 10)}\n"
    )
    meta = {"campus": "co_so_1", "year": 2025}

    splitter = _make_splitter()
    parents, children = splitter.split(
        md, file_id="f-meta", key="k", doc_meta=meta
    )

    assert parents and children
    for chunk in (*parents, *children):
        assert chunk.extra.get("campus") == "co_so_1"
        assert chunk.extra.get("year") == 2025
        # key is also stored for traceability
        assert chunk.extra.get("key") == "k"


def test_empty_input() -> None:
    """Empty / whitespace-only input → empty lists, no crash."""
    splitter = _make_splitter()
    for text in ("", "   ", "\n\n\t\n"):
        parents, children = splitter.split(text, file_id="f-empty", key="k")
        assert parents == []
        assert children == []


def test_no_header_input_synthesizes_from_key_and_first_line() -> None:
    """Plain text without any markdown headers → synthetic-header fallback.

    h1 is the filename stem from ``key``; h2 is the first meaningful line of
    the body. This guarantees ``header_path`` is never ``["(no header)"]`` when
    we have any context to work with (which protects downstream embedding/
    reranking from losing the section signal entirely).
    """
    md = "Trường Đại học Sư phạm Kỹ thuật Hưng Yên giới thiệu chương trình.\n\n" + \
         _long_paragraph("Thông tin chi tiết về các ngành đào tạo.", 30)

    splitter = _make_splitter()
    parents, children = splitter.split(
        md, file_id="f-noh", key="samples/wiki.md"
    )

    assert parents, "Even header-less input must yield at least one parent"
    for chunk in (*parents, *children):
        assert chunk.header_path[0] == "wiki", (
            f"h1 should be filename stem 'wiki', got {chunk.header_path!r}"
        )
        assert len(chunk.header_path) >= 2, (
            "Should synthesize at least h1 + h2 from filename + first line"
        )
        # Crucially: never the legacy placeholder when we have a key + body.
        assert chunk.header_path != ["(no header)"]
        # And the content prefix mirrors the synthesized path.
        expected_prefix = " > ".join(chunk.header_path) + "\n\n"
        assert chunk.content.startswith(expected_prefix)


def test_no_header_no_key_falls_back_to_placeholder() -> None:
    """When BOTH the source lacks headers AND no ``key`` is supplied, fall
    back to the legacy ``["(no header)"]`` placeholder rather than crashing.

    This is the only path that should still produce the placeholder — and it
    only happens for ad-hoc programmatic callers (the ingestion pipeline
    always supplies a key)."""
    md = _long_paragraph("plain prose without any markdown headers.", 30)

    splitter = _make_splitter()
    parents, children = splitter.split(md, file_id="f-noh", key="")

    assert parents
    # h2 (first line) is still synthesized; h1 collapses to the placeholder.
    for chunk in (*parents, *children):
        assert chunk.header_path[0] == "(no header)"


def test_filename_with_vietnamese_diacritics_used_as_h1() -> None:
    """The filename stem (with diacritics / spaces) survives unchanged as h1
    so the section signal stays human-readable for the LLM."""
    md = "Tuyển sinh Đại Học các ngành Năm 2024\n\n" + \
         _long_paragraph("Bảng điểm chuẩn các ngành.", 20)

    splitter = _make_splitter()
    parents, _ = splitter.split(
        md, file_id="f-vn", key="samples/Điểm 2024.md"
    )
    assert parents
    assert parents[0].header_path[0] == "Điểm 2024"


def test_real_markdown_headers_bypass_synthetic_fallback() -> None:
    """When the source DOES have proper markdown headers, the synthetic
    fallback must stay out of the way."""
    md = (
        "# Real Title\n"
        "## Real Section\n"
        f"{_long_paragraph('genuine markdown content', 15)}\n"
    )
    splitter = _make_splitter()
    parents, _ = splitter.split(md, file_id="f-real", key="samples/anything.md")
    assert parents
    # h1 must come from the markdown, not from the filename.
    assert parents[0].header_path[0] == "Real Title"
    assert "anything" not in parents[0].header_path
