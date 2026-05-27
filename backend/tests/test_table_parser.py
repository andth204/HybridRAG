"""Unit tests for :mod:`src.hybridrag.ingestion.chunking.table_parser`.

Self-contained: no on-disk fixtures, no network. Each test builds its own
markdown input.
"""
from __future__ import annotations

import logging

import pytest

from src.hybridrag.ingestion.chunking.table_parser import (
    TableBlock,
    TableChunk,
    find_tables,
    split_text_keeping_tables,
    table_to_row_chunks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_TABLE = (
    "| STT | Ngành | Điểm |\n"
    "| --- | --- | --- |\n"
    "| 1 | CNTT | 17 |\n"
    "| 2 | Kỹ thuật ô tô | 17 |\n"
    "| 3 | Cơ điện tử | 16 |\n"
    "| 4 | Quản trị KD | 15 |\n"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_find_simple_table() -> None:
    """One 3-column, 4-row table → detected with the correct shape."""
    tables = find_tables(_SIMPLE_TABLE)
    assert len(tables) == 1
    table = tables[0]
    assert table.header == ["STT", "Ngành", "Điểm"]
    assert len(table.rows) == 4
    assert table.rows[0] == ["1", "CNTT", "17"]
    assert table.rows[-1] == ["4", "Quản trị KD", "15"]


def test_alignment_parsing() -> None:
    """Separator ``| :--- | :---: | ---: |`` → left / center / right."""
    md = (
        "| A | B | C |\n"
        "| :--- | :---: | ---: |\n"
        "| 1 | 2 | 3 |\n"
    )
    tables = find_tables(md)
    assert len(tables) == 1
    assert tables[0].alignments == ["left", "center", "right"]


def test_caption_from_heading() -> None:
    """Nearest non-blank line above the header is a markdown heading."""
    md = (
        "## Bảng điểm 2024\n"
        "\n"
        + _SIMPLE_TABLE
    )
    tables = find_tables(md)
    assert len(tables) == 1
    assert tables[0].caption == "Bảng điểm 2024"


def test_caption_from_paragraph() -> None:
    """Nearest non-blank line above the header is a regular paragraph."""
    md = (
        "Đây là bảng điểm chuẩn năm 2024.\n"
        "\n"
        + _SIMPLE_TABLE
    )
    tables = find_tables(md)
    assert len(tables) == 1
    assert tables[0].caption == "Đây là bảng điểm chuẩn năm 2024."


def test_no_caption() -> None:
    """Table at the top of a document → caption is None."""
    tables = find_tables(_SIMPLE_TABLE)
    assert len(tables) == 1
    assert tables[0].caption is None


def test_row_chunk_content_includes_header() -> None:
    """Every emitted row chunk's content carries the rendered header row."""
    tables = find_tables(_SIMPLE_TABLE)
    chunks = table_to_row_chunks(tables[0], file_id="f-1", table_index=0)
    assert len(chunks) == 4
    expected_header_md = "| STT | Ngành | Điểm |"
    for chunk in chunks:
        assert expected_header_md in chunk.content, (
            f"Chunk {chunk.row_index} missing header row in content:\n{chunk.content!r}"
        )
        # And the row itself is in the rendered content.
        assert chunk.row[1] in chunk.content


def test_chunk_id_idempotent() -> None:
    """Same input twice → identical chunk_ids in the same order."""
    tables_a = find_tables(_SIMPLE_TABLE)
    tables_b = find_tables(_SIMPLE_TABLE)
    chunks_a = table_to_row_chunks(tables_a[0], file_id="f-id", table_index=2)
    chunks_b = table_to_row_chunks(tables_b[0], file_id="f-id", table_index=2)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    # And ids are unique within a single run.
    assert len({c.chunk_id for c in chunks_a}) == len(chunks_a)


def test_uneven_row_width(caplog: pytest.LogCaptureFixture) -> None:
    """Short rows get padded, long rows get truncated, each warns once."""
    md = (
        "| A | B | C |\n"
        "| --- | --- | --- |\n"
        "| 1 | 2 |\n"               # short row → pad
        "| 4 | 5 | 6 | 7 | 8 |\n"   # long row → truncate
    )
    with caplog.at_level(logging.WARNING, logger="src.hybridrag.ingestion.chunking.table_parser"):
        tables = find_tables(md)
    assert len(tables) == 1
    rows = tables[0].rows
    assert rows[0] == ["1", "2", ""]
    assert rows[1] == ["4", "5", "6"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 2


def test_skip_fenced_code_block() -> None:
    """A pipe-table inside a fenced code block must NOT be detected."""
    md = (
        "Some prose with a fenced sample below:\n"
        "\n"
        "```\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "```\n"
        "\n"
        "And a REAL table here:\n"
        "\n"
        "| X | Y |\n"
        "| --- | --- |\n"
        "| 9 | 8 |\n"
    )
    tables = find_tables(md)
    assert len(tables) == 1
    assert tables[0].header == ["X", "Y"]
    assert tables[0].rows == [["9", "8"]]


def test_split_text_keeping_tables() -> None:
    """Prose / table / prose → three segments in order."""
    md = (
        "prose A\n"
        "\n"
        "| H1 | H2 |\n"
        "| --- | --- |\n"
        "| r1 | r2 |\n"
        "\n"
        "prose B\n"
    )
    segments = list(split_text_keeping_tables(md))
    # Filter pure-whitespace separators that the generator may emit.
    assert len(segments) == 3
    prose_a, table_seg, prose_b = segments

    text_a, tbl_a = prose_a
    assert tbl_a is None
    assert "prose A" in text_a

    text_t, tbl_t = table_seg
    assert text_t == ""
    assert isinstance(tbl_t, TableBlock)
    assert tbl_t.header == ["H1", "H2"]
    assert tbl_t.rows == [["r1", "r2"]]

    text_b, tbl_b = prose_b
    assert tbl_b is None
    assert "prose B" in text_b
