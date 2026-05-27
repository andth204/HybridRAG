"""Markdown-table-aware parser + row-level chunker.

Phase 3A of the HybridRAG admissions chatbot roadmap (see
``backend/docs/ROADMAP_V2.md``). Admissions docs frequently encode
score / tuition / program tables as markdown pipe-tables. The default
recursive splitter cuts those tables in half and loses the
row -> header alignment, which makes structured lookups unreliable.

This module exposes three primitives:

* :func:`find_tables` -- detect every markdown table in a document and
  return a structured :class:`TableBlock` per table.
* :func:`table_to_row_chunks` -- turn one :class:`TableBlock` into one
  :class:`TableChunk` per data row, with the header row preserved on
  every chunk so the LLM never sees a headerless row.
* :func:`split_text_keeping_tables` -- iterate through a document
  yielding ``(prose, None)`` and ``("", TableBlock)`` pairs in order,
  so the higher-level chunker can route prose through normal recursive
  splitting and tables through row-level chunking.

Limitations
-----------
* The parser does a simple ``str.split("|")`` to extract cells. A literal
  ``|`` inside cell content (e.g. inside a URL) will be treated as a
  column boundary. In the admissions corpus this is rare; the cost of
  handling escaped pipes (``\\|``) is not worth the complexity right now.
* HTML tables (``<table>...``) are NOT detected. Markdown only.
* Code-fenced (``\`\`\` ... \`\`\``) blocks are skipped so a fake
  ``|---|---|`` inside a code sample is not mistaken for a real table.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


logger = logging.getLogger(__name__)


# A line that looks like a markdown table separator row, e.g.
# ``| --- | :---: | ---: |`` or ``|---|---|``.
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")

# A line that looks like a markdown table row (must contain at least one
# pipe and start/end with optional whitespace). Header rows and data rows
# both match this.
TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")

# A fenced code-block boundary (``` or ~~~), optionally followed by a
# language tag. We toggle a flag on each match to skip table detection
# inside the block.
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Stable namespace for the deterministic uuid5 chunk ids.
_NAMESPACE = uuid.NAMESPACE_OID


# --- public dataclasses --------------------------------------------------

@dataclass
class TableBlock:
    """A parsed markdown table.

    ``start_line`` and ``end_line`` are 0-indexed line numbers within the
    original text (line of header row .. last data row, both inclusive).
    """

    caption: Optional[str]
    header: list[str]
    alignments: list[str]
    rows: list[list[str]]
    start_line: int
    end_line: int
    raw: str


@dataclass
class TableChunk:
    """One row of a :class:`TableBlock` rendered as a self-contained chunk.

    ``content`` always carries the caption (if any) plus the header row
    plus this row, rendered as a minimal markdown sub-table so the LLM
    sees a row in context.
    """

    chunk_id: str
    caption: Optional[str]
    header: list[str]
    row: list[str]
    row_index: int
    columns_count: int
    content: str
    extra: dict[str, Any] = field(default_factory=dict)


# --- internal helpers ----------------------------------------------------

def _split_cells(row_line: str) -> list[str]:
    """Split a markdown row on ``|`` and trim whitespace per cell.

    Leading and trailing empty cells produced by leading/trailing ``|``
    are dropped (``"| a | b |"`` -> ``["a", "b"]``).
    """
    raw_cells = row_line.split("|")
    # Strip outer whitespace per cell first so a cell that is only spaces
    # is treated as empty for the leading/trailing trim.
    cells = [c.strip() for c in raw_cells]
    # Drop leading empty cells (from a leading ``|``).
    while cells and cells[0] == "":
        cells.pop(0)
    # Drop trailing empty cells (from a trailing ``|``).
    while cells and cells[-1] == "":
        cells.pop()
    return cells


def _parse_alignments(sep_line: str) -> list[str]:
    """Derive per-column alignment from a separator row.

    * ``:---``  -> ``"left"``
    * ``:---:`` -> ``"center"``
    * ``---:``  -> ``"right"``
    * ``---``   -> ``"left"`` (markdown default)
    """
    alignments: list[str] = []
    for cell in _split_cells(sep_line):
        c = cell.strip()
        starts = c.startswith(":")
        ends = c.endswith(":")
        if starts and ends:
            alignments.append("center")
        elif ends:
            alignments.append("right")
        else:
            # ``:---`` is explicit left; bare ``---`` is default-left.
            alignments.append("left")
    return alignments


def _normalize_row_width(row: list[str], width: int, *, where: str) -> list[str]:
    """Pad short rows with empty strings, truncate long ones; log mismatches."""
    if len(row) == width:
        return row
    if len(row) < width:
        logger.warning(
            "table row at %s has %d cells, expected %d -- padding with empties",
            where,
            len(row),
            width,
        )
        return row + [""] * (width - len(row))
    logger.warning(
        "table row at %s has %d cells, expected %d -- truncating",
        where,
        len(row),
        width,
    )
    return row[:width]


def _render_row_content(
    caption: Optional[str],
    header: list[str],
    row: list[str],
) -> str:
    """Render a minimal markdown sub-table that preserves the header."""
    header_md = "| " + " | ".join(header) + " |"
    row_md = "| " + " | ".join(row) + " |"
    if caption:
        return f"{caption}\n\n{header_md}\n{row_md}"
    return f"{header_md}\n{row_md}"


def _looks_like_header_line(line: str) -> bool:
    """True if the line is a markdown ATX header (``# ... ``, ``## ...``)."""
    return bool(re.match(r"^\s{0,3}#{1,6}\s+\S", line))


def _strip_header_marks(line: str) -> str:
    """Strip leading ``#`` and surrounding whitespace from a header line."""
    return re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip()


def _find_caption(lines: list[str], header_line_idx: int) -> Optional[str]:
    """Walk upward from the row above the header to find the caption.

    Rule:
    * Skip blank lines.
    * If the first non-blank line is a table row (would be part of another
      adjacent table) the caption is None.
    * If it is a markdown header, return the header text without the
      ``#`` marks.
    * Otherwise return the line stripped of whitespace.
    """
    i = header_line_idx - 1
    while i >= 0 and lines[i].strip() == "":
        i -= 1
    if i < 0:
        return None
    candidate = lines[i]
    if TABLE_ROW_RE.match(candidate) or TABLE_SEP_RE.match(candidate):
        return None
    if _looks_like_header_line(candidate):
        text = _strip_header_marks(candidate)
        return text or None
    text = candidate.strip()
    return text or None


def _fence_aware_iter(lines: list[str]) -> Iterator[tuple[int, str, bool]]:
    """Yield ``(idx, line, in_fence)`` for every line.

    ``in_fence`` is True while we are inside a fenced code block; the
    table detector uses it to skip detection in code blocks.
    """
    in_fence = False
    for idx, line in enumerate(lines):
        if _FENCE_RE.match(line):
            # The fence line itself is reported with the PRIOR flag so
            # that an opening fence doesn't accidentally hide itself.
            yield idx, line, in_fence
            in_fence = not in_fence
            continue
        yield idx, line, in_fence


# --- public API ----------------------------------------------------------

def find_tables(text: str) -> list[TableBlock]:
    """Scan ``text`` and return every markdown table as a :class:`TableBlock`.

    Detection rule (per the task contract):
      * a line is a SEPARATOR when it matches :data:`TABLE_SEP_RE` AND the
        line immediately above it matches :data:`TABLE_ROW_RE`;
      * the line above the separator is the HEADER row;
      * consecutive lines below the separator that match
        :data:`TABLE_ROW_RE` are DATA rows;
      * a table is closed by a blank line or any non-row line;
      * tables inside fenced code blocks (``\`\`\` ... \`\`\``) are skipped;
      * header-only tables (zero data rows) are still emitted.
    """
    if not text:
        return []

    lines = text.splitlines()
    tables: list[TableBlock] = []
    fence_flags = [in_fence for _idx, _line, in_fence in _fence_aware_iter(lines)]

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Need a separator line, but at least one prior line for the header.
        if (
            i >= 1
            and not fence_flags[i]
            and not fence_flags[i - 1]
            and TABLE_SEP_RE.match(line)
            and TABLE_ROW_RE.match(lines[i - 1])
        ):
            header_idx = i - 1
            sep_idx = i
            header_cells = _split_cells(lines[header_idx])
            alignments = _parse_alignments(lines[sep_idx])
            # Force alignments to match header width.
            alignments = _normalize_row_width(
                alignments, len(header_cells), where=f"separator line {sep_idx}"
            )

            data_rows: list[list[str]] = []
            j = sep_idx + 1
            while j < n:
                if fence_flags[j]:
                    break
                row_line = lines[j]
                if row_line.strip() == "" or not TABLE_ROW_RE.match(row_line):
                    break
                cells = _split_cells(row_line)
                cells = _normalize_row_width(
                    cells, len(header_cells), where=f"line {j}"
                )
                data_rows.append(cells)
                j += 1
            end_idx = j - 1 if data_rows else sep_idx

            caption = _find_caption(lines, header_idx)
            raw = "\n".join(lines[header_idx : end_idx + 1])
            tables.append(
                TableBlock(
                    caption=caption,
                    header=header_cells,
                    alignments=alignments,
                    rows=data_rows,
                    start_line=header_idx,
                    end_line=end_idx,
                    raw=raw,
                )
            )
            i = end_idx + 1
            continue
        i += 1

    return tables


def table_to_row_chunks(
    table: TableBlock,
    *,
    file_id: str,
    table_index: int,
    extra: dict[str, Any] | None = None,
) -> list[TableChunk]:
    """One row of ``table`` -> one :class:`TableChunk`.

    Each chunk's ``content`` carries the caption (if any) plus the
    header row plus this row, so a chunk surfaced by retrieval is
    self-contained when shown to the LLM. ``chunk_id`` is a stable
    uuid5 derived from ``file_id``, ``table_index`` and ``row_index``,
    so re-ingestion yields identical ids (idempotent indexing).
    """
    chunks: list[TableChunk] = []
    columns_count = len(table.header)
    base_extra: dict[str, Any] = dict(extra or {})
    for row_index, row in enumerate(table.rows):
        # Defensive: rows should already be normalized by find_tables, but
        # callers may construct TableBlocks themselves.
        row = _normalize_row_width(
            row, columns_count, where=f"table_index={table_index} row={row_index}"
        )
        content = _render_row_content(table.caption, table.header, row)
        seed = f"{file_id}::table::{table_index}::row::{row_index}"
        chunk_id = str(uuid.uuid5(_NAMESPACE, seed))
        chunks.append(
            TableChunk(
                chunk_id=chunk_id,
                caption=table.caption,
                header=list(table.header),
                row=list(row),
                row_index=row_index,
                columns_count=columns_count,
                content=content,
                extra=dict(base_extra),
            )
        )
    return chunks


def split_text_keeping_tables(
    text: str,
) -> Iterator[tuple[str, Optional[TableBlock]]]:
    """Yield ``(segment, table)`` pairs that reassemble to ``text``.

    Prose runs yield ``(prose, None)``; each detected table yields
    ``("", TableBlock)``. Line breaks inside prose runs are preserved
    exactly as in the input.

    The higher-level chunker is expected to:
      * Send prose segments through normal recursive splitting.
      * Send each table through :func:`table_to_row_chunks` (or render
        the full table separately).
    """
    if not text:
        return

    tables = find_tables(text)
    if not tables:
        yield text, None
        return

    lines = text.splitlines(keepends=True)
    cursor = 0  # index into lines

    def _slice_to_text(start: int, end: int) -> str:
        # ``end`` is exclusive. ``"".join`` preserves the original
        # newline characters captured by ``splitlines(keepends=True)``.
        return "".join(lines[start:end])

    for table in tables:
        if table.start_line > cursor:
            prose = _slice_to_text(cursor, table.start_line)
            # Yield even if it is whitespace-only; the consumer can
            # decide what to do with empty separators. We do NOT yield
            # truly empty strings (no characters at all).
            if prose:
                yield prose, None
        yield "", table
        cursor = table.end_line + 1

    if cursor < len(lines):
        tail = _slice_to_text(cursor, len(lines))
        if tail:
            yield tail, None


# Integration note: this module is consumed by
# :class:`hybridrag.ingestion.chunking.hierarchical.HierarchicalSplitter`
# when constructed with ``route_tables_to_row_chunks=True``. The flag is
# opt-in so existing splitter callers keep their current behaviour.


__all__ = [
    "TableBlock",
    "TableChunk",
    "TABLE_ROW_RE",
    "TABLE_SEP_RE",
    "find_tables",
    "table_to_row_chunks",
    "split_text_keeping_tables",
]
