# Next-gen markdown-aware hierarchical (parent/child) chunker. The legacy
# `splitter.py` in this package will be deprecated once Phase 2 integration wires
# this module into `processor.py` (see backend/docs/ROADMAP_V2.md, task 2B).
from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .table_parser import find_tables, table_to_row_chunks


# --- public dataclass -----------------------------------------------------

@dataclass
class HierarchicalChunk:
    """A single parent or child chunk emitted by :class:`HierarchicalSplitter`.

    ``parent_id`` is ``None`` on parent chunks themselves and points to the
    owning parent's ``chunk_id`` on child chunks.
    """

    chunk_id: str
    parent_id: Optional[str]
    content: str
    header_path: list[str]
    section: str
    chunk_level: str  # "parent" or "child"
    is_table: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# --- helpers --------------------------------------------------------------

# Markdown table separator row, e.g. `| --- | :---: | ---: |` or `|---|---|`.
_TABLE_SEP_RE = re.compile(
    r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$",
    re.MULTILINE,
)

_NAMESPACE = uuid.NAMESPACE_OID
_NO_HEADER = "(no header)"

# Strip leading list markers / numbering / bullet glyphs so the synthetic h2
# derived from the first line is a clean section title (e.g. "1. Foo" -> "Foo",
# "🔹 Tổng chỉ tiêu: ..." -> "Tổng chỉ tiêu: ...").
_FIRST_LINE_CLEAN_RE = re.compile(
    r"^[\s​]*"                       # leading whitespace / zero-width
    r"(?:[#>*\-•‣◦⁃]+\s*)?"  # bullets / md noise
    r"(?:\d+[\.\)]\s*)?"                  # "1." or "1)"
    r"(?:[\U0001F000-\U0001FFFF☀-➿⬀-⯿]+\s*)+?"  # emoji run
    r"|^[\s​]+"
)


def _derive_h1_from_key(key: str | None) -> str:
    """Filename (sans extension) extracted from a MinIO-style key.

    Falls back to ``_NO_HEADER`` only when ``key`` is missing/empty. Returns
    a human-readable string (never crashes on weird inputs).
    """
    if not key:
        return _NO_HEADER
    name = os.path.basename(str(key).replace("\\", "/"))
    stem, _ = os.path.splitext(name)
    stem = stem.strip()
    return stem or _NO_HEADER


def _first_meaningful_line(body: str, *, max_len: int = 120) -> str:
    """First non-empty line of ``body`` with bullet/numbering noise trimmed.

    Returns an empty string only if ``body`` is all whitespace. The result is
    truncated to ``max_len`` chars so an accidental long paragraph cannot
    pollute the header_path UI.
    """
    if not body:
        return ""
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip a leading run of bullets / numbering / emojis.
        cleaned = _FIRST_LINE_CLEAN_RE.sub("", line).strip()
        cleaned = cleaned or line  # fall back to original if regex over-stripped
        if len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 1].rstrip() + "…"
        return cleaned
    return ""


@lru_cache(maxsize=4)
def _get_encoder(encoding_name: str) -> tiktoken.Encoding:
    """Cache tiktoken encoders by name (process-global)."""
    return tiktoken.get_encoding(encoding_name)


def _token_len(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


def _contains_table(text: str) -> bool:
    """True if ``text`` contains at least one markdown table separator row."""
    return bool(_TABLE_SEP_RE.search(text))


def _build_header_path(metadata: dict[str, str], header_keys: list[str]) -> list[str]:
    """Order the header values according to ``header_keys`` (h1, h2, ...)."""
    path = [metadata[k] for k in header_keys if metadata.get(k)]
    if not path:
        return [_NO_HEADER]
    return path


def _header_prefix(header_path: list[str]) -> str:
    return " > ".join(header_path) + "\n\n"


def _stable_uuid(*parts: str) -> str:
    payload = "::".join(parts)
    return str(uuid.uuid5(_NAMESPACE, payload))


def _content_fingerprint(content: str) -> str:
    """sha1 of the first 512 chars of the content used as a stable id seed."""
    return hashlib.sha1(content[:512].encode("utf-8")).hexdigest()


# --- main splitter --------------------------------------------------------

class HierarchicalSplitter:
    """Markdown-aware parent/child chunker.

    Pipeline:
        1. :class:`MarkdownHeaderTextSplitter` (``strip_headers=False``) cuts the
           document on H1/H2/H3/H4 boundaries so each section keeps its header
           chain in ``metadata``.
        2. Each section is split into parent chunks of ~``parent_size`` tokens
           via :class:`RecursiveCharacterTextSplitter`, then each parent is
           split into child chunks of ~``child_size`` tokens.
        3. Every chunk's ``content`` is prefixed with the joined header path
           (``"Cơ sở 1 > Địa chỉ"``) so the embedder and the LLM always see
           the section context.

    Identifiers are deterministic UUIDv5 derived from ``file_id`` + a content
    fingerprint, so re-ingesting the same document yields the same ids
    (idempotent indexing).
    """

    DEFAULT_HEADERS: list[tuple[str, str]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]

    def __init__(
        self,
        parent_size: int = 1800,
        child_size: int = 350,
        parent_overlap: int = 100,
        child_overlap: int = 60,
        encoding_name: str = "cl100k_base",
        headers_to_split_on: list[tuple[str, str]] | None = None,
        route_tables_to_row_chunks: bool = True,
    ) -> None:
        self.parent_size = parent_size
        self.child_size = child_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        self.encoding_name = encoding_name
        self.headers_to_split_on = headers_to_split_on or list(self.DEFAULT_HEADERS)
        # Opt-in: when True, a parent flagged is_table=True whose body
        # contains parseable markdown tables emits one child per table
        # row (header preserved) via :mod:`table_parser`. The PARENT
        # chunk still contains the full table text so the LLM can see
        # the entire table when expanded.
        self.route_tables_to_row_chunks = route_tables_to_row_chunks
        # Keys ordered by header depth (h1, h2, ...) — used to assemble the path.
        self._header_keys: list[str] = [key for _, key in self.headers_to_split_on]

        encoder = _get_encoder(self.encoding_name)
        self._length_fn = lambda x: _token_len(x, encoder)

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_size,
            chunk_overlap=self.parent_overlap,
            length_function=self._length_fn,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_size,
            chunk_overlap=self.child_overlap,
            length_function=self._length_fn,
        )

    # -- public API --------------------------------------------------------

    def split(
        self,
        text: str,
        *,
        file_id: str,
        key: str,
        doc_meta: dict[str, Any] | None = None,
    ) -> tuple[list[HierarchicalChunk], list[HierarchicalChunk]]:
        """Split ``text`` into parent and child chunks.

        Parameters
        ----------
        text:
            Raw markdown content. Empty / whitespace-only input yields two
            empty lists (no error).
        file_id:
            Stable identifier of the source document. Used as the namespace
            seed for chunk ids, so two re-ingests of the same content produce
            identical ids.
        key:
            Original object key (e.g. MinIO key). Stored on each chunk's
            ``extra`` payload for downstream filtering / debugging.
        doc_meta:
            Optional per-document metadata (e.g. ``{"campus": "co_so_1"}``)
            merged into every emitted chunk's ``extra``.

        Returns
        -------
        (parents, children):
            Two lists. Parents are top-level chunks (~``parent_size`` tokens).
            Each child carries the ``chunk_id`` of its owning parent in
            ``parent_id``.
        """
        if not text or not text.strip():
            return [], []

        doc_meta = dict(doc_meta or {})
        parents: list[HierarchicalChunk] = []
        children: list[HierarchicalChunk] = []
        parent_index = 0
        child_index = 0

        sections = self._header_splitter.split_text(text)

        # Synthetic-header fallback: when the source has no markdown headers
        # at all (common for plain prose / scraped Vietnamese docs / .txt
        # exported as .md), the splitter emits a single section with empty
        # metadata. Every chunk would then carry header_path=["(no header)"],
        # destroying the section signal used by the embedder and reranker.
        # Detect this case once and synthesize:
        #   h1 = filename (sans extension) from ``key``
        #   h2 = first meaningful line of the section body
        synth_h1: str | None = None
        if not any(
            (section.metadata or {}).get(k) for section in sections for k in self._header_keys
        ):
            synth_h1 = _derive_h1_from_key(key)

        for section in sections:
            body = (section.page_content or "").strip()
            if not body:
                continue

            if synth_h1 is not None:
                synth_h2 = _first_meaningful_line(body)
                header_path = [synth_h1] + ([synth_h2] if synth_h2 else [])
                if not header_path:
                    header_path = [_NO_HEADER]
            else:
                header_path = _build_header_path(section.metadata or {}, self._header_keys)
            is_table = _contains_table(body)
            section_label = header_path[-1]
            prefix = _header_prefix(header_path)

            # Parent split inside this section.
            if self._length_fn(body) <= self.parent_size:
                parent_bodies = [body]
            else:
                parent_bodies = self._parent_splitter.split_text(body)

            for parent_body in parent_bodies:
                parent_body = parent_body.strip()
                if not parent_body:
                    continue

                parent_content = prefix + parent_body
                parent_id = _stable_uuid(
                    file_id,
                    "parent",
                    _content_fingerprint(parent_content),
                    str(parent_index),
                )
                parent_extra: dict[str, Any] = {"key": key, **doc_meta}

                parent_chunk = HierarchicalChunk(
                    chunk_id=parent_id,
                    parent_id=None,
                    content=parent_content,
                    header_path=list(header_path),
                    section=section_label,
                    chunk_level="parent",
                    is_table=is_table,
                    extra=parent_extra,
                )
                parents.append(parent_chunk)
                parent_index += 1

                # Child split inside this parent. When routing tables to
                # row chunks is enabled AND this parent contains parseable
                # markdown tables, emit one child per data row (header
                # preserved). Falls back to recursive splitting if the
                # table parser finds nothing usable.
                row_child_bodies: list[str] = []
                if self.route_tables_to_row_chunks and is_table:
                    for table in find_tables(parent_body):
                        row_chunks = table_to_row_chunks(
                            table,
                            file_id=file_id,
                            table_index=parent_index,
                        )
                        row_child_bodies.extend(rc.content for rc in row_chunks)

                if row_child_bodies:
                    child_bodies = row_child_bodies
                elif self._length_fn(parent_body) <= self.child_size:
                    child_bodies = [parent_body]
                else:
                    child_bodies = self._child_splitter.split_text(parent_body)

                for child_body in child_bodies:
                    child_body = child_body.strip()
                    if not child_body:
                        continue

                    child_content = prefix + child_body
                    child_id = _stable_uuid(
                        file_id,
                        "child",
                        parent_id,
                        _content_fingerprint(child_content),
                        str(child_index),
                    )
                    child_extra: dict[str, Any] = {"key": key, **doc_meta}

                    children.append(
                        HierarchicalChunk(
                            chunk_id=child_id,
                            parent_id=parent_id,
                            content=child_content,
                            header_path=list(header_path),
                            section=section_label,
                            chunk_level="child",
                            is_table=is_table,
                            extra=child_extra,
                        )
                    )
                    child_index += 1

        return parents, children


__all__ = ["HierarchicalChunk", "HierarchicalSplitter"]
