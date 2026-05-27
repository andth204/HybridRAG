"""Ingest-time metadata extractor for HybridRAG chunks.

Given a chunk's content, its hierarchical header path, and the source
filename, produce a small ``dict`` of structured metadata fields that
get attached to the chunk before indexing into Weaviate. The output
keys are kept in sync with the ``DocChunk`` schema defined in
``ROADMAP_V2.md`` (campus / doc_type / year / faculty / major).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from src.hybridrag.utils.entity_resolver import resolve, resolve_all


# A year token: "20XX" with XX in 00-99 — covers 2000-2099 admission cycles.
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# Campus header heuristic: any "Cơ sở N" / "co so N" pattern with N in 1..9.
CAMPUS_HEADER_PATTERN = re.compile(r"c[ơo]\s*s[ởo]\s*([1-9])", re.IGNORECASE)

# Doc-type heuristic patterns keyed by canonical doc_type. We scan the
# filename (and, as fallback, the first header line) for these.
# Vietnamese filenames typically use ``_`` or ``-`` between syllables
# (e.g. ``tuyen_sinh_247.md``, ``hoc-phi.md``), so the separator class
# accepts whitespace, underscores, and hyphens.
_SEP = r"[\s_\-]*"
_FILENAME_DOCTYPE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"điểm|diem|score", re.IGNORECASE),                       "diem_chuan"),
    (re.compile(rf"học{_SEP}phí|hoc{_SEP}phi|tuition", re.IGNORECASE),    "hoc_phi"),
    (re.compile(rf"tuyển{_SEP}sinh|tuyen{_SEP}sinh|admission", re.IGNORECASE), "tuyen_sinh"),
    (re.compile(rf"học{_SEP}bổng|hoc{_SEP}bong|scholarship", re.IGNORECASE),  "hoc_bong"),
    (re.compile(rf"liên{_SEP}hệ|lien{_SEP}he|contact", re.IGNORECASE),    "lien_he"),
)

# Minimum WRatio (0-100) for faculty/major aggregation to count as a "dominant" match.
_FACULTY_MAJOR_MIN_SCORE = 85
_CAMPUS_MIN_SCORE = 80
_DOCTYPE_HEADER_MIN_SCORE = 85


def _strip_diacritics(s: str) -> str:
    """Best-effort stripping of Vietnamese diacritics; used for filename heuristics."""
    s = s.replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _find_year(*candidates: str) -> Optional[int]:
    """Return the earliest 4-digit 20XX match across the given strings."""
    years: list[int] = []
    for s in candidates:
        if not s:
            continue
        for m in YEAR_PATTERN.finditer(s):
            years.append(int(m.group(1)))
    if not years:
        return None
    return min(years)


def _campus_from_header_regex(header_path: list[str]) -> Optional[str]:
    """Quick regex pass over header_path: '... Cơ sở 1 ...' → 'co_so_1'."""
    for h in header_path:
        if not h:
            continue
        m = CAMPUS_HEADER_PATTERN.search(h)
        if m:
            return f"co_so_{m.group(1)}"
    return None


def _doc_type_from_filename(filename: str) -> Optional[str]:
    """Heuristic doc_type lookup from filename (with-and-without diacritics)."""
    if not filename:
        return None
    candidates = [filename, _strip_diacritics(filename)]
    for cand in candidates:
        for pattern, dtype in _FILENAME_DOCTYPE_PATTERNS:
            if pattern.search(cand):
                return dtype
    return None


def _top_canonical(records: list[dict[str, Any]]) -> Optional[str]:
    """Pick the first record's canonical key, or None if the list is empty."""
    if not records:
        return None
    return records[0].get("canonical")


def extract_metadata(
    *,
    text: str,
    header_path: Optional[list[str]] = None,
    filename: Optional[str] = None,
) -> dict[str, Any]:
    """Extract structured metadata from a chunk for Weaviate indexing.

    Args:
        text:        The chunk's body text.
        header_path: Markdown header ancestors (e.g. ``["Cơ sở 1", "Liên hệ"]``).
        filename:    Source filename, used to derive year + doc_type cheaply.

    Returns:
        A dict containing some subset of:
          - ``year``      (int): earliest 20XX seen in header / filename / text
          - ``campus``    (str): canonical campus key, e.g. ``"co_so_1"``
          - ``doc_type``  (str): canonical doc_type key, e.g. ``"diem_chuan"``
          - ``faculty``   (str): canonical faculty key
          - ``major``     (str): canonical major key
        Empty / not-found keys are omitted (rather than set to ``None``)
        so the dict can be merged directly into the Weaviate object.
    """
    header_path = list(header_path or [])
    filename = filename or ""
    text = text or ""

    meta: dict[str, Any] = {}

    # ----- Year ----------------------------------------------------- #
    header_joined = " | ".join(header_path)
    year = _find_year(header_joined, filename, text[:500])
    if year is not None:
        meta["year"] = year

    # ----- Campus --------------------------------------------------- #
    campus = _campus_from_header_regex(header_path)
    if campus is None and header_joined:
        hits = resolve_all(
            header_joined,
            min_score=_CAMPUS_MIN_SCORE,
            entity_types=["campus"],
        )
        campus = _top_canonical(hits.get("campus", []))
    if campus is None:
        head_text = text[:500]
        if head_text:
            hits = resolve_all(
                head_text,
                min_score=_CAMPUS_MIN_SCORE,
                entity_types=["campus"],
            )
            campus = _top_canonical(hits.get("campus", []))
    if campus:
        meta["campus"] = campus

    # ----- Doc type ------------------------------------------------- #
    doc_type = _doc_type_from_filename(filename)
    if doc_type is None and header_path:
        # Try resolving the first header against the doc_type corpus.
        first_header = header_path[0]
        rec = resolve(first_header, "doc_type", min_score=_DOCTYPE_HEADER_MIN_SCORE)
        if rec is not None:
            doc_type = rec.get("canonical")
    if doc_type:
        meta["doc_type"] = doc_type

    # ----- Faculty / major ----------------------------------------- #
    fm_corpus = (header_joined + "\n" + text[:1000]).strip()
    if fm_corpus:
        hits = resolve_all(
            fm_corpus,
            min_score=_FACULTY_MAJOR_MIN_SCORE,
            entity_types=["faculty", "major"],
        )
        faculty = _top_canonical(hits.get("faculty", []))
        major = _top_canonical(hits.get("major", []))
        if faculty:
            meta["faculty"] = faculty
        if major:
            meta["major"] = major

    return meta


__all__ = [
    "extract_metadata",
    "YEAR_PATTERN",
    "CAMPUS_HEADER_PATTERN",
]
