"""Free-form lookup tools backed by the Phase 3B structured store.

Callers pass user-facing strings ("CNTT", "cơ sở 1", "khối kinh tế"),
this module canonicalizes via the entity resolver and queries the
repos. Returns plain dicts so the dialogue layer (Phase 4) can send
them straight back to the LLM via tool-call results.
"""
from __future__ import annotations

from typing import Any, Optional

from src.hybridrag.kg.scores_repo import AdmissionScore, AdmissionScoresRepo
from src.hybridrag.kg.tuition_repo import TuitionRepo, TuitionRow
from src.hybridrag.utils.entity_resolver import resolve


# --------------------------------------------------------------------- #
# Resolution helpers
# --------------------------------------------------------------------- #
def _resolve_canonical(text: str | None, entity_type: str) -> Optional[str]:
    """Fuzzy-resolve ``text`` to its canonical key for ``entity_type``.

    Returns None for empty inputs or when no alias matches above the
    resolver's default similarity floor. Callers should treat ``None`` as
    "no filter" rather than "filter for nothing".
    """
    if not text:
        return None
    rec = resolve(text, entity_type)
    if rec is None:
        return None
    canonical = rec.get("canonical")
    return canonical if isinstance(canonical, str) else None


def _score_to_dict(row: AdmissionScore) -> dict[str, Any]:
    """Render an ``AdmissionScore`` as a JSON-friendly dict."""
    return {
        "id": row.id,
        "campus": row.campus,
        "faculty": row.faculty,
        "major_canonical": row.major_canonical,
        "major_code": row.major_code,
        "year": row.year,
        "method": row.method,
        "subject_combo": row.subject_combo,
        "score": row.score,
        "note": row.note,
        "source_file": row.source_file,
        "source_chunk_id": row.source_chunk_id,
    }


def _tuition_to_dict(row: TuitionRow) -> dict[str, Any]:
    """Render a ``TuitionRow`` as a JSON-friendly dict."""
    return {
        "id": row.id,
        "major_canonical": row.major_canonical,
        "year": row.year,
        "amount_vnd": row.amount_vnd,
        "unit": row.unit,
        "note": row.note,
        "source_file": row.source_file,
    }


# --------------------------------------------------------------------- #
# Public tool surface
# --------------------------------------------------------------------- #
def lookup_score(
    *,
    major: str | None = None,
    year: int | None = None,
    campus: str | None = None,
    faculty: str | None = None,
    method: str | None = None,
    limit: int = 5,
    repo: AdmissionScoresRepo | None = None,
) -> list[dict[str, Any]]:
    """Look up admission cutoffs by free-form major / campus / faculty.

    Args:
        major:   Free-form major name (e.g. "CNTT", "Công nghệ thông tin").
        year:    Optional 4-digit year filter.
        campus:  Optional campus name (e.g. "cơ sở 1").
        faculty: Optional faculty name (e.g. "khoa CNTT").
        method:  Optional method code (already canonical, e.g. "THPT");
                 not resolved through the entity dictionary because the
                 set of methods is small and stable.
        limit:   Maximum rows to return.
        repo:    Override repo (mainly for tests).

    Returns:
        A list of plain dicts ready to be JSON-serialized as a tool result.
        Empty list if no row matches.
    """
    major_canonical = _resolve_canonical(major, "major")
    if major and major_canonical is None:
        # Caller asked about something the dictionary doesn't know yet —
        # the structured store can't help. Return [] so the dialogue layer
        # falls back to RAG instead of pretending the row is missing.
        return []

    campus_canonical = _resolve_canonical(campus, "campus")
    faculty_canonical = _resolve_canonical(faculty, "faculty")

    repo = repo or AdmissionScoresRepo()
    rows = repo.lookup(
        major_canonical=major_canonical,
        year=year,
        campus=campus_canonical,
        faculty=faculty_canonical,
        method=method,
        limit=limit,
    )
    return [_score_to_dict(r) for r in rows]


def lookup_tuition(
    *,
    major: str | None = None,
    year: int | None = None,
    unit: str | None = None,
    limit: int = 5,
    repo: TuitionRepo | None = None,
) -> list[dict[str, Any]]:
    """Look up tuition rates by free-form major.

    Mirrors :func:`lookup_score` for the ``tuition`` table.
    """
    major_canonical = _resolve_canonical(major, "major")
    if major and major_canonical is None:
        return []

    repo = repo or TuitionRepo()
    rows = repo.lookup(
        major_canonical=major_canonical,
        year=year,
        unit=unit,
        limit=limit,
    )
    return [_tuition_to_dict(r) for r in rows]


def list_majors_by_campus(
    campus: str,
    *,
    repo: AdmissionScoresRepo | None = None,
) -> list[str]:
    """Distinct majors offered at a given campus (from ``admission_scores``).

    ``campus`` is resolved through the entity dictionary so callers can
    pass user-facing strings ("cơ sở Khoái Châu", "cs1"). Returns an
    empty list if the campus name can't be resolved.
    """
    campus_canonical = _resolve_canonical(campus, "campus")
    if campus_canonical is None:
        return []
    repo = repo or AdmissionScoresRepo()
    return repo.list_majors_by_campus(campus_canonical)


__all__ = [
    "list_majors_by_campus",
    "lookup_score",
    "lookup_tuition",
]
