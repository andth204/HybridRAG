"""LLM extractor for admission-score rows.

Reads a chunk of Vietnamese admission markdown (typically a score-table
section from ``Điểm 2024.md`` etc.) and returns a list of normalized
row dicts ready for ``AdmissionScoresRepo.upsert(**row)``.

Design choices:

* JSON-only output via ``response_format={"type": "json_object"}``.
  The system prompt asks the model to wrap the array in
  ``{"rows": [...]}`` because the OpenAI JSON mode rejects bare arrays.
* Defensive parsing: a malformed JSON blob, a missing year, or a major
  that doesn't resolve through the entity dictionary all degrade
  silently (warning log + skip), never raising.
* Empty ``OPENAI_API_KEY`` short-circuits to an empty list so unit
  tests and offline ingestion runs don't try to hit the network.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from openai import OpenAI

from src.config.settings import settings
from src.hybridrag.utils.entity_resolver import resolve

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You extract structured admission score data from Vietnamese university markdown text.
Return JSON only, wrapped in a single top-level object: {"rows": [ ... ]}.
Each item in "rows" must be:
{"campus": str, "faculty": str, "major": str, "major_code": str|null, "year": int, "method": str, "subject_combo": str|null, "score": float|null, "note": str|null}
Rules:
- year MUST be a 4-digit int (e.g. 2024).
- score is a float (e.g. 21.5). null when the cutoff is unknown / blank.
- major is the verbatim major name as it appears (e.g. "Công nghệ thông tin"). Do NOT canonicalize.
- major_code is the "Mã ngành" (e.g. "7480201") when present, null otherwise.
- method is one of: "THPT", "HSA", "HSG", "CCQT", "XHCT", "XTKH", "GDR", or "" when unspecified.
- subject_combo is e.g. "A00", "D01" when specified for a single combo; for multiple combos give them as a comma-separated string. null otherwise.
- campus, faculty default to empty strings when not stated.
Return {"rows": []} if no scores found. Output JSON ONLY, no commentary, no markdown fencing."""


# Whitelist of method codes the SYSTEM_PROMPT instructs the model to use.
# Used purely for telemetry / logging — we don't drop unknown codes, we
# just keep a warning trail so the prompt author can extend the list.
_KNOWN_METHODS = {"THPT", "HSA", "HSG", "CCQT", "XHCT", "XTKH", "GDR", ""}


def _make_client() -> OpenAI | None:
    """Build an OpenAI client when an API key is configured, else None."""
    key = (settings.OPENAI_API_KEY or "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)


def _call_llm(client: OpenAI, text: str) -> str:
    """Single LLM round-trip; returns the raw JSON string."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content or "{}"


def _parse_rows(raw: str) -> list[dict[str, Any]]:
    """Extract the ``rows`` array from the model's JSON response.

    The model is *supposed* to emit ``{"rows": [...]}`` but we also
    accept a bare list as a forgiving fallback in case it slips through
    the JSON-mode wrapping.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("scores extractor: model returned non-JSON: %r", raw[:200])
        return []
    if isinstance(payload, dict):
        rows = payload.get("rows", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        log.warning("scores extractor: unexpected payload type: %s", type(payload))
        return []
    return [r for r in rows if isinstance(r, dict)]


def _coerce_year(raw_year: Any) -> int | None:
    """Convert the model's ``year`` field to a strict 4-digit int."""
    if isinstance(raw_year, bool):  # bool is a subclass of int — exclude.
        return None
    if isinstance(raw_year, int):
        year = raw_year
    elif isinstance(raw_year, str):
        try:
            year = int(raw_year.strip())
        except ValueError:
            return None
    else:
        return None
    if year < 2000 or year > 2099:
        return None
    return year


def _coerce_score(raw_score: Any) -> float | None:
    """Convert the model's ``score`` field to a float (or None)."""
    if raw_score is None:
        return None
    if isinstance(raw_score, bool):
        return None
    if isinstance(raw_score, (int, float)):
        return float(raw_score)
    if isinstance(raw_score, str):
        cleaned = raw_score.strip().replace(",", ".")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _coerce_str(value: Any) -> str:
    """Return ``value`` as a stripped string ("" for missing/None/non-str)."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _coerce_optional_str(value: Any) -> str | None:
    """Return a stripped string or None when empty/missing."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_row(raw: dict[str, Any], *, source_file: str | None) -> dict[str, Any] | None:
    """Validate + canonicalize one model row. Returns ``None`` to drop it."""
    year = _coerce_year(raw.get("year"))
    if year is None:
        log.warning("scores extractor: dropping row without valid year: %r", raw)
        return None

    major_text = _coerce_str(raw.get("major"))
    if not major_text:
        log.warning("scores extractor: dropping row without major: %r", raw)
        return None

    rec = resolve(major_text, "major")
    if rec is None:
        log.warning(
            "scores extractor: major %r did not resolve through entity dict; skipping",
            major_text,
        )
        return None
    major_canonical = rec.get("canonical")
    if not isinstance(major_canonical, str):
        log.warning("scores extractor: resolved major lacks canonical: %r", rec)
        return None

    method = _coerce_str(raw.get("method")).upper() if isinstance(raw.get("method"), str) else ""
    if method and method not in _KNOWN_METHODS:
        log.info("scores extractor: unfamiliar method %r (kept verbatim)", method)

    return {
        "campus": _coerce_str(raw.get("campus")),
        "faculty": _coerce_str(raw.get("faculty")),
        "major_canonical": major_canonical,
        "major_code": _coerce_optional_str(raw.get("major_code")) or rec.get("code"),
        "year": year,
        "method": method,
        "subject_combo": _coerce_optional_str(raw.get("subject_combo")),
        "score": _coerce_score(raw.get("score")),
        "note": _coerce_optional_str(raw.get("note")),
        "source_file": source_file,
    }


def extract_scores_from_text(
    text: str,
    *,
    source_file: str | None = None,
    client: OpenAI | None = None,
) -> list[dict[str, Any]]:
    """Run the LLM extractor on ``text`` and return upsert-ready row dicts.

    Args:
        text:        The markdown chunk to extract from.
        source_file: Filename to stamp on each row (for provenance + the
                     ``delete_by_source`` reindex flow).
        client:      Override OpenAI client (used by tests).

    Returns:
        A list of dicts, each with keys matching
        :meth:`AdmissionScoresRepo.upsert` kwargs. May be empty.
    """
    if not text or not text.strip():
        return []

    client = client if client is not None else _make_client()
    if client is None:
        log.warning(
            "scores extractor: OPENAI_API_KEY not set; returning [] for %s",
            source_file or "<inline>",
        )
        return []

    try:
        raw = _call_llm(client, text)
    except Exception as exc:  # noqa: BLE001 — never let an LLM error crash ingest
        log.warning("scores extractor: LLM call failed: %r", exc)
        return []

    rows = _parse_rows(raw)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        result = _normalize_row(row, source_file=source_file)
        if result is not None:
            normalized.append(result)
    return normalized


__all__ = [
    "SYSTEM_PROMPT",
    "extract_scores_from_text",
]
