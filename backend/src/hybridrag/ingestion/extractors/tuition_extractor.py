"""LLM extractor for tuition rows.

Mirrors :mod:`scores_extractor`, swapping in a prompt that targets the
``tuition`` table (major / year / amount_vnd / unit / note).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from src.config.settings import settings
from src.hybridrag.utils.entity_resolver import resolve

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You extract structured tuition data from Vietnamese university markdown text.
Return JSON only, wrapped in a single top-level object: {"rows": [ ... ]}.
Each item in "rows" must be:
{"major": str, "year": int, "amount_vnd": int|null, "unit": "per_credit"|"per_semester"|"per_year"|"per_month", "note": str|null}
Rules:
- year MUST be a 4-digit int (e.g. 2025).
- amount_vnd is an integer number of Vietnamese dong. Strip dots/commas/spaces ("1.790.000" -> 1790000). null when unknown.
- major is the verbatim major name OR a group description ("Nhóm ngành Công nghệ thông tin"). The downstream resolver will canonicalize it; do NOT canonicalize.
- unit MUST be one of "per_credit" (mỗi tín chỉ), "per_semester" (mỗi học kỳ), "per_year" (mỗi năm), "per_month" (mỗi tháng). Choose the closest match from the source.
- note is a free-form remark in Vietnamese (e.g. "chương trình tiếng Anh"); null otherwise.
Return {"rows": []} if no tuition data found. Output JSON ONLY, no commentary, no markdown fencing."""


_KNOWN_UNITS = {"per_credit", "per_semester", "per_year", "per_month"}


def _make_client() -> OpenAI | None:
    """Build an OpenAI client when an API key is configured, else None."""
    key = (settings.OPENAI_API_KEY or "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)


def _call_llm(client: OpenAI, text: str) -> str:
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
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("tuition extractor: model returned non-JSON: %r", raw[:200])
        return []
    if isinstance(payload, dict):
        rows = payload.get("rows", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        log.warning("tuition extractor: unexpected payload type: %s", type(payload))
        return []
    return [r for r in rows if isinstance(r, dict)]


def _coerce_year(raw_year: Any) -> int | None:
    if isinstance(raw_year, bool):
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


def _coerce_amount(raw_amount: Any) -> int | None:
    """Best-effort cast to ``int`` (VND), tolerating dot/comma group separators."""
    if raw_amount is None:
        return None
    if isinstance(raw_amount, bool):
        return None
    if isinstance(raw_amount, int):
        return raw_amount
    if isinstance(raw_amount, float):
        return int(raw_amount)
    if isinstance(raw_amount, str):
        cleaned = (
            raw_amount.strip()
            .replace(".", "")
            .replace(",", "")
            .replace(" ", "")
            .replace("đ", "")
            .replace("VND", "")
            .replace("vnd", "")
        )
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def _coerce_unit(raw_unit: Any) -> str:
    """Pin the unit to the known whitelist; default to ``per_credit``."""
    if isinstance(raw_unit, str) and raw_unit.strip() in _KNOWN_UNITS:
        return raw_unit.strip()
    return "per_credit"


def _coerce_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_row(raw: dict[str, Any], *, source_file: str | None) -> dict[str, Any] | None:
    year = _coerce_year(raw.get("year"))
    if year is None:
        log.warning("tuition extractor: dropping row without valid year: %r", raw)
        return None

    major_text = raw.get("major")
    if not isinstance(major_text, str) or not major_text.strip():
        log.warning("tuition extractor: dropping row without major: %r", raw)
        return None

    rec = resolve(major_text, "major")
    if rec is None:
        log.warning(
            "tuition extractor: major %r did not resolve; skipping", major_text,
        )
        return None
    major_canonical = rec.get("canonical")
    if not isinstance(major_canonical, str):
        log.warning("tuition extractor: resolved major lacks canonical: %r", rec)
        return None

    return {
        "major_canonical": major_canonical,
        "year": year,
        "amount_vnd": _coerce_amount(raw.get("amount_vnd")),
        "unit": _coerce_unit(raw.get("unit")),
        "note": _coerce_optional_str(raw.get("note")),
        "source_file": source_file,
    }


def extract_tuition_from_text(
    text: str,
    *,
    source_file: str | None = None,
    client: OpenAI | None = None,
) -> list[dict[str, Any]]:
    """Run the LLM extractor on ``text`` and return upsert-ready row dicts.

    Mirrors :func:`extract_scores_from_text` for the ``tuition`` table.
    """
    if not text or not text.strip():
        return []

    client = client if client is not None else _make_client()
    if client is None:
        log.warning(
            "tuition extractor: OPENAI_API_KEY not set; returning [] for %s",
            source_file or "<inline>",
        )
        return []

    try:
        raw = _call_llm(client, text)
    except Exception as exc:  # noqa: BLE001 — never let an LLM error crash ingest
        log.warning("tuition extractor: LLM call failed: %r", exc)
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
    "extract_tuition_from_text",
]
