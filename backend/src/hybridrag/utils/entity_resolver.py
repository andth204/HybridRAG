"""Fuzzy entity resolver backed by ``backend/data/dict/entities.yaml``.

Resolves free-form Vietnamese / English spans to a canonical entity
record (campus, faculty, major, doc_type, university). Used at ingest
time to populate chunk metadata and at query time to derive Weaviate
filter values from the user's question.

Matching is diacritic- and case-insensitive (via ``_normalize_key``)
and uses ``rapidfuzz.fuzz.WRatio`` so an alias like ``CNTT`` matches
the full ``công nghệ thông tin`` corpus entry.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from rapidfuzz import fuzz, process

from src.config.settings import settings
from src.hybridrag.utils.synonyms import _normalize_key


EntityType = str  # "campus" | "faculty" | "major" | "doc_type" | "university"

# Default minimum WRatio score to accept a match. WRatio returns 0-100;
# 80 is a good "clearly the same entity, allowing for typo / spacing" floor.
DEFAULT_MIN_SCORE = 80

# Sliding-window upper bound for ``resolve_all``. 4 tokens is enough to
# catch "công nghệ kỹ thuật điện" / "trường đại học sư phạm" without
# blowing up the search space.
DEFAULT_MAX_NGRAM = 4


# -------------------------------------------------------------------- #
# Loader
# -------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _load_entities(path: str | None = None) -> dict[EntityType, list[dict[str, Any]]]:
    """Load ``entities.yaml`` once per process and return the raw structure."""
    fp = Path(path) if path else (settings.BASE_DIR / "data" / "dict" / "entities.yaml")
    if not fp.exists():
        return {}
    try:
        data = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(data, dict):
        return {}
    for etype, records in data.items():
        if not isinstance(records, list):
            continue
        clean: list[dict[str, Any]] = []
        for rec in records:
            if isinstance(rec, dict) and rec.get("canonical"):
                clean.append(rec)
        if clean:
            out[etype] = clean
    return out


@functools.lru_cache(maxsize=8)
def _build_corpus(entity_type: EntityType) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Build a flat ``(normalized_alias, record_index)`` corpus for an entity type.

    Returns two parallel tuples (immutable so the result can be cached):
      - ``aliases``       — normalized strings to fuzzy-match against
      - ``record_index``  — index into ``_load_entities()[entity_type]``
    """
    records = _load_entities().get(entity_type, [])
    aliases: list[str] = []
    indices: list[int] = []
    for idx, rec in enumerate(records):
        candidates: list[str] = []
        for field in ("canonical", "display"):
            value = rec.get(field)
            if isinstance(value, str):
                candidates.append(value)
        # canonical is snake_case — replace underscores so fuzz scores
        # surface forms like "co so 1" against "co_so_1".
        canonical = rec.get("canonical")
        if isinstance(canonical, str) and "_" in canonical:
            candidates.append(canonical.replace("_", " "))
        for a in rec.get("aliases", []) or []:
            if isinstance(a, str):
                candidates.append(a)
        # Optional: also match against the `code` field for majors.
        code = rec.get("code")
        if isinstance(code, str):
            candidates.append(code)
        seen: set[str] = set()
        for cand in candidates:
            norm = _normalize_key(cand)
            if norm and norm not in seen:
                seen.add(norm)
                aliases.append(norm)
                indices.append(idx)
    return tuple(aliases), tuple(indices)


def reload_entities() -> None:
    """Drop the cached entity table (useful for tests that patch the YAML)."""
    _load_entities.cache_clear()
    _build_corpus.cache_clear()


# -------------------------------------------------------------------- #
# Public API
# -------------------------------------------------------------------- #
def list_canonical(entity_type: EntityType) -> list[str]:
    """Return all canonical keys defined for a given entity type."""
    return [rec["canonical"] for rec in _load_entities().get(entity_type, [])]


def resolve(
    text: str,
    entity_type: EntityType,
    *,
    min_score: int = DEFAULT_MIN_SCORE,
) -> Optional[dict[str, Any]]:
    """Resolve a free-form span to a canonical entity record.

    Args:
        text:        Free-form text (already isolated to a single entity span).
        entity_type: One of the top-level keys in ``entities.yaml``
                     ("campus", "faculty", "major", "doc_type", "university").
        min_score:   Lower bound on ``rapidfuzz.fuzz.WRatio`` (0-100) to accept.

    Returns:
        The full entity record dict (with ``canonical``/``display``/``aliases``),
        or ``None`` if no alias scores at or above ``min_score``.
    """
    if not text:
        return None
    aliases, indices = _build_corpus(entity_type)
    if not aliases:
        return None

    query = _normalize_key(text)
    if not query:
        return None

    match = process.extractOne(query, aliases, scorer=fuzz.WRatio)
    if match is None:
        return None
    _alias, score, pos = match
    if score < min_score:
        return None
    record = _load_entities()[entity_type][indices[pos]]
    return record


def resolve_all(
    text: str,
    *,
    min_score: int = DEFAULT_MIN_SCORE,
    max_ngram: int = DEFAULT_MAX_NGRAM,
    entity_types: Optional[list[EntityType]] = None,
) -> dict[EntityType, list[dict[str, Any]]]:
    """Scan a free-form passage and return ALL entity hits across types.

    Implementation: sliding window of 1..``max_ngram`` tokens; each
    candidate span is fuzzy-matched against every entity-type corpus;
    the best record per (type, canonical) pair is kept (highest score).

    Args:
        text:         Free-form text (a chunk, a header path, a user query).
        min_score:    Minimum WRatio score for a window to be considered a hit.
        max_ngram:    Largest sliding-window size to consider.
        entity_types: Restrict scanning to a subset of entity types.

    Returns:
        ``{entity_type: [record, ...]}``, with records sorted by descending
        score within each type. Empty types are omitted.
    """
    if not text:
        return {}

    types = entity_types or list(_load_entities().keys())
    if not types:
        return {}

    tokens = re.findall(r"\S+", text, flags=re.UNICODE)
    if not tokens:
        return {}

    n = len(tokens)
    window_cap = min(max_ngram, n)

    # (etype, canonical) → (score, record)
    best: dict[tuple[EntityType, str], tuple[float, dict[str, Any]]] = {}

    for size in range(1, window_cap + 1):
        for start in range(0, n - size + 1):
            span = " ".join(tokens[start : start + size])
            query = _normalize_key(span)
            if not query:
                continue
            for etype in types:
                aliases, indices = _build_corpus(etype)
                if not aliases:
                    continue
                match = process.extractOne(query, aliases, scorer=fuzz.WRatio)
                if match is None:
                    continue
                _alias, score, pos = match
                if score < min_score:
                    continue
                record = _load_entities()[etype][indices[pos]]
                key = (etype, record["canonical"])
                prev = best.get(key)
                if prev is None or score > prev[0]:
                    best[key] = (score, record)

    out: dict[EntityType, list[dict[str, Any]]] = {}
    for (etype, _canonical), (_score, record) in best.items():
        out.setdefault(etype, []).append(record)
    # Sort each list by score desc — preserve highest-confidence first.
    for etype, recs in out.items():
        recs.sort(
            key=lambda r: -max(
                (s for (t, c), (s, _) in best.items() if t == etype and c == r["canonical"]),
                default=0,
            )
        )
    return out


__all__ = [
    "EntityType",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_MAX_NGRAM",
    "_load_entities",
    "_build_corpus",
    "reload_entities",
    "list_canonical",
    "resolve",
    "resolve_all",
]
