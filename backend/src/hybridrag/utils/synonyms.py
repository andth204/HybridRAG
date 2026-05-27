"""Vietnamese synonym normalizer for the UTEHY admissions chatbot.

Loads ``backend/data/dict/synonyms_vn.yaml`` lazily and exposes a
``normalize_text`` helper that rewrites known abbreviations / variants
("CNTT", "diem chuan", "DHSPKT HY", ...) into their canonical Vietnamese
phrase. Designed for use at both index time (chunk text → expanded form
to feed BM25 / embeddings) and query time (user input → expanded form
for the rewriter and Weaviate hybrid search).
"""
from __future__ import annotations

import functools
import re
import unicodedata
from pathlib import Path
from typing import Optional

import yaml

from src.config.settings import settings


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #
def _normalize_key(s: str) -> str:
    """Lowercase, strip, NFD-strip-accents, đ→d, collapse whitespace.

    Used both to build the synonym lookup table keys and to look up
    arbitrary candidate spans from caller input. Stable across input
    encodings — "CNTT", "cntt", "Cntt" all map to "cntt"; "đại học" and
    "dai hoc" both map to "dai hoc".
    """
    if not s:
        return ""
    s = s.lower().strip()
    # Replace đ/Đ explicitly because NFD does not decompose them.
    s = s.replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s


# -------------------------------------------------------------------- #
# Lazy-loaded synonym map
# -------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _load_synonyms(path: str | None = None) -> dict[str, str]:
    """Load ``synonyms_vn.yaml`` once per process.

    Returns a dict whose **keys are normalized** (via ``_normalize_key``)
    so callers can look up arbitrary candidate spans without re-normalizing.
    """
    fp = Path(path) if path else (settings.BASE_DIR / "data" / "dict" / "synonyms_vn.yaml")
    if not fp.exists():
        return {}
    try:
        data = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    raw = data.get("synonyms") or {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        key = _normalize_key(str(k))
        if key:
            out[key] = str(v)
    return out


def _max_ngram_size(syn: dict[str, str]) -> int:
    """Largest token count among synonym keys (caps the sliding window)."""
    if not syn:
        return 1
    return max((len(k.split()) for k in syn), default=1)


# -------------------------------------------------------------------- #
# Public API
# -------------------------------------------------------------------- #
_TOKEN_PATTERN = re.compile(r"\S+", re.UNICODE)


def normalize_text(text: str, *, expand: bool = True) -> str:
    """Replace known synonyms in ``text`` with their canonical form.

    Matching is case-insensitive and diacritic-insensitive. The original
    whitespace structure (single-space-separated tokens) is preserved;
    unmatched tokens are emitted verbatim with their original casing /
    diacritics. Greedy longest-match: prefers 3-token spans over 2-token,
    over single tokens.

    Args:
        text:    Input string (Vietnamese, possibly mixed with English).
        expand:  If ``False``, returns the input unchanged.

    Returns:
        The expanded string; identical length-preservation is *not*
        guaranteed because some canonical forms are longer than their
        abbreviations (e.g. ``CNTT`` → ``công nghệ thông tin``).
    """
    if not text or not expand:
        return text

    syn = _load_synonyms()
    if not syn:
        return text

    # Tokenize by whitespace, keeping the surface form for unmatched tokens.
    tokens = _TOKEN_PATTERN.findall(text)
    if not tokens:
        return text

    max_n = min(_max_ngram_size(syn), len(tokens))
    if max_n < 1:
        return text

    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        matched = False
        # Try the largest n-gram first, down to single token.
        for size in range(min(max_n, n - i), 0, -1):
            span = " ".join(tokens[i : i + size])
            key = _normalize_key(span)
            if key in syn:
                out.append(syn[key])
                i += size
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1

    return " ".join(out)


def normalize_query(query: str) -> str:
    """Public convenience wrapper for query-time normalization."""
    return normalize_text(query, expand=True)


def reload_synonyms() -> None:
    """Drop the cached synonym map (useful for tests that patch the YAML)."""
    _load_synonyms.cache_clear()


__all__ = [
    "_normalize_key",
    "_load_synonyms",
    "normalize_text",
    "normalize_query",
    "reload_synonyms",
]
