"""Exact-match answer cache (v3.5).

Lightweight in-process TTLCache that lets us short-circuit the full
retrieve → rerank → compose pipeline when the SAME user question (with
the same intent + slot frame) was answered recently AND the KG hasn't
changed since.

Why not semantic / vector cache?
--------------------------------
Admission Q&A traffic has a tight head distribution — the same handful
of "Điểm chuẩn CNTT 2025?" queries get asked over and over. Exact match
on the normalized query + slot signature is enough to capture that
pattern without the complexity (and Redis dependency) of a vector store.

Cache key
---------
``md5(intent + sorted_slot_canonical + normalized_query + kg_version)``

The ``kg_version`` component is a cheap digest of
``admission_scores`` / ``tuition`` row counts + max(created_at). When
admin re-runs the score extractor, the version flips and every cached
answer that depended on KG data automatically invalidates without an
explicit cache.clear() call.

Behaviour contract
------------------
* All public functions are no-ops when
  ``settings.ANSWER_CACHE_ENABLED`` is False (returns None / does
  nothing). Lets ops disable the cache with a single flag flip if
  something misbehaves.
* The cache is thread-safe (cachetools.TTLCache is not, so we wrap
  every access in a ``threading.Lock``).
* Failures inside this module are swallowed and logged — they must
  never break the request pipeline.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Optional

from cachetools import TTLCache

from src.config.settings import settings


log = logging.getLogger(__name__)


# How often to re-check the KG version digest. Postgres roundtrip is
# fast (~5-20 ms) but doing it on every chat turn would add up — once
# a minute is plenty for an admin-driven KG that rarely changes.
_KG_VERSION_TTL_SEC: float = 60.0


_lock = threading.Lock()
_cache: Optional[TTLCache] = None
_kg_version: str = ""
_kg_version_checked_at: float = 0.0


def _get_cache() -> TTLCache:
    """Lazy-construct the TTLCache so settings can be tuned at boot."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = TTLCache(
                maxsize=int(getattr(settings, "ANSWER_CACHE_MAXSIZE", 1024)),
                ttl=int(getattr(settings, "ANSWER_CACHE_TTL_SEC", 86400)),
            )
        return _cache


def _compute_kg_version() -> str:
    """Cheap digest of the KG state for cache invalidation."""
    try:
        # Local import keeps this module loadable in test environments
        # without psycopg2 / DB credentials.
        from src.hybridrag.utils.db_pool import borrow

        with borrow(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM admission_scores),
                      (SELECT COUNT(*) FROM tuition),
                      (SELECT EXTRACT(EPOCH FROM COALESCE(MAX(created_at), 'epoch'))
                         FROM admission_scores),
                      (SELECT EXTRACT(EPOCH FROM COALESCE(MAX(created_at), 'epoch'))
                         FROM tuition)
                    """
                )
                row = cur.fetchone()
        return hashlib.md5(repr(row).encode("utf-8")).hexdigest()[:12]
    except Exception as exc:  # noqa: BLE001
        log.debug("answer_cache: KG version probe failed: %s", exc)
        return "unknown"


def get_kg_version(*, force: bool = False) -> str:
    """Return the cached KG version digest. Refreshes every ~60s."""
    global _kg_version, _kg_version_checked_at
    now = time.time()
    if not force and _kg_version and (now - _kg_version_checked_at) < _KG_VERSION_TTL_SEC:
        return _kg_version
    digest = _compute_kg_version()
    _kg_version = digest
    _kg_version_checked_at = now
    return digest


def invalidate_kg_version() -> None:
    """Force a refresh on the next call (use after a known KG write)."""
    global _kg_version_checked_at
    _kg_version_checked_at = 0.0


def _normalize_query(text: str) -> str:
    """Lowercase + whitespace-collapse the query for stable hashing."""
    return " ".join((text or "").lower().split())


def _slot_signature(slots: dict[str, Any] | None) -> str:
    """Stable string representation of the relevant slot frame.

    Only includes slots whose value is not None / empty so a missing
    slot doesn't differentiate from an absent one.
    """
    if not slots:
        return ""
    parts: list[str] = []
    for k in sorted(slots.keys()):
        v = slots[k]
        if v is None or v == "":
            continue
        parts.append(f"{k}={v}")
    return "|".join(parts)


def make_key(intent: str, slots: dict[str, Any] | None, query: str) -> str:
    """Return the deterministic cache key for this turn."""
    raw = (
        f"intent={intent}::"
        f"slots={_slot_signature(slots)}::"
        f"q={_normalize_query(query)}::"
        f"kg={get_kg_version()}"
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def lookup(
    intent: str,
    slots: dict[str, Any] | None,
    query: str,
) -> Optional[dict[str, Any]]:
    """Return the cached answer envelope for this (intent, slots, query)."""
    if not getattr(settings, "ANSWER_CACHE_ENABLED", False):
        return None
    try:
        key = make_key(intent, slots, query)
        cache = _get_cache()
        with _lock:
            return cache.get(key)
    except Exception as exc:  # noqa: BLE001
        log.debug("answer_cache.lookup failed: %s", exc)
        return None


def store(
    intent: str,
    slots: dict[str, Any] | None,
    query: str,
    *,
    answer: str,
    sources: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist this turn's answer envelope for future identical queries."""
    if not getattr(settings, "ANSWER_CACHE_ENABLED", False):
        return
    if not answer:
        return
    try:
        key = make_key(intent, slots, query)
        cache = _get_cache()
        payload = {
            "answer": answer,
            "sources": list(sources or []),
            "metadata": dict(metadata or {}),
        }
        with _lock:
            cache[key] = payload
    except Exception as exc:  # noqa: BLE001
        log.debug("answer_cache.store failed: %s", exc)


def stats() -> dict[str, Any]:
    """Diagnostic counters for /metrics dashboards."""
    cache = _get_cache()
    with _lock:
        return {
            "size": len(cache),
            "maxsize": cache.maxsize,
            "ttl": cache.ttl,
            "kg_version": _kg_version or "unset",
        }


__all__ = [
    "lookup",
    "store",
    "make_key",
    "get_kg_version",
    "invalidate_kg_version",
    "stats",
]
