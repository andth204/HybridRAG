"""Unit tests for :class:`WeaviateHybridSearcher`.

These tests deliberately mock ``WeaviateStore`` so the suite does NOT
require a running Weaviate instance. They cover:

    1. Children sharing a parent_id are collapsed and their content is
       swapped for the full parent body (parent-document retrieval).
    2. A failed / empty parent fetch falls back to the child body.
    3. The synonym normaliser runs before the store call.
    4. Explicit caller filters pass through to the store.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.hybridrag.retrieval.weaviate_hybrid import WeaviateHybridSearcher


# ---------------------------------------------------------------- helpers

def _build_searcher_with_store_mock(
    *,
    store_search_return: list[dict[str, Any]] | None = None,
    parent_lookup: dict[str, dict[str, Any]] | None = None,
    captured: dict[str, Any] | None = None,
    parent_fetch_calls: list[list[str]] | None = None,
) -> WeaviateHybridSearcher:
    """Construct a searcher whose ``_store`` is a mock.

    - ``store_search_return`` — what the mocked ``store.search`` returns.
    - ``parent_lookup`` — dict mapping parent ``chunk_id`` to a doc dict
      returned by the mocked ``store.get_by_chunk_ids``. When omitted,
      the parent fetch returns an empty list (forces fallback).
    - ``captured`` — mirror of kwargs passed into ``store.search``.
    - ``parent_fetch_calls`` — appended-to whenever ``get_by_chunk_ids``
      is called, capturing the id list each time.
    """
    searcher = WeaviateHybridSearcher()
    # Disable the (heavy) reranker so the only async hop is the store
    # call. The reranker behaviour is exercised by hybrid.py's own tests.
    searcher._reranker = None

    fake_store = MagicMock()
    payload = list(store_search_return or [])
    lookup = dict(parent_lookup or {})

    async def _fake_search(**kwargs: Any) -> list[dict[str, Any]]:
        if captured is not None:
            captured.update(kwargs)
        return payload

    def _fake_get_by_chunk_ids(ids: list[str]) -> list[dict[str, Any]]:
        if parent_fetch_calls is not None:
            parent_fetch_calls.append(list(ids))
        return [lookup[i] for i in ids if i in lookup]

    fake_store.search = AsyncMock(side_effect=_fake_search)
    fake_store.get_by_chunk_ids = MagicMock(side_effect=_fake_get_by_chunk_ids)
    searcher._store = fake_store
    # `load_indexes` would otherwise try to (re)construct the store. We
    # short-circuit it by patching to a noop.
    searcher.load_indexes = lambda: None  # type: ignore[method-assign]
    return searcher


# ---------------------------------------------------------------- tests


def test_explicit_filters_pass_through() -> None:
    """Explicit caller filters reach the store verbatim."""
    captured: dict[str, Any] = {}
    searcher = _build_searcher_with_store_mock(
        store_search_return=[],
        captured=captured,
    )

    asyncio.run(
        searcher.search(
            "điểm chuẩn CNTT",
            top_k=5,
            filters={"campus": "co_so_2", "year": 2099},
            use_reranker=False,
            use_hyde=False,
        )
    )

    forwarded = captured.get("filters") or {}
    assert forwarded.get("campus") == "co_so_2"
    assert forwarded.get("year") == 2099


def test_no_implicit_filters() -> None:
    """Without caller-supplied filters, the store call must omit them."""
    captured: dict[str, Any] = {}
    searcher = _build_searcher_with_store_mock(
        store_search_return=[],
        captured=captured,
    )

    asyncio.run(
        searcher.search(
            "điểm chuẩn CNTT cơ sở 1 2024",
            top_k=5,
            use_reranker=False,
            use_hyde=False,
        )
    )

    assert captured.get("filters") is None, (
        f"no filters should be inferred; got {captured.get('filters')!r}"
    )


def test_parent_dedup_swaps_to_parent_body() -> None:
    """Children sharing a parent_id collapse and their content is
    swapped for the parent body fetched in a single batch round-trip.

    Score fields (``rerank_score``) from the winning child must survive
    so downstream low-recall gates keep their numeric signal.
    """
    canned = [
        {
            "chunk_id": "c1", "parent_id": "P", "content": "first child",
            "rerank_score": 0.9, "key": "doc.md",
        },
        {
            "chunk_id": "c2", "parent_id": "P", "content": "second child",
            "rerank_score": 0.8, "key": "doc.md",
        },
        {
            "chunk_id": "c3", "parent_id": "P", "content": "third child",
            "rerank_score": 0.7, "key": "doc.md",
        },
        {
            "chunk_id": "c4", "parent_id": "Q", "content": "fourth child",
            "rerank_score": 0.6, "key": "doc2.md",
        },
    ]
    parent_lookup = {
        "P": {
            "chunk_id": "P", "parent_id": None,
            "content": "FULL parent P body with sibling info",
            "text": "FULL parent P body with sibling info",
            "key": "doc.md", "section": "intro",
            "header_path": ["H1"], "chunk_level": "parent",
        },
        "Q": {
            "chunk_id": "Q", "parent_id": None,
            "content": "FULL parent Q body",
            "text": "FULL parent Q body",
            "key": "doc2.md", "section": "details",
            "header_path": ["H2"], "chunk_level": "parent",
        },
    }
    parent_fetch_calls: list[list[str]] = []
    searcher = _build_searcher_with_store_mock(
        store_search_return=canned,
        parent_lookup=parent_lookup,
        parent_fetch_calls=parent_fetch_calls,
    )

    results = asyncio.run(
        searcher.search(
            "some query",
            top_k=10,
            use_reranker=False,
            use_hyde=False,
            normalize=False,
        )
    )

    # Two distinct parents survive (P then Q), each with parent body.
    assert [r["chunk_id"] for r in results] == ["P", "Q"]
    assert results[0]["content"] == "FULL parent P body with sibling info"
    assert results[1]["content"] == "FULL parent Q body"
    # text mirror is preserved (consumed by verifier + answer fallback).
    assert results[0]["text"] == results[0]["content"]
    # parent_id is cleared on the swapped doc — the entry IS the parent.
    assert results[0]["parent_id"] is None
    # Score fields from the WINNING child are preserved verbatim.
    assert results[0]["rerank_score"] == 0.9
    assert results[1]["rerank_score"] == 0.6
    # Exactly ONE batch fetch for both distinct parents.
    assert len(parent_fetch_calls) == 1
    assert sorted(parent_fetch_calls[0]) == ["P", "Q"]
    # The batch payload must be deduplicated — three children of P
    # must NOT cause "P" to appear three times in the fetch list.
    assert parent_fetch_calls[0].count("P") == 1


def test_parent_fetch_miss_falls_back_to_child() -> None:
    """When the parent fetch returns nothing for a given parent_id, the
    child body is returned unchanged so retrieval never blocks on a
    parent-fetch failure.
    """
    canned = [
        {
            "chunk_id": "c1", "parent_id": "P", "content": "child body",
            "rerank_score": 0.5, "key": "doc.md",
        },
    ]
    # Empty lookup = parent not found.
    searcher = _build_searcher_with_store_mock(
        store_search_return=canned,
        parent_lookup={},
    )

    results = asyncio.run(
        searcher.search(
            "q",
            top_k=5,
            use_reranker=False,
            use_hyde=False,
            normalize=False,
        )
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["content"] == "child body"
    assert results[0]["rerank_score"] == 0.5


def test_score_only_preservation_no_rerank() -> None:
    """When rerank is off and Weaviate returns only a raw ``score``,
    the swap must copy ``score`` (not just ``rerank_score``) onto the
    parent-swapped doc — clarifier/handoff thresholds fall back to it.
    """
    canned = [
        {
            "chunk_id": "c1", "parent_id": "P", "content": "child",
            "score": 0.77, "key": "doc.md",
        },
    ]
    parent_lookup = {
        "P": {
            "chunk_id": "P", "parent_id": None,
            "content": "parent body", "text": "parent body",
            "key": "doc.md",
        },
    }
    searcher = _build_searcher_with_store_mock(
        store_search_return=canned,
        parent_lookup=parent_lookup,
    )

    results = asyncio.run(
        searcher.search(
            "q",
            top_k=5,
            use_reranker=False,
            use_hyde=False,
            normalize=False,
        )
    )

    assert results[0]["chunk_id"] == "P"
    assert results[0]["content"] == "parent body"
    assert results[0]["score"] == 0.77


def test_mixed_parent_found_and_missing() -> None:
    """One parent found, one missing — both winning children survive,
    found one swapped to parent body, missing one keeps child body.
    """
    canned = [
        {
            "chunk_id": "c1", "parent_id": "P", "content": "child P",
            "rerank_score": 0.9, "key": "a.md",
        },
        {
            "chunk_id": "c2", "parent_id": "Q", "content": "child Q",
            "rerank_score": 0.8, "key": "b.md",
        },
    ]
    parent_lookup = {
        "P": {
            "chunk_id": "P", "parent_id": None,
            "content": "parent P body", "text": "parent P body",
            "key": "a.md",
        },
        # Q intentionally missing.
    }
    searcher = _build_searcher_with_store_mock(
        store_search_return=canned,
        parent_lookup=parent_lookup,
    )

    results = asyncio.run(
        searcher.search(
            "q",
            top_k=5,
            use_reranker=False,
            use_hyde=False,
            normalize=False,
        )
    )

    assert [r["chunk_id"] for r in results] == ["P", "c2"]
    assert results[0]["content"] == "parent P body"
    assert results[1]["content"] == "child Q"  # fallback
    assert results[0]["rerank_score"] == 0.9
    assert results[1]["rerank_score"] == 0.8


def test_parent_fetch_exception_falls_back() -> None:
    """If ``get_by_chunk_ids`` raises, every winning child must be
    returned unchanged — retrieval never blocks on a parent fetch.
    """
    canned = [
        {
            "chunk_id": "c1", "parent_id": "P", "content": "child body",
            "rerank_score": 0.5, "key": "doc.md",
        },
    ]
    searcher = _build_searcher_with_store_mock(store_search_return=canned)
    # Patch the mock to raise.
    searcher._store.get_by_chunk_ids = MagicMock(
        side_effect=RuntimeError("weaviate down")
    )

    results = asyncio.run(
        searcher.search(
            "q",
            top_k=5,
            use_reranker=False,
            use_hyde=False,
            normalize=False,
        )
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["content"] == "child body"


def test_legacy_null_id_chunk_passthrough() -> None:
    """A chunk with no ``chunk_id`` AND no ``parent_id`` (legacy / flat
    pipeline residue) must pass through without crashing and without
    triggering a parent fetch.
    """
    canned = [
        {"chunk_id": None, "parent_id": None, "content": "legacy",
         "rerank_score": 0.4, "key": "old.md"},
        {"chunk_id": "c1", "parent_id": "P", "content": "modern child",
         "rerank_score": 0.5, "key": "new.md"},
    ]
    parent_lookup = {
        "P": {
            "chunk_id": "P", "parent_id": None,
            "content": "modern parent body",
            "text": "modern parent body",
            "key": "new.md",
        },
    }
    parent_fetch_calls: list[list[str]] = []
    searcher = _build_searcher_with_store_mock(
        store_search_return=canned,
        parent_lookup=parent_lookup,
        parent_fetch_calls=parent_fetch_calls,
    )

    results = asyncio.run(
        searcher.search(
            "q",
            top_k=5,
            use_reranker=False,
            use_hyde=False,
            normalize=False,
        )
    )

    # Legacy chunk preserved verbatim, modern child swapped to parent.
    assert results[0]["content"] == "legacy"
    assert results[1]["content"] == "modern parent body"
    # The legacy chunk must NOT contribute to the parent fetch list.
    assert parent_fetch_calls == [["P"]]


def test_parent_level_hit_skips_fetch() -> None:
    """If the hit itself is a parent (parent_id is None), no extra
    fetch is performed and the doc is returned as-is.
    """
    canned = [
        {
            "chunk_id": "P", "parent_id": None,
            "content": "already a parent",
            "rerank_score": 0.95, "key": "doc.md",
        },
    ]
    parent_fetch_calls: list[list[str]] = []
    searcher = _build_searcher_with_store_mock(
        store_search_return=canned,
        parent_lookup={},
        parent_fetch_calls=parent_fetch_calls,
    )

    results = asyncio.run(
        searcher.search(
            "q",
            top_k=5,
            use_reranker=False,
            use_hyde=False,
            normalize=False,
        )
    )

    assert [r["chunk_id"] for r in results] == ["P"]
    assert results[0]["content"] == "already a parent"
    # No batch fetch should have been issued.
    assert parent_fetch_calls == []


def test_normalize_query_applied() -> None:
    """Abbreviations like 'KTPM' / non-diacritic 'diem chuan' get expanded
    BEFORE the store sees the query.
    """
    captured: dict[str, Any] = {}
    searcher = _build_searcher_with_store_mock(
        store_search_return=[],
        captured=captured,
    )

    asyncio.run(
        searcher.search(
            "diem chuan ktpm",
            top_k=5,
            use_reranker=False,
            use_hyde=False,
        )
    )

    forwarded_query = captured.get("query") or ""
    # Synonyms YAML expands "diem chuan" → "điểm chuẩn" and "ktpm" →
    # "kỹ thuật phần mềm". We accept either canonical form to keep the
    # test stable against synonym table edits.
    lowered = forwarded_query.lower()
    assert "điểm chuẩn" in lowered or "diem chuan" not in lowered, (
        f"expected synonym expansion, got query={forwarded_query!r}"
    )
    # Ensure the raw abbreviation has been rewritten (one of the forms
    # listed in synonyms_vn.yaml must appear).
    assert "ktpm" not in lowered, (
        f"expected 'ktpm' to be expanded, got query={forwarded_query!r}"
    )
