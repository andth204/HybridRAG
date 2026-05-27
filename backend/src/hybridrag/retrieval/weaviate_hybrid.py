"""Weaviate-backed hybrid searcher.

Pipeline:

1. **Query normalisation** -- synonym expansion via
   :func:`src.hybridrag.utils.synonyms.normalize_query` (e.g. ``"CNTT"`` →
   ``"công nghệ thông tin"``) so BM25 hits ASCII queries and embeddings
   get the canonical Vietnamese form.

2. **HyDE (optional)** -- generate a hypothetical passage that answers
   the query, then embed *that* for the vector side of the hybrid
   search. The original (normalised) query is still used for the BM25
   side. Falls back to embedding the query itself when HyDE generation
   fails or is disabled.

3. **Hybrid search** -- delegated to Weaviate's native hybrid query
   (BM25 + vector with an ``alpha`` blend).

4. **Parent/child de-duplication** -- collapse multiple child hits that
   share the same ``parent_id`` so the LLM sees one representative chunk
   per parent.

No metadata filters are applied here; the index is small and the LLM
already grounds answers in retrieved context. Explicit ``filters=`` from
the caller are still honoured.

The retrieval path is feature-flagged via ``settings.RETRIEVAL_BACKEND`` --
this class is only constructed when the flag is set to ``"weaviate"``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

from src.config.settings import settings
from src.hybridrag.ingestion.embedding import embedder
from src.hybridrag.ingestion.ingestion_service.entities.weaviate_store import WeaviateStore
from src.hybridrag.retrieval.hyde import generate_hyde
from src.hybridrag.retrieval.reranker import Reranker
from src.hybridrag.utils.synonyms import normalize_query

# Score fields we copy from the winning child hit onto the parent-swapped
# doc dict so downstream clarifier/handoff/handoff-low-recall thresholds
# still read the same numeric signal.
_SCORE_FIELDS = ("rerank_score", "rrf_score", "vector_score", "bm25_score", "score")

log = logging.getLogger(__name__)


class WeaviateHybridSearcher:
    """Hybrid retrieval over a Weaviate ``DocChunk`` collection."""

    def __init__(self) -> None:
        self._store: WeaviateStore | None = None
        # Mirror HybridSearcher: instantiate the reranker eagerly so the
        # ``model`` attribute is reachable; actual weights load on preload().
        self._reranker = Reranker() if settings.USE_RERANKER else None

    # ------------------------------------------------------------------ init
    def load_indexes(self) -> None:
        """Lazily build the WeaviateStore handle and preload the reranker.

        We do NOT open the Weaviate connection here; the store connects on
        first use inside :meth:`search`. This keeps unit tests that mock the
        store free of network requirements.
        """
        if self._store is None:
            self._store = WeaviateStore(
                embedding_fn=embedder.embed,
                embedding_dim=embedder.get_dimension(),
            )
        if self._reranker is not None:
            try:
                self._reranker.preload()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "WeaviateHybridSearcher: reranker preload failed (%s); "
                    "will skip rerank step",
                    exc,
                )
        log.info(
            "WeaviateHybridSearcher: ready (reranker=%s)",
            "on" if (self._reranker and self._reranker.model) else "off",
        )

    # ------------------------------------------------------------- search
    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        alpha: float | None = None,
        filters: dict[str, Any] | None = None,
        use_reranker: bool | None = None,
        rerank_top_k: int | None = None,
        normalize: bool = True,
        use_hyde: bool | None = None,
        search_timeout: float = 10.0,
        rerank_timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """Run a normalised hybrid query against Weaviate.

        Parameters
        ----------
        query:
            Raw user question. Will be passed through
            :func:`normalize_query` unless ``normalize=False``.
        top_k:
            Final number of results to return (after rerank + dedup).
        alpha:
            Hybrid blending coefficient (0=BM25 only, 1=vector only).
            Defaults to ``settings.WEAVIATE_HYBRID_ALPHA``.
        filters:
            Explicit ``{property: value}`` filter map passed through to
            the Weaviate store. No filter inference is performed.
        use_reranker:
            Override ``settings.USE_RERANKER`` for this single call.
        rerank_top_k:
            How many items to keep after reranking. Defaults to ``top_k``.
        normalize:
            If ``False``, skip synonym expansion (useful for evaluators
            that want to measure raw retrieval).
        use_hyde:
            Override ``settings.HYDE_ENABLED`` for this single call.

        Returns
        -------
        list[dict]
            Same shape as the legacy ``HybridSearcher.search`` output, with
            ``content``, ``key``, ``chunk_id``, ``parent_id`` and the
            structured metadata fields populated.
        """
        self.load_indexes()
        if self._store is None:
            raise RuntimeError("WeaviateStore not initialized")

        # 1. Synonym expansion.
        q = normalize_query(query) if normalize else query

        # 2. HyDE: build a hypothetical passage and embed it for the
        #    vector side of the hybrid search. BM25 still uses ``q``.
        do_hyde = use_hyde if use_hyde is not None else settings.HYDE_ENABLED
        vector_text: str | None = None
        if do_hyde:
            hyde_text = await generate_hyde(q)
            if hyde_text:
                # Concatenate the query with the HyDE passage so the
                # embedded vector retains the query's keyword signal in
                # addition to the hypothetical-answer wording. The
                # original query is appended at the front for emphasis.
                vector_text = f"{q}\n\n{hyde_text}"

        # 3. Oversample so dedup + rerank still has good candidates to
        #    choose from after parent collapsing. Children share parents
        #    ~5-6:1 in the production corpus, so we need a healthy
        #    multiplier to avoid landing on fewer than top_k distinct
        #    parents post-collapse.
        oversample = min(max(top_k * 6, 30), 80)
        effective_alpha = alpha if alpha is not None else settings.WEAVIATE_HYBRID_ALPHA
        explicit_filters = {k: v for k, v in (filters or {}).items() if v is not None}
        log.info(
            "WeaviateHybridSearcher.search query=%r filters=%s hyde=%s",
            q[:60], explicit_filters or None, bool(vector_text),
        )

        t0 = time.perf_counter()
        results = await self._store.search(
            query=q,
            top_k=oversample,
            filters=explicit_filters or None,
            alpha=effective_alpha,
            vector_query=vector_text,
        )
        log.info(
            "WeaviateHybridSearcher.search: hits=%d in %.1f ms",
            len(results), (time.perf_counter() - t0) * 1000.0,
        )

        if not results:
            return []

        # 4. Optional reranking. IMPORTANT: rerank operates on CHILD
        #    content because the cross-encoder tokenizer caps at 512
        #    tokens (see Reranker._score_pairs). Reranking a ~3k-char
        #    parent body would silently truncate the tail and flatten
        #    scores — so the order is: rerank child → collapse to parent
        #    → swap content. Parent body never enters the rerank loop.
        do_rerank = use_reranker if use_reranker is not None else settings.USE_RERANKER
        # Oversample for collapse: children of the same parent collapse
        # to one entry, so we need extra candidates upstream to still
        # hit ``top_k`` distinct parents after dedup. With a 5-6
        # children-per-parent fan-out (see the live corpus snapshot),
        # 4× is too tight; 6× keeps headroom for both the dedup loss
        # and any parent-fetch misses.
        collapse_pool = max(top_k * 6, (rerank_top_k or top_k), 30)
        if do_rerank and self._reranker is not None and self._reranker.model:
            results = await self._reranker.arerank(
                q, results,
                top_k=collapse_pool,
                timeout=rerank_timeout,
            )
        else:
            results = results[:collapse_pool]

        # 5. Collapse children sharing a parent and swap the winning
        #    entry's content for the full parent body fetched from
        #    Weaviate (parent-document retrieval). Score fields from the
        #    winning child are preserved so clarifier/handoff thresholds
        #    keep their semantics.
        results = await self._collapse_to_parents(results, top_k=top_k)

        return results[:top_k]

    # --------------------------------------------------------- parent expand
    async def _collapse_to_parents(
        self,
        docs: List[Dict[str, Any]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Group hits by canonical parent and swap content for parent body.

        Algorithm:

        1. Walk ``docs`` (already rerank-ordered). For each hit, compute
           a *canonical id* = ``parent_id or chunk_id`` (a child's canon
           is its parent's chunk_id; a parent-level hit OR a legacy flat
           chunk uses its own chunk_id).
        2. Keep the FIRST occurrence per canonical id — that's the
           highest-scoring hit in its group. All sibling children get
           folded into this entry.
        3. Collect canonical ids that point to a parent we still need
           to fetch (i.e. canonical id != hit's own chunk_id → the hit
           was a child; we need the parent body).
        4. One batch ``fetch_objects`` round-trip pulls every needed
           parent. Each surviving hit gets its ``content``/``text``/
           ``chunk_id``/``parent_id``/``header_path`` swapped with the
           parent's. The original child's score fields are preserved
           verbatim so downstream score-based gates (low-recall handoff,
           low-recall clarifier) see the same numeric signal they did
           pre-swap.
        5. If a parent fetch fails or returns nothing for a given id,
           the winning child is returned as-is (graceful degradation).
        """
        if not docs:
            return []

        # Step 1-2: dedupe by canonical id, preserve order. We do NOT
        # truncate to ``top_k`` here — if the parent fetch later misses
        # for some entries, we still want to have enough surviving
        # candidates to trim to ``top_k`` at the end.
        seen: set[str] = set()
        winners: List[Dict[str, Any]] = []
        for d in docs:
            chunk_id = (d.get("chunk_id") or "").strip()
            parent_id = (d.get("parent_id") or "").strip()
            canonical = parent_id or chunk_id
            if not canonical:
                # Legacy / null-id chunk; pass through without dedup.
                # Cap at top_k * 2 so a corrupt index full of null ids
                # cannot blow up memory.
                if len(winners) < top_k * 2:
                    winners.append(d)
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            winners.append(d)
            # Stop once we have a healthy oversample to feed the swap
            # step — enough headroom that a few parent-fetch misses
            # don't drop us below top_k.
            if len(winners) >= top_k * 2:
                break

        # Step 3: collect parent chunk_ids we need to fetch.
        # Skip hits whose own chunk_id already equals the canonical id
        # (the hit IS the parent → no fetch needed) and skip hits
        # without a parent_id (parent-level or legacy chunks).
        need_parents: List[str] = []
        for d in winners:
            chunk_id = (d.get("chunk_id") or "").strip()
            parent_id = (d.get("parent_id") or "").strip()
            if parent_id and parent_id != chunk_id:
                need_parents.append(parent_id)

        # Step 4: one batch round-trip. The underlying weaviate-client
        # call is sync (matches has_key/delete_by_key pattern); run it
        # off the event loop so we don't stall concurrent requests.
        parent_by_id: Dict[str, Dict[str, Any]] = {}
        if need_parents and self._store is not None:
            unique_ids = list(dict.fromkeys(need_parents))
            try:
                fetched = await asyncio.to_thread(
                    self._store.get_by_chunk_ids, unique_ids
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "WeaviateHybridSearcher: parent fetch failed (%s); "
                    "falling back to child bodies",
                    exc,
                )
                fetched = []
            for p in fetched:
                pid = p.get("chunk_id")
                if pid:
                    parent_by_id[pid] = p

        # Step 5: swap content for parent body. Preserve score fields
        # from the original child hit. If the parent fetch missed,
        # return the child as-is.
        out: List[Dict[str, Any]] = []
        for d in winners:
            chunk_id = (d.get("chunk_id") or "").strip()
            parent_id = (d.get("parent_id") or "").strip()
            if not parent_id or parent_id == chunk_id:
                # Hit was already a parent (or legacy chunk) — keep as is.
                out.append(d)
                continue
            parent = parent_by_id.get(parent_id)
            if parent is None or not (parent.get("content") or "").strip():
                # No parent found — keep child body.
                out.append(d)
                continue
            # Build the swapped dict: parent body + parent identity,
            # child's score fields, child's filter metadata when present.
            swapped: Dict[str, Any] = {
                "chunk_id":    parent.get("chunk_id"),
                "parent_id":   None,
                "file_id":     parent.get("file_id") or d.get("file_id"),
                "key":         parent.get("key") or d.get("key") or "",
                "content":     parent.get("content") or "",
                "text":        parent.get("text") or parent.get("content") or "",
                "section":     parent.get("section") or d.get("section"),
                "header_path": parent.get("header_path") or d.get("header_path"),
                "chunk_level": parent.get("chunk_level") or "parent",
                "is_table":    parent.get("is_table", d.get("is_table")),
                "campus":      parent.get("campus") or d.get("campus"),
                "doc_type":    parent.get("doc_type") or d.get("doc_type"),
                "faculty":     parent.get("faculty") or d.get("faculty"),
                "major":       parent.get("major") or d.get("major"),
                "year":        parent.get("year") or d.get("year"),
            }
            for sf in _SCORE_FIELDS:
                if sf in d:
                    swapped[sf] = d[sf]
            out.append(swapped)
        return out


__all__ = ["WeaviateHybridSearcher"]
