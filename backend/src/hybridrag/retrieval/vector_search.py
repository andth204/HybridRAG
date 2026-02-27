from __future__ import annotations
import asyncio
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional
import numpy as np
from src.config.settings import settings
from src.hybridrag.ingestion.embedding import embedder
from src.hybridrag.ingestion.ingestion_service.entities.faiss_store import FAISSStore
log = logging.getLogger(__name__)


class EmbeddingLRUCache:
    def __init__(self, maxsize: int = 2048) -> None:
        self._maxsize = max(0, int(maxsize))
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_many(self, texts: List[str], fetcher) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        pos_by_text: Dict[str, List[int]] = {}
        for i, t in enumerate(texts):
            pos_by_text.setdefault(t, []).append(i)

        out: List[Optional[np.ndarray]] = [None] * len(texts)
        missing: List[str] = []
        lock = self._get_lock()
        async with lock:
            for t, poss in pos_by_text.items():
                v = self._data.get(t)
                if v is not None:
                    self._data.move_to_end(t)
                    for p in poss:
                        out[p] = v
                else:
                    missing.append(t)

        if missing:
            fetched = np.asarray(await fetcher(missing), dtype=np.float32)
            if fetched.shape[0] != len(missing):
                raise RuntimeError("Embedding fetcher returned wrong batch size")
            async with lock:
                for t, vec in zip(missing, fetched):
                    if self._maxsize > 0:
                        self._data[t] = vec
                        self._data.move_to_end(t)
                        while len(self._data) > self._maxsize:
                            self._data.popitem(last=False)
                    for p in pos_by_text[t]:
                        out[p] = vec

        result = [v for v in out if v is not None]
        if len(result) != len(texts):
            raise RuntimeError("Missing embedding(s) after cache lookup")
        return np.stack(result, axis=0).astype(np.float32)


class VectorSearcher:
    def __init__(self) -> None:
        self._store: FAISSStore | None = None
        self._emb_cache = EmbeddingLRUCache(
            maxsize=int(getattr(settings, "EMBEDDING_LRU_MAXSIZE", 2048))
        )

    async def _embed(self, texts: List[str]) -> np.ndarray:
        return await self._emb_cache.get_many(texts, embedder.embed)

    def load_index(self) -> None:
        if self._store is not None:
            return
        self._store = FAISSStore(
            embedding_fn=self._embed,
            embedding_dim=int(getattr(settings, "EMBEDDING_DIMENSION", embedder.get_dimension())),
            cache_dir=str(settings.FAISS_CACHE_DIR),
            index_type=str(getattr(settings, "FAISS_INDEX_TYPE", "hnsw")),
            hnsw_m=int(getattr(settings, "FAISS_HNSW_M", 32)),
            hnsw_ef_construction=int(getattr(settings, "FAISS_HNSW_EF_CONSTRUCTION", 200)),
            hnsw_ef_search=int(getattr(settings, "FAISS_HNSW_EF_SEARCH", 64)),
            ivf_nlist=int(getattr(settings, "FAISS_IVF_NLIST", 4096)),
            ivf_nprobe=int(getattr(settings, "FAISS_IVF_NPROBE", 16)),
        )
        log.info("VectorSearcher: FAISS index loaded from %s (metric=IP, normalized)", settings.FAISS_CACHE_DIR)

    async def search(
        self,
        query: str,
        top_k: int,
        timeout: float = 5.0,
        *,
        nprobe: int | None = None,
        ef_search: int | None = None,
    ) -> List[Dict[str, Any]]:
        if self._store is None:
            raise RuntimeError("FAISS index not loaded. Call load_index() first.")

        raw = await asyncio.wait_for(
            self._store.search(query, top_k=top_k, nprobe=nprobe, ef_search=ef_search),
            timeout=timeout,
        )
        return [
            {
                "chunk_id": r["chunk_id"],
                "file_id": r.get("file_id", ""),
                "key": r.get("key", ""),
                "content": r["text"],
                "vector_score": float(r["score"]),
            }
            for r in raw
        ]

async def test_search(query: str, top_k: int) -> None:
    searcher = VectorSearcher()
    searcher.load_index()
    print(f"\nQuery: {query}  |  Top-K: {top_k}\n")
    for i, r in enumerate(await searcher.search(query=query, top_k=top_k), 1):
        print(f"[{i}] chunk_id={r['chunk_id']}  score={r['vector_score']:.4f}")
        print(f"     {r['content'][:500]}...\n")


if __name__ == "__main__":
    asyncio.run(test_search(query="Cac phuong thuc tuyen sinh", top_k=settings.VECTOR_SEARCH_K))