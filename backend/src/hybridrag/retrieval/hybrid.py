from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, List
from src.config.settings import settings
from src.hybridrag.retrieval.bm25_search import BM25Searcher
from src.hybridrag.retrieval.fusion import RRFFusion
from src.hybridrag.retrieval.reranker import Reranker
from src.hybridrag.retrieval.vector_search import VectorSearcher
log = logging.getLogger(__name__)


class HybridSearcher:
    def __init__(self) -> None:
        self.vector   = VectorSearcher()
        self.bm25     = BM25Searcher()
        self.rrf      = RRFFusion()
        self.reranker = Reranker()

    def load_indexes(self) -> None:
        self.vector.load_index()
        self.bm25.load_index()
        if settings.USE_RERANKER:
            self.reranker.preload()
        log.info("HybridSearcher: indexes ready (reranker=%s)", "on" if self.reranker.model else "off")

    async def search(
        self,
        query: str,
        *,
        vector_k: int | None = None,
        bm25_k: int | None = None,
        fusion_top_n: int | None = None,
        use_reranker: bool | None = None,
        rerank_top_k: int | None = None,
        search_timeout: float = 10.0,
        rerank_timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        vk = vector_k or settings.VECTOR_SEARCH_K
        bk = bm25_k   or settings.ELASTIC_SEARCH_K
        do_rerank = use_reranker if use_reranker is not None else settings.USE_RERANKER
        limit = rerank_top_k or settings.RERANK_TOP_K

        vector_res, bm25_res = await asyncio.gather(
            self.vector.search(query, top_k=vk, timeout=search_timeout),
            self.bm25.search(query,   top_k=bk, timeout=search_timeout),
            return_exceptions=True,
        )
        if isinstance(vector_res, Exception):
            log.warning("HybridSearcher: vector search failed (%s), using BM25 only", vector_res)
            vector_res = []
        if isinstance(bm25_res, Exception):
            log.warning("HybridSearcher: BM25 search failed (%s), using vector only", bm25_res)
            bm25_res = []
        if not vector_res and not bm25_res:
            raise RuntimeError("Both vector and BM25 searches failed.")

        fused = self.rrf.fuse(vector_res, bm25_res, top_n=fusion_top_n)

        if do_rerank and self.reranker.model:
            return await self.reranker.arerank(
                query, fused, top_k=limit, timeout=rerank_timeout
            )
        return fused[:limit]

async def _test(
    query: str,
    use_reranker: bool = settings.USE_RERANKER,
    vector_k: int = settings.VECTOR_SEARCH_K,
    bm25_k: int = settings.ELASTIC_SEARCH_K,
    top_k: int = settings.RERANK_TOP_K,
) -> None:
    searcher = HybridSearcher()
    searcher.load_indexes()

    t0 = time.perf_counter()
    results = await searcher.search(
        query=query,
        vector_k=vector_k,
        bm25_k=bm25_k,
        use_reranker=use_reranker,
        rerank_top_k=top_k,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"  Query    : {query}")
    print(f"  Reranker : {'ON' if use_reranker else 'OFF'}")
    print(f"  Time     : {elapsed*1000:.1f} ms")
    print(f"{'='*60}\n")

    for i, r in enumerate(results, 1):
        scores = []
        if "vector_score" in r: scores.append(f"vector={r['vector_score']:.4f}")
        if "bm25_score"   in r: scores.append(f"bm25={r['bm25_score']:.4f}")
        if "rrf_score"    in r: scores.append(f"rrf={r['rrf_score']:.6f}")
        if "rerank_score" in r: scores.append(f"rerank={r['rerank_score']:.4f}")
        print(f"[{i}] {r.get('key', 'N/A')}  |  {' | '.join(scores)}")
        print(f"     {r['content'][:600]}...\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    asyncio.run(_test(query="phương thức tuyển sinh moi nhat", use_reranker=True))