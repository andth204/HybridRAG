from __future__ import annotations
from typing import List, Dict, Any
from src.config.settings import settings

class RRFFusion:
    def __init__(self, k: int | None = None, top_n: int | None = None) -> None:
        self._k = k or settings.RRF_K
        self._top_n = top_n or settings.FUSION_K

    def fuse(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        k = self._k
        limit = top_n or self._top_n
        rrf_scores: Dict[str, float] = {}
        meta: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(vector_results, start=1):
            cid = doc["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in meta:
                meta[cid] = {**doc}

        for rank, doc in enumerate(bm25_results, start=1):
            cid = doc["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in meta:
                meta[cid] = {**doc}

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        results: List[Dict[str, Any]] = []
        for cid, score in ranked:
            entry = {**meta[cid]}
            # keep individual scores if present, add rrf
            entry["rrf_score"] = round(score, 6)
            results.append(entry)
        return results