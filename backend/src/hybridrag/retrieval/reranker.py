from __future__ import annotations
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from src.config.settings import settings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
log = logging.getLogger(__name__)
_RERANK_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")


class Reranker:
    def __init__(self, model_name: str | None = None, top_k: int | None = None) -> None:
        self._model_name = model_name or settings.RERANKER_MODEL
        self._top_k = top_k or settings.RERANK_TOP_K
        self.model: str | None = None
        self._tokenizer = None
        self._hf_model = None
        self._torch = None
        self._device = "cpu"
        self._loaded = False

    def preload(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        log.info("Reranker: loading model '%s' ...", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        self._hf_model = AutoModelForSequenceClassification.from_pretrained(self._model_name, trust_remote_code=True)
        self._hf_model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._hf_model = self._hf_model.to(self._device)
        self._torch = torch
        self.model = self._model_name
        log.info("Reranker: ready on device=%s", self._device)

    def _score_pairs(self, query: str, docs: List[str]) -> List[float]:
        torch = self._torch
        pairs = [[query, d] for d in docs]
        enc = self._tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._hf_model(**enc).logits
        scores = logits.squeeze(-1).tolist()
        return [scores] if isinstance(scores, float) else scores

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        limit = top_k or self._top_k
        if not self.model or not docs:
            return docs[:limit]

        scores = self._score_pairs(query, [d["content"] for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:limit]
        return [{**doc, "rerank_score": round(float(score), 6)} for doc, score in ranked]

    async def arerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int | None = None,
        timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        if not self.model or not docs:
            return docs[: (top_k or self._top_k)]

        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_RERANK_EXECUTOR, self.rerank, query, docs, top_k),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            log.error("Reranker: inference timed out after %.1fs — returning fused results", timeout)
            return docs[: (top_k or self._top_k)]