from __future__ import annotations
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List
from src.config.settings import settings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
log = logging.getLogger(__name__)
_RERANK_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(getattr(settings, "RERANK_WORKERS", 2)),
    thread_name_prefix="reranker",
)


class Reranker:
    _init_lock: Lock = Lock()
    _shared_loaded: bool = False
    _shared_model_name: str | None = None
    _shared_tokenizer = None
    _shared_hf_model = None
    _shared_torch = None
    _shared_device: str = "cuda"

    def __init__(self, model_name: str | None = None, top_k: int | None = None) -> None:
        self._model_name = model_name or settings.RERANKER_MODEL
        self._top_k = top_k or settings.RERANK_TOP_K
        self.model: str | None = None
        self._tokenizer = None
        self._hf_model = None
        self._torch = None
        self._device = "cuda"
        self._loaded = False

    def preload(self) -> None:
        """Load the cross-encoder onto CUDA if available, else skip.

        Behaviour:

        * GPU present → load the model, store the shared handles, set
          ``self.model`` so :meth:`rerank` runs.
        * GPU absent → log a warning and leave ``self.model = None``.
          :meth:`rerank` / :meth:`arerank` then short-circuit to a
          plain top-k cut, so the retrieval pipeline degrades
          gracefully instead of crashing.

        This is the contract HybridSearcher / WeaviateHybridSearcher
        rely on (they check ``reranker.model`` before invoking rerank).
        """
        if self._loaded:
            return

        with self.__class__._init_lock:
            if not self.__class__._shared_loaded:
                if not torch.cuda.is_available():
                    log.warning(
                        "Reranker: CUDA not available — skipping reranker load. "
                        "Retrieval will fall back to the raw hybrid score "
                        "ordering. Install a CUDA-enabled PyTorch build and "
                        "run on a GPU machine to enable reranking."
                    )
                    # Mark as loaded so we don't retry on every call;
                    # ``self.model`` stays None so rerank is skipped.
                    self.__class__._shared_loaded = True
                    self.__class__._shared_device = "cpu"
                    self._loaded = True
                    return

                log.info("Reranker: loading model '%s' ...", self._model_name)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
                    hf_model = AutoModelForSequenceClassification.from_pretrained(
                        self._model_name,
                        trust_remote_code=True,
                    )
                    hf_model.eval()
                    device = "cuda"
                    hf_model = hf_model.to(device)
                except Exception as exc:  # noqa: BLE001
                    # OOM, model-file corruption, network failure during
                    # download — all best-effort. Log and skip rerank.
                    log.warning(
                        "Reranker: model load failed (%s) — skipping rerank",
                        exc,
                    )
                    self.__class__._shared_loaded = True
                    self.__class__._shared_device = "cpu"
                    self._loaded = True
                    return

                self.__class__._shared_tokenizer = tokenizer
                self.__class__._shared_hf_model = hf_model
                self.__class__._shared_torch = torch
                self.__class__._shared_device = device
                self.__class__._shared_model_name = self._model_name
                self.__class__._shared_loaded = True
                log.info("Reranker: ready on device=%s", device)

        # Bind shared handles to this instance. When the GPU-fallback
        # branch above ran, ``_shared_model_name`` stays None and
        # ``self.model`` remains None — that's the signal the rest of
        # the pipeline uses to bypass rerank.
        self._tokenizer = self.__class__._shared_tokenizer
        self._hf_model = self.__class__._shared_hf_model
        self._torch = self.__class__._shared_torch
        self._device = self.__class__._shared_device
        self.model = self.__class__._shared_model_name
        self._loaded = True

    def _score_pairs(self, query: str, docs: List[str]) -> List[float]:
        """Return per-pair relevance probabilities in ``[0, 1]``.

        The cross-encoder head emits raw logits (Jina v2 typically in
        ``[-10, +10]``). We squash through sigmoid so downstream gates
        — clarifier ``LOW_RECALL_THRESHOLD=0.30`` and handoff
        ``LOW_RECALL_THRESHOLD=0.15`` — read a calibrated probability
        rather than a sign-ambiguous raw logit.
        """
        torch = self._torch
        pairs = [[query, d] for d in docs]
        enc = self._tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._hf_model(**enc).logits
            probs = torch.sigmoid(logits)
        scores = probs.squeeze(-1).tolist()
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
            log.error("Reranker: inference timed out after %.1fs, returning fused results", timeout)
            return docs[: (top_k or self._top_k)]