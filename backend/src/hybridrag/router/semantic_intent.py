"""Semantic intent fallback used when the keyword classifier is unsure.

Loads ``data/eval/golden_v0.jsonl`` lazily, groups queries by their
``intent`` label, embeds every query through the existing
:class:`src.hybridrag.ingestion.embedding.OpenAIEmbedder`, and averages
the per-intent vectors into a single prototype. The prototypes are
cached to disk under ``settings.ROUTER_EMBEDDINGS_DIR / 'intent_protos.pkl'``
so re-runs avoid a second embedding bill.

At classify time we embed the query once and pick the intent whose
prototype has the highest cosine similarity to it.

Defensive behaviour:

* If the golden file is missing or the embedder can't be reached, the
  classifier degrades to returning ``(DEFAULT_INTENT, 0.0, source="fallback")``
  rather than raising — the caller (combined router) treats that as
  "semantic has nothing to say" and falls back to the keyword result.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config.settings import settings

from .intents import DEFAULT_INTENT, Intent, IntentResult

log = logging.getLogger(__name__)


def _golden_path() -> Path:
    return settings.BASE_DIR / "data" / "eval" / "golden_v0.jsonl"


def _proto_path() -> Path:
    return Path(settings.ROUTER_EMBEDDINGS_DIR) / "intent_protos.pkl"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (NaN-safe)."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticIntentClassifier:
    """Cosine-similarity classifier over averaged intent prototypes.

    Parameters:
        embedder:    Any object exposing an async ``embed(text)`` coroutine
                     that returns a 1-D ``np.ndarray`` (the project's
                     :class:`OpenAIEmbedder` matches this contract).
        golden_path: Override the golden-set location (mostly for tests).
        proto_path:  Override the prototype-cache location (mostly for tests).
    """

    def __init__(
        self,
        embedder: Any = None,
        *,
        golden_path: Path | str | None = None,
        proto_path: Path | str | None = None,
    ) -> None:
        self._embedder = embedder
        self._golden_path = Path(golden_path) if golden_path else _golden_path()
        self._proto_path = Path(proto_path) if proto_path else _proto_path()
        # Loaded lazily on first classify() so import-time is cheap.
        self._protos: dict[Intent, np.ndarray] | None = None
        # If the golden file is absent we record that fact once and stop
        # trying — avoids spamming the log on every request.
        self._disabled: bool = False

    # ------------------------------------------------------------------ #
    # Lazy embedder
    # ------------------------------------------------------------------ #
    def _get_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        # Local import keeps the openai dep out of the import path for
        # consumers (tests) that mock the embedder.
        from src.hybridrag.ingestion.embedding import OpenAIEmbedder

        self._embedder = OpenAIEmbedder()
        return self._embedder

    # ------------------------------------------------------------------ #
    # Prototype management
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_golden(path: Path) -> dict[Intent, list[str]]:
        grouped: dict[Intent, list[str]] = {}
        if not path.exists():
            return grouped
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("SemanticIntentClassifier: bad JSON line in %s", path)
                    continue
                label = rec.get("intent")
                query = rec.get("query")
                if not label or not query:
                    continue
                try:
                    intent = Intent(label)
                except ValueError:
                    log.warning(
                        "SemanticIntentClassifier: unknown intent label %r in golden — skipped",
                        label,
                    )
                    continue
                grouped.setdefault(intent, []).append(str(query))
        return grouped

    def _load_protos_from_disk(self) -> Optional[dict[Intent, np.ndarray]]:
        if not self._proto_path.exists():
            return None
        try:
            with self._proto_path.open("rb") as fh:
                data = pickle.load(fh)
        except (pickle.PickleError, EOFError, OSError) as exc:
            log.warning(
                "SemanticIntentClassifier: failed to read proto cache %s: %s",
                self._proto_path,
                exc,
            )
            return None
        if not isinstance(data, dict):
            return None
        # Re-key string labels back to Intent enum members.
        out: dict[Intent, np.ndarray] = {}
        for k, v in data.items():
            try:
                intent = Intent(k) if isinstance(k, str) else k
            except ValueError:
                continue
            if isinstance(v, np.ndarray):
                out[intent] = v.astype(np.float32, copy=False)
        return out or None

    def _save_protos_to_disk(self, protos: dict[Intent, np.ndarray]) -> None:
        try:
            self._proto_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {k.value: v for k, v in protos.items()}
            with self._proto_path.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError as exc:
            log.warning(
                "SemanticIntentClassifier: failed to write proto cache %s: %s",
                self._proto_path,
                exc,
            )

    async def _build_protos(self) -> dict[Intent, np.ndarray]:
        grouped = self._load_golden(self._golden_path)
        if not grouped:
            log.warning(
                "SemanticIntentClassifier: golden file %s missing or empty",
                self._golden_path,
            )
            return {}

        embedder = self._get_embedder()
        protos: dict[Intent, np.ndarray] = {}
        for intent, queries in grouped.items():
            vecs: list[np.ndarray] = []
            for q in queries:
                try:
                    vec = await embedder.embed(q)
                except Exception as exc:  # noqa: BLE001 - external service
                    log.warning(
                        "SemanticIntentClassifier: embedding failed for %r: %s",
                        q,
                        exc,
                    )
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                vecs.append(arr)
            if vecs:
                protos[intent] = np.mean(np.vstack(vecs), axis=0).astype(np.float32)
        return protos

    async def _ensure_protos(self) -> dict[Intent, np.ndarray]:
        if self._protos is not None:
            return self._protos
        if self._disabled:
            return {}
        cached = self._load_protos_from_disk()
        if cached:
            self._protos = cached
            return self._protos
        protos = await self._build_protos()
        if not protos:
            self._disabled = True
            self._protos = {}
            return self._protos
        self._save_protos_to_disk(protos)
        self._protos = protos
        return self._protos

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #
    async def classify(self, query: str) -> IntentResult:
        """Embed ``query`` and return the most-similar intent prototype.

        When prototypes are unavailable (no golden file, no network, no
        embedder), this returns a sentinel ``IntentResult`` whose
        ``source`` is ``"fallback"`` and ``intent`` is
        :data:`DEFAULT_INTENT`. The caller treats that as "no semantic
        opinion" and keeps whatever the keyword classifier produced.
        """
        if not query or not query.strip():
            return IntentResult(
                intent=DEFAULT_INTENT,
                score=0.0,
                matched=[],
                source="fallback",
            )

        protos = await self._ensure_protos()
        if not protos:
            return IntentResult(
                intent=DEFAULT_INTENT,
                score=0.0,
                matched=[],
                source="fallback",
            )

        embedder = self._get_embedder()
        try:
            vec_raw = await embedder.embed(query)
        except Exception as exc:  # noqa: BLE001 - external service
            log.warning("SemanticIntentClassifier: embed(query) failed: %s", exc)
            return IntentResult(
                intent=DEFAULT_INTENT,
                score=0.0,
                matched=[],
                source="fallback",
            )
        vec = np.asarray(vec_raw, dtype=np.float32)
        if vec.ndim > 1:
            vec = vec.reshape(-1)

        best_intent: Intent = DEFAULT_INTENT
        best_score: float = -1.0
        for intent, proto in protos.items():
            sim = _cosine(vec, proto)
            if sim > best_score:
                best_score = sim
                best_intent = intent

        # Clamp cosine into the 0-1 band so it composes with the keyword
        # classifier's score on the same scale.
        score = max(0.0, min(1.0, best_score))
        return IntentResult(
            intent=best_intent,
            score=score,
            matched=[],
            source="semantic",
        )


__all__ = ["SemanticIntentClassifier"]
