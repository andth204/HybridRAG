"""Combined keyword-first / semantic-fallback intent router (Phase 4A).

Wraps :class:`KeywordIntentClassifier` and :class:`SemanticIntentClassifier`
into a single ``classify`` coroutine the dialogue layer can call. The
semantic classifier is only instantiated the first time it is actually
needed (lazy load) so cold-start latency stays low for queries that the
keyword bank can answer outright.

Dispatch rules (in order):

1. Keyword score >= :data:`INTENT_FALLBACK_THRESHOLD` and the top two
   intents are not within ``_AMBIGUITY_MARGIN`` of each other → trust
   the keyword result.
2. Otherwise consult the semantic classifier. Return whichever side has
   the higher score.
3. If both classifiers return ``score == 0``, return the keyword
   fallback (which is already pre-tagged with the default intent).
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from .intent_classifier import KeywordIntentClassifier
from .intents import INTENT_FALLBACK_THRESHOLD, IntentResult
from .semantic_intent import SemanticIntentClassifier

log = logging.getLogger(__name__)

# How close two keyword intents must be to be considered ambiguous
# (= worth checking the semantic classifier even though the top one is
# above the threshold). 0.05 picked empirically from the keyword eval —
# real near-ties are rare with the current YAML. A tiny epsilon is
# added to defeat float-precision artefacts in scoring (a margin of
# 0.05 + 4.4e-17 should still count as "within 0.05").
_AMBIGUITY_MARGIN: float = 0.05
_AMBIGUITY_EPS: float = 1e-9


class IntentRouter:
    """Keyword-first intent dispatch with semantic fallback.

    Both classifiers are dependency-injectable to make the unit tests
    cheap (no embedder, no YAML mocks).
    """

    def __init__(
        self,
        keyword: Optional[KeywordIntentClassifier] = None,
        semantic: Optional[SemanticIntentClassifier] = None,
    ) -> None:
        self._kw = keyword or KeywordIntentClassifier()
        # Track whether the caller explicitly handed us a semantic
        # instance — if so we never replace it with a default one even
        # if it's None-equivalent.
        self._sem: Optional[SemanticIntentClassifier] = semantic
        self._sem_provided = semantic is not None

    # ------------------------------------------------------------------ #
    # Lazy semantic instantiation
    # ------------------------------------------------------------------ #
    def _ensure_semantic(self) -> SemanticIntentClassifier:
        if self._sem is None:
            self._sem = SemanticIntentClassifier()
        return self._sem

    # ------------------------------------------------------------------ #
    # Ambiguity helper
    # ------------------------------------------------------------------ #
    def _ambiguous(self, query: str, top_score: float) -> bool:
        """Re-score the query with the keyword classifier and check if
        the runner-up is within :data:`_AMBIGUITY_MARGIN` of ``top_score``.

        We re-derive the second-best score from the same compiled rule
        bank instead of changing :meth:`KeywordIntentClassifier.classify`'s
        return shape — keeps the classifier's public contract minimal.
        """
        # Inline a stripped-down version of the classifier's scoring
        # loop so we can see the runner-up. Cheap because the rule bank
        # is small (8 intents).
        from src.hybridrag.utils.synonyms import _normalize_key

        rules = getattr(self._kw, "_rules", [])
        if not rules:
            return False

        normalised = _normalize_key(query)
        query_tokens = set(normalised.split())
        token_count = max(1, len(normalised.split()))
        # Match the classifier's sqrt-length penalty so the runner-up
        # score we compute here is on the same scale as the winner.
        denom = max(1.5, math.sqrt(token_count))

        # Mirror the classifier's short-keyword rule so the ambiguity
        # decision stays consistent with the primary scoring.
        from .intent_classifier import _SHORT_KEYWORD_LEN

        scores: list[float] = []
        for rule in rules:
            hits = 0
            for kw in rule.keywords:
                if not kw:
                    continue
                if len(kw) <= _SHORT_KEYWORD_LEN and " " not in kw:
                    if kw in query_tokens:
                        hits += 1
                elif kw in normalised:
                    hits += 1
            for pat in rule.regexes:
                if pat.search(query):
                    hits += 1
            if hits:
                scores.append((hits * rule.weight) / denom)
        if len(scores) < 2:
            return False
        scores.sort(reverse=True)
        # ``scores[0]`` may differ from ``top_score`` because the
        # classifier applies the chitchat priority rule; treat any
        # near-tie among the top two raw scores as ambiguous.
        return (scores[0] - scores[1]) <= _AMBIGUITY_MARGIN + _AMBIGUITY_EPS

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #
    async def classify(self, query: str) -> IntentResult:
        kw_res = self._kw.classify(query)

        confident = (
            kw_res.score >= INTENT_FALLBACK_THRESHOLD
            and kw_res.source == "keyword"
        )
        if confident and not self._ambiguous(query, kw_res.score):
            return kw_res

        sem = self._ensure_semantic()
        try:
            sem_res = await sem.classify(query)
        except Exception as exc:  # noqa: BLE001 - external service safety
            log.warning("IntentRouter: semantic classifier crashed: %s", exc)
            return kw_res

        # If semantic has nothing to say, keep the keyword pick (even if
        # it was below the threshold — it's still our best guess).
        if sem_res.source == "fallback":
            return kw_res

        return sem_res if sem_res.score > kw_res.score else kw_res


__all__ = ["IntentRouter"]
