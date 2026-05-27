"""Keyword-first fine-grained intent classifier (Phase 4A).

Loads a YAML config (``backend/data/dict/intent_keywords.yaml``) at
construction time and scores each intent by counting weighted keyword
and regex hits against the (diacritic-stripped) user query.

Design choices:

* Keywords are normalised through
  :func:`src.hybridrag.utils.synonyms._normalize_key` so the same lookup
  table matches "điểm chuẩn", "diem chuan", "Điểm Chuẩn".
* The normalisation strips diacritics from the *query* too, so we never
  miss a hit because the user typed plain ASCII.
* Regexes are matched against the **raw** query (with diacritics) so
  patterns like ``20\\d{2}`` can include diacritic context if needed.
* Scoring: ``hits * weight / max(1, len(query_tokens) * 0.5)`` — keeps
  long queries from artificially deflating shorter, more confident
  matches; the 0.5 factor lets a single keyword in a 2-token query
  score well above the fallback threshold.
* Ties: broken by the declaration order in the YAML, with one tweak —
  the ``chitchat`` intent only wins outright if **no** non-chitchat
  intent has a hit. That way "chào, cho mình hỏi điểm chuẩn" routes to
  ``score_lookup`` instead of ``chitchat``.

The classifier is stateless after construction; safe to reuse across
threads / requests.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml

from src.config.settings import settings
from src.hybridrag.utils.synonyms import _normalize_key

from .intents import (
    DEFAULT_INTENT,
    INTENT_FALLBACK_THRESHOLD,
    Intent,
    IntentResult,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- #
# Internal representation of a compiled intent rule.
# ---------------------------------------------------------------- #
@dataclass
class _IntentRule:
    intent: Intent
    keywords: list[str]                 # normalised, in declaration order
    keyword_set: frozenset[str]         # for O(1) "contains" probes
    regexes: list[re.Pattern[str]]      # compiled against raw text
    weight: float


# Short keywords (<= this many chars) must match on whole-word
# boundaries — otherwise "hi" matches the substring "thi" inside
# Vietnamese words and pollutes chitchat scores. Longer keywords
# (e.g. "điểm chuẩn") are safe to match as substrings because their
# false-positive rate is negligible.
_SHORT_KEYWORD_LEN: int = 4


def _default_yaml_path() -> Path:
    return settings.BASE_DIR / "data" / "dict" / "intent_keywords.yaml"


class KeywordIntentClassifier:
    """Score a query against the keyword bank of every fine-grained intent.

    Parameters:
        path: Override the YAML config location (mostly for tests).

    The classifier returns an :class:`IntentResult` for every query.
    When the top score is below :data:`INTENT_FALLBACK_THRESHOLD`, the
    intent is replaced with :data:`DEFAULT_INTENT` and ``source`` is set
    to ``"fallback"`` — the combined :class:`IntentRouter` then knows to
    consult the semantic classifier.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._yaml_path = Path(path) if path else _default_yaml_path()
        self._rules: list[_IntentRule] = self._load(self._yaml_path)

    # ------------------------------------------------------------------ #
    # Loading / compilation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load(path: Path) -> list[_IntentRule]:
        if not path.exists():
            log.warning("KeywordIntentClassifier: YAML not found at %s", path)
            return []
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            log.error("KeywordIntentClassifier: bad YAML %s: %s", path, exc)
            return []

        intent_block = raw.get("intents") or {}
        rules: list[_IntentRule] = []

        # Preserve declaration order so tie-breaks are deterministic.
        for intent_label, cfg in intent_block.items():
            try:
                intent = Intent(intent_label)
            except ValueError:
                log.warning(
                    "KeywordIntentClassifier: unknown intent label %r in YAML — skipped",
                    intent_label,
                )
                continue

            kws_raw: Iterable[str] = (cfg or {}).get("keywords") or []
            # Normalise each keyword the same way the query will be, and

            # drop blanks / dupes while preserving order.
            seen: set[str] = set()
            keywords: list[str] = []
            for kw in kws_raw:
                norm = _normalize_key(str(kw))
                if norm and norm not in seen:
                    seen.add(norm)
                    keywords.append(norm)

            regex_raw: Iterable[str] = (cfg or {}).get("regex") or []
            regexes: list[re.Pattern[str]] = []
            for pat in regex_raw:
                try:
                    regexes.append(re.compile(str(pat), re.IGNORECASE | re.UNICODE))
                except re.error as exc:
                    log.warning(
                        "KeywordIntentClassifier: bad regex %r for %s: %s",
                        pat,
                        intent_label,
                        exc,
                    )

            weight = float((cfg or {}).get("weight", 1.0))
            rules.append(
                _IntentRule(
                    intent=intent,
                    keywords=keywords,
                    keyword_set=frozenset(keywords),
                    regexes=regexes,
                    weight=weight,
                )
            )

        if not rules:
            log.warning("KeywordIntentClassifier: no intents loaded from %s", path)
        return rules

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #
    def classify(self, query: str) -> IntentResult:
        """Score every intent against ``query`` and return the best.

        Returns an :class:`IntentResult` with ``source="fallback"`` and
        ``intent=DEFAULT_INTENT`` if no intent crosses the threshold.
        """
        if not query or not query.strip():
            return IntentResult(
                intent=DEFAULT_INTENT,
                score=0.0,
                matched=[],
                source="fallback",
            )

        normalised = _normalize_key(query)
        # Length penalty divisor — square-root scaling so a 12-token
        # query (~3.5) is only ~2x harder than a 4-token query (~2.0),
        # rather than 3x under linear half-token. Empirically this is
        # what makes the keyword classifier hit the 75% accuracy bar
        # on the golden set without overflowing short-query scores.
        token_count = max(1, len(normalised.split()))
        denom = max(1.5, math.sqrt(token_count))

        # Tokenised view of the query, used to confirm whole-word matches
        # for short keywords (see ``_SHORT_KEYWORD_LEN``).
        query_tokens = set(normalised.split())

        scored: list[tuple[float, _IntentRule, list[str]]] = []
        for rule in self._rules:
            matched: list[str] = []
            hits = 0

            for kw in rule.keywords:
                if not kw:
                    continue
                if len(kw) <= _SHORT_KEYWORD_LEN and " " not in kw:
                    # Word-boundary probe: only count "hi" if the query
                    # actually has the standalone token "hi", not when
                    # it appears inside e.g. "thi" or "phải".
                    if kw in query_tokens:
                        hits += 1
                        matched.append(kw)
                else:
                    # Multi-word or longer single-word keyword — substring
                    # probe covers conjugations / punctuation gracefully.
                    if kw in normalised:
                        hits += 1
                        matched.append(kw)

            for pat in rule.regexes:
                if pat.search(query):
                    hits += 1
                    matched.append(f"re:{pat.pattern}")

            score = (hits * rule.weight) / denom if hits else 0.0
            scored.append((score, rule, matched))

        if not scored:
            return IntentResult(
                intent=DEFAULT_INTENT,
                score=0.0,
                matched=[],
                source="fallback",
            )

        # Find the best non-chitchat hit and the chitchat hit separately.
        best_non_chitchat: Optional[tuple[float, _IntentRule, list[str]]] = None
        best_chitchat: Optional[tuple[float, _IntentRule, list[str]]] = None
        for entry in scored:
            score, rule, _ = entry
            if score <= 0:
                continue
            if rule.intent is Intent.CHITCHAT:
                if best_chitchat is None or score > best_chitchat[0]:
                    best_chitchat = entry
            else:
                if best_non_chitchat is None or score > best_non_chitchat[0]:
                    best_non_chitchat = entry

        # Priority rule: a non-chitchat intent with a real hit always
        # beats chitchat, even if chitchat's raw score is higher (since
        # pleasantries are cheap and would otherwise dominate short
        # queries like "chào, cho mình hỏi điểm chuẩn"). The chitchat
        # weight is intentionally low so this rarely matters, but the
        # explicit rule documents the desired behaviour.
        winner: Optional[tuple[float, _IntentRule, list[str]]]
        if best_non_chitchat is not None:
            winner = best_non_chitchat
        else:
            winner = best_chitchat

        if winner is None or winner[0] < INTENT_FALLBACK_THRESHOLD:
            return IntentResult(
                intent=DEFAULT_INTENT,
                score=winner[0] if winner else 0.0,
                matched=winner[2] if winner else [],
                source="fallback",
            )

        score, rule, matched = winner
        # Clamp at 1.0 for sanity — heavy keyword stacks shouldn't yield
        # scores larger than the semantic similarities we compare against.
        return IntentResult(
            intent=rule.intent,
            score=min(1.0, score),
            matched=matched,
            source="keyword",
        )

    # ------------------------------------------------------------------ #
    # Debug helpers
    # ------------------------------------------------------------------ #
    @property
    def intent_keyword_counts(self) -> dict[str, int]:
        """Map intent label → number of compiled keyword phrases.

        Handy for the eval script to print configuration provenance.
        """
        return {rule.intent.value: len(rule.keywords) for rule in self._rules}


__all__ = ["KeywordIntentClassifier"]
