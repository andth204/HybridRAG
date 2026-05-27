"""Fine-grained intent definitions used by the Phase 4A classifier.

The labels here match exactly the ``intent`` field in
``backend/data/eval/golden_v0.jsonl`` so the eval harness and the
runtime router speak the same vocabulary.

Downstream code (chat router, tool dispatcher, prompts) imports the
:class:`Intent` enum rather than free-form strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    """Fine-grained intents for the admissions chatbot.

    Inherits from ``str`` so members serialize naturally to JSON / YAML
    and can be compared directly to the string labels in
    ``golden_v0.jsonl``.
    """

    CHITCHAT = "chitchat"
    SCORE_LOOKUP = "score_lookup"
    TUITION_LOOKUP = "tuition_lookup"
    PROGRAM_INFO = "program_info"
    ADMISSION_METHOD = "admission_method"
    DEADLINE = "deadline"
    COMPARE = "compare"
    GENERAL_QA = "general_qa"


@dataclass(frozen=True)
class IntentResult:
    """The output of an intent classifier.

    Attributes:
        intent:   Predicted :class:`Intent`.
        score:    Confidence in the 0-1 range. The interpretation depends
                  on ``source`` — keyword classifiers use a normalised
                  hit-rate, semantic classifiers use cosine similarity.
        matched:  For debugging: the keyword tokens / regex patterns that
                  triggered the match. Semantic classifiers usually leave
                  this empty.
        source:   ``"keyword" | "semantic" | "fallback"``. Lets the
                  combined router decide whether to escalate.
    """

    intent: Intent
    score: float
    matched: list[str] = field(default_factory=list)
    source: str = "keyword"


# Default intent emitted when no classifier has a useful signal — keeps
# downstream callers from special-casing ``None`` everywhere.
DEFAULT_INTENT: Intent = Intent.GENERAL_QA

# Below this keyword score the combined router falls back to the
# semantic classifier (see ``IntentRouter``). Tuned against
# ``golden_v0.jsonl`` with the sqrt length-penalty: a single 1.2-weight
# keyword in a 15-token query scores ~0.31, so 0.20 is the smallest
# threshold that catches them without admitting accidental matches.
INTENT_FALLBACK_THRESHOLD: float = 0.20


__all__ = [
    "Intent",
    "IntentResult",
    "DEFAULT_INTENT",
    "INTENT_FALLBACK_THRESHOLD",
]
