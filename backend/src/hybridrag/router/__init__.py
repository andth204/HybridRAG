from .route import Route
from .samples import ROUTES
from .keywords import KeywordRouter

try:
    from .semantic import SemanticRouter
except ImportError:
    SemanticRouter = None  # type: ignore

# Phase 4A: fine-grained intent classification (additive — does not
# alter the legacy KeywordRouter / SemanticRouter binary chitchat-vs-
# retrieval surface used elsewhere).
from .intents import (
    DEFAULT_INTENT,
    INTENT_FALLBACK_THRESHOLD,
    Intent,
    IntentResult,
)
from .intent_classifier import KeywordIntentClassifier
from .intent_router import IntentRouter
from .semantic_intent import SemanticIntentClassifier

__all__ = [
    "Route",
    "SemanticRouter",
    "ROUTES",
    "KeywordRouter",
    "Intent",
    "IntentResult",
    "DEFAULT_INTENT",
    "INTENT_FALLBACK_THRESHOLD",
    "KeywordIntentClassifier",
    "SemanticIntentClassifier",
    "IntentRouter",
]