"""Tool functions invoked by the dialogue layer.

These are sync callables today (Phase 3B). Phase 4 wraps them in
OpenAI function-calling adapters; the surface here is intentionally
plain-Python so it stays testable without an LLM in the loop.
"""

from src.hybridrag.tools.lookup import (
    list_majors_by_campus,
    lookup_score,
    lookup_tuition,
)

__all__ = [
    "list_majors_by_campus",
    "lookup_score",
    "lookup_tuition",
]
