"""Structured knowledge repositories backing the Phase 3B lookup layer.

These modules wrap the Postgres tables created by
``scripts/migrations/002_admission_scores.sql`` (``admission_scores`` and
``tuition``) so the rest of the codebase never touches raw SQL for the
structured store. The dialogue tool layer (Phase 4) calls into these via
``src.hybridrag.tools.lookup``.
"""

from src.hybridrag.kg.scores_repo import AdmissionScore, AdmissionScoresRepo
from src.hybridrag.kg.tuition_repo import TuitionRepo, TuitionRow

__all__ = [
    "AdmissionScore",
    "AdmissionScoresRepo",
    "TuitionRepo",
    "TuitionRow",
]
