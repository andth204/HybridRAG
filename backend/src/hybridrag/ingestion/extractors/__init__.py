"""LLM-based extractors that populate the Phase 3B structured store.

Each extractor is a small wrapper around an OpenAI JSON-mode call. The
prompt does the schema enforcement; the Python code does defensive
post-processing (dropping rows without a year, canonicalizing majors,
etc.) before handing back row dicts that map 1:1 onto the matching
repo's ``upsert`` kwargs.
"""
from src.hybridrag.ingestion.extractors.scores_extractor import (
    extract_scores_from_text,
)
from src.hybridrag.ingestion.extractors.tuition_extractor import (
    extract_tuition_from_text,
)

__all__ = [
    "extract_scores_from_text",
    "extract_tuition_from_text",
]
