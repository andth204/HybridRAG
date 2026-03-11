from __future__ import annotations
import re
import unicodedata


_NON_WORD_RE = re.compile(r"[^\w\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def normalize_vietnamese_for_search(text: str) -> str:
    normalized = text.replace("\u0110", "D").replace("\u0111", "d")
    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = _NON_WORD_RE.sub(" ", normalized.lower())
    return _MULTISPACE_RE.sub(" ", normalized).strip()


def tokenize_vietnamese_for_search(text: str) -> list[str]:
    normalized = normalize_vietnamese_for_search(text)
    return normalized.split() if normalized else []