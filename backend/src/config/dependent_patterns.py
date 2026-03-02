from __future__ import annotations
import re
from functools import lru_cache
from typing import Dict, List, Pattern


DEFAULT_DEPENDENT_DOMAIN = "admissions"
SHORT_QUERY_TOKEN_THRESHOLD = 2
DOMAIN_DEPENDENT_PATTERNS: Dict[str, List[str]] = {
"admissions": [
        "nganh nay",
        "nganh do",
        "nganh ay",
        "truong nay",
        "truong do",
        "truong ay",
        "khoa nay",
        "khoa do",
        "he nay",
        "he do",
        "to hop nay",
        "to hop do",
        "phuong thuc nay",
        "phuong thuc do",
        "dieu kien nay",
        "dieu kien do",
        "muc nay",
        "muc do",
        "ma nay",
        "cai nay",
        "cai do",
        "cai ay",
        "cai kia",
        "noi nay",
        "noi do",
        "phan nay",
        "phan do",
        "thong tin nay",
        "thong tin do",
        "nhu vay",
        "vay la",
        "vay a",
        "vay ha",
        "roi sao",
        "tiep theo",
        "the con",
        "con cai nay",
        "con nganh do",
        "con khoa do",
        "con he do",
        "nhu tren",
        "o tren",
        "ben tren",
        "o duoi",
        "ben duoi",
    ]
}
_DOMAIN_REGEX_CACHE: Dict[str, Pattern[str]] = {}

def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def _build_domain_regex(domain: str) -> Pattern[str]:
    patterns = DOMAIN_DEPENDENT_PATTERNS.get(domain) or DOMAIN_DEPENDENT_PATTERNS[DEFAULT_DEPENDENT_DOMAIN]
    normalized = sorted(
        {_normalize_spaces(p) for p in patterns if p and p.strip()},
        key=len,
        reverse=True,
    )
    if not normalized:
        return re.compile(r"$^")
    joined = "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in normalized)
    return re.compile(rf"(?:^|\b)(?:{joined})(?:\b|$)")

def _get_domain_regex(domain: str) -> Pattern[str]:
    key = domain if domain in DOMAIN_DEPENDENT_PATTERNS else DEFAULT_DEPENDENT_DOMAIN
    regex = _DOMAIN_REGEX_CACHE.get(key)
    if regex is None:
        regex = _build_domain_regex(key)
        _DOMAIN_REGEX_CACHE[key] = regex
    return regex

@lru_cache(maxsize=8192)
def contains_dependent_pattern(normalized_query: str, domain: str = DEFAULT_DEPENDENT_DOMAIN) -> bool:
    if not normalized_query:
        return False
    return _get_domain_regex(domain).search(normalized_query) is not None


__all__ = [
    "DEFAULT_DEPENDENT_DOMAIN",
    "SHORT_QUERY_TOKEN_THRESHOLD",
    "DOMAIN_DEPENDENT_PATTERNS",
    "contains_dependent_pattern",
]