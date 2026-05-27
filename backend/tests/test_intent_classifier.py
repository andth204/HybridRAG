"""Tests for ``src.hybridrag.router.intent_classifier``.

Drives the keyword-first classifier against representative queries
(both with and without diacritics) and the documented priority rule:
when a real info-intent keyword fires, the chitchat intent must not
suppress it.
"""
from __future__ import annotations

import pytest

from src.hybridrag.router.intent_classifier import KeywordIntentClassifier
from src.hybridrag.router.intents import (
    DEFAULT_INTENT,
    INTENT_FALLBACK_THRESHOLD,
    Intent,
    IntentResult,
)


@pytest.fixture(scope="module")
def clf() -> KeywordIntentClassifier:
    """A classifier loaded from the production YAML."""
    return KeywordIntentClassifier()


def test_yaml_loaded_for_every_intent(clf: KeywordIntentClassifier) -> None:
    """Every Intent enum value should have at least 8 keywords."""
    counts = clf.intent_keyword_counts
    # All 8 declared intents must be present.
    assert set(counts) == {i.value for i in Intent}
    for label, count in counts.items():
        assert count >= 8, f"{label} only has {count} keywords; aim for >= 8"


def test_chitchat_match(clf: KeywordIntentClassifier) -> None:
    """A bare greeting routes to CHITCHAT above the fallback threshold."""
    res = clf.classify("Xin chào")
    assert isinstance(res, IntentResult)
    assert res.intent is Intent.CHITCHAT
    assert res.source == "keyword"
    assert res.score >= INTENT_FALLBACK_THRESHOLD


def test_score_lookup_match(clf: KeywordIntentClassifier) -> None:
    res = clf.classify("điểm chuẩn CNTT 2024")
    assert res.intent is Intent.SCORE_LOOKUP
    assert res.source == "keyword"
    assert "diem chuan" in res.matched


def test_no_diacritic_input(clf: KeywordIntentClassifier) -> None:
    """ASCII-only Vietnamese should still classify correctly."""
    res = clf.classify("diem chuan cntt 2024")
    assert res.intent is Intent.SCORE_LOOKUP
    assert res.source == "keyword"


def test_tuition_lookup_match(clf: KeywordIntentClassifier) -> None:
    res = clf.classify("học phí ngành KTPM")
    assert res.intent is Intent.TUITION_LOOKUP
    assert res.source == "keyword"


def test_compare_match(clf: KeywordIntentClassifier) -> None:
    res = clf.classify("so sánh điểm chuẩn 2023 và 2024")
    assert res.intent is Intent.COMPARE


def test_low_score_falls_back(clf: KeywordIntentClassifier) -> None:
    """Gibberish queries should hit the fallback path."""
    res = clf.classify("qwerty asdf zxcv mnbv lkjh")
    assert res.source == "fallback" or res.intent is DEFAULT_INTENT


def test_intent_priority_chitchat_first(clf: KeywordIntentClassifier) -> None:
    """A pleasantry must NOT suppress a stronger info-intent keyword.

    "chào, cho mình hỏi điểm chuẩn" mixes a chitchat token ("chào") with
    a strong score_lookup phrase ("điểm chuẩn"). The classifier must
    route to SCORE_LOOKUP — pleasantries are not allowed to steal real
    info-intent queries.
    """
    res = clf.classify("chào, cho mình hỏi điểm chuẩn")
    assert res.intent is Intent.SCORE_LOOKUP


def test_empty_query_is_fallback(clf: KeywordIntentClassifier) -> None:
    res = clf.classify("")
    assert res.source == "fallback"
    assert res.intent is DEFAULT_INTENT


def test_admission_method_match(clf: KeywordIntentClassifier) -> None:
    res = clf.classify("Năm 2025 trường xét tuyển theo những phương thức nào?")
    assert res.intent is Intent.ADMISSION_METHOD


def test_deadline_match(clf: KeywordIntentClassifier) -> None:
    res = clf.classify("Khi nào trường công bố kết quả trúng tuyển sớm năm 2024?")
    assert res.intent is Intent.DEADLINE


def test_program_info_match(clf: KeywordIntentClassifier) -> None:
    res = clf.classify("Ngành Công nghệ giáo dục ra trường làm nghề gì?")
    assert res.intent is Intent.PROGRAM_INFO


def test_general_qa_match(clf: KeywordIntentClassifier) -> None:
    res = clf.classify("Mã trường của UTEHY là gì và tên tiếng Anh là gì?")
    assert res.intent is Intent.GENERAL_QA
