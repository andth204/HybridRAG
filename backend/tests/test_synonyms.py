"""Tests for ``src.hybridrag.utils.synonyms``."""
from __future__ import annotations

import pytest

from src.hybridrag.utils.synonyms import (
    _normalize_key,
    normalize_query,
    normalize_text,
    reload_synonyms,
)


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    """Drop the LRU cache before each test so changes-on-disk are picked up."""
    reload_synonyms()


# ---------------------------------------------------------------- #
# _normalize_key
# ---------------------------------------------------------------- #
def test_normalize_key_lowercases_and_strips_diacritics() -> None:
    assert _normalize_key("CNTT") == "cntt"
    assert _normalize_key("Công Nghệ Thông Tin") == "cong nghe thong tin"
    assert _normalize_key("đại học") == "dai hoc"
    assert _normalize_key("  Đại  học  ") == "dai hoc"
    assert _normalize_key("") == ""


# ---------------------------------------------------------------- #
# normalize_text
# ---------------------------------------------------------------- #
def test_basic_substitution() -> None:
    """Case-insensitive replacement of a known abbreviation."""
    out = normalize_text("Điểm chuẩn CNTT")
    assert "công nghệ thông tin" in out.lower()
    # "Điểm chuẩn" itself is in the dict (lowercased canonical),
    # so we should not see the raw abbreviation any more.
    assert "cntt" not in out.lower()


def test_no_diacritic_input() -> None:
    """Plain-ASCII Vietnamese input should still match the dictionary."""
    out = normalize_text("diem chuan ktpm 2024")
    assert "kỹ thuật phần mềm" in out.lower()
    assert "ktpm" not in out.lower()
    assert "2024" in out


def test_unknown_word_unchanged() -> None:
    """A passage with nothing in the synonym dict should round-trip."""
    src = "Hello world"
    assert normalize_text(src) == src


def test_multiword_alias() -> None:
    """Greedy match should cover multi-token aliases (3+ tokens)."""
    out = normalize_text("ĐHSPKT HY")
    assert "đại học sư phạm kỹ thuật hưng yên" in out.lower()


def test_empty_string() -> None:
    """Empty / whitespace-only input must be returned unchanged."""
    assert normalize_text("") == ""
    assert normalize_text("   ") == "   "


def test_expand_disabled_is_identity() -> None:
    assert normalize_text("Điểm chuẩn CNTT", expand=False) == "Điểm chuẩn CNTT"


def test_normalize_query_alias() -> None:
    """``normalize_query`` is a thin alias for ``normalize_text``."""
    assert normalize_query("CNTT") == normalize_text("CNTT")


def test_mixed_passage_preserves_unknown_tokens() -> None:
    """Unmatched tokens (e.g. years, numbers) should be emitted verbatim."""
    out = normalize_text("Học phí CNTT năm 2024 là bao nhiêu")
    low = out.lower()
    assert "công nghệ thông tin" in low
    assert "học phí" in low
    assert "2024" in out
