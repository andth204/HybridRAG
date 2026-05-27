"""Tests for ``src.hybridrag.utils.entity_resolver``."""
from __future__ import annotations

import pytest

from src.hybridrag.utils.entity_resolver import (
    list_canonical,
    reload_entities,
    resolve,
    resolve_all,
)


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    """Drop the cached entity table before each test."""
    reload_entities()


# ---------------------------------------------------------------- #
# resolve()
# ---------------------------------------------------------------- #
def test_resolve_campus_alias() -> None:
    rec = resolve("cơ sở 1", "campus")
    assert rec is not None
    assert rec["canonical"] == "co_so_1"
    assert "Cơ sở 1" == rec.get("display")


def test_resolve_faculty_abbrev() -> None:
    rec = resolve("CNTT", "faculty")
    assert rec is not None
    assert rec["canonical"] == "cntt"


def test_resolve_major_abbrev() -> None:
    rec = resolve("KTPM", "major")
    assert rec is not None
    assert rec["canonical"] == "ky_thuat_phan_mem"


def test_resolve_unknown() -> None:
    assert resolve("xyz unknown random gibberish", "campus") is None


def test_resolve_no_diacritic() -> None:
    rec = resolve("co so 1", "campus")
    assert rec is not None
    assert rec["canonical"] == "co_so_1"


def test_resolve_university_alias() -> None:
    rec = resolve("UTEHY", "university")
    assert rec is not None
    assert rec["canonical"] == "utehy"


def test_resolve_doc_type() -> None:
    rec = resolve("điểm chuẩn", "doc_type")
    assert rec is not None
    assert rec["canonical"] == "diem_chuan"


def test_resolve_empty_returns_none() -> None:
    assert resolve("", "campus") is None
    assert resolve("   ", "campus") is None


def test_resolve_respects_min_score_floor() -> None:
    """A very loose query should fail when ``min_score`` is cranked up."""
    # A random short string shouldn't pass 99/100 threshold even with fuzzy WRatio.
    assert resolve("zz", "campus", min_score=99) is None


# ---------------------------------------------------------------- #
# resolve_all()
# ---------------------------------------------------------------- #
def test_resolve_all_multi() -> None:
    """A typical user query should surface faculty + campus + doc_type hits."""
    hits = resolve_all("Điểm chuẩn CNTT cơ sở 1 năm 2024")
    canonicals = {etype: {r["canonical"] for r in recs} for etype, recs in hits.items()}

    assert "faculty" in canonicals, f"faculty missing — got {canonicals}"
    assert "cntt" in canonicals["faculty"]

    assert "campus" in canonicals, f"campus missing — got {canonicals}"
    assert "co_so_1" in canonicals["campus"]

    assert "doc_type" in canonicals, f"doc_type missing — got {canonicals}"
    assert "diem_chuan" in canonicals["doc_type"]


def test_resolve_all_empty() -> None:
    assert resolve_all("") == {}


def test_resolve_all_restrict_type() -> None:
    hits = resolve_all("Điểm chuẩn CNTT cơ sở 1", entity_types=["campus"])
    assert set(hits.keys()) <= {"campus"}
    assert hits.get("campus")
    assert hits["campus"][0]["canonical"] == "co_so_1"


# ---------------------------------------------------------------- #
# list_canonical()
# ---------------------------------------------------------------- #
def test_list_canonical_campus() -> None:
    canon = list_canonical("campus")
    assert "co_so_1" in canon
    assert "co_so_2" in canon
    assert "co_so_3" in canon


def test_list_canonical_unknown_type() -> None:
    assert list_canonical("does_not_exist") == []
