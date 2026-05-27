"""Tests for ``src.hybridrag.tools.lookup``.

These tests use a hand-rolled stub repo (no DB, no mocks libs beyond
``unittest.mock``) to verify that:

* free-form major / campus / faculty strings are resolved via the
  entity dictionary before being handed to the repo,
* unknown majors short-circuit to an empty list (the dialogue layer
  needs this signal to fall back to RAG),
* ``list_majors_by_campus`` flows through the campus resolver.
"""
from __future__ import annotations

from typing import Any

import pytest

from src.hybridrag.kg.scores_repo import AdmissionScore
from src.hybridrag.kg.tuition_repo import TuitionRow
from src.hybridrag.tools import lookup
from src.hybridrag.utils.entity_resolver import reload_entities


@pytest.fixture(autouse=True)
def _fresh_entity_cache() -> None:
    """Drop the cached entity dictionary so each test sees the YAML fresh."""
    reload_entities()


# ------------------------------------------------------------------ #
# Stub repos — capture every kwarg the lookup module passes through.
# ------------------------------------------------------------------ #
class StubScoresRepo:
    def __init__(self, rows: list[AdmissionScore] | None = None) -> None:
        self.rows = rows or []
        self.lookup_calls: list[dict[str, Any]] = []
        self.list_majors_by_campus_calls: list[str] = []
        self.list_majors_by_campus_return: list[str] = []

    def lookup(self, **kwargs: Any) -> list[AdmissionScore]:
        self.lookup_calls.append(kwargs)
        return list(self.rows)

    def list_majors_by_campus(self, campus: str) -> list[str]:
        self.list_majors_by_campus_calls.append(campus)
        return list(self.list_majors_by_campus_return)


class StubTuitionRepo:
    def __init__(self, rows: list[TuitionRow] | None = None) -> None:
        self.rows = rows or []
        self.lookup_calls: list[dict[str, Any]] = []

    def lookup(self, **kwargs: Any) -> list[TuitionRow]:
        self.lookup_calls.append(kwargs)
        return list(self.rows)


def _make_score(**overrides: Any) -> AdmissionScore:
    base: dict[str, Any] = {
        "id": 1,
        "campus": "",
        "faculty": "cntt",
        "major_canonical": "cong_nghe_thong_tin",
        "major_code": "7480201",
        "year": 2024,
        "method": "THPT",
        "subject_combo": "A00",
        "score": 17.0,
        "note": None,
        "source_file": "Điểm 2024.md",
        "source_chunk_id": None,
    }
    base.update(overrides)
    return AdmissionScore(**base)


def _make_tuition(**overrides: Any) -> TuitionRow:
    base: dict[str, Any] = {
        "id": 1,
        "major_canonical": "cong_nghe_thong_tin",
        "year": 2025,
        "amount_vnd": 510_000,
        "unit": "per_credit",
        "note": None,
        "source_file": "tuyen_sinh_247.md",
    }
    base.update(overrides)
    return TuitionRow(**base)


# ------------------------------------------------------------------ #
# lookup_score
# ------------------------------------------------------------------ #
def test_lookup_score_resolves_major_alias_to_canonical() -> None:
    """A user query of "CNTT" must be canonicalized before hitting the repo."""
    stub = StubScoresRepo(rows=[_make_score()])

    out = lookup.lookup_score(major="CNTT", year=2024, repo=stub)

    assert len(stub.lookup_calls) == 1
    call = stub.lookup_calls[0]
    assert call["major_canonical"] == "cong_nghe_thong_tin"
    assert call["year"] == 2024
    # Optional fields default to None when caller doesn't supply them.
    assert call["campus"] is None
    assert call["faculty"] is None
    assert call["method"] is None
    assert call["limit"] == 5

    assert isinstance(out, list) and len(out) == 1
    row = out[0]
    assert row["major_canonical"] == "cong_nghe_thong_tin"
    assert row["score"] == 17.0
    assert row["year"] == 2024


def test_lookup_score_resolves_campus_and_faculty_aliases() -> None:
    stub = StubScoresRepo(rows=[])
    lookup.lookup_score(
        major="Công nghệ thông tin",
        campus="cơ sở 1",
        faculty="khoa CNTT",
        method="THPT",
        repo=stub,
    )
    call = stub.lookup_calls[0]
    assert call["major_canonical"] == "cong_nghe_thong_tin"
    assert call["campus"] == "co_so_1"
    assert call["faculty"] == "cntt"
    assert call["method"] == "THPT"


def test_lookup_score_unresolved_major_returns_empty() -> None:
    """An unknown major must NOT silently query the repo without a filter."""
    stub = StubScoresRepo(rows=[_make_score()])

    out = lookup.lookup_score(major="zzz nonsense major name", repo=stub)

    assert out == []
    assert stub.lookup_calls == []


def test_lookup_score_no_filters_calls_repo_with_nones() -> None:
    """When the caller passes nothing, repo gets no filters (and all rows)."""
    stub = StubScoresRepo(rows=[_make_score(), _make_score(id=2, year=2023, score=17.5)])

    out = lookup.lookup_score(repo=stub)

    assert len(stub.lookup_calls) == 1
    call = stub.lookup_calls[0]
    assert call["major_canonical"] is None
    assert call["year"] is None
    assert call["campus"] is None
    assert call["faculty"] is None
    assert call["method"] is None
    assert len(out) == 2


# ------------------------------------------------------------------ #
# lookup_tuition
# ------------------------------------------------------------------ #
def test_lookup_tuition_resolves_major_alias() -> None:
    stub = StubTuitionRepo(rows=[_make_tuition()])

    out = lookup.lookup_tuition(major="CNTT", year=2025, repo=stub)

    assert len(stub.lookup_calls) == 1
    call = stub.lookup_calls[0]
    assert call["major_canonical"] == "cong_nghe_thong_tin"
    assert call["year"] == 2025
    assert call["unit"] is None
    assert call["limit"] == 5

    assert out == [
        {
            "id": 1,
            "major_canonical": "cong_nghe_thong_tin",
            "year": 2025,
            "amount_vnd": 510_000,
            "unit": "per_credit",
            "note": None,
            "source_file": "tuyen_sinh_247.md",
        }
    ]


def test_lookup_tuition_unresolved_major_short_circuits() -> None:
    stub = StubTuitionRepo(rows=[_make_tuition()])
    out = lookup.lookup_tuition(major="completely unknown major xyz", repo=stub)
    assert out == []
    assert stub.lookup_calls == []


def test_lookup_tuition_respects_unit_filter() -> None:
    stub = StubTuitionRepo(rows=[])
    lookup.lookup_tuition(major="CNTT", unit="per_credit", repo=stub)
    assert stub.lookup_calls[0]["unit"] == "per_credit"


# ------------------------------------------------------------------ #
# list_majors_by_campus
# ------------------------------------------------------------------ #
def test_list_majors_by_campus_flows_through_to_repo() -> None:
    stub = StubScoresRepo()
    stub.list_majors_by_campus_return = ["cong_nghe_thong_tin", "ke_toan"]

    out = lookup.list_majors_by_campus("co_so_1", repo=stub)

    assert stub.list_majors_by_campus_calls == ["co_so_1"]
    assert out == ["cong_nghe_thong_tin", "ke_toan"]


def test_list_majors_by_campus_resolves_freeform_input() -> None:
    """User-facing strings like "cơ sở Khoái Châu" must canonicalize first."""
    stub = StubScoresRepo()
    stub.list_majors_by_campus_return = ["cong_nghe_thong_tin"]

    out = lookup.list_majors_by_campus("Khoái Châu", repo=stub)

    assert stub.list_majors_by_campus_calls == ["co_so_1"]
    assert out == ["cong_nghe_thong_tin"]


def test_list_majors_by_campus_unresolved_returns_empty() -> None:
    stub = StubScoresRepo()
    out = lookup.list_majors_by_campus("a campus that does not exist xyz", repo=stub)
    assert out == []
    assert stub.list_majors_by_campus_calls == []
