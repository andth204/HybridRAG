"""Unit tests for ``src.hybridrag.kg.scores_repo`` and ``tuition_repo``.

These tests stub the ``borrow()`` context manager so they assert on the
SQL strings and bind parameters without touching Postgres. The key
contract we want to lock down:

* ``upsert`` uses ``INSERT ... ON CONFLICT (...) DO UPDATE``.
* writes (``upsert`` / ``delete_by_source``) call ``conn.commit()``;
  reads (``lookup`` / ``list_*``) do NOT.
* SQL goes out via ``%s`` placeholders (no f-string interpolation of
  user data).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.hybridrag.kg import scores_repo as scores_mod
from src.hybridrag.kg import tuition_repo as tuition_mod
from src.hybridrag.kg.scores_repo import AdmissionScoresRepo
from src.hybridrag.kg.tuition_repo import TuitionRepo


# --------------------------------------------------------------------- #
# Fake connection helpers
# --------------------------------------------------------------------- #
class FakeCursor:
    """Captures every ``execute`` call and replays a queued ``fetch*`` result."""

    def __init__(self, fetchone_results: list[Any], fetchall_results: list[Any]) -> None:
        self._fetchone_queue = list(fetchone_results)
        self._fetchall_queue = list(fetchall_results)
        self.executed: list[tuple[str, Any]] = []
        self.rowcount = 0

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchone(self) -> Any:
        return self._fetchone_queue.pop(0) if self._fetchone_queue else None

    def fetchall(self) -> Any:
        return self._fetchall_queue.pop(0) if self._fetchall_queue else []


class FakeConn:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.commit = MagicMock(name="conn.commit")
        self.rollback = MagicMock(name="conn.rollback")

    def cursor(self, *args: Any, **kwargs: Any) -> FakeCursor:
        return self._cursor


def install_fake_borrow(
    monkeypatch: pytest.MonkeyPatch,
    module,
    *,
    fetchone_results: list[Any] | None = None,
    fetchall_results: list[Any] | None = None,
) -> tuple[FakeCursor, FakeConn]:
    """Replace ``module.borrow`` with a context manager yielding a fake conn."""
    cursor = FakeCursor(
        fetchone_results=fetchone_results or [],
        fetchall_results=fetchall_results or [],
    )
    conn = FakeConn(cursor)

    @contextmanager
    def fake_borrow(_dsn: str):
        yield conn

    monkeypatch.setattr(module, "borrow", fake_borrow)
    return cursor, conn


# --------------------------------------------------------------------- #
# admission_scores: upsert
# --------------------------------------------------------------------- #
def test_admission_upsert_uses_on_conflict_and_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    """upsert must use ON CONFLICT DO UPDATE *and* commit the connection."""
    returned_row = {
        "id": 42,
        "campus": "co_so_1",
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
    cursor, conn = install_fake_borrow(
        monkeypatch, scores_mod, fetchone_results=[returned_row],
    )

    repo = AdmissionScoresRepo(dsn="postgres://test")
    row = repo.upsert(
        campus="co_so_1",
        faculty="cntt",
        major_canonical="cong_nghe_thong_tin",
        major_code="7480201",
        year=2024,
        method="THPT",
        subject_combo="A00",
        score=17.0,
        source_file="Điểm 2024.md",
    )

    assert len(cursor.executed) == 1
    sql, params = cursor.executed[0]
    assert "INSERT INTO admission_scores" in sql
    assert "ON CONFLICT" in sql
    assert "DO UPDATE" in sql
    assert "%s" in sql, "must be parameterized"
    assert params == (
        "co_so_1",
        "cntt",
        "cong_nghe_thong_tin",
        "7480201",
        2024,
        "THPT",
        "A00",
        17.0,
        None,         # note
        "Điểm 2024.md",
        None,         # source_chunk_id
    )

    conn.commit.assert_called_once()
    assert row.id == 42
    assert row.major_canonical == "cong_nghe_thong_tin"
    assert row.score == 17.0


def test_admission_lookup_builds_where_clause_and_does_not_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "id": 1, "campus": "", "faculty": "cntt",
            "major_canonical": "cong_nghe_thong_tin",
            "major_code": "7480201", "year": 2024, "method": "THPT",
            "subject_combo": "A00", "score": 17.0, "note": None,
            "source_file": None, "source_chunk_id": None,
        }
    ]
    cursor, conn = install_fake_borrow(
        monkeypatch, scores_mod, fetchall_results=[rows],
    )
    repo = AdmissionScoresRepo(dsn="postgres://test")

    out = repo.lookup(major_canonical="cong_nghe_thong_tin", year=2024, limit=10)

    sql, params = cursor.executed[0]
    assert "SELECT" in sql and "FROM admission_scores" in sql
    assert "major_canonical = %s" in sql
    assert "year = %s" in sql
    assert "ORDER BY year DESC" in sql
    assert "LIMIT %s" in sql
    assert params == ("cong_nghe_thong_tin", 2024, 10)
    conn.commit.assert_not_called()
    assert len(out) == 1
    assert out[0].score == 17.0


def test_admission_lookup_no_filters_emits_no_where(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor, conn = install_fake_borrow(
        monkeypatch, scores_mod, fetchall_results=[[]],
    )
    repo = AdmissionScoresRepo(dsn="postgres://test")
    repo.lookup(limit=5)
    sql, params = cursor.executed[0]
    assert "WHERE" not in sql
    assert params == (5,)
    conn.commit.assert_not_called()


def test_admission_list_majors_filtered_by_year(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor, conn = install_fake_borrow(
        monkeypatch, scores_mod, fetchall_results=[[("cong_nghe_thong_tin",), ("ke_toan",)]],
    )
    repo = AdmissionScoresRepo(dsn="postgres://test")
    out = repo.list_majors(year=2024)
    sql, params = cursor.executed[0]
    assert "DISTINCT major_canonical" in sql
    assert "year = %s" in sql
    assert params == (2024,)
    conn.commit.assert_not_called()
    assert out == ["cong_nghe_thong_tin", "ke_toan"]


def test_admission_list_years_filtered_by_major(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor, conn = install_fake_borrow(
        monkeypatch, scores_mod, fetchall_results=[[(2024,), (2023,)]],
    )
    repo = AdmissionScoresRepo(dsn="postgres://test")
    out = repo.list_years(major_canonical="cong_nghe_thong_tin")
    sql, params = cursor.executed[0]
    assert "DISTINCT year" in sql
    assert "major_canonical = %s" in sql
    assert params == ("cong_nghe_thong_tin",)
    conn.commit.assert_not_called()
    assert out == [2024, 2023]


def test_admission_delete_by_source_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor, conn = install_fake_borrow(monkeypatch, scores_mod)
    cursor.rowcount = 3
    repo = AdmissionScoresRepo(dsn="postgres://test")
    deleted = repo.delete_by_source("Điểm 2024.md")
    sql, params = cursor.executed[0]
    assert sql.strip().upper().startswith("DELETE FROM ADMISSION_SCORES")
    assert "source_file = %s" in sql
    assert params == ("Điểm 2024.md",)
    conn.commit.assert_called_once()
    assert deleted == 3


def test_admission_list_majors_by_campus(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor, conn = install_fake_borrow(
        monkeypatch, scores_mod, fetchall_results=[[("cong_nghe_thong_tin",)]],
    )
    repo = AdmissionScoresRepo(dsn="postgres://test")
    out = repo.list_majors_by_campus("co_so_1")
    sql, params = cursor.executed[0]
    assert "campus = %s" in sql
    assert params == ("co_so_1",)
    conn.commit.assert_not_called()
    assert out == ["cong_nghe_thong_tin"]


# --------------------------------------------------------------------- #
# tuition repo
# --------------------------------------------------------------------- #
def test_tuition_upsert_uses_on_conflict_and_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    returned_row = {
        "id": 7,
        "major_canonical": "cong_nghe_thong_tin",
        "year": 2025,
        "amount_vnd": 510_000,
        "unit": "per_credit",
        "note": None,
        "source_file": "tuyen_sinh_247.md",
    }
    cursor, conn = install_fake_borrow(
        monkeypatch, tuition_mod, fetchone_results=[returned_row],
    )

    repo = TuitionRepo(dsn="postgres://test")
    row = repo.upsert(
        major_canonical="cong_nghe_thong_tin",
        year=2025,
        amount_vnd=510_000,
        unit="per_credit",
        source_file="tuyen_sinh_247.md",
    )
    sql, params = cursor.executed[0]
    assert "INSERT INTO tuition" in sql
    assert "ON CONFLICT" in sql
    assert "DO UPDATE" in sql
    assert "%s" in sql
    assert params == (
        "cong_nghe_thong_tin",
        2025,
        510_000,
        "per_credit",
        None,
        "tuyen_sinh_247.md",
    )
    conn.commit.assert_called_once()
    assert row.id == 7
    assert row.amount_vnd == 510_000


def test_tuition_lookup_builds_where_clause_and_does_not_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
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
    cursor, conn = install_fake_borrow(
        monkeypatch, tuition_mod, fetchall_results=[rows],
    )
    repo = TuitionRepo(dsn="postgres://test")
    out = repo.lookup(major_canonical="cong_nghe_thong_tin", year=2025, limit=5)
    sql, params = cursor.executed[0]
    assert "SELECT" in sql and "FROM tuition" in sql
    assert "major_canonical = %s" in sql
    assert "year = %s" in sql
    assert "LIMIT %s" in sql
    assert params == ("cong_nghe_thong_tin", 2025, 5)
    conn.commit.assert_not_called()
    assert len(out) == 1
    assert out[0].amount_vnd == 510_000


def test_tuition_list_majors_does_not_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor, conn = install_fake_borrow(
        monkeypatch, tuition_mod, fetchall_results=[[("cong_nghe_thong_tin",)]],
    )
    repo = TuitionRepo(dsn="postgres://test")
    out = repo.list_majors()
    sql, _ = cursor.executed[0]
    assert "DISTINCT major_canonical" in sql
    conn.commit.assert_not_called()
    assert out == ["cong_nghe_thong_tin"]


def test_tuition_delete_by_source_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor, conn = install_fake_borrow(monkeypatch, tuition_mod)
    cursor.rowcount = 2
    repo = TuitionRepo(dsn="postgres://test")
    deleted = repo.delete_by_source("tuyen_sinh_247.md")
    sql, params = cursor.executed[0]
    assert sql.strip().upper().startswith("DELETE FROM TUITION")
    assert "source_file = %s" in sql
    assert params == ("tuyen_sinh_247.md",)
    conn.commit.assert_called_once()
    assert deleted == 2
