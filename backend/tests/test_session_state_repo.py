"""Unit tests for ``src.hybridrag.chat.session_state.SessionStateRepo``.

The tests stub ``db_pool.borrow`` so we assert on SQL strings + bind
parameters + commit-on-write without any live Postgres. Same harness
as ``test_kg_repos.py``.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.hybridrag.chat import session_state as ss_mod
from src.hybridrag.chat.session_state import (
    SessionState,
    SessionStateRepo,
    SlotValue,
)


# --------------------------------------------------------------------- #
# Fake connection helpers (mirror test_kg_repos.py)
# --------------------------------------------------------------------- #
class FakeCursor:
    def __init__(
        self,
        fetchone_results: list[Any] | None = None,
        fetchall_results: list[Any] | None = None,
    ) -> None:
        self._fetchone_queue = list(fetchone_results or [])
        self._fetchall_queue = list(fetchall_results or [])
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
    *,
    fetchone_results: list[Any] | None = None,
    fetchall_results: list[Any] | None = None,
) -> tuple[FakeCursor, FakeConn]:
    cursor = FakeCursor(fetchone_results=fetchone_results, fetchall_results=fetchall_results)
    conn = FakeConn(cursor)

    @contextmanager
    def fake_borrow(_dsn: str):
        yield conn

    monkeypatch.setattr(ss_mod, "borrow", fake_borrow)
    return cursor, conn


# --------------------------------------------------------------------- #
# get()
# --------------------------------------------------------------------- #
def test_get_missing_session_returns_default_and_does_not_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get() for a missing session must NOT issue an INSERT or commit."""
    cursor, conn = install_fake_borrow(monkeypatch, fetchone_results=[None])

    repo = SessionStateRepo(dsn="postgres://test")
    state = repo.get("00000000-0000-0000-0000-000000000001")

    # Exactly one statement was issued — the SELECT.
    assert len(cursor.executed) == 1, f"expected exactly one SQL exec, got {cursor.executed}"
    sql, params = cursor.executed[0]
    upper_sql = sql.upper()
    # Must be a read; ``updated_at`` appears in the column list, so we
    # check for the SQL statement-keyword forms with a trailing space.
    assert upper_sql.lstrip().startswith("SELECT"), f"expected SELECT, got: {sql!r}"
    assert "chat_session_state" in sql
    assert "INSERT " not in upper_sql
    assert "UPDATE " not in upper_sql
    assert "DELETE " not in upper_sql
    assert params == ("00000000-0000-0000-0000-000000000001",)
    conn.commit.assert_not_called()

    # And we get a default empty state back.
    assert isinstance(state, SessionState)
    assert state.session_id == "00000000-0000-0000-0000-000000000001"
    assert state.slots == {}
    assert state.last_intent is None
    assert state.last_query is None


def test_get_existing_row_hydrates_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    """A row with JSONB slots round-trips into typed SlotValue objects."""
    row = {
        "session_id": "abc",
        "slots": {
            "campus": {
                "value": "co_so_1",
                "display": "Cơ sở 1",
                "set_at": "2026-05-21T12:34:56+00:00",
                "confidence": 0.95,
                "turn": 3,
            },
            "year": {
                "value": 2024,
                "display": "2024",
                "confidence": 1.0,
                "turn": 3,
            },
        },
        "last_intent": "score_lookup",
        "last_query": "điểm chuẩn 2024",
        "updated_at": None,
    }
    cursor, conn = install_fake_borrow(monkeypatch, fetchone_results=[row])

    repo = SessionStateRepo(dsn="postgres://test")
    state = repo.get("abc")

    conn.commit.assert_not_called()
    assert state.last_intent == "score_lookup"
    assert state.last_query == "điểm chuẩn 2024"
    assert "campus" in state.slots
    assert state.slots["campus"].value == "co_so_1"
    assert state.slots["campus"].display == "Cơ sở 1"
    assert state.slots["campus"].turn == 3
    assert state.slots["year"].value == 2024
    assert state.slots["year"].turn == 3


# --------------------------------------------------------------------- #
# upsert()
# --------------------------------------------------------------------- #
def test_upsert_emits_on_conflict_and_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    returned_row = {
        "session_id": "abc",
        "slots": {"campus": {"value": "co_so_1", "display": "Cơ sở 1",
                              "set_at": None, "confidence": 1.0, "turn": 1}},
        "last_intent": None,
        "last_query": "x",
        "updated_at": None,
    }
    cursor, conn = install_fake_borrow(monkeypatch, fetchone_results=[returned_row])

    repo = SessionStateRepo(dsn="postgres://test")
    state = SessionState(
        session_id="abc",
        slots={"campus": SlotValue(value="co_so_1", display="Cơ sở 1", turn=1)},
        last_query="x",
    )
    out = repo.upsert(state)

    assert len(cursor.executed) == 1
    sql, params = cursor.executed[0]
    assert "INSERT INTO chat_session_state" in sql
    assert "ON CONFLICT (session_id) DO UPDATE" in sql
    assert "%s" in sql
    # ``upsert`` replaces the slot map wholesale (not a JSONB merge) —
    # that's set_slot's job.
    assert params[0] == "abc"
    # params[1] is psycopg2.extras.Json — its ``adapted`` attr exposes the dict
    json_arg = params[1]
    assert hasattr(json_arg, "adapted")
    assert "campus" in json_arg.adapted
    assert json_arg.adapted["campus"]["value"] == "co_so_1"
    assert params[2] is None  # last_intent
    assert params[3] == "x"   # last_query

    conn.commit.assert_called_once()
    assert isinstance(out, SessionState)
    assert "campus" in out.slots


# --------------------------------------------------------------------- #
# set_slot() — the JSONB merge case the dialogue layer cares about
# --------------------------------------------------------------------- #
def test_set_slot_uses_jsonb_merge_and_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    """set_slot must use ``chat_session_state.slots || EXCLUDED.slots`` to merge."""
    returned_row = {
        "session_id": "abc",
        "slots": {"major": {"value": "cong_nghe_thong_tin", "display": "Công nghệ thông tin",
                             "set_at": None, "confidence": 1.0, "turn": 2}},
        "last_intent": None,
        "last_query": None,
        "updated_at": None,
    }
    cursor, conn = install_fake_borrow(monkeypatch, fetchone_results=[returned_row])

    repo = SessionStateRepo(dsn="postgres://test")
    out = repo.set_slot(
        "abc",
        "major",
        SlotValue(value="cong_nghe_thong_tin", display="Công nghệ thông tin", turn=2),
    )

    assert len(cursor.executed) == 1
    sql, params = cursor.executed[0]
    # The crucial guarantee: existing slots survive a single-slot update.
    assert "chat_session_state.slots || EXCLUDED.slots" in sql, (
        "set_slot must use the JSONB || merge so other slots are preserved"
    )
    assert "INSERT INTO chat_session_state" in sql
    assert "ON CONFLICT (session_id) DO UPDATE" in sql

    assert params[0] == "abc"
    json_arg = params[1]
    assert hasattr(json_arg, "adapted")
    # Only the new slot is in the payload — the merge happens server-side.
    assert json_arg.adapted == {
        "major": {
            "value": "cong_nghe_thong_tin",
            "display": "Công nghệ thông tin",
            "set_at": None,
            "confidence": 1.0,
            "turn": 2,
        }
    }

    conn.commit.assert_called_once()
    assert "major" in out.slots


def test_set_slot_invalid_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown slot names must raise ValueError before touching the DB."""
    cursor, conn = install_fake_borrow(monkeypatch)

    repo = SessionStateRepo(dsn="postgres://test")
    with pytest.raises(ValueError, match="Unsupported slot name"):
        repo.set_slot("abc", "not_a_real_slot", SlotValue(value="x"))

    assert cursor.executed == [], "no SQL should be executed for an invalid slot name"
    conn.commit.assert_not_called()


# --------------------------------------------------------------------- #
# clear_slot()
# --------------------------------------------------------------------- #
def test_clear_slot_uses_jsonb_minus_and_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    returned_row = {
        "session_id": "abc",
        "slots": {},
        "last_intent": None,
        "last_query": None,
        "updated_at": None,
    }
    cursor, conn = install_fake_borrow(monkeypatch, fetchone_results=[returned_row])

    repo = SessionStateRepo(dsn="postgres://test")
    repo.clear_slot("abc", "campus")

    sql, params = cursor.executed[0]
    assert "UPDATE chat_session_state" in sql
    assert "slots - %s" in sql
    assert params == ("campus", "abc")
    conn.commit.assert_called_once()


def test_clear_slot_invalid_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor, conn = install_fake_borrow(monkeypatch)
    repo = SessionStateRepo(dsn="postgres://test")
    with pytest.raises(ValueError):
        repo.clear_slot("abc", "bogus_slot")
    assert cursor.executed == []
    conn.commit.assert_not_called()


# --------------------------------------------------------------------- #
# reset() / touch()
# --------------------------------------------------------------------- #
def test_reset_deletes_row_and_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor, conn = install_fake_borrow(monkeypatch)
    repo = SessionStateRepo(dsn="postgres://test")
    repo.reset("abc")

    sql, params = cursor.executed[0]
    assert sql.strip().upper().startswith("DELETE FROM CHAT_SESSION_STATE")
    assert params == ("abc",)
    conn.commit.assert_called_once()


def test_touch_inserts_with_empty_slots_and_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor, conn = install_fake_borrow(monkeypatch)
    repo = SessionStateRepo(dsn="postgres://test")
    repo.touch("abc", last_intent="score_lookup", last_query="điểm 2024")

    sql, params = cursor.executed[0]
    assert "INSERT INTO chat_session_state" in sql
    assert "ON CONFLICT (session_id) DO UPDATE" in sql
    assert "'{}'::jsonb" in sql  # default empty slot map
    assert params == ("abc", "score_lookup", "điểm 2024")
    conn.commit.assert_called_once()
