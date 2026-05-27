"""Tests for ``src.hybridrag.chat.slot_filler.SlotFiller``.

The filler is exercised against the real entity resolver (so we
verify the YAML dictionary actually canonicalizes "CNTT" →
"cong_nghe_thong_tin") but the repo layer is replaced with an
in-memory fake so no live Postgres is needed.
"""
from __future__ import annotations

from typing import Optional

import pytest

from src.hybridrag.chat.session_state import (
    SessionState,
    SessionStateRepo,
    SlotValue,
)
from src.hybridrag.chat.slot_filler import SlotFiller
from src.hybridrag.utils.entity_resolver import reload_entities


@pytest.fixture(autouse=True)
def _fresh_entity_cache() -> None:
    reload_entities()


# --------------------------------------------------------------------- #
# In-memory repo fake
# --------------------------------------------------------------------- #
class FakeSessionStateRepo:
    """Minimal in-memory stand-in for ``SessionStateRepo``."""

    SUPPORTED_SLOTS = SessionStateRepo.SUPPORTED_SLOTS

    def __init__(self) -> None:
        self.store: dict[str, SessionState] = {}
        self.reset_calls: list[str] = []
        self.upsert_calls: list[SessionState] = []

    def get(self, session_id: str) -> SessionState:
        existing = self.store.get(session_id)
        if existing is not None:
            # Return a shallow copy so the caller can mutate without
            # corrupting our store.
            return SessionState(
                session_id=existing.session_id,
                slots=dict(existing.slots),
                last_intent=existing.last_intent,
                last_query=existing.last_query,
                updated_at=existing.updated_at,
            )
        return SessionState(session_id=session_id)

    def upsert(self, state: SessionState) -> SessionState:
        self.upsert_calls.append(state)
        stored = SessionState(
            session_id=state.session_id,
            slots=dict(state.slots),
            last_intent=state.last_intent,
            last_query=state.last_query,
            updated_at=state.updated_at,
        )
        self.store[state.session_id] = stored
        return stored

    def reset(self, session_id: str) -> None:
        self.reset_calls.append(session_id)
        self.store.pop(session_id, None)


# --------------------------------------------------------------------- #
# extract_slots
# --------------------------------------------------------------------- #
def test_extract_year_from_query() -> None:
    filler = SlotFiller(repo=FakeSessionStateRepo())
    out = filler.extract_slots("Điểm chuẩn 2024", turn_index=1)
    assert "year" in out
    assert out["year"].value == 2024
    assert out["year"].confidence == 1.0
    assert out["year"].turn == 1
    assert out["year"].display == "2024"


def test_extract_campus_alias() -> None:
    filler = SlotFiller(repo=FakeSessionStateRepo())
    out = filler.extract_slots("Cơ sở 1 dạy ngành gì", turn_index=2)
    assert "campus" in out
    assert out["campus"].value == "co_so_1"
    assert out["campus"].display == "Cơ sở 1"
    assert out["campus"].turn == 2


def test_extract_major_abbrev() -> None:
    filler = SlotFiller(repo=FakeSessionStateRepo())
    out = filler.extract_slots("CNTT có khó không", turn_index=1)
    # "CNTT" can match either faculty.cntt or major.cong_nghe_thong_tin.
    # The slot filler stores both — the dialogue layer decides which one
    # to use depending on the intent. The deliverable explicitly asks
    # for the major slot to be populated.
    assert "major" in out, f"major missing — got {out.keys()}"
    assert out["major"].value == "cong_nghe_thong_tin"
    assert out["major"].display == "Công nghệ thông tin"


def test_extract_multiple_slots() -> None:
    filler = SlotFiller(repo=FakeSessionStateRepo())
    out = filler.extract_slots(
        "điểm chuẩn ngành CNTT cơ sở 1 năm 2024", turn_index=3
    )
    # year (regex)
    assert "year" in out and out["year"].value == 2024
    # campus
    assert "campus" in out and out["campus"].value == "co_so_1"
    # major (CNTT → cong_nghe_thong_tin)
    assert "major" in out and out["major"].value == "cong_nghe_thong_tin"


def test_no_slots() -> None:
    filler = SlotFiller(repo=FakeSessionStateRepo())
    out = filler.extract_slots("Xin chào", turn_index=1)
    # "Xin chào" carries no slot signal; allowed to be empty.
    assert "year" not in out
    assert "campus" not in out
    assert "major" not in out


def test_extract_empty_query() -> None:
    filler = SlotFiller(repo=FakeSessionStateRepo())
    assert filler.extract_slots("", turn_index=0) == {}
    assert filler.extract_slots("   ", turn_index=0) == {}


# --------------------------------------------------------------------- #
# update — full read/merge/write loop
# --------------------------------------------------------------------- #
def test_update_writes_to_repo() -> None:
    fake = FakeSessionStateRepo()
    filler = SlotFiller(repo=fake)

    state = filler.update("sess-1", "Điểm chuẩn CNTT 2024", turn_index=1)

    assert len(fake.upsert_calls) == 1
    assert "year" in state.slots
    assert state.slots["year"].value == 2024
    assert "major" in state.slots
    assert state.slots["major"].value == "cong_nghe_thong_tin"
    assert state.last_query == "Điểm chuẩn CNTT 2024"


def test_reset_detection() -> None:
    """An explicit reset phrase nukes the slot frame."""
    fake = FakeSessionStateRepo()
    # Pre-populate state with a stale slot.
    fake.upsert(SessionState(
        session_id="sess-1",
        slots={"campus": SlotValue(value="co_so_1", display="Cơ sở 1", turn=1)},
        last_query="trước đó",
    ))

    filler = SlotFiller(repo=fake)
    out = filler.update("sess-1", "quên đi câu hỏi cũ giúp tôi", turn_index=2)

    assert "sess-1" in fake.reset_calls, "reset() must be called on explicit reset phrases"
    assert out.slots == {}, f"expected empty slot frame after reset, got {out.slots}"


def test_reset_detection_no_diacritics() -> None:
    """'quen di' (no accents) also triggers reset."""
    fake = FakeSessionStateRepo()
    fake.upsert(SessionState(
        session_id="sess-1",
        slots={"campus": SlotValue(value="co_so_1", display="Cơ sở 1", turn=1)},
    ))
    filler = SlotFiller(repo=fake)
    filler.update("sess-1", "quen di nhe", turn_index=2)
    assert "sess-1" in fake.reset_calls


# --------------------------------------------------------------------- #
# decay
# --------------------------------------------------------------------- #
def test_decay_drops_stale_and_keeps_fresh() -> None:
    """Slot set at turn=2, current turn=9, decay_turns=6 → dropped.

    A newly-extracted slot at the current turn survives.
    """
    fake = FakeSessionStateRepo()
    fake.upsert(SessionState(
        session_id="sess-1",
        slots={
            "campus": SlotValue(value="co_so_1", display="Cơ sở 1", turn=2),
        },
    ))

    filler = SlotFiller(repo=fake, decay_turns=6)
    # Query that introduces a fresh major slot at turn 9.
    out = filler.update("sess-1", "CNTT là gì", turn_index=9)

    # Stale campus (turn=2, age=7 > decay_turns=6) gets decayed.
    assert "campus" not in out.slots, (
        f"stale campus should be dropped at age=7 with decay_turns=6 — got {out.slots}"
    )
    # Fresh major slot survives.
    assert "major" in out.slots
    assert out.slots["major"].turn == 9


def test_decay_keeps_within_window() -> None:
    """A slot exactly at the window edge survives."""
    fake = FakeSessionStateRepo()
    fake.upsert(SessionState(
        session_id="sess-1",
        slots={"campus": SlotValue(value="co_so_1", display="Cơ sở 1", turn=4)},
    ))
    filler = SlotFiller(repo=fake, decay_turns=6)
    # turn 10, age 6, == decay_turns → still kept (the bound is inclusive).
    out = filler.update("sess-1", "xin chào", turn_index=10)
    assert "campus" in out.slots, f"slot at boundary age==decay_turns should survive, got {out.slots}"


def test_decay_missing_turn_field_treated_as_ancient() -> None:
    """Legacy SlotValue with turn=0 (the default) decays away naturally."""
    fake = FakeSessionStateRepo()
    fake.upsert(SessionState(
        session_id="sess-1",
        slots={
            # turn defaults to 0 — simulating a row written before
            # Phase 4B introduced the turn field.
            "campus": SlotValue(value="co_so_1", display="Cơ sở 1"),
        },
    ))
    filler = SlotFiller(repo=fake, decay_turns=6)
    out = filler.update("sess-1", "xin chào", turn_index=20)
    # age = 20 - 0 = 20 > 6 → decayed.
    assert "campus" not in out.slots


# --------------------------------------------------------------------- #
# filters_for_retrieval
# --------------------------------------------------------------------- #
def test_filters_for_retrieval() -> None:
    filler = SlotFiller(repo=FakeSessionStateRepo())
    state = SessionState(
        session_id="sess-1",
        slots={
            "campus": SlotValue(value="co_so_1", display="Cơ sở 1", turn=1),
            "year":   SlotValue(value=2024,        display="2024",     turn=1),
        },
    )
    filters = filler.filters_for_retrieval(state)
    assert filters == {"campus": "co_so_1", "year": 2024}


def test_filters_for_retrieval_empty_state() -> None:
    filler = SlotFiller(repo=FakeSessionStateRepo())
    state = SessionState(session_id="sess-1")
    assert filler.filters_for_retrieval(state) == {}


def test_filters_for_retrieval_skips_null_values() -> None:
    filler = SlotFiller(repo=FakeSessionStateRepo())
    state = SessionState(
        session_id="sess-1",
        slots={
            "campus": SlotValue(value=None, display=None, turn=1),
            "year":   SlotValue(value=2024, display="2024", turn=1),
        },
    )
    filters = filler.filters_for_retrieval(state)
    assert filters == {"year": 2024}


# --------------------------------------------------------------------- #
# decay_turns floor (defensive)
# --------------------------------------------------------------------- #
def test_decay_turns_zero_floored_to_one() -> None:
    """decay_turns=0 would erase every slot; defensively floored to 1."""
    filler = SlotFiller(repo=FakeSessionStateRepo(), decay_turns=0)
    assert filler.decay_turns == 1
