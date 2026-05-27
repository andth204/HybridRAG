"""Tests for ``src.hybridrag.chat.slot_filler.expand_coreference``."""
from __future__ import annotations

from src.hybridrag.chat.session_state import SessionState, SlotValue
from src.hybridrag.chat.slot_filler import expand_coreference


# --------------------------------------------------------------------- #
# campus
# --------------------------------------------------------------------- #
def test_expand_campus_coref() -> None:
    state = SessionState(
        session_id="sess-1",
        slots={"campus": SlotValue(value="co_so_1", display="Cơ sở 1", turn=1)},
    )
    out = expand_coreference("Ngành nào dạy ở đó", state)
    assert "Cơ sở 1" in out
    assert "ở đó" not in out
    assert out == "Ngành nào dạy ở Cơ sở 1"


def test_expand_co_so_kia_coref() -> None:
    """'cơ sở đó' / 'cơ sở kia' also expand to the campus slot."""
    state = SessionState(
        session_id="sess-1",
        slots={"campus": SlotValue(value="co_so_3", display="Cơ sở 3", turn=1)},
    )
    out = expand_coreference("Học phí cơ sở đó bao nhiêu", state)
    assert "Cơ sở 3" in out
    assert "cơ sở đó" not in out.lower()


# --------------------------------------------------------------------- #
# major
# --------------------------------------------------------------------- #
def test_expand_major_coref() -> None:
    state = SessionState(
        session_id="sess-1",
        slots={
            "major": SlotValue(
                value="cong_nghe_thong_tin",
                display="Công nghệ thông tin",
                turn=1,
            )
        },
    )
    out = expand_coreference("Ngành đó học phí bao nhiêu?", state)
    assert "Công nghệ thông tin" in out
    assert "Ngành Công nghệ thông tin" in out


# --------------------------------------------------------------------- #
# year / faculty
# --------------------------------------------------------------------- #
def test_expand_year_coref() -> None:
    state = SessionState(
        session_id="sess-1",
        slots={"year": SlotValue(value=2024, display="2024", turn=1)},
    )
    out = expand_coreference("Điểm chuẩn năm đó là bao nhiêu", state)
    assert "năm 2024" in out


def test_expand_faculty_coref() -> None:
    state = SessionState(
        session_id="sess-1",
        slots={"faculty": SlotValue(value="cntt", display="Công nghệ thông tin", turn=1)},
    )
    out = expand_coreference("Khoa đó có bao nhiêu giảng viên?", state)
    assert "khoa Công nghệ thông tin" in out.lower() or "Khoa Công nghệ thông tin" in out


# --------------------------------------------------------------------- #
# no-op cases
# --------------------------------------------------------------------- #
def test_no_state_no_change() -> None:
    """Coref expression but the matching slot isn't set → no rewrite."""
    state = SessionState(session_id="sess-1")  # empty slots
    q = "Ngành đó học phí bao nhiêu?"
    assert expand_coreference(q, state) == q


def test_no_coref_no_change() -> None:
    """Plain query without any coref expressions is returned verbatim."""
    state = SessionState(
        session_id="sess-1",
        slots={"campus": SlotValue(value="co_so_1", display="Cơ sở 1", turn=1)},
    )
    q = "Học phí ngành Công nghệ thông tin"
    assert expand_coreference(q, state) == q


def test_partial_state_only_expands_known_slots() -> None:
    """If campus is known but major isn't, only ``ở đó`` is expanded."""
    state = SessionState(
        session_id="sess-1",
        slots={"campus": SlotValue(value="co_so_1", display="Cơ sở 1", turn=1)},
    )
    out = expand_coreference("Ngành đó ở đó học phí bao nhiêu", state)
    # campus coref expanded
    assert "ở Cơ sở 1" in out
    # major coref kept verbatim (no major slot)
    assert "Ngành đó" in out or "ngành đó" in out


def test_empty_query_is_returned_as_is() -> None:
    state = SessionState(session_id="sess-1")
    assert expand_coreference("", state) == ""
