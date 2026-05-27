"""Tests for ``src.hybridrag.chat.clarifier`` (Phase 4C).

These tests are intentionally hermetic — anything that would hit
``entities.yaml`` for major / campus resolution is monkey-patched on
``resolve_all`` so the test stays deterministic regardless of dict
content changes. The popular-major fallback path DOES read the YAML
because that file is checked in and stable.

NOTE (2026-05-22): the public ``Clarifier.check`` entry point is
short-circuited to return ``None`` per product decision (the bot now
always answers from retrieval instead of asking clarifying questions
back). The internal helpers (``_check_ambiguous_entity``,
``_check_missing_year`` etc.) are intact and still callable, but the
suite below — which exercises ``check`` — is skipped wholesale. To
re-enable, remove the ``pytest.skip`` line below AND remove the early
``return None`` in :meth:`Clarifier.check`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

pytest.skip(
    "Clarifier.check disabled — bot answers from retrieval instead of asking back.",
    allow_module_level=True,
)

from src.hybridrag.chat import clarifier as clarifier_mod
from src.hybridrag.chat.clarifier import (
    DEFAULT_AMBIGUITY_GAP,
    ClarificationRequest,
    Clarifier,
)


# -------------------------------------------------------------------- #
# Fixtures
# -------------------------------------------------------------------- #
@pytest.fixture
def clf() -> Clarifier:
    return Clarifier()


def _major_rec(canonical: str, display: str, score: float | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "canonical": canonical,
        "display": display,
        "aliases": [display],
    }
    if score is not None:
        rec["_score"] = score
    return rec


def _campus_rec(canonical: str, display: str, score: float | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "canonical": canonical,
        "display": display,
        "aliases": [display],
    }
    if score is not None:
        rec["_score"] = score
    return rec


def _patch_resolve_all(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, list[dict[str, Any]]]) -> None:
    """Replace ``resolve_all`` to return the given (entity_type → records) map."""

    def fake_resolve_all(
        text: str,
        *,
        min_score: int = 80,
        max_ngram: int = 4,
        entity_types: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        if entity_types is None:
            return dict(mapping)
        return {k: v for k, v in mapping.items() if k in entity_types}

    monkeypatch.setattr(clarifier_mod, "resolve_all", fake_resolve_all)


# -------------------------------------------------------------------- #
# 1. Ambiguous major
# -------------------------------------------------------------------- #
def test_ambiguous_major_returns_clarify(
    clf: Clarifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_resolve_all(
        monkeypatch,
        {
            "major": [
                _major_rec("ky_thuat_phan_mem",                "Kỹ thuật phần mềm",                85),
                _major_rec("cong_nghe_ky_thuat_co_khi",        "Công nghệ kỹ thuật cơ khí",        83),
                _major_rec("cong_nghe_ky_thuat_dien_dien_tu",  "Công nghệ kỹ thuật điện, điện tử", 80),
            ]
        },
    )
    req = clf.check(
        query="học phí ngành kỹ thuật",
        intent="tuition_lookup",
        session_slots={},
    )
    assert isinstance(req, ClarificationRequest)
    assert req.reason == "ambiguous_major"
    assert req.slot == "major"
    assert len(req.options) == 3
    # First option corresponds to the highest mock score.
    assert req.options[0]["value"] == "ky_thuat_phan_mem"
    # Question must be Vietnamese and mention "ngành".
    assert "ngành" in req.question.lower() or "ngành" in req.question


# -------------------------------------------------------------------- #
# 2. Single specific major — no clarify.
# -------------------------------------------------------------------- #
def test_specific_major_no_clarify(
    clf: Clarifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_resolve_all(
        monkeypatch,
        {
            "major": [
                _major_rec("cong_nghe_thong_tin", "Công nghệ thông tin", 95),
            ]
        },
    )
    req = clf.check(
        query="học phí CNTT",
        intent="tuition_lookup",
        session_slots={"year": 2024},  # avoid the missing_year branch
    )
    assert req is None


# -------------------------------------------------------------------- #
# 3. Missing year on score_lookup.
# -------------------------------------------------------------------- #
def test_missing_year_score_lookup(
    clf: Clarifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CNTT resolves singularly → no ambiguous_major.
    _patch_resolve_all(
        monkeypatch,
        {
            "major": [
                _major_rec("cong_nghe_thong_tin", "Công nghệ thông tin", 95),
            ]
        },
    )
    current = datetime.now(timezone.utc).year
    req = clf.check(
        query="điểm chuẩn CNTT",
        intent="score_lookup",
        session_slots={},
    )
    assert isinstance(req, ClarificationRequest)
    assert req.reason == "missing_year"
    assert req.slot == "year"
    values = [opt["value"] for opt in req.options]
    assert values == [str(current), str(current - 1), str(current - 2)]
    # Labels should be Vietnamese "Năm YYYY".
    for opt in req.options:
        assert opt["label"].startswith("Năm")
    assert "năm" in req.question.lower()


# -------------------------------------------------------------------- #
# 4. Session-supplied year suppresses missing_year.
# -------------------------------------------------------------------- #
def test_session_year_satisfies(
    clf: Clarifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_resolve_all(
        monkeypatch,
        {
            "major": [
                _major_rec("cong_nghe_thong_tin", "Công nghệ thông tin", 95),
            ]
        },
    )
    req = clf.check(
        query="điểm chuẩn CNTT",
        intent="score_lookup",
        session_slots={"year": 2024},
    )
    # No retrieval docs and no remaining ambiguity → expect None.
    assert req is None


# -------------------------------------------------------------------- #
# 5. Missing major on score_lookup.
# -------------------------------------------------------------------- #
def test_missing_major_score_lookup(
    clf: Clarifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No majors resolve from the query at all.
    _patch_resolve_all(monkeypatch, {})
    req = clf.check(
        query="điểm chuẩn năm 2024",
        intent="score_lookup",
        session_slots={},
    )
    assert isinstance(req, ClarificationRequest)
    assert req.reason == "missing_major"
    assert req.slot == "major"
    assert len(req.options) >= 1
    # The popular-majors loader pulls from entities.yaml; CNTT should be top.
    values = {opt["value"] for opt in req.options}
    assert "cong_nghe_thong_tin" in values
    # Question references "ngành".
    assert "ngành" in req.question


# -------------------------------------------------------------------- #
# 6. Ambiguous campus.
# -------------------------------------------------------------------- #
def test_ambiguous_campus(
    clf: Clarifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_resolve_all(
        monkeypatch,
        {
            # The clarifier should also not trip on a single confident major
            # — keep majors empty here so we exercise the campus branch.
            "major": [],
            "campus": [
                _campus_rec("co_so_1", "Cơ sở 1", 90),
                _campus_rec("co_so_2", "Cơ sở 2", 88),
            ],
        },
    )
    req = clf.check(
        query="cơ sở dạy CNTT",
        intent="program_info",
        session_slots={},
    )
    assert isinstance(req, ClarificationRequest)
    assert req.reason == "ambiguous_campus"
    assert req.slot == "campus"
    assert len(req.options) == 2
    assert {opt["value"] for opt in req.options} == {"co_so_1", "co_so_2"}


# -------------------------------------------------------------------- #
# 7. Low retrieval recall.
# -------------------------------------------------------------------- #
def test_low_recall_triggers(
    clf: Clarifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_resolve_all(monkeypatch, {})  # nothing else fires
    req = clf.check(
        query="trường có gì hay không",
        intent="general_qa",
        session_slots={"major": "cong_nghe_thong_tin"},  # avoid missing_major
        retrieval_docs=[{"rerank_score": 0.1, "content": "..."}],
    )
    assert isinstance(req, ClarificationRequest)
    assert req.reason == "low_recall"
    assert req.options == []
    assert req.slot is None
    # Question is a generic Vietnamese rephrasing prompt.
    assert "rõ" in req.question or "chi tiết" in req.question


# -------------------------------------------------------------------- #
# 8. Chitchat skips year (and everything else).
# -------------------------------------------------------------------- #
def test_chitchat_skips_year_check(
    clf: Clarifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_resolve_all(monkeypatch, {})
    req = clf.check(
        query="Chào bạn",
        intent="chitchat",
        session_slots={},
    )
    assert req is None


# -------------------------------------------------------------------- #
# 9. Priority — ambiguous_major beats missing_year.
# -------------------------------------------------------------------- #
def test_priority_major_over_year(
    clf: Clarifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two close major candidates → ambiguous_major.
    # Intent is score_lookup with NO year → missing_year would also fire.
    _patch_resolve_all(
        monkeypatch,
        {
            "major": [
                _major_rec("ky_thuat_phan_mem",         "Kỹ thuật phần mềm",         90),
                _major_rec("cong_nghe_ky_thuat_co_khi", "Công nghệ kỹ thuật cơ khí", 88),
            ],
        },
    )
    req = clf.check(
        query="điểm chuẩn ngành kỹ thuật",
        intent="score_lookup",
        session_slots={},
    )
    assert isinstance(req, ClarificationRequest)
    assert req.reason == "ambiguous_major"


# -------------------------------------------------------------------- #
# 10. Options text format — numbered list with "1." / "2.".
# -------------------------------------------------------------------- #
def test_options_text_format(
    clf: Clarifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_resolve_all(
        monkeypatch,
        {
            "major": [
                _major_rec("ky_thuat_phan_mem",                "Kỹ thuật phần mềm",                85),
                _major_rec("cong_nghe_ky_thuat_co_khi",        "Công nghệ kỹ thuật cơ khí",        83),
            ],
        },
    )
    req = clf.check(
        query="học phí ngành kỹ thuật",
        intent="tuition_lookup",
        session_slots={"year": 2024},
    )
    assert isinstance(req, ClarificationRequest)
    assert "1." in req.question
    assert "2." in req.question
    # Both display names should be present in the rendered text.
    assert "Kỹ thuật phần mềm" in req.question
    assert "Công nghệ kỹ thuật cơ khí" in req.question
