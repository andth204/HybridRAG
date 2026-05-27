"""Tests for ``src.hybridrag.ingestion.extractors.scores_extractor``.

All tests inject a stub OpenAI client; the real ``openai`` library is
never called. The unit under test is the *normalization* layer that
sits between the model's free-form JSON and the repo's strict schema.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.hybridrag.ingestion.extractors import scores_extractor
from src.hybridrag.ingestion.extractors.scores_extractor import (
    extract_scores_from_text,
)
from src.hybridrag.utils.entity_resolver import reload_entities


@pytest.fixture(autouse=True)
def _fresh_entity_cache() -> None:
    reload_entities()


def _make_client(payload: dict[str, Any] | str) -> Any:
    """Build a stub OpenAI client whose .chat.completions.create returns ``payload``."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    message = SimpleNamespace(content=raw)
    choice = SimpleNamespace(message=message)
    completion = SimpleNamespace(choices=[choice])
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


# ------------------------------------------------------------------ #
# Happy-path normalization
# ------------------------------------------------------------------ #
def test_extract_normalizes_and_canonicalizes_major() -> None:
    """A well-formed model response yields one normalized row per item."""
    payload = {
        "rows": [
            {
                "campus": "",
                "faculty": "",
                "major": "Công nghệ thông tin",
                "major_code": "7480201",
                "year": 2024,
                "method": "THPT",
                "subject_combo": "A00",
                "score": 17.0,
                "note": None,
            },
            {
                "campus": "",
                "faculty": "",
                "major": "CNTT",  # alias should still resolve.
                "major_code": None,
                "year": "2023",   # stringified year should be coerced.
                "method": "thpt",  # case-insensitive method.
                "subject_combo": None,
                "score": "17.5",  # stringified score.
                "note": "ghi chú",
            },
        ]
    }
    client = _make_client(payload)

    out = extract_scores_from_text(
        "Bảng điểm chuẩn 2024 CNTT", source_file="Điểm 2024.md", client=client,
    )

    assert len(out) == 2
    row0, row1 = out

    # Row 0
    assert row0["major_canonical"] == "cong_nghe_thong_tin"
    assert row0["major_code"] == "7480201"
    assert row0["year"] == 2024
    assert row0["method"] == "THPT"
    assert row0["subject_combo"] == "A00"
    assert row0["score"] == 17.0
    assert row0["source_file"] == "Điểm 2024.md"

    # Row 1 — stringified inputs survive coercion
    assert row1["major_canonical"] == "cong_nghe_thong_tin"
    # Falls back to the dictionary's `code` when the model said null.
    assert row1["major_code"] == "7480201"
    assert row1["year"] == 2023
    assert row1["method"] == "THPT"  # uppercased
    assert row1["score"] == 17.5
    assert row1["note"] == "ghi chú"
    assert row1["source_file"] == "Điểm 2024.md"


def test_extract_passes_through_to_admission_repo_kwargs() -> None:
    """Every key in the result must be a kwarg accepted by ``AdmissionScoresRepo.upsert``."""
    from inspect import signature
    from src.hybridrag.kg.scores_repo import AdmissionScoresRepo

    payload = {
        "rows": [
            {
                "campus": "co_so_1",
                "faculty": "cntt",
                "major": "Công nghệ thông tin",
                "major_code": "7480201",
                "year": 2024,
                "method": "THPT",
                "subject_combo": "A00",
                "score": 17.0,
                "note": None,
            }
        ]
    }
    out = extract_scores_from_text(
        "x", source_file="Điểm 2024.md", client=_make_client(payload),
    )
    accepted = set(signature(AdmissionScoresRepo.upsert).parameters.keys()) - {"self"}
    for row in out:
        assert set(row.keys()) <= accepted, f"unknown keys: {set(row.keys()) - accepted}"


# ------------------------------------------------------------------ #
# Defensive behavior
# ------------------------------------------------------------------ #
def test_extract_drops_row_without_year() -> None:
    payload = {
        "rows": [
            {
                "major": "Công nghệ thông tin",
                "year": None,
                "score": 17.0,
            },
            {
                "major": "Công nghệ thông tin",
                "year": 2024,
                "score": 17.0,
            },
        ]
    }
    out = extract_scores_from_text("x", client=_make_client(payload))
    assert len(out) == 1
    assert out[0]["year"] == 2024


def test_extract_drops_row_with_unresolvable_major() -> None:
    payload = {
        "rows": [
            {
                "major": "asdfghjkl-does-not-exist",
                "year": 2024,
                "score": 17.0,
            },
            {
                "major": "Công nghệ thông tin",
                "year": 2024,
                "score": 17.0,
            },
        ]
    }
    out = extract_scores_from_text("x", client=_make_client(payload))
    assert len(out) == 1
    assert out[0]["major_canonical"] == "cong_nghe_thong_tin"


def test_extract_handles_invalid_json() -> None:
    """A malformed model response degrades to []; never raises."""
    client = _make_client("definitely not json")
    out = extract_scores_from_text("x", client=client)
    assert out == []


def test_extract_handles_bare_array_response() -> None:
    """Defensive: if the model emits a bare JSON array, we still parse it."""
    payload = [
        {"major": "Công nghệ thông tin", "year": 2024, "score": 17.0},
    ]
    out = extract_scores_from_text("x", client=_make_client(payload))
    assert len(out) == 1
    assert out[0]["major_canonical"] == "cong_nghe_thong_tin"


def test_extract_empty_input_short_circuits() -> None:
    client = MagicMock()
    assert extract_scores_from_text("", client=client) == []
    assert extract_scores_from_text("   \n  ", client=client) == []
    client.chat.completions.create.assert_not_called()


def test_extract_no_api_key_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an OPENAI_API_KEY the extractor never tries to call the API."""
    monkeypatch.setattr(scores_extractor.settings, "OPENAI_API_KEY", "")
    out = extract_scores_from_text(
        "Bảng điểm CNTT 2024.", source_file="Điểm 2024.md",
    )
    assert out == []


def test_extract_swallows_llm_exception() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    out = extract_scores_from_text("anything", client=client)
    assert out == []


def test_extract_rejects_out_of_range_year() -> None:
    payload = {
        "rows": [
            {"major": "Công nghệ thông tin", "year": 1999, "score": 17.0},
            {"major": "Công nghệ thông tin", "year": 2100, "score": 17.0},
            {"major": "Công nghệ thông tin", "year": 2024, "score": 17.0},
        ]
    }
    out = extract_scores_from_text("x", client=_make_client(payload))
    assert [r["year"] for r in out] == [2024]


def test_extract_passes_correct_call_args() -> None:
    """The OpenAI call must use gpt-4o-mini, temp=0, JSON mode."""
    payload = {"rows": []}
    client = _make_client(payload)
    extract_scores_from_text("x", client=client)
    client.chat.completions.create.assert_called_once()
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0
    assert kwargs["response_format"] == {"type": "json_object"}
    # System + user message in that order.
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "x"
