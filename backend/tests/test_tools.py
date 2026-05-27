"""Tests for ``src.hybridrag.chat.tools``.

The dispatcher is intentionally thin (parse JSON → call function →
re-serialise) so the tests focus on the error envelope contract and
the intent-to-tool routing.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from src.hybridrag.chat import tools as tools_mod


# ---------------------------------------------------------------- #
# Patch helpers
# ---------------------------------------------------------------- #
@pytest.fixture
def patched_registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the live tool registry with stubs so the test never
    touches the real KG / DB layer."""
    calls: dict[str, list[dict[str, Any]]] = {}

    def fake_lookup_score(**kwargs: Any) -> list[dict[str, Any]]:
        calls.setdefault("lookup_score", []).append(kwargs)
        return [
            {
                "major_canonical": "cong_nghe_thong_tin",
                "year": kwargs.get("year", 2024),
                "score": 17.0,
            }
        ]

    def fake_lookup_tuition(**kwargs: Any) -> list[dict[str, Any]]:
        calls.setdefault("lookup_tuition", []).append(kwargs)
        return [
            {
                "major_canonical": "cong_nghe_thong_tin",
                "year": kwargs.get("year", 2025),
                "amount_vnd": 1_790_000,
            }
        ]

    def fake_list_majors_by_campus(campus: str, **kwargs: Any) -> list[str]:
        calls.setdefault("list_majors_by_campus", []).append({"campus": campus, **kwargs})
        return ["cong_nghe_thong_tin", "ke_toan"]

    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup_score", fake_lookup_score)
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup_tuition", fake_lookup_tuition)
    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "list_majors_by_campus",
        fake_list_majors_by_campus,
    )
    return calls


# ---------------------------------------------------------------- #
# execute_tool
# ---------------------------------------------------------------- #
def test_execute_lookup_score(patched_registry: dict[str, Any]) -> None:
    out = tools_mod.execute_tool(
        "lookup_score",
        json.dumps({"major": "CNTT", "year": 2024}),
    )
    parsed = json.loads(out)
    assert "data" in parsed
    assert isinstance(parsed["data"], list)
    assert parsed["data"][0]["score"] == 17.0
    assert patched_registry["lookup_score"][0] == {"major": "CNTT", "year": 2024}


def test_execute_lookup_tuition(patched_registry: dict[str, Any]) -> None:
    out = tools_mod.execute_tool(
        "lookup_tuition",
        json.dumps({"major": "CNTT", "year": 2025}),
    )
    parsed = json.loads(out)
    assert parsed["data"][0]["amount_vnd"] == 1_790_000


def test_execute_list_majors_by_campus(patched_registry: dict[str, Any]) -> None:
    out = tools_mod.execute_tool(
        "list_majors_by_campus",
        json.dumps({"campus": "co_so_1"}),
    )
    parsed = json.loads(out)
    assert parsed["data"] == ["cong_nghe_thong_tin", "ke_toan"]


def test_execute_invalid_json() -> None:
    """Malformed JSON arguments must surface as ``invalid_arguments``."""
    out = tools_mod.execute_tool("lookup_score", "{not valid json")
    parsed = json.loads(out)
    assert parsed == {"error": "invalid_arguments"}


def test_execute_empty_arguments_string(patched_registry: dict[str, Any]) -> None:
    """Empty arguments string means "call with no kwargs"."""
    out = tools_mod.execute_tool("lookup_score", "")
    parsed = json.loads(out)
    # The stub returns at least one row even without kwargs.
    assert "data" in parsed
    assert patched_registry["lookup_score"][0] == {}


def test_execute_non_object_arguments() -> None:
    """A JSON value that is not an object must surface as invalid args."""
    out = tools_mod.execute_tool("lookup_score", json.dumps([1, 2, 3]))
    parsed = json.loads(out)
    assert parsed == {"error": "invalid_arguments"}


def test_execute_unknown_tool() -> None:
    out = tools_mod.execute_tool("not_a_tool", json.dumps({}))
    parsed = json.loads(out)
    assert parsed["error"].startswith("unknown_tool:")


def test_execute_bad_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling a registered tool with the wrong kwarg yields ``bad_arguments``."""

    def real_signature_lookup_score(*, major: str, **_: Any) -> list[dict[str, Any]]:
        # ``major`` is required — calling without it must raise TypeError.
        return [{"major": major}]

    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "lookup_score",
        real_signature_lookup_score,
    )
    out = tools_mod.execute_tool("lookup_score", json.dumps({"year": 2024}))
    parsed = json.loads(out)
    assert parsed["error"].startswith("bad_arguments")


def test_execute_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected exception inside the tool is wrapped, not propagated."""

    def boom(**_: Any) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup_score", boom)
    out = tools_mod.execute_tool("lookup_score", json.dumps({"major": "CNTT"}))
    parsed = json.loads(out)
    assert parsed["error"] == "tool_runtime_error"
    assert "kaboom" in parsed["detail"]


# ---------------------------------------------------------------- #
# Schema invariants
# ---------------------------------------------------------------- #
def test_schemas_cover_registered_tools() -> None:
    """Every advertised schema must point to a registered function."""
    advertised = {entry["function"]["name"] for entry in tools_mod.OPENAI_TOOL_SCHEMAS}
    registered = set(tools_mod.TOOL_REGISTRY)
    assert advertised <= registered, advertised - registered


def test_schemas_have_required_shape() -> None:
    """Quick smoke test on the OpenAI tool-schema shape."""
    for entry in tools_mod.OPENAI_TOOL_SCHEMAS:
        assert entry["type"] == "function"
        fn = entry["function"]
        assert "name" in fn and isinstance(fn["name"], str)
        assert "description" in fn and isinstance(fn["description"], str)
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params


# ---------------------------------------------------------------- #
# tools_for_intent
# ---------------------------------------------------------------- #
def test_tools_for_intent_score_lookup() -> None:
    out = tools_mod.tools_for_intent("score_lookup")
    assert len(out) == 1
    assert out[0]["function"]["name"] == "lookup_score"


def test_tools_for_intent_tuition_lookup() -> None:
    out = tools_mod.tools_for_intent("tuition_lookup")
    assert len(out) == 1
    assert out[0]["function"]["name"] == "lookup_tuition"


def test_tools_for_intent_compare_returns_two() -> None:
    out = tools_mod.tools_for_intent("compare")
    names = {entry["function"]["name"] for entry in out}
    assert names == {"lookup_score", "lookup_tuition"}


def test_tools_for_intent_unknown() -> None:
    """Unknown intent label → empty list, never an error."""
    assert tools_mod.tools_for_intent("not_a_real_intent") == []


def test_tools_for_intent_chitchat_yields_nothing() -> None:
    """Chitchat and other RAG-only intents must not expose any tool."""
    assert tools_mod.tools_for_intent("chitchat") == []
    assert tools_mod.tools_for_intent("general_qa") == []
    assert tools_mod.tools_for_intent("deadline") == []
    assert tools_mod.tools_for_intent("admission_method") == []
