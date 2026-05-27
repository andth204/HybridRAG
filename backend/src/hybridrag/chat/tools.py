"""OpenAI function-calling tool schemas + dispatcher (Phase 4D).

This module is the glue between the LLM's tool-call protocol and the
structured-store lookups in :mod:`src.hybridrag.tools.lookup`. Three
public surfaces:

* :data:`OPENAI_TOOL_SCHEMAS` — the JSON-Schema descriptors the LLM
  needs to know about; passed straight to ``client.chat.completions``
  via the ``tools=`` parameter.
* :func:`execute_tool` — dispatch a single tool call by name and return
  a JSON string (the LLM's tool-result payload).
* :func:`tools_for_intent` — narrow the schema list to the tools that
  are relevant for the predicted intent, so the LLM doesn't see
  irrelevant options.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from src.hybridrag.tools.lookup import (
    list_majors_by_campus,
    lookup_score,
    lookup_tuition,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- #
# OpenAI function-calling schemas
# ---------------------------------------------------------------- #
OPENAI_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_score",
            "description": (
                "Look up admission scores (điểm chuẩn) for a given major, "
                "year, optionally filtered by campus or faculty. Use this "
                "when the user asks for specific score numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "major": {
                        "type": "string",
                        "description": (
                            "Major name in Vietnamese (e.g. 'Công nghệ "
                            "thông tin') or abbreviation (e.g. 'CNTT')"
                        ),
                    },
                    "year": {
                        "type": "integer",
                        "description": (
                            "4-digit year e.g. 2024. If omitted, latest "
                            "year is used."
                        ),
                    },
                    "campus": {
                        "type": "string",
                        "description": (
                            "Campus identifier or display name "
                            "(e.g. 'cơ sở 1', 'Hải Dương'). Optional."
                        ),
                    },
                    "faculty": {
                        "type": "string",
                        "description": "Faculty name. Optional.",
                    },
                },
                "required": ["major"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_tuition",
            "description": (
                "Look up tuition (học phí) for a given major and year. "
                "Returns amount in VND."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "major": {
                        "type": "string",
                        "description": "Major name or abbreviation",
                    },
                    "year": {
                        "type": "integer",
                        "description": "4-digit year",
                    },
                },
                "required": ["major"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_majors_by_campus",
            "description": "List all majors offered at a specific campus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "campus": {
                        "type": "string",
                        "description": "Campus identifier or display name",
                    },
                },
                "required": ["campus"],
            },
        },
    },
]


# ---------------------------------------------------------------- #
# Dispatch table
# ---------------------------------------------------------------- #
TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "lookup_score": lookup_score,
    "lookup_tuition": lookup_tuition,
    "list_majors_by_campus": list_majors_by_campus,
}


def execute_tool(name: str, arguments_json: str) -> str:
    """Dispatch a tool call and serialise the result for the LLM.

    The return shape is always a JSON object containing either ``data``
    (success) or ``error`` (failure) so the LLM sees a stable contract.

    Args:
        name:           Tool name as advertised in :data:`OPENAI_TOOL_SCHEMAS`.
        arguments_json: The raw ``arguments`` string the LLM emitted
                        (already JSON-encoded per the OpenAI protocol).

    Returns:
        A JSON-encoded string.
    """
    # 1) Parse arguments. The LLM is supposed to emit valid JSON, but
    #    bad outputs happen and we don't want them to crash the chat.
    try:
        kwargs: dict[str, Any] = json.loads(arguments_json) if arguments_json else {}
    except Exception as exc:  # noqa: BLE001 - intentionally broad
        log.warning("execute_tool: invalid arguments JSON for %s: %s", name, exc)
        return json.dumps({"error": "invalid_arguments"}, ensure_ascii=False)

    if not isinstance(kwargs, dict):
        log.warning("execute_tool: arguments JSON for %s is not an object", name)
        return json.dumps({"error": "invalid_arguments"}, ensure_ascii=False)

    # 2) Look up the function.
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return json.dumps(
            {"error": f"unknown_tool:{name}"},
            ensure_ascii=False,
        )

    # 3) Invoke. Catch TypeError separately so the LLM can be told its
    #    arguments were malformed and try again with a corrected call;
    #    other exceptions become a generic runtime-error envelope.
    try:
        result = func(**kwargs)
    except TypeError as exc:
        return json.dumps(
            {"error": f"bad_arguments:{exc}"},
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001 - external code surface
        log.exception("execute_tool: %s failed", name)
        return json.dumps(
            {"error": "tool_runtime_error", "detail": str(exc)},
            ensure_ascii=False,
        )

    return json.dumps({"data": result}, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- #
# Intent → tool routing
# ---------------------------------------------------------------- #
# Restrict the tools the LLM sees per intent. Tools not listed here are
# never offered, so the model can't waste tokens debating between
# unrelated functions. Intents that resolve through plain RAG (chitchat,
# general_qa, deadline, admission_method) intentionally map to no tools.
INTENT_TO_TOOLS: dict[str, list[str]] = {
    "score_lookup": ["lookup_score"],
    "tuition_lookup": ["lookup_tuition"],
    "program_info": ["list_majors_by_campus"],
    "compare": ["lookup_score", "lookup_tuition"],
}


def tools_for_intent(intent: str) -> list[dict[str, Any]]:
    """Return only the OpenAI tool schemas relevant to ``intent``.

    Accepts the raw string label (matches :class:`Intent` enum values
    via ``Intent.SCORE_LOOKUP.value == "score_lookup"``).
    """
    names = set(INTENT_TO_TOOLS.get(intent, []))
    if not names:
        return []
    return [t for t in OPENAI_TOOL_SCHEMAS if t["function"]["name"] in names]


__all__ = [
    "OPENAI_TOOL_SCHEMAS",
    "TOOL_REGISTRY",
    "INTENT_TO_TOOLS",
    "execute_tool",
    "tools_for_intent",
]
