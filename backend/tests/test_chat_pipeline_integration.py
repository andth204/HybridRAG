"""Integration tests for the Phase 4 chat dispatcher.

These tests exercise :func:`src.api.routers.chat.chat_answer` end-to-end
through FastAPI's ``TestClient``, but every collaborator (Postgres,
OpenAI, retrieval, reranker) is replaced with an in-memory or callable
stub so the suite stays hermetic.

Coverage targets — one test per acceptance criterion in the Phase 4
integration spec:

1. ``test_clarification_short_circuits`` — clarifier fires → no LLM
   chat-completion call, no answer generation, persisted message is
   marked ``type="clarification"``.
2. ``test_intent_chitchat_routes_chitchat`` — chitchat path uses
   ``stream_chitchat_answer`` and never invokes retrieval.
3. ``test_intent_score_lookup_calls_tool`` — score_lookup path advertises
   the ``lookup_score`` schema, executes the tool, and surfaces the
   tool answer.
4. ``test_tool_error_falls_back_to_rag`` — tool returns ``{"error":...}``
   → answer is generated from the RAG branch instead.
5. ``test_metadata_includes_intent_slots_tools`` — assistant message
   metadata contains all the Phase 4 keys.
6. ``test_slot_failure_does_not_break_pipeline`` — ``SlotFiller.update``
   raises → pipeline continues with an empty state and logs a warning.
7. ``test_coreference_expands_query`` — "ngành đó học phí" + state
   with major.display="CNTT" → the rewriter sees the expanded query.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.core.dependencies import AuthContext, get_auth_context
from src.api.routers import chat as chat_module
from src.hybridrag.chat.clarifier import ClarificationRequest
from src.hybridrag.chat.message import ChatMessage
from src.hybridrag.chat.session import ChatSession
from src.hybridrag.chat.session_state import SessionState, SlotValue
from src.hybridrag.router.intent_router import IntentRouter
from src.hybridrag.router.intents import Intent, IntentResult


# -------------------------------------------------------------------- #
# Fake repos / collaborators (process-wide, never touch Postgres)
# -------------------------------------------------------------------- #
class FakeChatSessionRepo:
    """In-memory stand-in for ``ChatSessionRepo``."""

    def __init__(self, *_, **__) -> None:  # signature parity with the real repo
        self.sessions: dict[str, ChatSession] = {}
        self.touch_calls: list[str] = []

    def create(self, user_id: str, title: Optional[str] = None) -> ChatSession:
        sid = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        session = ChatSession(id=sid, user_id=user_id, title=title, created_at=now, updated_at=now)
        self.sessions[sid] = session
        return session

    def get(self, session_id: str, user_id: Optional[str] = None) -> Optional[ChatSession]:
        sess = self.sessions.get(session_id)
        if sess is None:
            return None
        if user_id and sess.user_id != user_id:
            return None
        return sess

    def rename(self, session_id: str, title: str, user_id: Optional[str] = None) -> bool:
        sess = self.sessions.get(session_id)
        if sess is None:
            return False
        self.sessions[session_id] = ChatSession(
            id=sess.id, user_id=sess.user_id, title=title,
            created_at=sess.created_at, updated_at=datetime.now(timezone.utc),
        )
        return True

    def touch(self, session_id: str) -> bool:
        self.touch_calls.append(session_id)
        return session_id in self.sessions

    def delete(self, session_id: str, user_id: Optional[str] = None) -> bool:
        return self.sessions.pop(session_id, None) is not None


class FakeChatMessageRepo:
    """In-memory stand-in for ``ChatMessageRepo`` (records every create)."""

    def __init__(self, *_, **__) -> None:
        self.messages: list[ChatMessage] = []
        self.create_calls: list[dict[str, Any]] = []

    def create(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        parent_message_id: Optional[str] = None,
        revision_number: int = 1,
        is_edited: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            parent_message_id=parent_message_id,
            revision_number=revision_number,
            is_edited=is_edited,
            metadata=metadata,
            created_at=datetime.now(timezone.utc),
        )
        self.messages.append(msg)
        self.create_calls.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "metadata": metadata,
            }
        )
        return msg

    def load_history(self, session_id: str, *, limit: int = 200, offset: int = 0, ascending: bool = True) -> list[ChatMessage]:
        return [m for m in self.messages if m.session_id == session_id and m.role in {"user", "assistant"}]


@dataclass
class _ReflectStub:
    """Replacement for ``query_reflection`` exposing the same ``reflect`` API."""

    calls: list[tuple[str, list[dict[str, str]]]]

    async def reflect(self, query: str, history: list[dict[str, str]]) -> str:
        self.calls.append((query, list(history)))
        return query


class _FakeIntentRouter:
    """Fixed-result intent router."""

    def __init__(self, result: IntentResult) -> None:
        self._result = result
        self.classify_calls: list[str] = []

    async def classify(self, query: str) -> IntentResult:
        self.classify_calls.append(query)
        return self._result


class _FakeClarifier:
    def __init__(self, result: ClarificationRequest | None) -> None:
        self._result = result
        self.check_calls: list[dict[str, Any]] = []

    def check(
        self,
        *,
        query: str,
        intent: str | None,
        session_slots: dict[str, Any] | None,
        retrieval_docs: list[dict[str, Any]] | None,
    ) -> ClarificationRequest | None:
        self.check_calls.append(
            {
                "query": query,
                "intent": intent,
                "session_slots": dict(session_slots or {}),
                "retrieval_docs": list(retrieval_docs or []),
            }
        )
        return self._result


class _FakeSlotFiller:
    """A slot filler that returns a fixed state (or raises) and records calls."""

    def __init__(
        self,
        state: SessionState | None = None,
        raise_on_update: Exception | None = None,
    ) -> None:
        self._state = state
        self._raise = raise_on_update
        self.update_calls: list[dict[str, Any]] = []
        self.repo = _FakeStateRepo()

    def update(self, session_id: str, query: str, *, turn_index: int) -> SessionState:
        self.update_calls.append(
            {"session_id": session_id, "query": query, "turn_index": turn_index}
        )
        if self._raise is not None:
            raise self._raise
        if self._state is not None:
            self._state.session_id = session_id
            return self._state
        return SessionState(session_id=session_id)

    def filters_for_retrieval(self, state: SessionState) -> dict[str, Any]:
        """Phase 5.5 contract — project ``state.slots`` into a flat
        ``{slot: value}`` dict so the chat router can forward it to the
        retrieval backend. Mirrors ``SlotFiller.filters_for_retrieval``.
        """
        out: dict[str, Any] = {}
        for name, sv in (state.slots or {}).items():
            if sv is None or sv.value is None:
                continue
            out[name] = sv.value
        return out


class _FakeStateRepo:
    """Just enough to honour ``filler.repo.touch(...)`` from chat.py."""

    def __init__(self) -> None:
        self.touch_calls: list[dict[str, Any]] = []

    def touch(
        self,
        session_id: str,
        *,
        last_intent: str | None = None,
        last_query: str | None = None,
    ) -> None:
        self.touch_calls.append(
            {"session_id": session_id, "last_intent": last_intent, "last_query": last_query}
        )


class _StubAnswerGenerator:
    """Minimal :class:`AnswerGenerator` replacement."""

    def __init__(self, *, sync_text: str = "RAG answer text", stream_pieces: list[str] | None = None) -> None:
        self.sync_text = sync_text
        self.stream_pieces = stream_pieces or ["RAG ", "answer ", "text"]
        self.answer_text_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def answer_text(self, *, query: str, retrieved_docs: list[dict[str, Any]], **_: Any) -> str:
        self.answer_text_calls.append({"query": query, "retrieved_docs": retrieved_docs})
        return self.sync_text

    async def stream_answer(
        self, *, query: str, retrieved_docs: list[dict[str, Any]], **_: Any
    ) -> AsyncIterator[str]:
        self.stream_calls.append({"query": query, "retrieved_docs": retrieved_docs})
        for piece in self.stream_pieces:
            yield piece

    def build_context(self, docs: list[dict[str, Any]], **_: Any) -> str:
        return "fake-context"


# -------------------------------------------------------------------- #
# Pytest fixtures
# -------------------------------------------------------------------- #
@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Wire up an isolated FastAPI app with the chat router + every
    collaborator stubbed out. Each test gets a fresh harness."""

    fake_session_repo = FakeChatSessionRepo()
    fake_message_repo = FakeChatMessageRepo()
    reflect_stub = _ReflectStub(calls=[])

    monkeypatch.setattr(chat_module, "ChatSessionRepo", lambda *a, **kw: fake_session_repo)
    monkeypatch.setattr(chat_module, "ChatMessageRepo", lambda *a, **kw: fake_message_repo)
    monkeypatch.setattr(chat_module, "query_reflection", reflect_stub)
    monkeypatch.setattr(chat_module, "_assert_openai_key", lambda: None)

    # Default doc set — individual tests can override via canned_docs.
    canned_docs: list[dict[str, Any]] = [{"key": "doc1.md", "content": "..."}]

    async def fake_retrieve(
        query: str,
        search_mode: str,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        # Phase 5.5: the production signature now accepts filters; the
        # legacy tests don't care about the value, just keyword tolerance.
        return list(canned_docs)

    monkeypatch.setattr(chat_module, "_retrieve_docs", fake_retrieve)
    # ``compact_sources`` happens to be pure, but stub it for stable assertions.
    monkeypatch.setattr(chat_module, "compact_sources", lambda docs, limit=5: [d.get("key", "") for d in docs])

    # generate_with_tools / stream_chitchat_answer / get_answer_generator
    # default to silently fall through to RAG. Tests override these via the
    # ``harness`` dict.
    answer_gen = _StubAnswerGenerator()
    monkeypatch.setattr(chat_module, "get_answer_generator", lambda: answer_gen)

    async def default_chitchat_stream(query: str, timeout: float = 20.0):
        # No-op default. Chitchat-specific tests override this.
        yield "hi from chitchat"

    monkeypatch.setattr(chat_module, "stream_chitchat_answer", default_chitchat_stream)

    async def default_gen_with_tools(**kwargs):  # pragma: no cover - defensive default
        return "", [], False

    monkeypatch.setattr(chat_module, "generate_with_tools", default_gen_with_tools)

    # Intent / clarifier / slot caches — tests override via patched getters.
    default_intent_router = _FakeIntentRouter(
        IntentResult(intent=Intent.GENERAL_QA, score=0.5, source="keyword")
    )
    default_clarifier = _FakeClarifier(None)
    default_slot_filler = _FakeSlotFiller(state=SessionState(session_id="placeholder"))

    monkeypatch.setattr(chat_module, "get_intent_router", lambda: default_intent_router)
    monkeypatch.setattr(chat_module, "get_clarifier", lambda: default_clarifier)
    monkeypatch.setattr(chat_module, "get_slot_filler", lambda: default_slot_filler)

    # Auth override on a fresh app per test (TestClient lifespan is opt-in).
    app = FastAPI()
    app.include_router(chat_module.router)

    def fake_auth() -> AuthContext:
        return AuthContext(
            user_id=str(uuid.uuid4()),
            access_token="test-token",
            user_role="user",
        )

    app.dependency_overrides[get_auth_context] = fake_auth

    client = TestClient(app)
    return {
        "app": app,
        "client": client,
        "session_repo": fake_session_repo,
        "message_repo": fake_message_repo,
        "reflect_stub": reflect_stub,
        "canned_docs": canned_docs,
        "answer_gen": answer_gen,
        "intent_router": default_intent_router,
        "clarifier": default_clarifier,
        "slot_filler": default_slot_filler,
    }


def _set_intent(harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch, intent: Intent, score: float = 0.85) -> _FakeIntentRouter:
    router = _FakeIntentRouter(IntentResult(intent=intent, score=score, source="keyword"))
    monkeypatch.setattr(chat_module, "get_intent_router", lambda: router)
    harness["intent_router"] = router
    return router


def _set_clarifier(harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch, req: ClarificationRequest | None) -> _FakeClarifier:
    clarifier = _FakeClarifier(req)
    monkeypatch.setattr(chat_module, "get_clarifier", lambda: clarifier)
    harness["clarifier"] = clarifier
    return clarifier


def _set_slot_filler(
    harness: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: SessionState | None = None,
    raise_on_update: Exception | None = None,
) -> _FakeSlotFiller:
    filler = _FakeSlotFiller(state=state, raise_on_update=raise_on_update)
    monkeypatch.setattr(chat_module, "get_slot_filler", lambda: filler)
    harness["slot_filler"] = filler
    return filler


def _last_assistant_message(message_repo: FakeChatMessageRepo) -> ChatMessage:
    for msg in reversed(message_repo.messages):
        if msg.role == "assistant":
            return msg
    raise AssertionError("no assistant message persisted")


# -------------------------------------------------------------------- #
# 1. Clarification short-circuits the pipeline.
# -------------------------------------------------------------------- #
def test_clarification_short_circuits(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_intent(harness, monkeypatch, Intent.SCORE_LOOKUP, score=0.9)
    clar_req = ClarificationRequest(
        reason="missing_year",
        question="Bạn muốn biết thông tin của năm nào?\n1. Năm 2024",
        options=[{"value": "2024", "label": "Năm 2024"}],
        slot="year",
    )
    _set_clarifier(harness, monkeypatch, clar_req)

    # If the answer generator / chitchat / tool branch was called we
    # would notice — they all raise.
    called: list[str] = []

    async def boom_chitchat(q: str, timeout: float = 20.0):  # pragma: no cover
        called.append("chitchat")
        yield ""

    async def boom_tools(**_: Any) -> tuple[str, list[dict[str, Any]], bool]:
        called.append("tool")
        return "", [], False

    monkeypatch.setattr(chat_module, "stream_chitchat_answer", boom_chitchat)
    monkeypatch.setattr(chat_module, "generate_with_tools", boom_tools)
    monkeypatch.setattr(
        chat_module,
        "get_answer_generator",
        lambda: pytest.fail("answer_generator must not be built during clarification short-circuit"),
    )

    resp = harness["client"].post(
        "/api/v1/chat/answer",
        json={"question": "điểm chuẩn CNTT", "search_mode": "hybrid"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Response surface: clarification mirrors the request, answer is the question.
    assert body["clarification"] is not None
    assert body["clarification"]["reason"] == "missing_year"
    assert "Năm 2024" in body["clarification"]["question"]
    assert body["answer"] == clar_req.question
    assert body["intent"] == "score_lookup"

    # No chitchat / no tool call branch executed.
    assert called == []

    # Persisted assistant message marked as clarification.
    assistant_msg = _last_assistant_message(harness["message_repo"])
    assert assistant_msg.metadata is not None
    assert assistant_msg.metadata.get("type") == "clarification"
    assert assistant_msg.metadata.get("clarification") is not None
    assert assistant_msg.metadata["clarification"]["options"] == [
        {"value": "2024", "label": "Năm 2024"}
    ]
    # last_intent / last_query touch happened for the next turn.
    assert harness["slot_filler"].repo.touch_calls
    assert harness["slot_filler"].repo.touch_calls[-1]["last_intent"] == "score_lookup"


# -------------------------------------------------------------------- #
# 2. Chitchat short-circuit — uses stream_chitchat_answer, no retrieval.
# -------------------------------------------------------------------- #
def test_intent_chitchat_routes_chitchat(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_intent(harness, monkeypatch, Intent.CHITCHAT, score=0.95)

    pieces = ["Xin ", "chào ", "bạn"]
    captured_chitchat: list[str] = []

    async def fake_chitchat(query: str, timeout: float = 20.0):
        captured_chitchat.append(query)
        for p in pieces:
            yield p

    retrieved_called: list[Any] = []

    async def fake_retrieve(
        query: str,
        search_mode: str,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:  # pragma: no cover
        retrieved_called.append(query)
        return []

    monkeypatch.setattr(chat_module, "stream_chitchat_answer", fake_chitchat)
    monkeypatch.setattr(chat_module, "_retrieve_docs", fake_retrieve)
    monkeypatch.setattr(
        chat_module,
        "get_answer_generator",
        lambda: pytest.fail("answer_generator must not be built on chitchat"),
    )

    resp = harness["client"].post(
        "/api/v1/chat/answer",
        json={"question": "xin chào", "search_mode": "hybrid"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == "Xin chào bạn"
    assert body["route_name"] == "chitchat"
    assert body["intent"] == "chitchat"
    assert body["retrieved_count"] == 0
    assert body["sources"] == []
    assert retrieved_called == [], "chitchat must not invoke retrieval"
    assert captured_chitchat == ["xin chào"]


# -------------------------------------------------------------------- #
# 3. score_lookup intent → tool call.
# -------------------------------------------------------------------- #
def test_intent_score_lookup_calls_tool(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config.settings import settings as _s
    monkeypatch.setattr(_s, "KG_ENABLED", True)
    _set_intent(harness, monkeypatch, Intent.SCORE_LOOKUP, score=0.9)
    # Provide a year via session state so the clarifier doesn't fire on
    # missing_year — but we still also stub the clarifier to be safe.
    state = SessionState(
        session_id="placeholder",
        slots={
            "major": SlotValue(value="cong_nghe_thong_tin", display="Công nghệ thông tin", turn=1),
            "year": SlotValue(value=2024, display="2024", turn=1),
        },
    )
    _set_slot_filler(harness, monkeypatch, state=state)

    captured: dict[str, Any] = {}

    # Note: Phase 5.2 verifier runs on tool answers too -- include a
    # citation that matches the canned doc so the answer is "verified"
    # and not annotated with the unverified-numbers warning.
    harness["canned_docs"][0]["content"] = "Điểm chuẩn ngành CNTT năm 2024 là 27.0."

    async def fake_tool_branch(**kwargs) -> tuple[str, list[dict[str, Any]], bool]:
        captured.update(kwargs)
        return (
            "Điểm chuẩn CNTT 2024 là 27.0 [1]",
            [
                {
                    "name": "lookup_score",
                    "arguments": json.dumps({"major": "CNTT", "year": 2024}),
                    "result_summary": {"row_count": 1, "preview": [{"score": 27.0}]},
                }
            ],
            True,
        )

    monkeypatch.setattr(chat_module, "generate_with_tools", fake_tool_branch)
    monkeypatch.setattr(
        chat_module,
        "get_answer_generator",
        lambda: pytest.fail("RAG generator should not be built when tool branch succeeds"),
    )

    resp = harness["client"].post(
        "/api/v1/chat/answer",
        json={"question": "điểm chuẩn CNTT 2024", "search_mode": "hybrid"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["intent"] == "score_lookup"
    assert body["route_name"] == "retrieval"
    assert body["answer"] == "Điểm chuẩn CNTT 2024 là 27.0 [1]"
    assert body["tool_calls"], "tool_calls must be reported in the response"
    assert body["tool_calls"][0]["name"] == "lookup_score"
    # The tool branch received the intent and the slot context.
    assert captured["intent"] == "score_lookup"
    # display values surface in the slot context dict
    assert captured["session_slots"].get("major") == "Công nghệ thông tin"
    # Persisted metadata mirrors the wire response.
    assistant_msg = _last_assistant_message(harness["message_repo"])
    assert assistant_msg.metadata is not None
    assert assistant_msg.metadata.get("intent") == "score_lookup"
    assert assistant_msg.metadata.get("tool_calls"), "tool_calls must be saved on the message"


# -------------------------------------------------------------------- #
# 4. Tool error falls back to RAG.
# -------------------------------------------------------------------- #
def test_tool_error_falls_back_to_rag(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config.settings import settings as _s
    monkeypatch.setattr(_s, "KG_ENABLED", True)
    _set_intent(harness, monkeypatch, Intent.SCORE_LOOKUP, score=0.9)
    # Avoid clarifier firing
    state = SessionState(
        session_id="placeholder",
        slots={
            "major": SlotValue(value="cong_nghe_thong_tin", display="Công nghệ thông tin", turn=1),
            "year": SlotValue(value=2024, display="2024", turn=1),
        },
    )
    _set_slot_filler(harness, monkeypatch, state=state)

    # Tool branch: signal an error envelope (used_tool=False, empty answer,
    # but tool_call_log captures the failure).
    async def failing_tools(**kwargs) -> tuple[str, list[dict[str, Any]], bool]:
        return (
            "",
            [
                {
                    "name": "lookup_score",
                    "arguments": "{}",
                    "result_summary": {"error": "tool_runtime_error"},
                }
            ],
            False,
        )

    monkeypatch.setattr(chat_module, "generate_with_tools", failing_tools)
    rag_generator = _StubAnswerGenerator(sync_text="RAG fallback answer")
    monkeypatch.setattr(chat_module, "get_answer_generator", lambda: rag_generator)

    resp = harness["client"].post(
        "/api/v1/chat/answer",
        json={"question": "điểm chuẩn CNTT 2024", "search_mode": "hybrid"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The user sees the RAG answer, never the tool's error envelope.
    assert body["answer"] == "RAG fallback answer"
    assert body["intent"] == "score_lookup"
    assert rag_generator.answer_text_calls, "RAG branch must be invoked on tool error"
    # The audit log still surfaces the failed tool call.
    assert body["tool_calls"], "tool_calls audit log must persist even on fallback"
    assert body["tool_calls"][0]["result_summary"].get("error") == "tool_runtime_error"


# -------------------------------------------------------------------- #
# 5. Metadata includes intent / slots / tool_calls keys.
# -------------------------------------------------------------------- #
def test_metadata_includes_intent_slots_tools(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config.settings import settings as _s
    monkeypatch.setattr(_s, "KG_ENABLED", True)
    _set_intent(harness, monkeypatch, Intent.SCORE_LOOKUP, score=0.85)
    state = SessionState(
        session_id="placeholder",
        slots={
            "major": SlotValue(value="cong_nghe_thong_tin", display="Công nghệ thông tin", turn=1),
            "year": SlotValue(value=2024, display="2024", turn=1),
        },
    )
    _set_slot_filler(harness, monkeypatch, state=state)

    async def fake_tool(**_: Any) -> tuple[str, list[dict[str, Any]], bool]:
        return (
            "Final answer",
            [
                {
                    "name": "lookup_score",
                    "arguments": "{}",
                    "result_summary": {"row_count": 2, "preview": []},
                }
            ],
            True,
        )

    monkeypatch.setattr(chat_module, "generate_with_tools", fake_tool)

    resp = harness["client"].post(
        "/api/v1/chat/answer",
        json={"question": "điểm chuẩn CNTT 2024", "search_mode": "hybrid"},
    )
    assert resp.status_code == 200, resp.text

    assistant_msg = _last_assistant_message(harness["message_repo"])
    meta = assistant_msg.metadata or {}
    # Mandatory Phase 4 metadata surface.
    for key in (
        "intent",
        "intent_score",
        "slots_snapshot",
        "clarification",
        "route",
        "pipeline",
        "rewrite_ms",
        "route_ms",
        "generate_ms",
        "search_ms",
        "tool_calls",
    ):
        assert key in meta, f"metadata missing key {key!r}; got {sorted(meta.keys())}"
    assert meta["intent"] == "score_lookup"
    assert meta["slots_snapshot"]["major"]["value"] == "cong_nghe_thong_tin"
    assert meta["clarification"] is None
    assert isinstance(meta["tool_calls"], list) and len(meta["tool_calls"]) == 1


# -------------------------------------------------------------------- #
# 6. Slot failure does not break the pipeline.
# -------------------------------------------------------------------- #
def test_slot_failure_does_not_break_pipeline(
    harness: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_intent(harness, monkeypatch, Intent.GENERAL_QA, score=0.4)
    _set_slot_filler(harness, monkeypatch, raise_on_update=RuntimeError("postgres down"))

    rag = _StubAnswerGenerator(sync_text="answer despite slot failure")
    monkeypatch.setattr(chat_module, "get_answer_generator", lambda: rag)

    caplog.set_level(logging.WARNING, logger=chat_module.log.name)
    resp = harness["client"].post(
        "/api/v1/chat/answer",
        json={"question": "thông tin tuyển sinh", "search_mode": "hybrid"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == "answer despite slot failure"
    # Empty slots snapshot — the filler raised, the pipeline degrades gracefully.
    assert body["slots_snapshot"] == {}
    # A warning was emitted.
    assert any("slot_filler.update failed" in rec.message for rec in caplog.records)


# -------------------------------------------------------------------- #
# 7. Coreference expansion runs before the rewriter sees the query.
# -------------------------------------------------------------------- #
def test_coreference_expands_query(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_intent(harness, monkeypatch, Intent.TUITION_LOOKUP, score=0.85)
    state = SessionState(
        session_id="placeholder",
        slots={
            "major": SlotValue(value="cong_nghe_thong_tin", display="CNTT", turn=1),
            "year": SlotValue(value=2024, display="2024", turn=1),
        },
    )
    _set_slot_filler(harness, monkeypatch, state=state)

    # Stub the tool branch so the pipeline reaches it without doing RAG.
    async def fake_tool(**kwargs) -> tuple[str, list[dict[str, Any]], bool]:
        return "ok", [], True

    monkeypatch.setattr(chat_module, "generate_with_tools", fake_tool)

    resp = harness["client"].post(
        "/api/v1/chat/answer",
        json={"question": "ngành đó học phí bao nhiêu", "search_mode": "hybrid"},
    )
    assert resp.status_code == 200, resp.text

    # The rewriter (mocked) recorded the query it was asked to reflect on.
    # That query MUST be the coreference-expanded form.
    assert harness["reflect_stub"].calls, "reflect must have been called"
    expanded_query, _history = harness["reflect_stub"].calls[-1]
    assert "CNTT" in expanded_query, f"expected coref expansion, got: {expanded_query!r}"
    assert "ngành đó" not in expanded_query
