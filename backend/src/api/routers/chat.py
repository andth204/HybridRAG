from __future__ import annotations
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.api.core.dependencies import AuthContext, get_auth_context, normalize_uuid_or_400
from src.api.core.runtime import (
    compact_sources,
    generate_with_tools,
    get_active_searcher,
    get_answer_generator,
    get_clarifier,
    get_intent_router,
    get_slot_filler,
    stream_chitchat_answer,
)
from src.config.settings import settings
from src.hybridrag.chat.clarifier import ClarificationRequest
from src.hybridrag.chat.feedback import ChatFeedbackRepo, FeedbackRecord
from src.hybridrag.chat.message import ChatMessage, ChatMessageRepo
from src.hybridrag.chat.session import ChatSession, ChatSessionRepo
from src.hybridrag.chat.session_state import SessionState
from src.hybridrag.chat.slot_filler import expand_coreference
from src.hybridrag.chat.tools import INTENT_TO_TOOLS
from src.hybridrag.chat.verifier import annotate_answer, verify_answer
from src.hybridrag.chat.handoff import build_handoff_payload, should_handoff
from src.hybridrag.retrieval.hybrid import HybridSearcher
from src.hybridrag.rewriter import query_reflection
from src.hybridrag.router.intents import DEFAULT_INTENT, Intent
from src.hybridrag.utils.pii_scrub import scrub, scrub_dict

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
CHAT_HISTORY_LIMIT = int(getattr(settings, "CHAT_HISTORY_LIMIT", 200))
REWRITE_HISTORY_LIMIT = int(getattr(settings, "REWRITE_HISTORY_LIMIT", 16))
SearchMode = Literal["hybrid"]
UNTITLED_SESSION_TITLE = "Untitled conversation"
SESSION_TITLE_MAX_LENGTH = 120
EMPTY_ANSWER_FALLBACK = "Xin lỗi, hệ thống chưa tạo được câu trả lời, vui lòng thử lại."

# Keys whose string values inside a metadata blob may contain raw
# user-typed PII (phone / email / CCCD). The PII scrubber rewrites these
# leaves before they hit structured logs / observability.
_LOG_PII_KEYS: set[str] = {
    "question",
    "query",
    "content",
    "answer",
    "arguments",
    "detail",
    "slots",
    "clarification",
    "rewritten",
    "rewritten_query",
}


def _log_scrub(payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively scrub PII from a log/metadata payload.

    Use this helper at every ``log.info``/``log.warning`` call site that
    forwards a metadata dict (or a dict derived from user input). Keeps
    chat persistence verbatim — only what would leak to observability
    paths gets sanitized.
    """
    return scrub_dict(payload, keys=_LOG_PII_KEYS)


def _escape_for_sse(text: str) -> str:
    """Escape ``<``/``>`` in a free-form string before it lands in an SSE payload.

    Defense-in-depth XSS guard for the streaming sources list — a
    markdown filename that contains ``<script>`` would otherwise reach
    the frontend verbatim and could be inlined into the DOM by a
    misbehaving renderer. ``sanitize_user_text`` already does the same
    thing for prompt embedding; this helper closes the SSE wire.
    """
    if not isinstance(text, str):
        return text
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _safe_sources_for_sse(sources: list[str]) -> list[str]:
    """Apply :func:`_escape_for_sse` to every source string in ``sources``."""
    return [_escape_for_sse(s) for s in sources]


def _normalize_search_mode(raw_mode: str | None) -> SearchMode:
    value = (raw_mode or "hybrid").strip().lower()
    if value == "hybrid":
        return "hybrid"
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Only search_mode='hybrid' is supported for answer endpoints",
    )


async def _retrieve_docs(
    query: str,
    search_mode: SearchMode,
    *,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Forward retrieval to the active backend, threading through filters.

    ``filters`` carries the projection of session slots (year / campus /
    major / faculty / doc_type) into the structured-filter dict the
    Weaviate searcher accepts. The legacy FAISS+BM25 ``HybridSearcher``
    does not expose metadata filters, so the dict is dropped on that
    path with no behaviour change.
    """
    if search_mode != "hybrid":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only search_mode='hybrid' is supported for answer endpoints",
        )
    searcher = get_active_searcher()
    # Use the concrete type so static analysis + behaviour both match —
    # WeaviateHybridSearcher takes ``filters``, the legacy one doesn't.
    if isinstance(searcher, HybridSearcher):
        return await searcher.search(
            query=query,
            vector_k=settings.VECTOR_SEARCH_K,
            bm25_k=settings.ELASTIC_SEARCH_K,
            fusion_top_n=settings.FUSION_K,
            use_reranker=settings.USE_RERANKER,
            rerank_top_k=settings.RERANK_TOP_K,
        )
    # The Weaviate path uses Weaviate's native hybrid scoring (alpha
    # blend) instead of FAISS+BM25+RRF. vector_k/bm25_k/fusion_top_n
    # are ignored here -- the oversample / dedup logic lives inside
    # WeaviateHybridSearcher.
    return await searcher.search(
        query=query,
        top_k=settings.RERANK_TOP_K,
        filters=filters or None,
        use_reranker=settings.USE_RERANKER,
        rerank_top_k=settings.RERANK_TOP_K,
    )


async def _cancel_task_silent(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=500)


class ChatSessionResponse(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    metadata: dict[str, Any] | None = None
    created_at: datetime


class ChatSessionsListResponse(BaseModel):
    items: list[ChatSessionResponse]


class ChatMessagesListResponse(BaseModel):
    items: list[ChatMessageResponse]


class ChatAnswerRequest(BaseModel):
    session_id: str | None = None
    question: str = Field(..., min_length=1, max_length=4000)
    search_mode: SearchMode = "hybrid"


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)


class ChatAnswerResponse(BaseModel):
    session_id: str
    question: str
    search_mode: str
    rewritten_query: str
    route_name: str
    route_score: float
    answer: str
    retrieved_count: int
    sources: list[str]
    # ----- Phase 4 additive fields -----
    # Older frontends may ignore these; they are populated whenever the
    # dialogue layer has produced something meaningful for the turn.
    intent: str | None = None
    intent_score: float | None = None
    clarification: dict[str, Any] | None = None
    slots_snapshot: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None


def _session_to_response(session: ChatSession) -> ChatSessionResponse:
    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _message_to_response(message: ChatMessage) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        metadata=message.metadata,
        created_at=message.created_at,
    )


def _to_rewriter_history(messages: list[ChatMessage]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        if message.role not in {"user", "assistant"}:
            continue
        history.append({"role": message.role, "content": content})
    return history


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _normalize_answer_text(answer_text: str) -> str:
    normalized = answer_text.strip()
    return normalized or EMPTY_ANSWER_FALLBACK


async def _get_owned_session_or_404(
    *,
    session_repo: ChatSessionRepo,
    user_id: str,
    raw_session_id: str,
) -> ChatSession:
    session_id = normalize_uuid_or_400(raw_session_id, "session_id")
    session = await asyncio.to_thread(session_repo.get, session_id, user_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return session


async def _route_query(query: str) -> tuple[float, str]:
    """Phase 4 dispatcher: classify a query into one of the 8 fine-grained intents.

    Returns ``(score, intent_str)`` so callers stay drop-in compatible
    with the legacy ``(score, route_name)`` contract. The intent string
    is the :class:`Intent` enum's value (e.g. ``"score_lookup"``); the
    legacy ``route_name`` (``"chitchat" | "retrieval"``) is computed
    separately via :func:`_intent_to_legacy_route`.
    """
    intent_router = get_intent_router()
    result = await intent_router.classify(query)
    return float(result.score), result.intent.value


# -------------------------------------------------------------------- #
# Phase 4 helpers
# -------------------------------------------------------------------- #
def _intent_to_legacy_route(intent_str: str) -> str:
    """Project a fine-grained intent into the binary legacy bucket.

    Older frontends only know about ``"chitchat" | "retrieval"`` so we
    keep emitting that label alongside the new ``intent`` field.
    """
    return "chitchat" if intent_str == Intent.CHITCHAT.value else "retrieval"


def _serialise_slots(state: SessionState | None) -> dict[str, Any]:
    """JSON-friendly view of the current slot frame for assistant metadata."""
    if state is None or not state.slots:
        return {}
    return state.slots_jsonable()


def _serialise_clarification(req: ClarificationRequest | None) -> dict[str, Any] | None:
    """Render a :class:`ClarificationRequest` for the wire / metadata."""
    if req is None:
        return None
    return {
        "reason": req.reason,
        "question": req.question,
        "options": list(req.options or []),
        "slot": req.slot,
    }


async def _safe_slot_update(
    session_id: str, query: str, turn_index: int
) -> SessionState:
    """Run the slot filler, swallowing any backend error.

    A Postgres outage / connection-pool error must not kill the request:
    the slot frame is an optional input to clarifier + retrieval filters,
    so we degrade gracefully to an empty state and continue.
    """
    filler = get_slot_filler()
    try:
        return await asyncio.to_thread(filler.update, session_id, query, turn_index=turn_index)
    except Exception as exc:  # noqa: BLE001 - defensive
        # PII guard: the user query is logged via the structured ``query``
        # field so the scrubber redacts phones / emails / IDs before the
        # message hits observability.
        log.warning(
            "slot_filler.update failed for session=%s — falling back to empty state: %s | ctx=%s",
            session_id,
            exc,
            _log_scrub({"query": query}),
        )
        return SessionState(session_id=session_id)


async def _safe_session_touch(
    session_id: str,
    *,
    last_intent: str | None,
    last_query: str | None,
) -> None:
    """Best-effort write of last_intent / last_query into session_state."""
    repo = get_slot_filler().repo
    try:
        await asyncio.to_thread(
            repo.touch,
            session_id,
            last_intent=last_intent,
            last_query=last_query,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "session_state.touch failed for session=%s: %s | ctx=%s",
            session_id,
            exc,
            _log_scrub({"query": last_query}),
        )


def _turn_index_from_history(history: list[ChatMessage]) -> int:
    """Number of user turns already on record + 1 (the current turn)."""
    user_turns = sum(1 for msg in history if msg.role == "user")
    return user_turns + 1


def _flat_slot_values(state: SessionState | None) -> dict[str, Any]:
    """Project session slots into the flat ``{slot: value}`` dict the
    clarifier and tool prompt builder expect.
    """
    if state is None or not state.slots:
        return {}
    out: dict[str, Any] = {}
    for name, sv in state.slots.items():
        if sv is None or sv.value is None:
            continue
        out[name] = sv.value
    return out


def _slot_display_values(state: SessionState | None) -> dict[str, Any]:
    """Display-name view of the slot frame (used in tool prompt)."""
    if state is None or not state.slots:
        return {}
    out: dict[str, Any] = {}
    for name, sv in state.slots.items():
        if sv is None:
            continue
        if sv.display:
            out[name] = sv.display
        elif sv.value is not None:
            out[name] = sv.value
    return out


def _build_verification_summary(answer_text: str, retrieved_docs: list[dict[str, Any]]):
    """Run the Phase 5.2 verifier and project it into the compact metadata blob.

    Returns ``(annotated_answer, summary_dict)`` where ``summary_dict``
    is the small payload we persist on the assistant message (overall
    verdict + counts only — never the full claim list, that can be
    huge). Annotation appends the Vietnamese unverified-numbers warning
    when ``report.overall == "warning"``.
    """
    report = verify_answer(answer_text, retrieved_docs)
    summary = {
        "overall": report.overall,
        "claim_count": len(report.claims),
        "unverified_count": sum(
            1 for c in report.claims if c.status == "unverified"
        ),
        "no_citation_count": sum(
            1 for c in report.claims if c.status == "no_citation"
        ),
        "refusal_detected": report.refusal_detected,
    }
    annotated = annotate_answer(answer_text, report)
    return annotated, summary


def _assert_openai_key() -> None:
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is empty",
        )


def _build_session_title_from_question(question: str, max_length: int = SESSION_TITLE_MAX_LENGTH) -> str:
    compact = " ".join(question.strip().split())
    if not compact:
        return UNTITLED_SESSION_TITLE
    if len(compact) <= max_length:
        return compact
    suffix = "..."
    cut = max(1, max_length - len(suffix))
    return f"{compact[:cut].rstrip()}{suffix}"


async def _ensure_session_title_from_first_question(
    *,
    session_repo: ChatSessionRepo,
    session: ChatSession,
    user_id: str,
    question: str,
) -> None:
    existing_title = (session.title or "").strip()
    if existing_title:
        return
    title = _build_session_title_from_question(question)
    await asyncio.to_thread(session_repo.rename, session.id, title, user_id)


async def _resolve_answer_session(
    *,
    session_repo: ChatSessionRepo,
    user_id: str,
    raw_session_id: str | None,
    question: str,
) -> ChatSession:
    normalized_session_id = (raw_session_id or "").strip()
    if not normalized_session_id:
        title = _build_session_title_from_question(question)
        return await asyncio.to_thread(session_repo.create, user_id, title)

    session = await _get_owned_session_or_404(
        session_repo=session_repo,
        user_id=user_id,
        raw_session_id=normalized_session_id,
    )
    await _ensure_session_title_from_first_question(
        session_repo=session_repo,
        session=session,
        user_id=user_id,
        question=question,
    )
    return session


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    payload: CreateSessionRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> ChatSessionResponse:
    title = payload.title.strip() if payload.title else None
    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    session = await asyncio.to_thread(
        session_repo.create,
        auth.user_id,
        title,
    )
    return _session_to_response(session)


@router.get("/sessions", response_model=ChatSessionsListResponse)
async def list_chat_sessions(
    auth: AuthContext = Depends(get_auth_context),
) -> ChatSessionsListResponse:
    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    sessions = await asyncio.to_thread(session_repo.list_by_user, auth.user_id, limit=20, offset=0)
    return ChatSessionsListResponse(items=[_session_to_response(item) for item in sessions])


@router.patch("/sessions/{session_id}", response_model=ChatSessionResponse)
async def rename_chat_session(
    session_id: str,
    payload: RenameSessionRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> ChatSessionResponse:
    title = payload.title.strip()
    if not title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="title must not be empty",
        )

    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    session = await _get_owned_session_or_404(
        session_repo=session_repo,
        user_id=auth.user_id,
        raw_session_id=session_id,
    )
    renamed = await asyncio.to_thread(session_repo.rename, session.id, title, auth.user_id)
    if not renamed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    updated = await asyncio.to_thread(session_repo.get, session.id, auth.user_id)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return _session_to_response(updated)


@router.get("/sessions/{session_id}/messages", response_model=ChatMessagesListResponse)
async def get_session_messages(
    session_id: str,
    auth: AuthContext = Depends(get_auth_context),
) -> ChatMessagesListResponse:
    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    message_repo = ChatMessageRepo(settings.DATABASE_URL)
    session = await _get_owned_session_or_404(
        session_repo=session_repo,
        user_id=auth.user_id,
        raw_session_id=session_id,
    )
    history = await asyncio.to_thread(
        message_repo.load_history,
        session.id,
        limit=CHAT_HISTORY_LIMIT,
        offset=0,
        ascending=True,
    )
    return ChatMessagesListResponse(items=[_message_to_response(item) for item in history])


@router.get("/messages/search", response_model=ChatMessagesListResponse)
async def search_chat_messages(
    q: str = Query(..., min_length=1, max_length=2000),
    session_id: str | None = Query(default=None),
    auth: AuthContext = Depends(get_auth_context),
) -> ChatMessagesListResponse:
    query_text = q.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="q must not be empty",
        )

    normalized_session_id: str | None = None
    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    if session_id:
        owned_session = await _get_owned_session_or_404(
            session_repo=session_repo,
            user_id=auth.user_id,
            raw_session_id=session_id,
        )
        normalized_session_id = owned_session.id

    message_repo = ChatMessageRepo(settings.DATABASE_URL)
    messages = await asyncio.to_thread(
        message_repo.search,
        auth.user_id,
        query_text,
        session_id=normalized_session_id,
        limit=50,
        offset=0,
    )
    return ChatMessagesListResponse(items=[_message_to_response(item) for item in messages])


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: str,
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    normalized_session_id = normalize_uuid_or_400(session_id, "session_id")
    deleted = await asyncio.to_thread(session_repo.delete, normalized_session_id, auth.user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/answer", response_model=ChatAnswerResponse)
async def chat_answer(
    payload: ChatAnswerRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> ChatAnswerResponse:
    _assert_openai_key()
    question = payload.question.strip()
    search_mode = _normalize_search_mode(payload.search_mode)
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question must not be empty",
        )

    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    message_repo = ChatMessageRepo(settings.DATABASE_URL)
    session = await _resolve_answer_session(
        session_repo=session_repo,
        user_id=auth.user_id,
        raw_session_id=payload.session_id,
        question=question,
    )

    stored_messages = await asyncio.to_thread(
        message_repo.load_history,
        session.id,
        limit=REWRITE_HISTORY_LIMIT,
        offset=0,
        ascending=True,
    )
    chat_history = _to_rewriter_history(stored_messages)
    turn_index = _turn_index_from_history(stored_messages)
    await asyncio.to_thread(
        message_repo.create,
        session.id,
        "user",
        question,
        metadata={
            "type": "question",
            "pipeline": "api.chat.answer",
            "history_count_before": len(chat_history),
            "search_mode": search_mode,
        },
    )

    # Phase 4B: slot extraction + coreference expansion. The filler is
    # resilient to DB failure (returns an empty SessionState) so a
    # Postgres outage will not crash the request.
    session_state = await _safe_slot_update(session.id, question, turn_index)
    expanded_query = expand_coreference(question, session_state)

    # Session slots are tracked for display + clarifier hints (see
    # ``slots_snapshot`` in the response metadata) but we DO NOT pass
    # them as hard filters to the retrieval layer. Live audit showed
    # that AND-ing slot filters against the Weaviate index collapses
    # most queries to 0 hits because metadata tagging at ingest time
    # is sparse (~50% of chunks lack ``year``, etc.). Hybrid search
    # (BM25 + vector) already ranks by relevance — hard filters were
    # net-negative for recall.
    slot_filters: dict[str, Any] = get_slot_filler().filters_for_retrieval(session_state)

    # Speculatively start retrieval on the expanded question in parallel
    # with the rewrite LLM call. If the rewrite turns out to be a no-op,
    # we already have the retrieval ready -> saves ~200-1500ms. If the
    # rewrite diverges, we cancel and re-run retrieval against the
    # rewritten query.
    raw_retrieval_task: asyncio.Task[list[dict[str, Any]]] | None = (
        asyncio.create_task(
            _retrieve_docs(expanded_query, search_mode, filters=None)
        )
    )

    rewrite_t0 = time.perf_counter()
    rewritten_query = await query_reflection.reflect(expanded_query, chat_history)
    rewrite_ms = (time.perf_counter() - rewrite_t0) * 1000

    # Phase 4A: intent classification on the rewritten query.
    route_t0 = time.perf_counter()
    intent_router = get_intent_router()
    intent_result = await intent_router.classify(rewritten_query)
    intent_str = intent_result.intent.value
    intent_score = float(intent_result.score)
    route_name = _intent_to_legacy_route(intent_str)
    route_ms = (time.perf_counter() - route_t0) * 1000

    # Phase 4C: clarification gate. We peek at retrieval BEFORE the
    # expensive generation call so the clarifier can use a top-doc score
    # to decide if recall is too poor to answer. Chitchat short-circuits
    # without retrieval — we never clarify in that path.
    answer_text = ""
    retrieved_docs: list[dict[str, Any]] = []
    search_ms = 0.0
    generate_t0 = time.perf_counter()
    clarification: ClarificationRequest | None = None
    tool_call_log: list[dict[str, Any]] = []
    used_tool = False

    if route_name == "chitchat":
        # Chitchat does not need retrieval - cancel the speculative task.
        await _cancel_task_silent(raw_retrieval_task)
        raw_retrieval_task = None
    else:
        # We need preview docs both for clarifier (low_recall check) and
        # for the downstream answer path. Reuse the speculative task if
        # the rewrite did not diverge; otherwise re-run against the
        # rewritten query.
        search_t0 = time.perf_counter()
        rewrite_unchanged = rewritten_query.strip() == expanded_query.strip()
        if rewrite_unchanged:
            try:
                retrieved_docs = await raw_retrieval_task
            except (asyncio.CancelledError, Exception):
                retrieved_docs = await _retrieve_docs(
                    rewritten_query, search_mode, filters=None
                )
            raw_retrieval_task = None
        else:
            await _cancel_task_silent(raw_retrieval_task)
            raw_retrieval_task = None
            retrieved_docs = await _retrieve_docs(
                rewritten_query, search_mode, filters=None
            )
        search_ms = (time.perf_counter() - search_t0) * 1000

        # Now ask the clarifier whether we should short-circuit.
        clarifier = get_clarifier()
        clarification = clarifier.check(
            query=rewritten_query,
            intent=intent_str,
            session_slots=_flat_slot_values(session_state),
            retrieval_docs=retrieved_docs,
        )

    if clarification is not None:
        # Short-circuit: persist the clarification as the assistant message
        # and skip generation entirely. The verifier is NOT run on a
        # clarification — the assistant text is the clarifier's question,
        # not a generated answer.
        answer_text = _normalize_answer_text(clarification.question)
        generate_ms = (time.perf_counter() - generate_t0) * 1000
        sources = compact_sources(retrieved_docs)
        metadata: dict[str, Any] = {
            "type": "clarification",
            "route": route_name,
            "pipeline": "api.chat.answer",
            "search_mode": search_mode,
            "rewrite_ms": round(rewrite_ms, 2),
            "route_ms": round(route_ms, 2),
            "generate_ms": round(generate_ms, 2),
            "intent": intent_str,
            "intent_score": round(intent_score, 4),
            "slots_snapshot": _serialise_slots(session_state),
            "clarification": _serialise_clarification(clarification),
        }
        if route_name != "chitchat":
            metadata["n_retrieved"] = len(retrieved_docs)
            metadata["sources"] = sources
            metadata["search_ms"] = round(search_ms, 2)

        await asyncio.to_thread(
            message_repo.create,
            session.id,
            "assistant",
            answer_text,
            metadata=metadata,
        )
        await asyncio.to_thread(session_repo.touch, session.id)
        await _safe_session_touch(
            session.id, last_intent=intent_str, last_query=rewritten_query
        )
        return ChatAnswerResponse(
            session_id=session.id,
            question=question,
            search_mode=search_mode,
            rewritten_query=rewritten_query,
            route_name=route_name,
            route_score=intent_score,
            answer=answer_text,
            retrieved_count=len(retrieved_docs),
            sources=sources,
            intent=intent_str,
            intent_score=intent_score,
            clarification=_serialise_clarification(clarification),
            slots_snapshot=_serialise_slots(session_state),
            tool_calls=None,
        )

    # Branch by intent.
    if route_name == "chitchat":
        parts: list[str] = []
        async for piece in stream_chitchat_answer(rewritten_query):
            parts.append(piece)
        answer_text = "".join(parts).strip()
    else:
        # Tool-call branch (Phase 4D) for intents bound to structured store.
        # Gated by KG_ENABLED — keep off until admission_scores/tuition tables
        # are populated. Off: all retrieval intents go straight to RAG.
        if settings.KG_ENABLED and intent_str in INTENT_TO_TOOLS:
            tool_answer, tool_call_log, used_tool = await generate_with_tools(
                rewritten_query=rewritten_query,
                intent=intent_str,
                retrieved_docs=retrieved_docs,
                session_slots=_slot_display_values(session_state),
            )
            if used_tool and tool_answer:
                answer_text = tool_answer.strip()

        if not answer_text:
            # Fall through to RAG branch (no tool / model declined / tool error).
            generator = get_answer_generator()
            answer_text = (
                await generator.answer_text(
                    query=rewritten_query,
                    retrieved_docs=retrieved_docs,
                )
            ).strip()

    answer_text = _normalize_answer_text(answer_text)

    # Phase 5.2 — verify numeric claims against retrieved docs and
    # annotate the answer with a Vietnamese warning marker for any
    # unverified ones. Skip the verifier when the answer doesn't have a
    # citable evidence base (chitchat OR no retrieval).
    verification_summary: dict[str, Any] | None = None
    if route_name != "chitchat" and retrieved_docs:
        answer_text, verification_summary = _build_verification_summary(
            answer_text, retrieved_docs
        )

    # Human handoff fallback — when bot cannot answer (refusal / degenerate
    # / no docs / low recall), surface the advisor contact card.
    handoff_payload: dict[str, Any] | None = None
    needs_handoff, handoff_reason = should_handoff(
        intent=intent_str,
        answer_text=answer_text,
        retrieved_docs=retrieved_docs,
        verification_overall=(verification_summary or {}).get("overall"),
    )
    if needs_handoff and handoff_reason is not None:
        handoff_payload = build_handoff_payload(handoff_reason)
        # Append a short contact block to the answer text so old clients
        # without UI support still see the contact info.
        contact = handoff_payload["contact"]
        suffix = (
            f"\n\n📞 {handoff_payload['message']}\n"
            f"- Hotline: {contact['hotline']}\n"
            f"- Văn phòng: {contact['phone']}\n"
            f"- Email: {contact['email']}\n"
            f"- Fanpage: {contact['fanpage']}\n"
            f"- Giờ làm việc: {contact['office_hours']}"
        )
        # Avoid double-appending if the refusal template already contained it.
        if "Hotline" not in answer_text:
            answer_text = answer_text.rstrip() + suffix

    generate_ms = (time.perf_counter() - generate_t0) * 1000
    sources = compact_sources(retrieved_docs)

    metadata = {
        "type": "answer",
        "route": route_name,
        "pipeline": "api.chat.answer",
        "search_mode": search_mode,
        "rewrite_ms": round(rewrite_ms, 2),
        "route_ms": round(route_ms, 2),
        "generate_ms": round(generate_ms, 2),
        "intent": intent_str,
        "intent_score": round(intent_score, 4),
        "slots_snapshot": _serialise_slots(session_state),
        "clarification": None,
    }
    if route_name != "chitchat":
        metadata["n_retrieved"] = len(retrieved_docs)
        metadata["sources"] = sources
        metadata["search_ms"] = round(search_ms, 2)
    if tool_call_log:
        metadata["tool_calls"] = tool_call_log
    if verification_summary is not None:
        metadata["verification"] = verification_summary
    if handoff_payload is not None:
        metadata["handoff"] = handoff_payload

    await asyncio.to_thread(
        message_repo.create,
        session.id,
        "assistant",
        answer_text,
        metadata=metadata,
    )
    await asyncio.to_thread(session_repo.touch, session.id)
    await _safe_session_touch(
        session.id, last_intent=intent_str, last_query=rewritten_query
    )

    return ChatAnswerResponse(
        session_id=session.id,
        question=question,
        search_mode=search_mode,
        rewritten_query=rewritten_query,
        route_name=route_name,
        route_score=intent_score,
        answer=answer_text,
        retrieved_count=len(retrieved_docs),
        sources=sources,
        intent=intent_str,
        intent_score=intent_score,
        clarification=None,
        slots_snapshot=_serialise_slots(session_state),
        tool_calls=tool_call_log or None,
    )


@router.post("/answer/stream")
async def chat_answer_stream(
    payload: ChatAnswerRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> StreamingResponse:
    _assert_openai_key()
    question = payload.question.strip()
    search_mode = _normalize_search_mode(payload.search_mode)
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question must not be empty",
        )

    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    message_repo = ChatMessageRepo(settings.DATABASE_URL)
    session = await _resolve_answer_session(
        session_repo=session_repo,
        user_id=auth.user_id,
        raw_session_id=payload.session_id,
        question=question,
    )

    async def event_stream():
        answer_parts: list[str] = []
        rewritten_query = question
        intent_str = DEFAULT_INTENT.value
        intent_score = 0.0
        route_name = "pending"
        retrieved_docs: list[dict[str, Any]] = []
        raw_retrieval_task: asyncio.Task[list[dict[str, Any]]] | None = None
        tool_call_log: list[dict[str, Any]] = []
        session_state: SessionState | None = None
        clarification: ClarificationRequest | None = None
        used_tool = False
        try:
            yield _sse(
                "start",
                {
                    "session_id": session.id,
                    "question": question,
                    "search_mode": search_mode,
                },
            )

            stored_messages = await asyncio.to_thread(
                message_repo.load_history,
                session.id,
                limit=REWRITE_HISTORY_LIMIT,
                offset=0,
                ascending=True,
            )
            chat_history = _to_rewriter_history(stored_messages)
            turn_index = _turn_index_from_history(stored_messages)
            await asyncio.to_thread(
                message_repo.create,
                session.id,
                "user",
                question,
                metadata={
                    "type": "question",
                    "pipeline": "api.chat.answer.stream",
                    "history_count_before": len(chat_history),
                    "search_mode": search_mode,
                },
            )

            # Phase 4B: slot extraction + coreference expansion.
            session_state = await _safe_slot_update(session.id, question, turn_index)
            slots_snapshot = _serialise_slots(session_state)
            yield _sse(
                "status",
                {"stage": "slot_update", "slots": slots_snapshot},
            )
            expanded_query = expand_coreference(question, session_state)

            # Phase 5.5: pull current slot filters before retrieval so the
            # speculative task gets the same projection as the eventual
            # one. The implied-year fallback is layered in once we know
            # the intent.
            slot_filters: dict[str, Any] = (
                get_slot_filler().filters_for_retrieval(session_state)
            )

            # Speculatively start retrieval on the expanded question.
            raw_retrieval_task = asyncio.create_task(
                _retrieve_docs(
                    expanded_query, search_mode, filters=None
                )
            )

            # Stage 1: rewriting
            yield _sse("status", {"stage": "rewriting"})
            rewritten_query = await query_reflection.reflect(expanded_query, chat_history)

            # Stage 2: intent classification
            intent_router = get_intent_router()
            intent_result = await intent_router.classify(rewritten_query)
            intent_str = intent_result.intent.value
            intent_score = float(intent_result.score)
            route_name = _intent_to_legacy_route(intent_str)
            yield _sse(
                "status",
                {
                    "stage": "intent",
                    "intent": intent_str,
                    "score": round(intent_score, 4),
                },
            )
            # Keep legacy "routing" event so existing frontends still react.
            yield _sse(
                "status",
                {
                    "stage": "routing",
                    "rewritten": rewritten_query,
                    "route_name": route_name,
                },
            )

            if route_name == "chitchat":
                # Cancel speculative retrieval before announcing next stage to
                # avoid leaking a stale retrieve in flight.
                await _cancel_task_silent(raw_retrieval_task)
                raw_retrieval_task = None
            else:
                # Sequence cancellation BEFORE emitting "retrieving" so the
                # frontend never sees retrieving-twice for a divergent rewrite.
                rewrite_unchanged = rewritten_query.strip() == expanded_query.strip()
                if rewrite_unchanged:
                    yield _sse("status", {"stage": "retrieving"})
                    try:
                        retrieved_docs = await raw_retrieval_task
                    except (asyncio.CancelledError, Exception):
                        retrieved_docs = await _retrieve_docs(
                            rewritten_query, search_mode, filters=None
                        )
                    raw_retrieval_task = None
                else:
                    await _cancel_task_silent(raw_retrieval_task)
                    raw_retrieval_task = None
                    yield _sse("status", {"stage": "retrieving"})
                    retrieved_docs = await _retrieve_docs(
                        rewritten_query, search_mode, filters=None
                    )

                if settings.USE_RERANKER and len(retrieved_docs) > 0:
                    yield _sse("status", {"stage": "reranking"})

            sources = compact_sources(retrieved_docs)
            # Defense-in-depth: any '<'/'>' inside source labels must not
            # reach the SSE wire un-escaped (frontend XSS guard).
            safe_sources = _safe_sources_for_sse(sources)

            # Phase 4C: clarification gate (skipped on chitchat).
            if route_name != "chitchat":
                clarifier = get_clarifier()
                clarification = clarifier.check(
                    query=rewritten_query,
                    intent=intent_str,
                    session_slots=_flat_slot_values(session_state),
                    retrieval_docs=retrieved_docs,
                )

            if clarification is not None:
                clar_payload = _serialise_clarification(clarification)
                yield _sse("clarification", clar_payload or {})
                clarification_text = _normalize_answer_text(clarification.question)
                clar_metadata: dict[str, Any] = {
                    "type": "clarification",
                    "route": route_name,
                    "pipeline": "api.chat.answer.stream",
                    "search_mode": search_mode,
                    "n_retrieved": len(retrieved_docs),
                    "sources": sources,
                    "intent": intent_str,
                    "intent_score": round(intent_score, 4),
                    "slots_snapshot": slots_snapshot,
                    "clarification": clar_payload,
                }
                stored_clar = await asyncio.to_thread(
                    message_repo.create,
                    session.id,
                    "assistant",
                    clarification_text,
                    metadata=clar_metadata,
                )
                await asyncio.to_thread(session_repo.touch, session.id)
                await _safe_session_touch(
                    session.id,
                    last_intent=intent_str,
                    last_query=rewritten_query,
                )
                yield _sse(
                    "done",
                    {
                        "session_id": session.id,
                        "route_name": route_name,
                        "search_mode": search_mode,
                        "answer": clarification_text,
                        "sources": safe_sources,
                        "answer_id": getattr(stored_clar, "id", None),
                        "route": route_name,
                        "intent": intent_str,
                        "intent_score": intent_score,
                        "clarification": clar_payload,
                        "slots_snapshot": slots_snapshot,
                        "metadata": clar_metadata,
                    },
                )
                return

            # Preserve the existing "meta" event for backward compatibility.
            yield _sse(
                "meta",
                {
                    "session_id": session.id,
                    "question": question,
                    "rewritten_query": rewritten_query,
                    "route_name": route_name,
                    "route_score": intent_score,
                    "search_mode": search_mode,
                    "retrieved_count": len(retrieved_docs),
                    "intent": intent_str,
                    "intent_score": intent_score,
                },
            )

            # Tool-call branch (Phase 4D). We compute the answer
            # non-streamingly inside generate_with_tools, then replay it
            # to the client as tokens so the wire protocol stays the
            # same. If the tool branch declines (no schema / model
            # didn't call / tool error), we fall through to RAG.
            tool_answer = ""
            if (
                settings.KG_ENABLED
                and route_name != "chitchat"
                and intent_str in INTENT_TO_TOOLS
            ):
                # Emit tool_call status BEFORE we kick off the LLM, so
                # the frontend can show "Đang tra cứu...". We can't
                # surface arguments before the LLM picks them — emit
                # the high-level event with name=<first tool> as a hint.
                hint_name = INTENT_TO_TOOLS[intent_str][0]
                yield _sse(
                    "status",
                    {"stage": "tool_call", "name": hint_name, "arguments": {}},
                )
                tool_answer, tool_call_log, used_tool = await generate_with_tools(
                    rewritten_query=rewritten_query,
                    intent=intent_str,
                    retrieved_docs=retrieved_docs,
                    session_slots=_slot_display_values(session_state),
                )
                # Emit a tool_result status per executed call (row count only).
                for entry in tool_call_log:
                    summary = entry.get("result_summary") or {}
                    yield _sse(
                        "status",
                        {
                            "stage": "tool_result",
                            "name": entry.get("name"),
                            "row_count": int(summary.get("row_count", 0)),
                        },
                    )

            generating_payload: dict[str, Any] = {"stage": "generating"}
            if route_name != "chitchat":
                generating_payload["sources"] = safe_sources
            yield _sse("status", generating_payload)

            if route_name == "chitchat":
                async for piece in stream_chitchat_answer(rewritten_query):
                    answer_parts.append(piece)
                    yield _sse("token", {"text": piece})
            elif used_tool and tool_answer:
                # Replay the pre-computed answer as token events so
                # the wire protocol matches the streaming path.
                answer_parts.append(tool_answer)
                yield _sse("token", {"text": tool_answer})
            else:
                generator = get_answer_generator()
                async for piece in generator.stream_answer(
                    query=rewritten_query,
                    retrieved_docs=retrieved_docs,
                ):
                    answer_parts.append(piece)
                    yield _sse("token", {"text": piece})

            answer_text = _normalize_answer_text("".join(answer_parts))

            # Phase 5.2 — verify the LLM's numeric claims against the
            # retrieved docs. Skip on chitchat / no-retrieval paths.
            verification_summary: dict[str, Any] | None = None
            if route_name != "chitchat" and retrieved_docs:
                # Announce that the verification stage is running BEFORE
                # we annotate — the user sees a progress signal even
                # though the verifier itself is millisecond-cheap.
                yield _sse("status", {"stage": "verifying"})
                answer_text, verification_summary = _build_verification_summary(
                    answer_text, retrieved_docs
                )

            # Human handoff fallback.
            handoff_payload: dict[str, Any] | None = None
            needs_handoff, handoff_reason = should_handoff(
                intent=intent_str,
                answer_text=answer_text,
                retrieved_docs=retrieved_docs,
                verification_overall=(verification_summary or {}).get("overall"),
            )
            if needs_handoff and handoff_reason is not None:
                handoff_payload = build_handoff_payload(handoff_reason)
                contact = handoff_payload["contact"]
                suffix = (
                    f"\n\n📞 {handoff_payload['message']}\n"
                    f"- Hotline: {contact['hotline']}\n"
                    f"- Văn phòng: {contact['phone']}\n"
                    f"- Email: {contact['email']}\n"
                    f"- Fanpage: {contact['fanpage']}\n"
                    f"- Giờ làm việc: {contact['office_hours']}"
                )
                if "Hotline" not in answer_text:
                    answer_text = answer_text.rstrip() + suffix
                # Emit a dedicated SSE event so the frontend can render
                # a contact card alongside the plain-text suffix.
                yield _sse("handoff", handoff_payload)

            metadata: dict[str, Any] = {
                "type": "answer",
                "route": route_name,
                "pipeline": "api.chat.answer.stream",
                "search_mode": search_mode,
                "n_retrieved": len(retrieved_docs),
                "sources": sources,
                "intent": intent_str,
                "intent_score": round(intent_score, 4),
                "slots_snapshot": slots_snapshot,
                "clarification": None,
            }
            if tool_call_log:
                metadata["tool_calls"] = tool_call_log
            if verification_summary is not None:
                metadata["verification"] = verification_summary
            if handoff_payload is not None:
                metadata["handoff"] = handoff_payload
            stored_assistant = await asyncio.to_thread(
                message_repo.create,
                session.id,
                "assistant",
                answer_text,
                metadata=metadata,
            )
            await asyncio.to_thread(session_repo.touch, session.id)
            await _safe_session_touch(
                session.id, last_intent=intent_str, last_query=rewritten_query
            )
            yield _sse(
                "done",
                {
                    "session_id": session.id,
                    "route_name": route_name,
                    "search_mode": search_mode,
                    "answer": answer_text,
                    "sources": safe_sources,
                    "answer_id": getattr(stored_assistant, "id", None),
                    "route": route_name,
                    "intent": intent_str,
                    "intent_score": intent_score,
                    "slots_snapshot": slots_snapshot,
                    "tool_calls": tool_call_log or None,
                    "metadata": metadata,
                },
            )
        except Exception as exc:
            await _cancel_task_silent(raw_retrieval_task)
            log.warning(
                "chat_answer_stream failed: %s",
                _log_scrub({"detail": str(exc)}),
            )
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------- #
# Phase 5.6 — Feedback endpoint
# ---------------------------------------------------------------------- #
class FeedbackCreateRequest(BaseModel):
    """Body for ``POST /messages/{message_id}/feedback``.

    Pydantic validates ``rating`` against the literal enum so an invalid
    rating returns 422 without touching the DB. ``comment`` is bounded
    at 2000 chars to keep abuse + storage cost low; the API layer
    PII-scrubs it before persistence.
    """

    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    """Wire-format projection of :class:`FeedbackRecord`."""

    id: int
    message_id: str
    session_id: str
    user_id: str | None
    rating: str
    comment: str | None


@router.post(
    "/messages/{message_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_message_feedback(
    message_id: str,
    payload: FeedbackCreateRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> FeedbackResponse:
    """Persist a thumbs up/down (with optional comment) for an assistant message.

    Authorization: the message must belong to a session owned by the
    caller. Cross-user feedback is rejected with 404 so we don't leak
    the existence of foreign messages.

    Privacy: the ``comment`` field is passed through
    :func:`pii_scrub.scrub` BEFORE it touches the database. We never
    want phones / emails / national-ID numbers persisted in a
    feedback-style "free text" column where they would also surface
    inside analytics queries.
    """
    normalized_message_id = normalize_uuid_or_400(message_id, "message_id")

    message_repo = ChatMessageRepo(settings.DATABASE_URL)
    session_repo = ChatSessionRepo(settings.DATABASE_URL)

    # 1) Resolve the target message. We hide non-existent and not-owned
    #    messages behind the same 404 so the API never reveals the
    #    existence of foreign data.
    message = await asyncio.to_thread(message_repo.get, normalized_message_id)
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )

    # 2) Re-check that the session is owned by the caller.
    session = await asyncio.to_thread(
        session_repo.get, message.session_id, auth.user_id
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )

    # 3) PII scrub the user-supplied comment before storage.
    scrubbed_comment: str | None = None
    if payload.comment is not None:
        scrubbed_comment = scrub(payload.comment)

    feedback_repo = ChatFeedbackRepo(settings.DATABASE_URL)
    try:
        record = await asyncio.to_thread(
            feedback_repo.create,
            message_id=message.id,
            session_id=message.session_id,
            user_id=auth.user_id,
            rating=payload.rating,
            comment=scrubbed_comment,
        )
    except ValueError as exc:
        # Repo-level validation rejected the rating (defensive — Pydantic
        # should already catch this above).
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return FeedbackResponse(
        id=record.id,
        message_id=record.message_id,
        session_id=record.session_id,
        user_id=record.user_id,
        rating=record.rating,
        comment=record.comment,
    )
