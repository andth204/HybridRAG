from __future__ import annotations
import asyncio
import json
import time
from datetime import datetime
from typing import Any, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.api.core.dependencies import AuthContext, get_auth_context, normalize_uuid_or_400
from src.api.core.runtime import (
    compact_sources,
    get_answer_generator,
    get_hybrid_searcher,
    get_keyword_router,
    get_router_type,
    get_semantic_router,
    stream_chitchat_answer,
)
from src.config.settings import settings
from src.hybridrag.chat.message import ChatMessage, ChatMessageRepo
from src.hybridrag.chat.session import ChatSession, ChatSessionRepo
from src.hybridrag.rewriter import query_reflection

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
CHAT_HISTORY_LIMIT = int(getattr(settings, "CHAT_HISTORY_LIMIT", 200))
SearchMode = Literal["keyword", "semantic", "hybrid"]


def _normalize_search_mode(raw_mode: str | None) -> SearchMode:
    value = (raw_mode or "hybrid").strip().lower()
    if value in {"hybrid"}:
        return "hybrid"
    if value in {"semantic", "vector"}:
        return "semantic"
    if value == "keyword":
        return "keyword"
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="search_mode must be one of: keyword, semantic, hybrid",
    )


async def _retrieve_docs(query: str, search_mode: SearchMode) -> list[dict[str, Any]]:
    searcher = get_hybrid_searcher()
    single_mode_limit = max(1, int(getattr(settings, "SINGLE_MODE_SEARCH_MAX_K", 3)))
    if search_mode == "keyword":
        return await searcher.bm25.search(
            query=query,
            top_k=min(int(settings.ELASTIC_SEARCH_K), single_mode_limit),
        )
    if search_mode == "semantic":
        return await searcher.vector.search(
            query=query,
            top_k=min(int(settings.VECTOR_SEARCH_K), single_mode_limit),
        )
    return await searcher.search(
        query=query,
        vector_k=settings.VECTOR_SEARCH_K,
        bm25_k=settings.ELASTIC_SEARCH_K,
        fusion_top_n=settings.FUSION_K,
        use_reranker=settings.USE_RERANKER,
        rerank_top_k=settings.RERANK_TOP_K,
    )


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
    session_id: str
    question: str = Field(..., min_length=1, max_length=4000)
    search_mode: str = Field(default="hybrid", min_length=1, max_length=32)


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
    router_type = get_router_type()
    if router_type == "semantic":
        semantic_router = get_semantic_router()
        return await semantic_router.guide(query)
    keyword_router = get_keyword_router()
    score, route_name = keyword_router.guide(query)
    return float(score), route_name


def _assert_openai_key() -> None:
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is empty",
        )


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
    session = await _get_owned_session_or_404(
        session_repo=session_repo,
        user_id=auth.user_id,
        raw_session_id=payload.session_id,
    )

    stored_messages = await asyncio.to_thread(
        message_repo.load_history,
        session.id,
        limit=CHAT_HISTORY_LIMIT,
        offset=0,
        ascending=True,
    )
    chat_history = _to_rewriter_history(stored_messages)
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

    rewrite_t0 = time.perf_counter()
    rewritten_query = await query_reflection.reflect(question, chat_history)
    rewrite_ms = (time.perf_counter() - rewrite_t0) * 1000

    route_t0 = time.perf_counter()
    route_score, route_name = await _route_query(rewritten_query)
    route_ms = (time.perf_counter() - route_t0) * 1000

    answer_text = ""
    retrieved_docs: list[dict[str, Any]] = []
    search_ms = 0.0
    generate_t0 = time.perf_counter()

    if route_name == "chitchat":
        parts: list[str] = []
        async for piece in stream_chitchat_answer(rewritten_query):
            parts.append(piece)
        answer_text = "".join(parts).strip()
    else:
        search_t0 = time.perf_counter()
        retrieved_docs = await _retrieve_docs(rewritten_query, search_mode)
        search_ms = (time.perf_counter() - search_t0) * 1000
        generator = get_answer_generator()
        answer_text = (
            await generator.answer_text(
                query=rewritten_query,
                retrieved_docs=retrieved_docs,
            )
        ).strip()

    generate_ms = (time.perf_counter() - generate_t0) * 1000
    sources = compact_sources(retrieved_docs)

    metadata: dict[str, Any] = {
        "type": "answer",
        "route": route_name,
        "pipeline": "api.chat.answer",
        "search_mode": search_mode,
        "rewrite_ms": round(rewrite_ms, 2),
        "route_ms": round(route_ms, 2),
        "generate_ms": round(generate_ms, 2),
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

    return ChatAnswerResponse(
        session_id=session.id,
        question=question,
        search_mode=search_mode,
        rewritten_query=rewritten_query,
        route_name=route_name,
        route_score=route_score,
        answer=answer_text,
        retrieved_count=len(retrieved_docs),
        sources=sources,
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
    session = await _get_owned_session_or_404(
        session_repo=session_repo,
        user_id=auth.user_id,
        raw_session_id=payload.session_id,
    )

    stored_messages = await asyncio.to_thread(
        message_repo.load_history,
        session.id,
        limit=CHAT_HISTORY_LIMIT,
        offset=0,
        ascending=True,
    )
    chat_history = _to_rewriter_history(stored_messages)
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

    rewritten_query = await query_reflection.reflect(question, chat_history)
    route_score, route_name = await _route_query(rewritten_query)
    retrieved_docs: list[dict[str, Any]] = []
    if route_name != "chitchat":
        retrieved_docs = await _retrieve_docs(rewritten_query, search_mode)

    async def event_stream():
        answer_parts: list[str] = []
        try:
            yield _sse(
                "start",
                {
                    "session_id": session.id,
                    "question": question,
                    "rewritten_query": rewritten_query,
                    "route_name": route_name,
                    "route_score": route_score,
                    "search_mode": search_mode,
                    "retrieved_count": len(retrieved_docs),
                },
            )

            if route_name == "chitchat":
                async for piece in stream_chitchat_answer(rewritten_query):
                    answer_parts.append(piece)
                    yield _sse("chunk", {"content": piece})
            else:
                generator = get_answer_generator()
                async for piece in generator.stream_answer(
                    query=rewritten_query,
                    retrieved_docs=retrieved_docs,
                ):
                    answer_parts.append(piece)
                    yield _sse("chunk", {"content": piece})

            answer_text = "".join(answer_parts).strip()
            sources = compact_sources(retrieved_docs)
            metadata: dict[str, Any] = {
                "type": "answer",
                "route": route_name,
                "pipeline": "api.chat.answer.stream",
                "search_mode": search_mode,
                "n_retrieved": len(retrieved_docs),
                "sources": sources,
            }
            await asyncio.to_thread(
                message_repo.create,
                session.id,
                "assistant",
                answer_text,
                metadata=metadata,
            )
            await asyncio.to_thread(session_repo.touch, session.id)
            yield _sse(
                "done",
                {
                    "session_id": session.id,
                    "route_name": route_name,
                    "search_mode": search_mode,
                    "answer": answer_text,
                    "sources": sources,
                },
            )
        except Exception as exc:
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
