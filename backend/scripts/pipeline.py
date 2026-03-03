import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Iterable
from openai import AsyncOpenAI
from src.config.prompts import get_prompt
from src.config.settings import settings
from src.hybridrag.chat.answer import AnswerGenerator
from src.hybridrag.chat.message import ChatMessage, ChatMessageRepo
from src.hybridrag.chat.session import ChatSession, ChatSessionRepo
from src.hybridrag.retrieval.hybrid import HybridSearcher
from src.hybridrag.rewriter import query_reflection
from src.hybridrag.router import KeywordRouter, ROUTES, SemanticRouter
try:
    from create_user import compact_sources, ensure_fake_user
except ModuleNotFoundError:
    from backend.scripts.create_user import compact_sources, ensure_fake_user
_SEARCHER_SINGLETON: HybridSearcher | None = None


def _resolve_session(
    *,
    session_repo: ChatSessionRepo,
    user_id: str,
    session_id: str | None,
    title_hint: str,
) -> ChatSession:
    if session_id:
        try:
            normalized_session_id = str(uuid.UUID(session_id))
        except ValueError as exc:
            raise ValueError(
                f"Invalid session_id '{session_id}'. "
                "session_id must be a valid UUID, or empty/None to create a new session."
            ) from exc
        existing = session_repo.get(session_id=normalized_session_id, user_id=user_id)
        if existing is None:
            raise ValueError(
                f"Session '{normalized_session_id}' does not exist for user '{user_id}'. "
                "Use an existing session UUID or leave session_id empty to create a new session."
            )
        return existing
    return session_repo.create(user_id=user_id, title=title_hint[:120])


def _to_rewriter_history(messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        if message.role not in {"user", "assistant"}:
            continue
        history.append({"role": message.role, "content": content})
    return history


def _persist_assistant_message(
    *,
    message_repo: ChatMessageRepo,
    session_repo: ChatSessionRepo,
    session_id: str,
    content: str,
    metadata: dict[str, Any],
) -> None:
    message_repo.create(
        session_id=session_id,
        role="assistant",
        content=content.strip(),
        metadata=metadata,
    )
    session_repo.touch(session_id)


async def stream_chitchat_answer(query: str, timeout: float = 20.0) -> AsyncIterator[str]:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    prompt = get_prompt("answer_generation_chitchat", query=query)
    stream = await client.chat.completions.create(
        model=settings.GENERATE_MODEL,
        messages=[
            {"role": "system", "content": "You are a friendly university assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=settings.TEMPERATURE_CHITCHAT,
        max_tokens=settings.MAX_GEN_CHITCHAT,
        stream=True,
        timeout=timeout,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta or not delta.content:
            continue
        yield delta.content


def get_hybrid_searcher_singleton() -> HybridSearcher:
    global _SEARCHER_SINGLETON
    if _SEARCHER_SINGLETON is None:
        _SEARCHER_SINGLETON = HybridSearcher()
    return _SEARCHER_SINGLETON


def _extract_chunk_content(doc: dict[str, Any]) -> str:
    for key in ("content", "text", "chunk", "document"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _format_doc_scores(doc: dict[str, Any]) -> str:
    scores: list[str] = []
    if "vector_score" in doc:
        scores.append(f"vector={float(doc['vector_score']):.4f}")
    if "bm25_score" in doc:
        scores.append(f"bm25={float(doc['bm25_score']):.4f}")
    if "rrf_score" in doc:
        scores.append(f"rrf={float(doc['rrf_score']):.6f}")
    if "rerank_score" in doc:
        scores.append(f"rerank={float(doc['rerank_score']):.4f}")
    return " | ".join(scores) if scores else "N/A"


async def run_pipeline(
    query: str,
    *,
    router_type: str,
    use_history: bool,
    session_id: str | None = None,
) -> None:
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty. Please set it in backend/.env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    total_t0 = time.perf_counter()

    user_id = ensure_fake_user(
        email=FAKE_USER_EMAIL,
        username=FAKE_USER_NAME,
        google_id=FAKE_USER_GOOGLE_ID,
    )
    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    message_repo = ChatMessageRepo(settings.DATABASE_URL)
    session = _resolve_session(
        session_repo=session_repo,
        user_id=user_id,
        session_id=session_id,
        title_hint=query,
    )

    stored_messages = message_repo.load_history(session.id, limit=HISTORY_MESSAGE_LIMIT) if use_history else []
    chat_history = _to_rewriter_history(stored_messages)
    message_repo.create(
        session_id=session.id,
        role="user",
        content=query,
        metadata={
            "type": "question",
            "pipeline": "pipeline.py",
            "router_type": router_type,
            "use_history": use_history,
            "history_count_before": len(chat_history),
        },
    )

    searcher = get_hybrid_searcher_singleton()

    # Warm up reranker on the same singleton searcher used by the pipeline.
    warmup_t0 = time.perf_counter()
    searcher.reranker.preload()
    warmup_ms = (time.perf_counter() - warmup_t0) * 1000
    print(f"Reranker warmup done: {warmup_ms:.1f} ms | model={searcher.reranker.model}")

    # Warm up retrieval indexes once before route/search to avoid cold start.
    index_t0 = time.perf_counter()
    searcher.load_indexes()
    index_ms = (time.perf_counter() - index_t0) * 1000
    print(f"Searcher indexes ready: {index_ms:.1f} ms")

    t0 = time.perf_counter()
    rewritten_query = await query_reflection.reflect(query, chat_history)
    rewrite_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    if router_type == "semantic":
        router = SemanticRouter(routes=ROUTES, embeddings_dir=settings.ROUTER_EMBEDDINGS_DIR)
        route_score, route_name = await router.guide(rewritten_query)
    else:
        router = KeywordRouter(routes=ROUTES)
        route_score, route_name = router.guide(rewritten_query)
    route_ms = (time.perf_counter() - t1) * 1000

    print("\n=== PIPELINE INPUT ===")
    print(f"User ID        : {user_id}")
    print(f"Session ID     : {session.id}")
    print(f"History loaded : {len(chat_history)}")
    print(f"Original query : {query}")
    print(f"Rewritten query: {rewritten_query}")
    print(f"Route          : {route_name} (score={route_score})")

    if route_name == "chitchat":
        print("\n=== ANSWER (CHITCHAT STREAM) ===")
        answer_parts: list[str] = []
        t2 = time.perf_counter()
        async for piece in stream_chitchat_answer(rewritten_query):
            answer_parts.append(piece)
            print(piece, end="", flush=True)
        answer_ms = (time.perf_counter() - t2) * 1000
        answer_text = "".join(answer_parts).strip()
        _persist_assistant_message(
            message_repo=message_repo,
            session_repo=session_repo,
            session_id=session.id,
            content=answer_text,
            metadata={
                "type": "answer",
                "route": route_name,
                "pipeline": "pipeline.py",
                "rewrite_ms": round(rewrite_ms, 2),
                "route_ms": round(route_ms, 2),
                "generate_ms": round(answer_ms, 2),
            },
        )

        print("\n\n=== TIMING (ms) ===")
        print(f"warmup  : {warmup_ms:.1f}")
        print(f"indexes : {index_ms:.1f}")
        print(f"rewrite : {rewrite_ms:.1f}")
        print(f"route   : {route_ms:.1f}")
        print(f"answer  : {answer_ms:.1f}")
        print(f"total   : {(time.perf_counter() - total_t0) * 1000:.1f}")
        return

    t3 = time.perf_counter()
    retrieved_docs = await searcher.search(
        query=rewritten_query,
        use_reranker=ENABLE_RERANK_IN_SEARCH,
    )
    search_ms = (time.perf_counter() - t3) * 1000

    print("\n=== SEARCH RESULT ===")
    print(f"Retrieved docs: {len(retrieved_docs)}")
    for i, item in enumerate(retrieved_docs, start=1):
        label = item.get("key") or item.get("title") or item.get("id") or "N/A"
        chunk_id = item.get("chunk_id") or "N/A"
        scores = _format_doc_scores(item)
        content = _extract_chunk_content(item)
        preview = content[:SEARCH_CHUNK_PREVIEW_CHARS]
        if len(content) > SEARCH_CHUNK_PREVIEW_CHARS:
            preview += "..."

        print(f"\n  [{i}] {label}")
        print(f"      chunk_id: {chunk_id}")
        print(f"      scores  : {scores}")
        print(f"      content : {preview or '[empty]'}")

    generator = AnswerGenerator()
    print("\n=== ANSWER (RAG STREAM) ===")
    answer_parts: list[str] = []
    t4 = time.perf_counter()
    async for piece in generator.stream_answer(query=rewritten_query, retrieved_docs=retrieved_docs):
        answer_parts.append(piece)
        print(piece, end="", flush=True)
    answer_ms = (time.perf_counter() - t4) * 1000
    answer_text = "".join(answer_parts).strip()
    _persist_assistant_message(
        message_repo=message_repo,
        session_repo=session_repo,
        session_id=session.id,
        content=answer_text,
        metadata={
            "type": "answer",
            "route": route_name,
            "pipeline": "pipeline.py",
            "n_retrieved": len(retrieved_docs),
            "sources": compact_sources(retrieved_docs),
            "rewrite_ms": round(rewrite_ms, 2),
            "route_ms": round(route_ms, 2),
            "search_ms": round(search_ms, 2),
            "generate_ms": round(answer_ms, 2),
        },
    )

    print("\n\n=== TIMING (ms) ===")
    print(f"warmup  : {warmup_ms:.1f}")
    print(f"indexes : {index_ms:.1f}")
    print(f"rewrite : {rewrite_ms:.1f}")
    print(f"route   : {route_ms:.1f}")
    print(f"search  : {search_ms:.1f}")
    print(f"answer  : {answer_ms:.1f}")
    print(f"total   : {(time.perf_counter() - total_t0) * 1000:.1f}")


# ==== Pipeline Config (edit these values directly) ====
QUERY = "còn mtt?"
ROUTER_TYPE = "keyword"  # "keyword" | "semantic"
USE_HISTORY = True
ENABLE_RERANK_IN_SEARCH = True
SEARCH_CHUNK_PREVIEW_CHARS = 1000
SESSION_ID: str | None = "33bbd11d-d488-4b6e-bb78-779b8ecb4307"  # Set existing UUID to continue a session.
HISTORY_MESSAGE_LIMIT = 200
FAKE_USER_EMAIL = "fake.user@utehy.local"
FAKE_USER_NAME = "Fake User"
FAKE_USER_GOOGLE_ID = "fake-google-id-001"


if __name__ == "__main__":
    asyncio.run(
        run_pipeline(
            QUERY.strip(),
            router_type=ROUTER_TYPE,
            use_history=USE_HISTORY,
            session_id=SESSION_ID,
        )
    )