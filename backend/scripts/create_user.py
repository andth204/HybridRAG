import asyncio
import logging
import sys
import time
from typing import Any
import psycopg2
from src.config.settings import settings
from src.hybridrag.chat.answer import AnswerGenerator
from src.hybridrag.chat.message import ChatMessageRepo
from src.hybridrag.chat.session import ChatSessionRepo
from src.hybridrag.retrieval.hybrid import HybridSearcher


def ensure_fake_user(
    *,
    email: str = "fake.user@utehy.local",
    username: str = "Fake User",
    google_id: str = "fake-google-id-001",
) -> str:
    sql = """
    INSERT INTO users (google_id, email, username)
    VALUES (%s, %s, %s)
    ON CONFLICT (email)
    DO UPDATE SET username = EXCLUDED.username, updated_at = NOW()
    RETURNING id
    """

    with psycopg2.connect(settings.DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (google_id, email, username))
            return str(cur.fetchone()[0])


def compact_sources(items: list[dict[str, Any]], limit: int = 5) -> list[str]:
    names: list[str] = []
    for item in items:
        name = (
            item.get("title")
            or item.get("source")
            or item.get("key")
            or item.get("id")
            or "unknown"
        )
        value = str(name).strip()
        if value and value not in names:
            names.append(value)
        if len(names) >= limit:
            break
    return names


async def main(question: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    user_id = ensure_fake_user()
    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    message_repo = ChatMessageRepo(settings.DATABASE_URL)

    session = session_repo.create(user_id=user_id, title=question[:120])
    message_repo.create(
        session_id=session.id,
        role="user",
        content=question,
        metadata={"type": "question", "pipeline": "hybrid-search"},
    )

    searcher = HybridSearcher()
    searcher.load_indexes()

    t_search = time.perf_counter()
    retrieved = await searcher.search(query=question)
    search_ms = (time.perf_counter() - t_search) * 1000

    generator = AnswerGenerator()
    print(f"\nQuestion: {question}")
    print(f"User ID: {user_id}")
    print(f"Session ID: {session.id}")
    print(f"Retrieved docs: {len(retrieved)} | search: {search_ms:.1f} ms")
    print("\nAnswer (stream):")

    answer_parts: list[str] = []
    t_gen = time.perf_counter()
    async for chunk in generator.stream_answer(
        query=question,
        retrieved_docs=retrieved,
        timeout=25.0,
    ):
        answer_parts.append(chunk)
        print(chunk, end="", flush=True)

    answer_text = "".join(answer_parts).strip()
    gen_ms = (time.perf_counter() - t_gen) * 1000

    message_repo.create(
        session_id=session.id,
        role="assistant",
        content=answer_text,
        metadata={
            "type": "answer",
            "n_retrieved": len(retrieved),
            "sources": compact_sources(retrieved),
            "search_ms": round(search_ms, 2),
            "generate_ms": round(gen_ms, 2),
        },
    )
    session_repo.touch(session.id)

    history = message_repo.load_history(session.id)
    print(f"\nDone | generation: {gen_ms:.1f} ms | history_count: {len(history)}")



if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or "cntt la gi"
    asyncio.run(main(q))