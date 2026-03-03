from __future__ import annotations
from functools import lru_cache
from typing import Any, AsyncIterator
from openai import AsyncOpenAI
from src.config.prompts import get_prompt
from src.config.settings import settings
from src.hybridrag.chat.answer import AnswerGenerator
from src.hybridrag.retrieval.hybrid import HybridSearcher
from src.hybridrag.router import KeywordRouter, ROUTES, SemanticRouter


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


def get_router_type() -> str:
    configured = str(getattr(settings, "ROUTER_TYPE", "keyword")).strip().lower()
    return "semantic" if configured == "semantic" else "keyword"


@lru_cache(maxsize=1)
def get_hybrid_searcher() -> HybridSearcher:
    searcher = HybridSearcher()
    searcher.load_indexes()
    return searcher


@lru_cache(maxsize=1)
def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator()


@lru_cache(maxsize=1)
def get_keyword_router() -> KeywordRouter:
    return KeywordRouter(routes=ROUTES)


@lru_cache(maxsize=1)
def get_semantic_router() -> SemanticRouter:
    return SemanticRouter(
        routes=ROUTES,
        embeddings_dir=settings.ROUTER_EMBEDDINGS_DIR,
    )


def warm_up_runtime() -> None:
    get_hybrid_searcher()
    if get_router_type() == "semantic":
        get_semantic_router()
    else:
        get_keyword_router()


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