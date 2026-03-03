from __future__ import annotations
import re
from typing import Any, AsyncIterator, Dict, List, Optional
from openai import AsyncOpenAI
from src.config.prompts import get_prompt
from src.config.settings import settings


class AnswerGenerator:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    @staticmethod
    def _basename(path_like: str) -> str:
        cleaned = path_like.strip()
        if not cleaned:
            return ""
        cleaned = cleaned.split("?", 1)[0].split("#", 1)[0]
        parts = re.split(r"[\\/]+", cleaned)
        return parts[-1].strip() if parts and parts[-1].strip() else cleaned

    @staticmethod
    def _get_source_id(doc: Dict[str, Any], fallback_index: int) -> str:
        for key in ("key", "source", "title", "document", "doc_id", "id"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"Source-{fallback_index}"

    @staticmethod
    def _get_source_name(doc: Dict[str, Any], fallback_index: int) -> str:
        key_value = doc.get("key")
        if isinstance(key_value, str) and key_value.strip():
            return AnswerGenerator._basename(key_value)

        for key in ("source", "title", "document", "doc_id", "id"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return AnswerGenerator._basename(value)
        return f"Source-{fallback_index}"

    @staticmethod
    def _get_content(doc: Dict[str, Any]) -> str:
        for key in ("content", "text", "chunk", "document"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def build_context(
        self,
        docs: List[Dict[str, Any]],
        *,
        max_docs: int = 6,
        max_chars_per_doc: int = 1200,
    ) -> str:
        lines: List[str] = []
        used_source_ids: set[str] = set()
        kept = 0
        for i, doc in enumerate(docs, start=1):
            if kept >= max_docs:
                break
            source_id = self._get_source_id(doc, i)
            source_name = self._get_source_name(doc, i)
            if source_id in used_source_ids:
                continue

            content = self._get_content(doc)
            if not content:
                continue

            used_source_ids.add(source_id)
            kept += 1
            content = content[:max_chars_per_doc]
            lines.append(f"[{kept}] {source_name}\n{content}")
        return "\n\n".join(lines)

    async def stream_answer(
        self,
        *,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        timeout: float = 20.0,
        max_docs: int = 6,
        max_chars_per_doc: int = 1200,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        context = self.build_context(
            retrieved_docs,
            max_docs=max_docs,
            max_chars_per_doc=max_chars_per_doc,
        )
        prompt = get_prompt(
            "answer_generation_rag",
            context=context,
            query=query,
        )
        stream = await self.client.chat.completions.create(
            model=settings.GENERATE_MODEL,
            messages=[
                {"role": "system", "content": "You are a retrieval-grounded assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=(temperature if temperature is not None else settings.TEMPERATURE_MAIN),
            max_tokens=(max_tokens if max_tokens is not None else settings.MAX_GEN_MAIN),
            stream=True,
            timeout=timeout,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue
            piece = delta.content
            if piece:
                yield piece

    async def answer_text(
        self,
        *,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        timeout: float = 20.0,
        max_docs: int = 6,
        max_chars_per_doc: int = 1200,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        chunks: List[str] = []
        async for piece in self.stream_answer(
            query=query,
            retrieved_docs=retrieved_docs,
            timeout=timeout,
            max_docs=max_docs,
            max_chars_per_doc=max_chars_per_doc,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            chunks.append(piece)
        return "".join(chunks)

answer_generator = AnswerGenerator()
__all__ = ["AnswerGenerator", "answer_generator"]


if __name__ == "__main__":
    import asyncio
    import time
    from src.hybridrag.retrieval.hybrid import HybridSearcher

    async def _test_end_to_end(
        query: str = "Thông tin các phương thức tuyển sinh",
        *,
        use_reranker: bool = True,
        vector_k: int | None = None,
        bm25_k: int | None = None,
        rerank_top_k: int | None = None,
        max_docs: int = 6,
        max_chars_per_doc: int = 1200,
        gen_timeout: float = 25.0,
    ) -> None:
        # --- Load searcher once (indexes + reranker) ---
        searcher = HybridSearcher()
        searcher.load_indexes()

        # --- Retrieval (vector + bm25 + fuse + optional rerank) ---
        t0 = time.perf_counter()
        results = await searcher.search(
            query=query,
            vector_k=vector_k,
            bm25_k=bm25_k,
            use_reranker=use_reranker,
            rerank_top_k=rerank_top_k,
        )
        t_search = time.perf_counter() - t0

        print(f"\nQuery: {query}")
        print(f"Retrieved: {len(results)} docs | Search time: {t_search*1000:.1f} ms")
        for i, item in enumerate(results[:3], start=1):
            key = item.get("key", item.get("title", "N/A"))
            print(f"  [{i}] {key}")

        generator = AnswerGenerator()

        print("\nAnswer (stream):")
        t1 = time.perf_counter()
        async for chunk in generator.stream_answer(
            query=query,
            retrieved_docs=results,
            timeout=gen_timeout,
            max_docs=max_docs,
            max_chars_per_doc=max_chars_per_doc,
        ):
            print(chunk, end="", flush=True)
        t_gen = time.perf_counter() - t1
        print(f"\n\nGeneration time: {t_gen*1000:.1f} ms")

    asyncio.run(_test_end_to_end())