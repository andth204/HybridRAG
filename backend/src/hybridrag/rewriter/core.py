from __future__ import annotations
import asyncio
import hashlib
import logging
import re
import time
import unicodedata
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from cachetools import TTLCache
from src.config.prompts import get_prompt
from src.config.settings import settings
from src.hybridrag.utils.metrics import count_tokens, truncate_text
log = logging.getLogger(__name__)


class QueryReflection:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            max_retries=0,
        )
        self._cache_size: int = max(0, int(settings.REWRITER_CACHE_SIZE))
        self._cache_ttl_seconds: int = max(0, int(getattr(settings, "REWRITER_CACHE_TTL_SECONDS", 86400)))
        self._k_rewrite: int = max(1, int(getattr(settings, "K_REWRITE", 5)))
        self._max_history_tokens: int = max(0, int(getattr(settings, "MAX_HISTORY_TOKENS_REWRITE", 200)))
        self._max_history_chars: int = int(getattr(settings, "MAX_HISTORY_CHARS_REWRITE", 1200))
        self._rewriter_model: str = str(getattr(settings, "REWRITER_MODEL", settings.GENERATE_MODEL))
        self._temperature: float = min(float(settings.TEMPERATURE_REWRITER), 0.2)
        self._max_output_tokens: int = int(getattr(settings, "MAX_REWRITE_OUTPUT_TOKENS", 56))
        self._timeout: float = float(getattr(settings, "REWRITER_TIMEOUT", 1.8))
        self._emoji_pattern = re.compile(
            "["
            "\U0001F1E6-\U0001F1FF"  # flags
            "\U0001F300-\U0001FAFF"  # symbols & pictographs
            "\U00002600-\U000027BF"  # misc symbols/dingbats
            "\U00002300-\U000023FF"  # technical symbols
            "]+",
            flags=re.UNICODE,
        )
        self._small_talk_short_queries = {
            "xin chao",
            "chao",
            "hello",
            "hi",
            "alo",
            "cam on",
            "thanks",
            "ok",
            "oke",
            "bye",
        }
        self._cache_enabled: bool = self._cache_size > 0 and self._cache_ttl_seconds > 0
        self._cache: TTLCache[str, str] | None = (
            TTLCache(maxsize=self._cache_size, ttl=self._cache_ttl_seconds) if self._cache_enabled else None
        )
        self._cache_lock = asyncio.Lock()
        self._inflight: Dict[str, asyncio.Future[str]] = {}
        self._inflight_lock = asyncio.Lock()

    def _make_cache_key(self, current_query: str, recent_queries: List[str]) -> str:
        current = current_query.strip()
        history = "\x00".join(q.strip() for q in recent_queries)
        raw = f"{current}\x00\x00{history}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _normalize_for_smalltalk(self, text: str) -> str:
        lowered = text.lower()
        no_emoji = self._emoji_pattern.sub(" ", lowered)
        no_emoji = no_emoji.replace("\u200d", " ").replace("\ufe0f", " ")
        nfd = unicodedata.normalize("NFD", no_emoji)
        no_accent = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        no_punct = re.sub(r"[^\w\s]", " ", no_accent)
        return re.sub(r"\s+", " ", no_punct).strip()

    async def _cache_get(self, key: str) -> Optional[str]:
        if not self._cache_enabled or self._cache is None:
            return None
        async with self._cache_lock:
            try:
                return self._cache[key]
            except KeyError:
                return None

    async def _cache_set(self, key: str, value: str) -> None:
        if not self._cache_enabled or self._cache is None:
            return
        async with self._cache_lock:
            self._cache[key] = value

    def _safe_count_tokens(self, text: str) -> int:
        try:
            return count_tokens(text, model=self._rewriter_model)
        except Exception:
            return len(re.findall(r"\w+", text))

    def _safe_truncate_tokens(self, text: str, max_tokens: int) -> str:
        try:
            return truncate_text(text, max_tokens=max_tokens, model=self._rewriter_model)
        except Exception:
            return text[: self._max_history_chars]

    def _format_query_history(self, queries: List[str]) -> str:
        if not queries:
            return "No previous questions."

        selected_queries = queries
        if self._max_history_tokens > 0:
            kept_rev: List[str] = []
            used_tokens = 0
            for q in reversed(queries):
                q_tokens = self._safe_count_tokens(q)
                if kept_rev and used_tokens + q_tokens > self._max_history_tokens:
                    break
                if not kept_rev and q_tokens > self._max_history_tokens:
                    kept_rev.append(self._safe_truncate_tokens(q, self._max_history_tokens))
                    break
                kept_rev.append(q)
                used_tokens += q_tokens

            if kept_rev:
                selected_queries = list(reversed(kept_rev))
            else:
                selected_queries = queries[-1:]

        return "\n".join(f"{i + 1}. {q}" for i, q in enumerate(selected_queries))

    def _get_recent_user_queries(
        self,
        chat_history: List[Dict[str, str]],
        current_query: Optional[str] = None,
    ) -> List[str]:
        needed = max(2, self._k_rewrite + 1)
        recent_rev: List[str] = []
        for msg in reversed(chat_history):
            if msg.get("role") == "user" and msg.get("content"):
                recent_rev.append(msg["content"])
                if len(recent_rev) >= needed:
                    break
        if not recent_rev:
            return []

        recent_user_queries = list(reversed(recent_rev))
        if current_query and recent_user_queries:
            if recent_user_queries[-1].strip() == current_query.strip():
                recent_user_queries = recent_user_queries[:-1]
        return recent_user_queries[-self._k_rewrite :]

    def _normalize_rewrite_output(self, raw_output: str, current_query: str) -> str:
        if not raw_output:
            return current_query

        lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
        if not lines:
            return current_query
        candidate = lines[0]
        candidate = re.sub(r"^[-*+\d\.\)\s]+", "", candidate).strip()
        candidate = candidate.strip("`").strip().strip('"').strip("'")
        if not candidate:
            return current_query

        if ":" in candidate:
            prefix, remainder = candidate.split(":", 1)
            if prefix.strip().lower() in {
                "rewrite",
                "rewritten question",
                "rewritten",
                "output",
                "cau hoi viet lai",
                "cau hoi",
            }:
                candidate = remainder.strip()

        if not candidate:
            return current_query

        lowered = candidate.strip().lower()
        if lowered.startswith(("answer", "tra loi", "tham khao", "reference")):
            return current_query

        if len(candidate) > max(180, len(current_query) * 4):
            return current_query
        return candidate

    async def _call_rewriter(self, current_query: str, recent_queries: List[str], cache_key: str) -> str:
        query_history_string = self._format_query_history(recent_queries)
        if len(query_history_string) > self._max_history_chars:
            query_history_string = query_history_string[-self._max_history_chars :]

        result = current_query
        t0 = time.perf_counter()
        try:
            prompt = get_prompt(
                "query_reflection",
                query_history=query_history_string,
                current_query=current_query,
            )
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self._rewriter_model,
                    messages=[
                        {"role": "system", "content": "You rewrite context-dependent questions."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_output_tokens,
                ),
                timeout=self._timeout,
            )
            result = self._normalize_rewrite_output(
                response.choices[0].message.content or "",
                current_query=current_query,
            )
            if result != current_query:
                await self._cache_set(cache_key, result)

            log.debug(
                "QueryReflection completed in %.1f ms | '%s' -> '%s'",
                (time.perf_counter() - t0) * 1000,
                current_query,
                result,
            )
        except asyncio.TimeoutError:
            log.warning(
                "QueryReflection timed out after %.1f ms - returning original query",
                (time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            log.warning(
                "QueryReflection error (%.1f ms): %s - returning original query",
                (time.perf_counter() - t0) * 1000,
                exc,
            )
        return result

    async def reflect(self, current_query: str, chat_history: List[Dict[str, str]]) -> str:
        recent_queries = self._get_recent_user_queries(chat_history, current_query=current_query)
        if not recent_queries:
            log.debug("QueryReflection skipped: no history")
            return current_query

        query_text = current_query.strip()
        if not query_text:
            log.debug("QueryReflection skipped: empty query")
            return current_query

        normalized_smalltalk = self._normalize_for_smalltalk(query_text)
        if normalized_smalltalk in self._small_talk_short_queries:
            log.debug("QueryReflection skipped: small-talk short query")
            return current_query

        cache_key = self._make_cache_key(current_query, recent_queries)
        cached = await self._cache_get(cache_key)
        if cached is not None:
            log.debug("QueryReflection cache hit for key=%s", cache_key)
            return cached

        loop = asyncio.get_running_loop()
        async with self._inflight_lock:
            in_flight = self._inflight.get(cache_key)
            if in_flight is None:
                in_flight = loop.create_future()
                self._inflight[cache_key] = in_flight
                is_leader = True
            else:
                is_leader = False

        if not is_leader:
            log.debug("QueryReflection joined in-flight request for key=%s", cache_key)
            return await in_flight

        result = current_query
        try:
            result = await self._call_rewriter(current_query, recent_queries, cache_key)
        finally:
            if not in_flight.done():
                in_flight.set_result(result)
            async with self._inflight_lock:
                self._inflight.pop(cache_key, None)
        return result


query_reflection = QueryReflection()


if __name__ == "__main__":
    async def _main() -> None:
        q = "cntt??"
        chat_history = [
            {"role": "user", "content": "Truong co nhung nganh nao dang tuyen sinh?"},
            {"role": "assistant", "content": "Co CNTT, KTPM, Ke toan, QTKD va mot so nganh khac."},
            {"role": "user", "content": "Minh quan tam den CNTT."},
            {"role": "assistant", "content": "Co CNTT, KTPM, Ke toan, QTKD va mot so nganh khac."},
        ]

        rewritten_query = await query_reflection.reflect(q, chat_history)
        print(f"Original query : {q}")
        print(f"Rewritten query: {rewritten_query}")

    asyncio.run(_main())