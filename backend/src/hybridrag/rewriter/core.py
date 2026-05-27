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
from src.config.prompts import SAFETY_RULES, get_prompt
from src.config.settings import settings
from src.hybridrag.utils.metrics import count_tokens, truncate_text
from src.hybridrag.utils.prompt_security import sanitize_user_text
log = logging.getLogger(__name__)


# System message for the rewriter call. Embeds the safety preamble in
# the most trusted role so injection inside the user-content body still
# cannot override the rewrite contract.
_REWRITER_SYSTEM_MESSAGE = (
    "Bạn là bộ viết lại câu hỏi cho hệ thống tư vấn tuyển sinh tiếng Việt. "
    "Nhiệm vụ duy nhất: viết lại câu hỏi hiện tại thành MỘT câu tiếng Việt "
    "đầy đủ ngữ cảnh, không trả lời câu hỏi, không thêm giải thích.\n\n"
    f"{SAFETY_RULES}"
)


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
            "toi hieu",
            "co roi",
            "duoc roi",
            "vang",
            "da",
            "uh",
            "tam biet",
        }
        self._cache_enabled: bool = self._cache_size > 0 and self._cache_ttl_seconds > 0
        self._cache: TTLCache[str, str] | None = (
            TTLCache(maxsize=self._cache_size, ttl=self._cache_ttl_seconds) if self._cache_enabled else None
        )
        self._cache_lock = asyncio.Lock()
        self._inflight: Dict[str, asyncio.Future[str]] = {}
        self._inflight_lock = asyncio.Lock()

    def _make_cache_key(self, current_query: str, recent_turns: List[Dict[str, str]]) -> str:
        current = current_query.strip()
        history = "\x00".join(
            f"{t.get('role','')}:{(t.get('content') or '').strip()}"
            for t in recent_turns
        )
        raw = f"{current}\x00\x00{history}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _normalize_for_smalltalk(self, text: str) -> str:
        lowered = text.lower()
        no_emoji = self._emoji_pattern.sub(" ", lowered)
        no_emoji = no_emoji.replace("\u200d", " ").replace("\ufe0f", " ")
        nfd = unicodedata.normalize("NFD", no_emoji)
        no_accent = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        # "đ" (U+0111) does not decompose under NFD — transliterate explicitly
        no_accent = no_accent.replace("\u0111", "d").replace("\u0110", "D")
        no_punct = re.sub(r"[^\w\s]", " ", no_accent)
        return re.sub(r"\s+", " ", no_punct).strip()

    def _is_small_talk(self, normalized: str) -> bool:
        if normalized in self._small_talk_short_queries:
            return True
        words = normalized.split()
        if len(words) <= 6:
            for phrase in self._small_talk_short_queries:
                phrase_words = phrase.split()
                if words[: len(phrase_words)] == phrase_words:
                    return True
        return False

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

    def _format_query_history(self, turns: List[Dict[str, str]]) -> str:
        """Render the last N conversation turns as numbered Q/A pairs.

        Each turn is a dict ``{"role": "user"|"assistant", "content": ...}``
        in chronological order. The rendered block uses ``USER:`` /
        ``ASSISTANT:`` prefixes so the rewriter LLM can distinguish who
        said what — essential for resolving coreferences like
        ``"ở đó"`` against the MOST RECENT entity the assistant just
        mentioned.

        A token budget keeps the rendered block bounded; older turns
        are dropped first (we always keep the tail).
        """
        if not turns:
            return "No previous turns."

        selected = turns
        if self._max_history_tokens > 0:
            kept_rev: List[Dict[str, str]] = []
            used_tokens = 0
            for turn in reversed(turns):
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                t_tokens = self._safe_count_tokens(content)
                if kept_rev and used_tokens + t_tokens > self._max_history_tokens:
                    break
                if not kept_rev and t_tokens > self._max_history_tokens:
                    kept_rev.append({
                        "role": turn.get("role") or "user",
                        "content": self._safe_truncate_tokens(content, self._max_history_tokens),
                    })
                    break
                kept_rev.append({
                    "role": turn.get("role") or "user",
                    "content": content,
                })
                used_tokens += t_tokens

            if kept_rev:
                selected = list(reversed(kept_rev))
            else:
                selected = turns[-1:]

        lines: List[str] = []
        for i, turn in enumerate(selected, start=1):
            role = (turn.get("role") or "user").upper()
            label = "USER" if role == "USER" else "ASSISTANT"
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{i}. {label}: {content}")
        return "\n".join(lines)

    def _get_recent_turns(
        self,
        chat_history: List[Dict[str, str]],
        current_query: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Return the last ``_k_rewrite`` exchanges as alternating
        user/assistant turns. We slice on USER count so the budget is
        defined in "questions answered" not raw message count.

        The most recent user message — if it duplicates the current
        query — is stripped so the rewriter doesn't see the question it
        is asked to rewrite as its own history.
        """
        # Walk backwards collecting USER turns + their immediately
        # preceding ASSISTANT turn (if any).
        if not chat_history:
            return []

        # Index of each user turn in chat_history.
        user_indices = [i for i, m in enumerate(chat_history) if m.get("role") == "user" and (m.get("content") or "").strip()]
        if not user_indices:
            return []

        # Strip the trailing user turn if it duplicates current_query.
        if current_query and user_indices:
            last_idx = user_indices[-1]
            if (chat_history[last_idx].get("content") or "").strip() == current_query.strip():
                user_indices = user_indices[:-1]

        if not user_indices:
            return []

        # Keep last k user turns + any assistant turn that lives between
        # two kept user turns OR after the kept user turn but before the
        # current question.
        keep_users = user_indices[-self._k_rewrite :]
        start_idx = keep_users[0]
        end_idx = user_indices[-1] if user_indices else (len(chat_history) - 1)

        # Also include any assistant message that comes AFTER the last
        # kept user turn so the rewriter sees what was actually answered.
        # We extend the slice to include trailing assistant messages.
        slice_end = end_idx + 1
        while slice_end < len(chat_history) and chat_history[slice_end].get("role") == "assistant":
            slice_end += 1

        out: List[Dict[str, str]] = []
        for m in chat_history[start_idx:slice_end]:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role not in {"user", "assistant"}:
                continue
            out.append({"role": role, "content": content})
        return out

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

    @staticmethod
    def _sanitize_history_block(history_block: str) -> str:
        """Sanitize each numbered line of the rendered history string.

        ``_format_query_history`` already prefixes each turn with a
        line number, so we sanitize the content portion only and rejoin
        with newlines. This stops a previous user message from
        injecting prompt-protocol tokens / closing tags into the
        rewriter prompt.
        """
        if not history_block:
            return history_block
        out_lines: List[str] = []
        for line in history_block.splitlines():
            # Preserve the leading numeric prefix ("1. ", "2. ", ...) so
            # the model still sees structured history. Strip and
            # re-attach so sanitize only sees the user content.
            match = re.match(r"^(\s*\d+\.\s*)(.*)$", line)
            if match:
                prefix, body = match.group(1), match.group(2)
                out_lines.append(f"{prefix}{sanitize_user_text(body)}")
            else:
                out_lines.append(sanitize_user_text(line))
        return "\n".join(out_lines)

    async def _call_rewriter(self, current_query: str, recent_turns: List[Dict[str, str]], cache_key: str) -> str:
        query_history_string = self._format_query_history(recent_turns)
        if len(query_history_string) > self._max_history_chars:
            query_history_string = query_history_string[-self._max_history_chars :]
        safe_history = self._sanitize_history_block(query_history_string)
        safe_current_query = sanitize_user_text(current_query)

        result = current_query
        t0 = time.perf_counter()
        try:
            prompt = get_prompt(
                "query_reflection",
                query_history=safe_history,
                current_query=safe_current_query,
            )
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self._rewriter_model,
                    messages=[
                        {"role": "system", "content": _REWRITER_SYSTEM_MESSAGE},
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
            usage = getattr(response, "usage", None)
            if usage is not None:
                try:
                    from src.hybridrag.utils.cost_tracker import record as _cost_record
                    _cost_record(
                        model=self._rewriter_model,
                        tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
                        tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
                        feature="rewrite",
                    )
                except Exception:
                    pass
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
        recent_turns = self._get_recent_turns(chat_history, current_query=current_query)
        if not recent_turns:
            log.debug("QueryReflection skipped: no history")
            return current_query

        query_text = current_query.strip()
        if not query_text:
            log.debug("QueryReflection skipped: empty query")
            return current_query

        normalized_smalltalk = self._normalize_for_smalltalk(query_text)
        if self._is_small_talk(normalized_smalltalk):
            log.debug("QueryReflection skipped: small-talk short query")
            return current_query

        cache_key = self._make_cache_key(current_query, recent_turns)
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
            result = await self._call_rewriter(current_query, recent_turns, cache_key)
        finally:
            if not in_flight.done():
                in_flight.set_result(result)
            async with self._inflight_lock:
                self._inflight.pop(cache_key, None)
        return result


query_reflection = QueryReflection()


if __name__ == "__main__":
    async def _main() -> None:
        q = "Cơ sở đào tạo"
        chat_history = [
            {"role": "user", "content": "Phương thức & điều kiện xét tuyển??"},
            {"role": "assistant", "content": "Co CNTT, KTPM, Ke toan, QTKD va mot so nganh khac."},
            {"role": "user", "content": "Minh quan tam den CNTT."},
            {"role": "assistant", "content": "Co CNTT, KTPM, Ke toan, QTKD va mot so nganh khac."},
        ]

        rewritten_query = await query_reflection.reflect(q, chat_history)
        print(f"Original query : {q}")
        print(f"Rewritten query: {rewritten_query}")

    asyncio.run(_main())