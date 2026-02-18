import re
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from src.config.settings import settings
from src.config.prompts import get_prompt
from src.hybridrag.utils.metrics import count_tokens, truncate_text


class QueryReflection:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._last_query: Optional[str] = None
        self._last_rewritten: Optional[str] = None

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _extract_user_queries(self, chat_history: List[Dict[str, str]]) -> List[str]:
        return [msg["content"] for msg in chat_history if msg.get("role") == "user" and msg.get("content")]

    def _format_query_history(self, queries: List[str]) -> str:
        if not queries:
            return "No previous questions."
        return "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

    def _get_recent_user_queries(self, chat_history: List[Dict[str, str]]) -> List[str]:
        all_user_queries = self._extract_user_queries(chat_history)
        if len(all_user_queries) <= 1:
            return []
        return all_user_queries[-(min(settings.K_REWRITE, len(all_user_queries) - 1) + 1):-1]

    async def reflect(self, current_query: str, chat_history: List[Dict[str, str]]) -> str:
        recent_queries = self._get_recent_user_queries(chat_history)
        if not recent_queries:
            return current_query

        if self._last_query is not None and self._normalize(current_query) == self._normalize(self._last_query):
            return self._last_rewritten
        else:
            self._last_query = None
            self._last_rewritten = None

        query_history_string = self._format_query_history(recent_queries)
        token_count = count_tokens(query_history_string)
        if token_count > settings.MAX_HISTORY_TOKENS_REWRITE:
            query_history_string = truncate_text(query_history_string, settings.MAX_HISTORY_TOKENS_REWRITE)

        try:
            prompt = get_prompt("query_reflection", query_history=query_history_string, current_query=current_query)
            response = await self.client.chat.completions.create(
                model=settings.GENERATE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that reformulates questions based on previous context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.TEMPERATURE_REWRITER,
                max_tokens=settings.MAX_HISTORY_TOKENS_REWRITE,
                timeout=2.0
            )
            result = response.choices[0].message.content.strip()
            self._last_query = current_query
            self._last_rewritten = result
            return result
        except Exception:
            return current_query

query_reflection = QueryReflection()