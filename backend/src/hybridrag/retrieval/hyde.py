"""HyDE — Hypothetical Document Embeddings.

Generate a short hypothetical passage that *would* answer the user
query, then embed that passage for vector search. The hypothetical
answer typically shares more domain vocabulary with real documents than
the bare question, which improves recall on terse / under-specified
queries.

The generation is best-effort: any failure (timeout, API error, empty
output) returns ``""`` and the caller is expected to fall back to
embedding the raw query.
"""
from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI

from src.config.settings import settings

log = logging.getLogger(__name__)


_HYDE_SYSTEM = (
    "Bạn là chuyên viên tư vấn tuyển sinh của Trường Đại học Sư phạm "
    "Kỹ thuật Hưng Yên (UTEHY). Khi nhận một câu hỏi, hãy viết MỘT đoạn "
    "văn ngắn (3-6 câu) đóng vai trò là phần trích từ tài liệu tuyển "
    "sinh có thể trả lời câu hỏi đó. Đoạn văn phải mang giọng văn tài "
    "liệu chính thức, sử dụng các thuật ngữ chuyên ngành phù hợp "
    "(ngành, mã ngành, điểm chuẩn, học phí, phương thức xét tuyển, "
    "chỉ tiêu, ...). Không bắt đầu bằng lời chào hay xin lỗi. KHÔNG "
    "được nói rằng bạn không biết — hãy giả lập nội dung tài liệu kể "
    "cả khi không chắc chắn về số liệu chính xác."
)


_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


async def generate_hyde(query: str) -> str:
    """Return a hypothetical passage for ``query`` (or ``""`` on failure)."""
    q = (query or "").strip()
    if not q:
        return ""

    try:
        response = await _get_client().chat.completions.create(
            model=settings.HYDE_MODEL,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": q},
            ],
            temperature=settings.HYDE_TEMPERATURE,
            max_tokens=settings.HYDE_MAX_TOKENS,
            timeout=settings.HYDE_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("HyDE generation failed (%s); falling back to raw query", exc)
        return ""

    choice = (response.choices or [None])[0]
    if choice is None or choice.message is None:
        return ""
    text = (choice.message.content or "").strip()

    # Cost tracking — best-effort; never raise.
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            from src.hybridrag.utils.cost_tracker import record as _cost_record
            _cost_record(
                model=settings.HYDE_MODEL,
                tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
                tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
                feature="hyde",
            )
    except Exception:
        log.debug("hyde cost tracking failed", exc_info=True)

    return text


__all__ = ["generate_hyde"]
