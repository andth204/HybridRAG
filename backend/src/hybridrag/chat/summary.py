"""Rolling conversation summary (v3.5).

Generates a compact Vietnamese summary of older user/assistant turns so
the answer-compose prompt can carry long-session context without
re-feeding raw history every turn. We only kick in once a session has
crossed ``settings.SUMMARY_TRIGGER_TURN`` (default 12) and refresh every
``settings.SUMMARY_REFRESH_EVERY`` turns thereafter — short sessions
never pay the LLM cost.

The summary itself is persisted on ``chat_session_state`` via
:meth:`SessionStateRepo.update_summary`. Compose / rewriter prompts can
pull it from :attr:`SessionState.conversation_summary`.

Best-effort: any failure (timeout, OpenAI error, repo write failure)
returns the existing summary so a failed refresh never breaks the
request.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from openai import AsyncOpenAI

from src.config.settings import settings
from src.hybridrag.chat.session_state import SessionState, SessionStateRepo
from src.hybridrag.utils.prompt_security import sanitize_user_text

log = logging.getLogger(__name__)


_SUMMARY_SYSTEM_PROMPT = (
    "Bạn là bộ tóm tắt hội thoại tuyển sinh UTEHY. Đầu vào là lịch sử "
    "đối thoại giữa USER và ASSISTANT. Hãy viết một đoạn tiếng Việt "
    "ngắn (3-5 câu) tóm tắt: (a) chủ đề USER đang quan tâm, (b) các "
    "thực thể đã được đề cập (ngành, năm, cơ sở, phương thức xét "
    "tuyển), (c) thông tin chính ASSISTANT đã cung cấp. KHÔNG bịa "
    "thêm thông tin chưa có trong lịch sử. KHÔNG dùng emoji. Trả về "
    "đoạn văn thuần (plain text), không định dạng markdown."
)


_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _format_history(turns: list[dict[str, str]]) -> str:
    """Render turns as numbered USER/ASSISTANT lines for the prompt."""
    lines: list[str] = []
    for i, turn in enumerate(turns, start=1):
        role = (turn.get("role") or "user").upper()
        label = "USER" if role == "USER" else "ASSISTANT"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{i}. {label}: {content}")
    return "\n".join(lines)


def should_refresh(state: SessionState, *, current_turn: int) -> bool:
    """True when this turn should trigger a summary refresh.

    Rules:
      * Below ``SUMMARY_TRIGGER_TURN`` — never refresh (short session).
      * No summary yet AND past the trigger — refresh.
      * Have a summary — refresh every ``SUMMARY_REFRESH_EVERY`` turns
        based on ``state.summary_turn_count``.
    """
    if not getattr(settings, "SUMMARY_ENABLED", False):
        return False
    trigger = int(getattr(settings, "SUMMARY_TRIGGER_TURN", 12))
    if current_turn < trigger:
        return False
    if not state.conversation_summary:
        return True
    refresh_every = max(1, int(getattr(settings, "SUMMARY_REFRESH_EVERY", 4)))
    return (current_turn - int(state.summary_turn_count or 0)) >= refresh_every


async def generate_summary(history: list[dict[str, str]]) -> Optional[str]:
    """Run the LLM summarizer. Returns the new summary or None on failure."""
    if not history:
        return None
    if not settings.OPENAI_API_KEY:
        log.debug("summary.generate skipped: no OPENAI_API_KEY")
        return None

    rendered = _format_history(history)
    if not rendered.strip():
        return None
    # Sanitize the entire history body — same defense-in-depth as the
    # rewriter: a previous user message must not be able to slip a
    # prompt-protocol token into the summarizer's user role.
    safe_history = sanitize_user_text(rendered)
    model = str(getattr(settings, "SUMMARY_MODEL", "gpt-4o-mini"))
    max_tokens = int(getattr(settings, "SUMMARY_MAX_TOKENS", 200))

    t0 = time.perf_counter()
    try:
        resp = await asyncio.wait_for(
            _get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"<history>\n{safe_history}\n</history>"},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            ),
            timeout=6.0,
        )
    except Exception as exc:  # noqa: BLE001 — never block the chat path
        log.warning("summary.generate failed after %.1f ms: %s",
                    (time.perf_counter() - t0) * 1000, exc)
        return None

    text = (resp.choices[0].message.content if resp.choices else "") or ""
    text = text.strip()
    if not text:
        return None

    # Cost tracking — best-effort.
    try:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            from src.hybridrag.utils.cost_tracker import record as _cost_record
            _cost_record(
                model=model,
                tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
                tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
                feature="summary",
            )
    except Exception:
        log.debug("summary cost record failed", exc_info=True)

    return text


async def refresh_if_due(
    *,
    session_id: str,
    state: SessionState,
    history: list[dict[str, str]],
    current_turn: int,
    repo: SessionStateRepo,
) -> Optional[str]:
    """End-to-end refresh: gen + persist when due. Returns new summary or None."""
    if not should_refresh(state, current_turn=current_turn):
        return None
    new_summary = await generate_summary(history)
    if not new_summary:
        return None
    try:
        await asyncio.to_thread(
            repo.update_summary,
            session_id,
            summary=new_summary,
            turn_count=current_turn,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("summary.persist failed for session=%s: %s", session_id, exc)
        return None
    return new_summary


__all__ = [
    "should_refresh",
    "generate_summary",
    "refresh_if_due",
]
