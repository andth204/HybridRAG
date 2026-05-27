"""Human handoff: detect when bot cannot answer and surface advisor contact.

A turn qualifies for handoff when ANY of these conditions hold:
  1. ``verification_report.overall == "refusal"`` (answer matched the
     canonical refusal sentence).
  2. Retrieval returned zero docs AND intent is not chitchat.
  3. Top retrieval doc score below ``LOW_RECALL_THRESHOLD`` (only meaningful
     for backends that surface a normalised score).
  4. The answer text is shorter than 16 chars (degenerate LLM output).
  5. Explicit "không biết" / "không có thông tin" markers in the answer.

When triggered we attach a structured ``handoff`` block to the assistant
message metadata + emit it on SSE so the frontend can render an advisor
card (phone / hotline / email / fanpage / office hours).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Any, Optional
from src.config.settings import settings


LOW_RECALL_THRESHOLD: float = 0.15
DEGENERATE_ANSWER_MIN_CHARS: int = 16

_IDK_MARKERS = (
    "chưa có thông tin",
    "không có thông tin",
    "tôi không biết",
    "mình chưa biết",
    "mình không rõ",
    "liên hệ trực tiếp",
)


@dataclass(frozen=True)
class HandoffContact:
    advisor_name: str
    phone: str
    hotline: str
    email: str
    fanpage: str
    website: str
    office_hours: str
    zalo: str
    address: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_contact() -> HandoffContact:
    return HandoffContact(
        advisor_name=settings.HANDOFF_ADVISOR_NAME,
        phone=settings.HANDOFF_PHONE,
        hotline=settings.HANDOFF_HOTLINE,
        email=settings.HANDOFF_EMAIL,
        fanpage=settings.HANDOFF_FANPAGE,
        website=settings.HANDOFF_WEBSITE,
        office_hours=settings.HANDOFF_OFFICE_HOURS,
        zalo=settings.HANDOFF_ZALO,
        address=settings.HANDOFF_ADDRESS,
    )


def should_handoff(
    *,
    intent: str | None,
    answer_text: str,
    retrieved_docs: list[dict[str, Any]],
    verification_overall: str | None = None,
) -> tuple[bool, str | None]:
    """Return (trigger, reason) — `reason` is one of:
    ``refusal`` | ``no_docs`` | ``low_recall`` | ``degenerate`` | ``idk_marker`` | None.
    """
    if not settings.HUMAN_HANDOFF_ENABLED:
        return False, None
    if intent == "chitchat":
        return False, None

    if verification_overall == "refusal":
        return True, "refusal"

    answer_norm = (answer_text or "").strip().lower()
    if not answer_norm or len(answer_norm) < DEGENERATE_ANSWER_MIN_CHARS:
        return True, "degenerate"
    for marker in _IDK_MARKERS:
        if marker in answer_norm:
            return True, "idk_marker"

    if not retrieved_docs:
        return True, "no_docs"

    top_score = _top_score(retrieved_docs)
    if top_score is not None and top_score < LOW_RECALL_THRESHOLD:
        return True, "low_recall"

    return False, None


def build_handoff_payload(reason: str) -> dict[str, Any]:
    contact = get_contact()
    return {
        "reason": reason,
        "message": _vi_message_for(reason),
        "contact": contact.to_dict(),
    }


def _top_score(docs: list[dict[str, Any]]) -> Optional[float]:
    if not docs:
        return None
    d = docs[0]
    for key in ("rerank_score", "score", "rrf_score", "vector_score", "bm25_score"):
        v = d.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _vi_message_for(reason: str) -> str:
    base = (
        "Câu hỏi của bạn vượt phạm vi mình có dữ liệu. "
        "Bạn vui lòng liên hệ trực tiếp Phòng Tuyển sinh UTEHY để được tư vấn chi tiết:"
    )
    if reason == "refusal":
        return base
    if reason == "no_docs":
        return (
            "Mình chưa tìm thấy tài liệu phù hợp cho câu hỏi này. "
            "Vui lòng liên hệ trực tiếp Phòng Tuyển sinh:"
        )
    if reason == "low_recall":
        return (
            "Mình không chắc về câu trả lời cho thắc mắc này. "
            "Để có thông tin chính xác, bạn liên hệ Phòng Tuyển sinh:"
        )
    if reason == "idk_marker":
        return base
    if reason == "degenerate":
        return base
    return base


__all__ = [
    "HandoffContact",
    "get_contact",
    "should_handoff",
    "build_handoff_payload",
    "LOW_RECALL_THRESHOLD",
]
