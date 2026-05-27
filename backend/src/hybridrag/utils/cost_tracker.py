"""Phase 6.3 — OpenAI token usage + cost tracking."""
from __future__ import annotations
import datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional
from src.config.settings import settings
from src.hybridrag.utils.db_pool import borrow

log = logging.getLogger(__name__)

# USD per 1K tokens. Update as OpenAI changes prices.
PRICING_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o-mini":            {"in": 0.000150, "out": 0.000600},
    "gpt-4o":                 {"in": 0.005000, "out": 0.015000},
    "text-embedding-3-small": {"in": 0.000020, "out": 0.000000},
    "text-embedding-3-large": {"in": 0.000130, "out": 0.000000},
}


@dataclass(frozen=True)
class CostRecord:
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = PRICING_PER_1K.get(model)
    if not p:
        log.debug("Unknown model %s — cost=0", model)
        return 0.0
    return (tokens_in / 1000.0) * p["in"] + (tokens_out / 1000.0) * p["out"]


def record(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    model: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    feature: str = "answer",
) -> CostRecord:
    """Persist a usage row + emit metrics. Never raises."""
    cost = estimate_cost(model, tokens_in, tokens_out)
    rec = CostRecord(model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost)
    try:
        with borrow(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_cost_log
                        (session_id, user_id, model, tokens_in, tokens_out, cost_usd, feature, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (session_id, user_id, model, tokens_in, tokens_out, cost, feature),
                )
            conn.commit()
    except Exception as exc:
        log.warning("cost_tracker.record persist failed: %s (model=%s)", exc, model)
    try:
        from src.api.core.metrics import get_metrics
        m = get_metrics()
        if m.enabled:
            m.llm_tokens.labels(model=model, direction="in").inc(tokens_in)
            m.llm_tokens.labels(model=model, direction="out").inc(tokens_out)
            m.llm_cost.labels(model=model).inc(cost)
    except Exception:
        pass
    return rec


def daily_total(date: Optional[dt.date] = None) -> float:
    target = date or dt.datetime.now(dt.timezone.utc).date()
    try:
        with borrow(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(cost_usd), 0)::float
                    FROM chat_cost_log
                    WHERE created_at::date = %s
                    """,
                    (target,),
                )
                row = cur.fetchone()
                return float(row[0]) if row else 0.0
    except Exception as exc:
        log.warning("cost_tracker.daily_total failed: %s", exc)
        return 0.0


def session_total(session_id: str) -> float:
    try:
        with borrow(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0)::float FROM chat_cost_log WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
                return float(row[0]) if row else 0.0
    except Exception as exc:
        log.warning("cost_tracker.session_total failed: %s", exc)
        return 0.0
