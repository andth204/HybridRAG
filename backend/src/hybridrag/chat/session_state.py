"""Per-session slot state repository (Phase 4B dialogue manager).

Stores the *current* slot frame for a chat session so multi-turn
follow-ups can reuse extracted entities (campus / major / year /
faculty / doc_type) without re-asking the user. Backed by Postgres
``chat_session_state`` (see ``scripts/migrations/003_chat_session_state.sql``).

Slot frame shape (JSONB)::

    {
      "campus": {
        "value":      "co_so_1",
        "display":    "Cơ sở 1",
        "set_at":     "2026-05-21T12:34:56+00:00",
        "confidence": 0.95,
        "turn":       3
      },
      "major":    {...},
      "year":     {...},
      "faculty":  {...},
      "doc_type": {...}
    }

Concurrency / commit semantics:

  * All writes (``upsert``, ``set_slot``, ``clear_slot``, ``reset``,
    ``touch``) must call ``conn.commit()`` explicitly because the
    pooled ``borrow()`` context manager in
    ``src/hybridrag/utils/db_pool.py`` does *not* auto-commit.
  * Reads use ``RealDictCursor`` and never commit.
  * ``set_slot`` is atomic at the SQL layer — the JSONB merge
    ``chat_session_state.slots || EXCLUDED.slots`` runs in a single
    ``INSERT ... ON CONFLICT DO UPDATE`` so a concurrent reader can
    never observe a half-applied slot frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from psycopg2.extras import Json, RealDictCursor

from src.config.settings import settings
from src.hybridrag.utils.db_pool import borrow


# -------------------------------------------------------------------- #
# Slot value
# -------------------------------------------------------------------- #
@dataclass
class SlotValue:
    """One slot entry inside the JSONB ``slots`` map.

    ``value`` is the canonical key (e.g. ``"co_so_1"``) or a primitive
    (year as ``int``). ``display`` is the user-facing label injected
    back into rewrites/answers. ``set_at`` is an ISO-8601 UTC timestamp
    so the column round-trips cleanly through ``json.dumps``.

    ``mention_count`` was added in v3.5: it tracks how many times the
    user has referenced this canonical value. Slots with ``mention_count
    >= 2`` survive the regular age-based decay because two mentions are
    a much stronger signal that the entity is the conversational focus,
    not a passing reference. New rows default to ``1`` so legacy state
    that lacks the field is treated as a single mention.
    """

    value: Any
    display: Optional[str] = None
    set_at: Optional[str] = None
    confidence: float = 1.0
    turn: int = 0
    mention_count: int = 1

    def to_jsonable(self) -> dict[str, Any]:
        """Project this entry into a plain dict suitable for ``psycopg2.extras.Json``."""
        return {
            "value": self.value,
            "display": self.display,
            "set_at": self.set_at,
            "confidence": float(self.confidence),
            "turn": int(self.turn),
            "mention_count": int(self.mention_count),
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "SlotValue":
        """Reverse of ``to_jsonable``. Tolerates legacy/missing fields.

        Legacy rows may not have every field — in particular ``turn``
        was added in Phase 4B and older payloads default to ``0``. We
        never raise here so that a stale slot frame from a previous
        deploy round-trips cleanly.
        """
        if not isinstance(data, dict):
            # Defensive: a corrupt row should not crash the dialogue manager.
            return cls(value=data)
        return cls(
            value=data.get("value"),
            display=data.get("display"),
            set_at=data.get("set_at"),
            confidence=float(data.get("confidence", 1.0)),
            turn=int(data.get("turn", 0)),
            mention_count=int(data.get("mention_count", 1)),
        )


# -------------------------------------------------------------------- #
# Session state
# -------------------------------------------------------------------- #
@dataclass
class SessionState:
    """In-memory view of one row of ``chat_session_state``.

    v3.5 adds the three summary fields. ``conversation_summary`` is a
    rolling LLM-generated digest of the user/assistant turns older than
    the rewriter's window, so the answer-compose prompt can still see
    long-session context without re-feeding raw history every turn.
    ``summary_turn_count`` records the turn index at which the summary
    was last refreshed; the dialogue manager uses it to decide whether
    a refresh is due (every ``SUMMARY_REFRESH_EVERY`` turns after the
    ``SUMMARY_TRIGGER_TURN`` threshold).
    """

    session_id: str
    slots: dict[str, SlotValue] = field(default_factory=dict)
    last_intent: Optional[str] = None
    last_query: Optional[str] = None
    updated_at: Optional[datetime] = None
    conversation_summary: Optional[str] = None
    summary_updated_at: Optional[datetime] = None
    summary_turn_count: int = 0

    def slots_jsonable(self) -> dict[str, dict[str, Any]]:
        """Serialize the slot map for storage."""
        return {name: sv.to_jsonable() for name, sv in self.slots.items()}


# -------------------------------------------------------------------- #
# Repository
# -------------------------------------------------------------------- #
class SessionStateRepo:
    """CRUD wrapper around ``chat_session_state``."""

    SUPPORTED_SLOTS: set[str] = {"campus", "major", "year", "faculty", "doc_type"}

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.DATABASE_URL

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _validate_slot_name(self, name: str) -> None:
        if name not in self.SUPPORTED_SLOTS:
            raise ValueError(
                f"Unsupported slot name: {name!r}. "
                f"Supported: {sorted(self.SUPPORTED_SLOTS)}"
            )

    @staticmethod
    def _row_to_state(session_id: str, row: dict[str, Any] | None) -> SessionState:
        if row is None:
            return SessionState(session_id=session_id)
        raw_slots = row.get("slots") or {}
        slots: dict[str, SlotValue] = {}
        if isinstance(raw_slots, dict):
            for name, payload in raw_slots.items():
                if isinstance(payload, dict):
                    slots[name] = SlotValue.from_jsonable(payload)
        return SessionState(
            session_id=str(row.get("session_id") or session_id),
            slots=slots,
            last_intent=row.get("last_intent"),
            last_query=row.get("last_query"),
            updated_at=row.get("updated_at"),
            conversation_summary=row.get("conversation_summary"),
            summary_updated_at=row.get("summary_updated_at"),
            summary_turn_count=int(row.get("summary_turn_count") or 0),
        )

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def get(self, session_id: str) -> SessionState:
        """Return the current state, or an empty default if no row exists.

        A missing row is the common bootstrap case for the first turn of
        a fresh session — we deliberately do NOT insert a placeholder
        here. The first write through ``upsert`` / ``set_slot`` will
        create the row.
        """
        sql = """
        SELECT session_id, slots, last_intent, last_query, updated_at,
               conversation_summary, summary_updated_at, summary_turn_count
        FROM chat_session_state
        WHERE session_id = %s
        """
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (session_id,))
                row = cur.fetchone()
        return self._row_to_state(session_id, row)

    def update_summary(
        self,
        session_id: str,
        *,
        summary: str,
        turn_count: int,
    ) -> None:
        """Persist a freshly generated conversation summary.

        Creates the row if missing (empty slot map) so the summary can
        outlive a slot-less session. Idempotent on the (session_id) key
        thanks to ON CONFLICT DO UPDATE.
        """
        sql = """
        INSERT INTO chat_session_state (
            session_id, slots, conversation_summary,
            summary_updated_at, summary_turn_count, updated_at
        )
        VALUES (%s, '{}'::jsonb, %s, NOW(), %s, NOW())
        ON CONFLICT (session_id) DO UPDATE SET
            conversation_summary = EXCLUDED.conversation_summary,
            summary_updated_at   = NOW(),
            summary_turn_count   = EXCLUDED.summary_turn_count,
            updated_at           = NOW()
        """
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id, summary, int(turn_count)))
            conn.commit()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert(self, state: SessionState) -> SessionState:
        """Insert or fully replace the row's slots / last_* fields.

        This is the right call when the slot filler has computed an
        entire new merged frame (decay + new extractions). For merging a
        single slot into an existing frame, prefer ``set_slot`` (which
        uses a JSONB ``||`` to avoid clobbering concurrent updates).
        """
        sql = """
        INSERT INTO chat_session_state (session_id, slots, last_intent, last_query, updated_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (session_id) DO UPDATE SET
            slots       = EXCLUDED.slots,
            last_intent = COALESCE(EXCLUDED.last_intent, chat_session_state.last_intent),
            last_query  = COALESCE(EXCLUDED.last_query,  chat_session_state.last_query),
            updated_at  = NOW()
        RETURNING session_id, slots, last_intent, last_query, updated_at
        """
        params = (
            state.session_id,
            Json(state.slots_jsonable()),
            state.last_intent,
            state.last_query,
        )
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
            conn.commit()
        return self._row_to_state(state.session_id, row)

    def set_slot(self, session_id: str, name: str, value: SlotValue) -> SessionState:
        """Atomically merge one slot into the existing frame.

        Uses ``chat_session_state.slots || EXCLUDED.slots`` so existing
        keys for other slots are preserved. ``ValueError`` if the slot
        name isn't in ``SUPPORTED_SLOTS``.
        """
        self._validate_slot_name(name)
        sql = """
        INSERT INTO chat_session_state (session_id, slots, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (session_id) DO UPDATE SET
            slots = chat_session_state.slots || EXCLUDED.slots,
            updated_at = NOW()
        RETURNING session_id, slots, last_intent, last_query, updated_at
        """
        payload = {name: value.to_jsonable()}
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (session_id, Json(payload)))
                row = cur.fetchone()
            conn.commit()
        return self._row_to_state(session_id, row)

    def clear_slot(self, session_id: str, name: str) -> SessionState:
        """Drop one slot key from the JSONB frame.

        No-op (but still touches ``updated_at``) if the row doesn't
        exist or the slot wasn't set.
        """
        self._validate_slot_name(name)
        sql = """
        UPDATE chat_session_state
        SET slots = slots - %s, updated_at = NOW()
        WHERE session_id = %s
        RETURNING session_id, slots, last_intent, last_query, updated_at
        """
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (name, session_id))
                row = cur.fetchone()
            conn.commit()
        return self._row_to_state(session_id, row)

    def reset(self, session_id: str) -> None:
        """Drop the row entirely (used by 'quên đi' / 'reset' intent)."""
        sql = "DELETE FROM chat_session_state WHERE session_id = %s"
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id,))
            conn.commit()

    def touch(
        self,
        session_id: str,
        *,
        last_intent: str | None = None,
        last_query: str | None = None,
    ) -> None:
        """Update only the advisory ``last_intent`` / ``last_query`` fields.

        Creates the row if missing (with an empty slot map). Useful for
        the dialogue manager to record intent classification even when
        no slot moved this turn.
        """
        sql = """
        INSERT INTO chat_session_state (session_id, slots, last_intent, last_query, updated_at)
        VALUES (%s, '{}'::jsonb, %s, %s, NOW())
        ON CONFLICT (session_id) DO UPDATE SET
            last_intent = COALESCE(EXCLUDED.last_intent, chat_session_state.last_intent),
            last_query  = COALESCE(EXCLUDED.last_query,  chat_session_state.last_query),
            updated_at  = NOW()
        """
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id, last_intent, last_query))
            conn.commit()


__all__ = [
    "SlotValue",
    "SessionState",
    "SessionStateRepo",
]
