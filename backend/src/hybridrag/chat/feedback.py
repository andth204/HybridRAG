"""Phase 5.6 — Chat feedback repository.

Persists user-supplied thumbs-up / thumbs-down ratings for assistant
messages plus an optional free-text comment. The repo lives next to the
other chat repos so it shares the same ``borrow()`` connection pool and
the same DSN/commit semantics:

* ``create()`` is a write — it MUST call ``conn.commit()`` explicitly,
  because the pooled context manager in
  :mod:`src.hybridrag.utils.db_pool` does not auto-commit.
* ``get_for_message`` / ``aggregate_by_session`` are reads — no commit.

PII scrubbing is the responsibility of the route layer (we don't want
the storage layer to silently mutate caller input). Callers pass an
already-scrubbed comment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from psycopg2.extras import RealDictCursor

from src.config.settings import settings
from src.hybridrag.utils.db_pool import borrow


@dataclass(frozen=True)
class FeedbackRecord:
    """One row of ``chat_feedback`` projected into Python."""

    id: int
    message_id: str
    session_id: str
    user_id: Optional[str]
    rating: str
    comment: Optional[str]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view of the record (used by the API serializer)."""
        return {
            "id": self.id,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "comment": self.comment,
        }


class ChatFeedbackRepo:
    """CRUD wrapper around the ``chat_feedback`` table.

    Schema is defined by ``scripts/migrations/004_chat_feedback.sql``.
    """

    ALLOWED_RATINGS: frozenset[str] = frozenset({"up", "down"})

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.DATABASE_URL

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _validate_rating(cls, rating: str) -> None:
        if rating not in cls.ALLOWED_RATINGS:
            raise ValueError(
                f"rating must be one of {sorted(cls.ALLOWED_RATINGS)}, got {rating!r}"
            )

    @staticmethod
    def _row_to_record(row: dict) -> FeedbackRecord:
        return FeedbackRecord(
            id=int(row["id"]),
            message_id=str(row["message_id"]),
            session_id=str(row["session_id"]),
            user_id=str(row["user_id"]) if row.get("user_id") else None,
            rating=row["rating"],
            comment=row.get("comment"),
        )

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def create(
        self,
        *,
        message_id: str,
        session_id: str,
        user_id: str | None,
        rating: str,
        comment: str | None = None,
    ) -> FeedbackRecord:
        """Insert a new feedback row.

        ``rating`` is validated both here and at the DB layer (CHECK
        constraint) so we surface a clean ``ValueError`` on the client
        side without a roundtrip.
        """
        self._validate_rating(rating)
        sql = """
        INSERT INTO chat_feedback (message_id, session_id, user_id, rating, comment)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id, message_id, session_id, user_id, rating, comment, created_at
        """
        params = (message_id, session_id, user_id, rating, comment)
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                record = self._row_to_record(row)
            conn.commit()
            return record

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def get_for_message(self, message_id: str) -> list[FeedbackRecord]:
        """Return every feedback row attached to ``message_id`` (newest first)."""
        sql = """
        SELECT id, message_id, session_id, user_id, rating, comment, created_at
        FROM chat_feedback
        WHERE message_id = %s
        ORDER BY created_at DESC
        """
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (message_id,))
                rows = cur.fetchall()
                return [self._row_to_record(row) for row in rows]

    def aggregate_by_session(self, session_id: str) -> dict[str, int]:
        """Return ``{"up": N, "down": M}`` for a given session.

        Both keys are always present (zero when no feedback of that
        rating exists) so callers can render the summary without
        defensive ``get(..., 0)`` lookups.
        """
        sql = """
        SELECT rating, COUNT(*) AS count
        FROM chat_feedback
        WHERE session_id = %s
        GROUP BY rating
        """
        out: dict[str, int] = {"up": 0, "down": 0}
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (session_id,))
                rows = cur.fetchall()
        for row in rows:
            rating = row.get("rating")
            if rating in out:
                out[rating] = int(row.get("count", 0))
        return out


__all__ = ["ChatFeedbackRepo", "FeedbackRecord"]
