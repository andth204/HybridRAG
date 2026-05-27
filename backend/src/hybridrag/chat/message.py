from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from psycopg2.extras import Json, RealDictCursor

from src.hybridrag.utils.db_pool import borrow


@dataclass(frozen=True)
class ChatMessage:
    id: str
    session_id: str
    role: str
    content: str
    parent_message_id: Optional[str]
    revision_number: int
    is_edited: bool
    metadata: Optional[dict[str, Any]]
    created_at: datetime


class ChatMessageRepo:
    def __init__(self, dsn: str):
        self.dsn = dsn

    @staticmethod
    def _row_to_message(row: dict) -> ChatMessage:
        return ChatMessage(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            role=row["role"],
            content=row["content"],
            parent_message_id=str(row["parent_message_id"]) if row.get("parent_message_id") else None,
            revision_number=row["revision_number"],
            is_edited=row["is_edited"],
            metadata=row.get("metadata"),
            created_at=row["created_at"],
        )

    @staticmethod
    def _validate_role(role: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")

    def create(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        parent_message_id: Optional[str] = None,
        revision_number: int = 1,
        is_edited: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ChatMessage:
        self._validate_role(role)
        sql = """
        INSERT INTO chat_messages (
            session_id, role, content, parent_message_id,
            revision_number, is_edited, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id, session_id, role, content, parent_message_id,
                  revision_number, is_edited, metadata, created_at
        """
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    sql,
                    (
                        session_id,
                        role,
                        content,
                        parent_message_id,
                        revision_number,
                        is_edited,
                        Json(metadata) if metadata is not None else None,
                    ),
                )
                message = self._row_to_message(cur.fetchone())
            conn.commit()
            return message

    def get(self, message_id: str) -> Optional[ChatMessage]:
        sql = """
        SELECT id, session_id, role, content, parent_message_id,
               revision_number, is_edited, metadata, created_at
        FROM chat_messages
        WHERE id = %s
        """
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (message_id,))
                row = cur.fetchone()
                return self._row_to_message(row) if row else None

    def load_history(
        self,
        session_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
        ascending: bool = True,
    ) -> list[ChatMessage]:
        order = "ASC" if ascending else "DESC"
        sql = f"""
        SELECT id, session_id, role, content, parent_message_id,
               revision_number, is_edited, metadata, created_at
        FROM chat_messages
        WHERE session_id = %s
        ORDER BY created_at {order}
        LIMIT %s OFFSET %s
        """
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (session_id, limit, offset))
                rows = cur.fetchall()
                return [self._row_to_message(row) for row in rows]

    def latest(self, session_id: str, n: int = 20) -> list[ChatMessage]:
        sql = """
        SELECT * FROM (
            SELECT id, session_id, role, content, parent_message_id,
                   revision_number, is_edited, metadata, created_at
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        ) sub
        ORDER BY created_at ASC
        """
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (session_id, n))
                rows = cur.fetchall()
                return [self._row_to_message(row) for row in rows]

    def search(
        self,
        user_id: str,
        query: str,
        *,
        session_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ChatMessage]:
        base_sql = """
        SELECT m.id, m.session_id, m.role, m.content, m.parent_message_id,
               m.revision_number, m.is_edited, m.metadata, m.created_at
        FROM chat_messages m
        JOIN chat_sessions s ON s.id = m.session_id
        WHERE s.user_id = %s AND m.content ILIKE %s
        """

        params: tuple[Any, ...]
        if session_id:
            sql = (
                base_sql
                + " AND m.session_id = %s ORDER BY m.created_at DESC LIMIT %s OFFSET %s"
            )
            params = (user_id, f"%{query}%", session_id, limit, offset)
        else:
            sql = base_sql + " ORDER BY m.created_at DESC LIMIT %s OFFSET %s"
            params = (user_id, f"%{query}%", limit, offset)

        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_message(row) for row in rows]

    def delete(self, message_id: str, *, session_id: Optional[str] = None) -> bool:
        if session_id:
            sql = "DELETE FROM chat_messages WHERE id = %s AND session_id = %s"
            params = (message_id, session_id)
        else:
            sql = "DELETE FROM chat_messages WHERE id = %s"
            params = (message_id,)
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted

    def delete_by_session(self, session_id: str) -> int:
        sql = "DELETE FROM chat_messages WHERE session_id = %s"
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id,))
                rowcount = cur.rowcount
            conn.commit()
            return rowcount