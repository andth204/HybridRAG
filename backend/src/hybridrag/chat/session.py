from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from psycopg2.extras import RealDictCursor

from src.hybridrag.utils.db_pool import borrow


@dataclass(frozen=True)
class ChatSession:
    id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime


class ChatSessionRepo:
    def __init__(self, dsn: str):
        self.dsn = dsn

    @staticmethod
    def _row_to_session(row: dict) -> ChatSession:
        return ChatSession(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            title=row.get("title"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create(self, user_id: str, title: Optional[str] = None) -> ChatSession:
        sql = """
        INSERT INTO chat_sessions (user_id, title)
        VALUES (%s, %s)
        RETURNING id, user_id, title, created_at, updated_at
        """
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (user_id, title))
                session = self._row_to_session(cur.fetchone())
            conn.commit()
            return session

    def get(self, session_id: str, user_id: Optional[str] = None) -> Optional[ChatSession]:
        if user_id:
            sql = """
            SELECT id, user_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE id = %s AND user_id = %s
            """
            params = (session_id, user_id)
        else:
            sql = """
            SELECT id, user_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE id = %s
            """
            params = (session_id,)

        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return self._row_to_session(row) if row else None

    def list_by_user(
        self,
        user_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        title_query: Optional[str] = None,
    ) -> list[ChatSession]:
        if title_query:
            sql = """
            SELECT id, user_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = %s AND title ILIKE %s
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
            """
            params = (user_id, f"%{title_query}%", limit, offset)
        else:
            sql = """
            SELECT id, user_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
            """
            params = (user_id, limit, offset)

        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_session(row) for row in rows]

    def search_by_title(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ChatSession]:
        return self.list_by_user(
            user_id,
            limit=limit,
            offset=offset,
            title_query=query,
        )

    def rename(self, session_id: str, title: str, user_id: Optional[str] = None) -> bool:
        if user_id:
            sql = """
            UPDATE chat_sessions
            SET title = %s
            WHERE id = %s AND user_id = %s
            """
            params = (title, session_id, user_id)
        else:
            sql = """
            UPDATE chat_sessions
            SET title = %s
            WHERE id = %s
            """
            params = (title, session_id)

        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                changed = cur.rowcount > 0
            conn.commit()
            return changed

    def touch(self, session_id: str) -> bool:
        sql = """
        UPDATE chat_sessions
        SET updated_at = NOW()
        WHERE id = %s
        """
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id,))
                changed = cur.rowcount > 0
            conn.commit()
            return changed

    def delete(self, session_id: str, user_id: Optional[str] = None) -> bool:
        if user_id:
            sql = "DELETE FROM chat_sessions WHERE id = %s AND user_id = %s"
            params = (session_id, user_id)
        else:
            sql = "DELETE FROM chat_sessions WHERE id = %s"
            params = (session_id,)

        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted
