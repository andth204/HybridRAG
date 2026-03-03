from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass(frozen=True)
class User:
    id: str
    google_id: Optional[str]
    email: str
    username: Optional[str]
    created_at: datetime
    updated_at: datetime


class UserRepo:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _conn(self):
        return psycopg2.connect(self.dsn)

    @staticmethod
    def _row_to_user(row: dict) -> User:
        return User(
            id=str(row["id"]),
            google_id=row.get("google_id"),
            email=row["email"],
            username=row.get("username"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_by_id(self, user_id: str) -> Optional[User]:
        sql = """
        SELECT id, google_id, email, username, created_at, updated_at
        FROM users
        WHERE id = %s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
                return self._row_to_user(row) if row else None

    def get_by_email(self, email: str) -> Optional[User]:
        sql = """
        SELECT id, google_id, email, username, created_at, updated_at
        FROM users
        WHERE email = %s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (email,))
                row = cur.fetchone()
                return self._row_to_user(row) if row else None

    def upsert_google_user(
        self,
        *,
        google_id: str,
        email: str,
        username: Optional[str],
    ) -> User:
        sql = """
        INSERT INTO users (google_id, email, username)
        VALUES (%s, %s, %s)
        ON CONFLICT (email)
        DO UPDATE SET
            google_id = EXCLUDED.google_id,
            username = EXCLUDED.username,
            updated_at = NOW()
        RETURNING id, google_id, email, username, created_at, updated_at
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (google_id, email, username))
                return self._row_to_user(cur.fetchone())
