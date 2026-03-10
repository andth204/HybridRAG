from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor

USER_ROLE_MANAGER = "manager"
USER_ROLE_USER = "user"


@dataclass(frozen=True)
class User:
    id: str
    google_id: Optional[str]
    email: str
    username: Optional[str]
    role: str
    is_blocked: bool
    session_count: int
    message_count: int
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
            role=row.get("role") or USER_ROLE_USER,
            is_blocked=bool(row.get("is_blocked", False)),
            session_count=int(row.get("session_count") or 0),
            message_count=int(row.get("message_count") or 0),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def ensure_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'users'
                      AND table_schema = ANY(current_schemas(FALSE))
                      AND column_name IN ('role', 'is_blocked')
                    """
                )
                available_columns = {str(row["column_name"]) for row in cur.fetchall()}

                if not available_columns:
                    cur.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.tables
                            WHERE table_name = 'users'
                              AND table_schema = ANY(current_schemas(FALSE))
                        ) AS has_users_table
                        """
                    )
                    row = cur.fetchone()
                    if not row or not bool(row["has_users_table"]):
                        raise RuntimeError("Required table 'users' is missing from the current schema search path")

                if "role" not in available_columns:
                    cur.execute(
                        """
                        ALTER TABLE users
                        ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'user'
                        """
                    )

                if "is_blocked" not in available_columns:
                    cur.execute(
                        """
                        ALTER TABLE users
                        ADD COLUMN IF NOT EXISTS is_blocked BOOLEAN NOT NULL DEFAULT FALSE
                        """
                    )

    def get_by_id(self, user_id: str) -> Optional[User]:
        sql = """
        SELECT
            u.id,
            u.google_id,
            u.email,
            u.username,
            u.role,
            u.is_blocked,
            u.created_at,
            u.updated_at,
            COALESCE((
                SELECT COUNT(*)
                FROM chat_sessions cs
                WHERE cs.user_id = u.id
            ), 0) AS session_count,
            COALESCE((
                SELECT COUNT(*)
                FROM chat_messages cm
                JOIN chat_sessions cs ON cs.id = cm.session_id
                WHERE cs.user_id = u.id
            ), 0) AS message_count
        FROM users u
        WHERE u.id = %s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
                return self._row_to_user(row) if row else None

    def get_by_id_basic(self, user_id: str) -> Optional[User]:
        sql = """
        SELECT
            u.id,
            u.google_id,
            u.email,
            u.username,
            u.role,
            u.is_blocked,
            u.created_at,
            u.updated_at,
            0::BIGINT AS session_count,
            0::BIGINT AS message_count
        FROM users u
        WHERE u.id = %s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
                return self._row_to_user(row) if row else None

    def get_by_email(self, email: str) -> Optional[User]:
        sql = """
        SELECT
            u.id,
            u.google_id,
            u.email,
            u.username,
            u.role,
            u.is_blocked,
            u.created_at,
            u.updated_at,
            COALESCE((
                SELECT COUNT(*)
                FROM chat_sessions cs
                WHERE cs.user_id = u.id
            ), 0) AS session_count,
            COALESCE((
                SELECT COUNT(*)
                FROM chat_messages cm
                JOIN chat_sessions cs ON cs.id = cm.session_id
                WHERE cs.user_id = u.id
            ), 0) AS message_count
        FROM users u
        WHERE u.email = %s
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
        update_sql = """
        UPDATE users
        SET
            google_id = %s,
            username = COALESCE(%s, users.username),
            updated_at = NOW()
        WHERE email = %s
        RETURNING
            id,
            google_id,
            email,
            username,
            role,
            is_blocked,
            0::BIGINT AS session_count,
            0::BIGINT AS message_count,
            created_at,
            updated_at
        """
        insert_sql = """
        INSERT INTO users (google_id, email, username, role)
        VALUES (%s, %s, %s, %s)
        RETURNING
            id,
            google_id,
            email,
            username,
            role,
            is_blocked,
            0::BIGINT AS session_count,
            0::BIGINT AS message_count,
            created_at,
            updated_at
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Serialize first-user role assignment so only one first login gets manager.
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (915202603,))
                cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                existing_user = cur.fetchone()
                if existing_user:
                    cur.execute(update_sql, (google_id, username, email))
                    return self._row_to_user(cur.fetchone())

                cur.execute("SELECT COUNT(*) AS total_users FROM users")
                total_users = int(cur.fetchone()["total_users"])
                assigned_role = USER_ROLE_MANAGER if total_users == 0 else USER_ROLE_USER
                cur.execute(insert_sql, (google_id, email, username, assigned_role))
                return self._row_to_user(cur.fetchone())

    def list_users(self, *, limit: int = 100, offset: int = 0) -> list[User]:
        sql = """
        SELECT
            u.id,
            u.google_id,
            u.email,
            u.username,
            u.role,
            u.is_blocked,
            u.created_at,
            u.updated_at,
            COALESCE((
                SELECT COUNT(*)
                FROM chat_sessions cs
                WHERE cs.user_id = u.id
            ), 0) AS session_count,
            COALESCE((
                SELECT COUNT(*)
                FROM chat_messages cm
                JOIN chat_sessions cs ON cs.id = cm.session_id
                WHERE cs.user_id = u.id
            ), 0) AS message_count
        FROM users u
        ORDER BY u.created_at DESC
        LIMIT %s OFFSET %s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (limit, offset))
                rows = cur.fetchall()
                return [self._row_to_user(row) for row in rows]

    def update_role(self, *, user_id: str, role: str) -> Optional[User]:
        if role not in {USER_ROLE_MANAGER, USER_ROLE_USER}:
            raise ValueError("Invalid role")

        sql = """
        UPDATE users
        SET role = %s, updated_at = NOW()
        WHERE id = %s
        RETURNING
            id,
            google_id,
            email,
            username,
            role,
            is_blocked,
            0::BIGINT AS session_count,
            0::BIGINT AS message_count,
            created_at,
            updated_at
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (role, user_id))
                row = cur.fetchone()
                return self._row_to_user(row) if row else None

    def update_login_access(self, *, user_id: str, is_blocked: bool) -> Optional[User]:
        sql = """
        UPDATE users
        SET is_blocked = %s, updated_at = NOW()
        WHERE id = %s
        RETURNING
            id,
            google_id,
            email,
            username,
            role,
            is_blocked,
            0::BIGINT AS session_count,
            0::BIGINT AS message_count,
            created_at,
            updated_at
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (is_blocked, user_id))
                row = cur.fetchone()
                return self._row_to_user(row) if row else None

    def delete_user(self, *, user_id: str) -> bool:
        sql = """
        DELETE FROM users
        WHERE id = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.rowcount > 0

    def delete_by_email(self, *, email: str) -> bool:
        sql = """
        DELETE FROM users
        WHERE email = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (email.strip().lower(),))
                return cur.rowcount > 0