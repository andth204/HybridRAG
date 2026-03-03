from __future__ import annotations
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import psycopg2
from psycopg2.extras import Json, RealDictCursor


@dataclass(frozen=True)
class AuthTokenRecord:
    token_hash: str
    user_id: str
    token_type: str
    expires_at: datetime
    revoked: bool
    created_at: datetime
    metadata: Optional[dict]


class AuthTokenRepo:
    def __init__(self, dsn: str, schema: str = "public", table: str = "auth_tokens"):
        self.dsn = dsn
        self.schema = schema
        self.table = table
        self._ensure_table()

    def _conn(self):
        return psycopg2.connect(self.dsn)

    @staticmethod
    def _hash_token(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_record(row: dict) -> AuthTokenRecord:
        return AuthTokenRecord(
            token_hash=row["token_hash"],
            user_id=str(row["user_id"]),
            token_type=row["token_type"],
            expires_at=row["expires_at"],
            revoked=row["revoked"],
            created_at=row["created_at"],
            metadata=row.get("metadata"),
        )

    def _ensure_table(self) -> None:
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema}.{self.table} (
            token_hash TEXT PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_type VARCHAR(16) NOT NULL CHECK (token_type IN ('access', 'refresh')),
            expires_at TIMESTAMPTZ NOT NULL,
            revoked BOOLEAN NOT NULL DEFAULT FALSE,
            metadata JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_{self.table}_user_id ON {self.schema}.{self.table}(user_id);
        CREATE INDEX IF NOT EXISTS idx_{self.table}_expires_at ON {self.schema}.{self.table}(expires_at);
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

    def issue(
        self,
        *,
        user_id: str,
        token_type: str,
        expires_at: datetime,
        metadata: Optional[dict] = None,
    ) -> str:
        raw_token = secrets.token_urlsafe(48)
        token_hash = self._hash_token(raw_token)
        sql = f"""
        INSERT INTO {self.schema}.{self.table} (
            token_hash, user_id, token_type, expires_at, metadata
        ) VALUES (%s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        token_hash,
                        user_id,
                        token_type,
                        expires_at,
                        Json(metadata) if metadata is not None else None,
                    ),
                )
        return raw_token

    def get_valid(self, raw_token: str, token_type: str) -> Optional[AuthTokenRecord]:
        token_hash = self._hash_token(raw_token)
        sql = f"""
        SELECT token_hash, user_id, token_type, expires_at, revoked, created_at, metadata
        FROM {self.schema}.{self.table}
        WHERE token_hash = %s
          AND token_type = %s
          AND revoked = FALSE
          AND expires_at > NOW()
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (token_hash, token_type))
                row = cur.fetchone()
                return self._row_to_record(row) if row else None

    def revoke(self, raw_token: str) -> bool:
        token_hash = self._hash_token(raw_token)
        sql = f"""
        UPDATE {self.schema}.{self.table}
        SET revoked = TRUE
        WHERE token_hash = %s AND revoked = FALSE
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (token_hash,))
                return cur.rowcount > 0

    def consume_refresh(self, raw_token: str) -> Optional[AuthTokenRecord]:
        token_hash = self._hash_token(raw_token)
        select_sql = f"""
        SELECT token_hash, user_id, token_type, expires_at, revoked, created_at, metadata
        FROM {self.schema}.{self.table}
        WHERE token_hash = %s
          AND token_type = 'refresh'
          AND revoked = FALSE
          AND expires_at > NOW()
        FOR UPDATE
        """
        update_sql = f"""
        UPDATE {self.schema}.{self.table}
        SET revoked = TRUE
        WHERE token_hash = %s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_sql, (token_hash,))
                row = cur.fetchone()
                if row is None:
                    return None
                cur.execute(update_sql, (token_hash,))
                return self._row_to_record(row)

    def revoke_user_tokens(self, user_id: str, token_type: Optional[str] = None) -> int:
        if token_type:
            sql = f"""
            UPDATE {self.schema}.{self.table}
            SET revoked = TRUE
            WHERE user_id = %s AND token_type = %s AND revoked = FALSE
            """
            params = (user_id, token_type)
        else:
            sql = f"""
            UPDATE {self.schema}.{self.table}
            SET revoked = TRUE
            WHERE user_id = %s AND revoked = FALSE
            """
            params = (user_id,)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.rowcount