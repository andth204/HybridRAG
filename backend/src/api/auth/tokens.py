from __future__ import annotations
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor

from src.hybridrag.utils.db_pool import borrow


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

    def ensure_schema(self) -> None:
        qualified_table_name = f"{self.schema}.{self.table}"
        user_index_name = f"idx_{self.table}_user_id"
        expires_index_name = f"idx_{self.table}_expires_at"
        qualified_user_index_name = f"{self.schema}.{user_index_name}"
        qualified_expires_index_name = f"{self.schema}.{expires_index_name}"
        table_identifier = sql.Identifier(self.schema, self.table)

        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s)", (qualified_table_name,))
                table_exists = cur.fetchone()[0] is not None
                if not table_exists:
                    cur.execute(
                        sql.SQL(
                            """
                            CREATE TABLE {} (
                                token_hash TEXT PRIMARY KEY,
                                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                                token_type VARCHAR(16) NOT NULL CHECK (token_type IN ('access', 'refresh')),
                                expires_at TIMESTAMPTZ NOT NULL,
                                revoked BOOLEAN NOT NULL DEFAULT FALSE,
                                metadata JSONB,
                                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                            )
                            """
                        ).format(table_identifier)
                    )

                cur.execute("SELECT to_regclass(%s)", (qualified_user_index_name,))
                user_index_exists = cur.fetchone()[0] is not None
                if not user_index_exists:
                    cur.execute(
                        sql.SQL("CREATE INDEX {} ON {}(user_id)").format(
                            sql.Identifier(user_index_name),
                            table_identifier,
                        )
                    )

                cur.execute("SELECT to_regclass(%s)", (qualified_expires_index_name,))
                expires_index_exists = cur.fetchone()[0] is not None
                if not expires_index_exists:
                    cur.execute(
                        sql.SQL("CREATE INDEX {} ON {}(expires_at)").format(
                            sql.Identifier(expires_index_name),
                            table_identifier,
                        )
                    )
            conn.commit()

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
        with borrow(self.dsn) as conn:
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
            conn.commit()
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
        with borrow(self.dsn) as conn:
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
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (token_hash,))
                revoked = cur.rowcount > 0
            conn.commit()
            return revoked

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
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_sql, (token_hash,))
                row = cur.fetchone()
                if row is None:
                    return None
                cur.execute(update_sql, (token_hash,))
                record = self._row_to_record(row)
            conn.commit()
            return record

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
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rowcount = cur.rowcount
            conn.commit()
            return rowcount
