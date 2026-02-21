import logging
from dataclasses import dataclass
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor

@dataclass(frozen=True)
class FileState:
    bucket: str
    key: str
    etag: Optional[str]
    version_id: Optional[str]
    file_id: str

class FileStateRepo:
    def __init__(self, dsn: str, schema: str = "public", table: str = "file_index_state"):
        self.dsn = dsn
        self.schema = schema
        self.table = table
        self._ensure_table()

    def _conn(self):
        return psycopg2.connect(self.dsn)

    def _ensure_table(self):
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema}.{self.table} (
            bucket      TEXT NOT NULL,
            object_key  TEXT NOT NULL,
            etag        TEXT,
            version_id  TEXT,
            file_id     TEXT NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (bucket, object_key)
        );
        """
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(sql)

    def get(self, bucket: str, key: str) -> Optional[FileState]:
        sql = f"""
        SELECT bucket, object_key, etag, version_id, file_id
        FROM {self.schema}.{self.table}
        WHERE bucket=%s AND object_key=%s
        """
        with self._conn() as c:
            with c.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (bucket, key))
                row = cur.fetchone()
                if not row:
                    return None
                return FileState(
                    bucket=row["bucket"],
                    key=row["object_key"],
                    etag=row.get("etag"),
                    version_id=row.get("version_id"),
                    file_id=row["file_id"],
                )

    def upsert(self, state: FileState) -> None:
        sql = f"""
        INSERT INTO {self.schema}.{self.table} (bucket, object_key, etag, version_id, file_id)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (bucket, object_key)
        DO UPDATE SET etag=EXCLUDED.etag, version_id=EXCLUDED.version_id,
                      file_id=EXCLUDED.file_id, updated_at=NOW()
        """
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(sql, (state.bucket, state.key, state.etag, state.version_id, state.file_id))

    def delete(self, bucket: str, key: str) -> None:
        sql = f"DELETE FROM {self.schema}.{self.table} WHERE bucket=%s AND object_key=%s"
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(sql, (bucket, key))