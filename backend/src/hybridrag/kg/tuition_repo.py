"""Repository for the ``tuition`` table.

Companion to ``AdmissionScoresRepo``. Same connection / commit / SQL
contract: the repo owns the SQL, callers never see it. Reads do NOT
commit; writes always do.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from psycopg2.extras import RealDictCursor

from src.config.settings import settings
from src.hybridrag.utils.db_pool import borrow


@dataclass(frozen=True)
class TuitionRow:
    """One row of the ``tuition`` table."""

    id: int
    major_canonical: str
    year: int
    amount_vnd: Optional[int]
    unit: str
    note: Optional[str]
    source_file: Optional[str]


_SELECT_COLUMNS = (
    "id, major_canonical, year, amount_vnd, unit, note, source_file"
)


def _row_to_tuition(row: dict[str, Any]) -> TuitionRow:
    amount = row.get("amount_vnd")
    return TuitionRow(
        id=int(row["id"]),
        major_canonical=row["major_canonical"],
        year=int(row["year"]),
        amount_vnd=int(amount) if amount is not None else None,
        unit=row.get("unit") or "per_credit",
        note=row.get("note"),
        source_file=row.get("source_file"),
    )


class TuitionRepo:
    """CRUD wrapper around ``tuition``."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.DATABASE_URL

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert(
        self,
        *,
        major_canonical: str,
        year: int,
        amount_vnd: int | None = None,
        unit: str = "per_credit",
        note: str | None = None,
        source_file: str | None = None,
    ) -> TuitionRow:
        """Insert or update, keyed by ``(major_canonical, year, unit)``."""
        sql = f"""
        INSERT INTO tuition (
            major_canonical, year, amount_vnd, unit, note, source_file
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (major_canonical, year, unit)
        DO UPDATE SET
            amount_vnd  = EXCLUDED.amount_vnd,
            note        = EXCLUDED.note,
            source_file = EXCLUDED.source_file
        RETURNING {_SELECT_COLUMNS}
        """
        params = (major_canonical, year, amount_vnd, unit, note, source_file)
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
            conn.commit()
        return _row_to_tuition(row)

    def delete_by_source(self, source_file: str) -> int:
        """Drop every row attributed to ``source_file``. Returns count."""
        sql = "DELETE FROM tuition WHERE source_file = %s"
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (source_file,))
                deleted = cur.rowcount
            conn.commit()
        return deleted

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def lookup(
        self,
        *,
        major_canonical: str | None = None,
        year: int | None = None,
        unit: str | None = None,
        limit: int = 50,
    ) -> list[TuitionRow]:
        """Filter ``tuition`` rows by any combination of canonical keys."""
        clauses: list[str] = []
        params: list[Any] = []
        if major_canonical is not None:
            clauses.append("major_canonical = %s")
            params.append(major_canonical)
        if year is not None:
            clauses.append("year = %s")
            params.append(year)
        if unit is not None:
            clauses.append("unit = %s")
            params.append(unit)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT {_SELECT_COLUMNS} FROM tuition "
            f"{where} ORDER BY year DESC, major_canonical, unit LIMIT %s"
        )
        params.append(int(limit))

        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        return [_row_to_tuition(r) for r in rows]

    def list_majors(self) -> list[str]:
        """Distinct ``major_canonical`` values found in the table."""
        sql = (
            "SELECT DISTINCT major_canonical FROM tuition "
            "ORDER BY major_canonical"
        )
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        return [r[0] for r in rows]


__all__ = [
    "TuitionRow",
    "TuitionRepo",
]
