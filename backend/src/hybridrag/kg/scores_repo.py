"""Repository for the ``admission_scores`` table.

The table is the structured source of truth for "điểm chuẩn" (admission
cutoff) values. The dialogue tool layer queries it via
``src.hybridrag.tools.lookup.lookup_score``; the ingest extractor in
``src.hybridrag.ingestion.extractors.scores_extractor`` populates it via
``upsert``.

All write methods explicitly ``conn.commit()`` because ``db_pool.borrow``
does not auto-commit (see the comment block at the top of
``src/hybridrag/utils/db_pool.py``). Reads never commit.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Optional

from psycopg2.extras import RealDictCursor

from src.config.settings import settings
from src.hybridrag.utils.db_pool import borrow


@dataclass(frozen=True)
class AdmissionScore:
    """One row of the ``admission_scores`` table.

    Mirrors the migration in ``scripts/migrations/002_admission_scores.sql``.
    ``score`` is exposed as a float (the DB column is ``NUMERIC(5,2)``) so
    downstream code doesn't have to import ``decimal.Decimal`` everywhere.
    """

    id: int
    campus: str
    faculty: str
    major_canonical: str
    major_code: Optional[str]
    year: int
    method: str
    subject_combo: Optional[str]
    score: Optional[float]
    note: Optional[str]
    source_file: Optional[str]
    source_chunk_id: Optional[str]


# The shared SELECT column list — keeping it in one constant means the
# DB schema is described once and the row-to-dataclass mapper stays in
# lock-step with every read query.
_SELECT_COLUMNS = (
    "id, campus, faculty, major_canonical, major_code, year, method, "
    "subject_combo, score, note, source_file, source_chunk_id"
)


def _row_to_score(row: dict[str, Any]) -> AdmissionScore:
    """Cast a ``RealDictCursor`` row to an ``AdmissionScore`` dataclass."""
    score_value = row.get("score")
    score_float: Optional[float]
    if score_value is None:
        score_float = None
    elif isinstance(score_value, Decimal):
        score_float = float(score_value)
    else:
        score_float = float(score_value)
    return AdmissionScore(
        id=int(row["id"]),
        campus=row.get("campus") or "",
        faculty=row.get("faculty") or "",
        major_canonical=row["major_canonical"],
        major_code=row.get("major_code"),
        year=int(row["year"]),
        method=row.get("method") or "",
        subject_combo=row.get("subject_combo"),
        score=score_float,
        note=row.get("note"),
        source_file=row.get("source_file"),
        source_chunk_id=row.get("source_chunk_id"),
    )


class AdmissionScoresRepo:
    """CRUD wrapper around ``admission_scores``."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.DATABASE_URL

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert(
        self,
        *,
        campus: str = "",
        faculty: str = "",
        major_canonical: str,
        major_code: str | None = None,
        year: int,
        method: str = "",
        subject_combo: str | None = None,
        score: float | None = None,
        note: str | None = None,
        source_file: str | None = None,
        source_chunk_id: str | None = None,
    ) -> AdmissionScore:
        """Insert or update a row, keyed by the table's UNIQUE constraint.

        Returns the row as stored after the upsert (so callers always see
        the canonical id even on conflict).
        """
        sql = f"""
        INSERT INTO admission_scores (
            campus, faculty, major_canonical, major_code, year, method,
            subject_combo, score, note, source_file, source_chunk_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (campus, major_canonical, year, method, subject_combo)
        DO UPDATE SET
            faculty         = EXCLUDED.faculty,
            major_code      = EXCLUDED.major_code,
            score           = EXCLUDED.score,
            note            = EXCLUDED.note,
            source_file     = EXCLUDED.source_file,
            source_chunk_id = EXCLUDED.source_chunk_id
        RETURNING {_SELECT_COLUMNS}
        """
        params = (
            campus,
            faculty,
            major_canonical,
            major_code,
            year,
            method,
            subject_combo,
            score,
            note,
            source_file,
            source_chunk_id,
        )
        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
            conn.commit()
        return _row_to_score(row)

    def delete_by_source(self, source_file: str) -> int:
        """Drop every row attributed to ``source_file``. Returns count."""
        sql = "DELETE FROM admission_scores WHERE source_file = %s"
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
        campus: str | None = None,
        faculty: str | None = None,
        method: str | None = None,
        limit: int = 50,
    ) -> list[AdmissionScore]:
        """Filter the table by any combination of canonical keys.

        All filters are optional. ``limit`` caps the returned rows.
        Results are ordered by year DESC then major then method so the
        caller can rely on a stable, "most recent first" view.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if major_canonical is not None:
            clauses.append("major_canonical = %s")
            params.append(major_canonical)
        if year is not None:
            clauses.append("year = %s")
            params.append(year)
        if campus is not None:
            clauses.append("campus = %s")
            params.append(campus)
        if faculty is not None:
            clauses.append("faculty = %s")
            params.append(faculty)
        if method is not None:
            clauses.append("method = %s")
            params.append(method)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT {_SELECT_COLUMNS} FROM admission_scores "
            f"{where} ORDER BY year DESC, major_canonical, method LIMIT %s"
        )
        params.append(int(limit))

        with borrow(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        return [_row_to_score(r) for r in rows]

    def list_majors(self, year: int | None = None) -> list[str]:
        """Distinct ``major_canonical`` values, optionally filtered by year."""
        if year is None:
            sql = (
                "SELECT DISTINCT major_canonical FROM admission_scores "
                "ORDER BY major_canonical"
            )
            params: tuple[Any, ...] = ()
        else:
            sql = (
                "SELECT DISTINCT major_canonical FROM admission_scores "
                "WHERE year = %s ORDER BY major_canonical"
            )
            params = (year,)
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [r[0] for r in rows]

    def list_years(self, major_canonical: str | None = None) -> list[int]:
        """Distinct ``year`` values, optionally filtered by major."""
        if major_canonical is None:
            sql = "SELECT DISTINCT year FROM admission_scores ORDER BY year DESC"
            params: tuple[Any, ...] = ()
        else:
            sql = (
                "SELECT DISTINCT year FROM admission_scores "
                "WHERE major_canonical = %s ORDER BY year DESC"
            )
            params = (major_canonical,)
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [int(r[0]) for r in rows]

    def list_majors_by_campus(self, campus: str) -> list[str]:
        """Distinct ``major_canonical`` values offered at a given campus."""
        sql = (
            "SELECT DISTINCT major_canonical FROM admission_scores "
            "WHERE campus = %s ORDER BY major_canonical"
        )
        with borrow(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (campus,))
                rows = cur.fetchall()
        return [r[0] for r in rows]


__all__ = [
    "AdmissionScore",
    "AdmissionScoresRepo",
]
