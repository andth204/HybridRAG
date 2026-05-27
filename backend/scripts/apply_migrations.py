"""Apply SQL migrations under ``scripts/migrations/*.sql``.

Lightweight, dependency-free migration runner. Each ``.sql`` file is
executed against ``settings.DATABASE_URL`` exactly once; subsequent
runs detect prior application via a ``schema_migrations`` table and
skip already-applied files.

Usage::

    python -m scripts.apply_migrations            # apply pending
    python -m scripts.apply_migrations --dry-run  # show plan only

Exit code is non-zero if any file errored (we still attempt later files
so a single bad migration doesn't block the rest, mirroring how
``alembic upgrade --sql`` reports partial failure).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from src.config.settings import settings
from src.hybridrag.utils.db_pool import borrow


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

# DDL for the bookkeeping table. We create it on first run so the
# migration runner is self-contained and doesn't need a pre-seeded schema.
SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);
"""


def discover_migrations(migrations_dir: Path) -> list[Path]:
    """Return every ``*.sql`` file under ``migrations_dir`` in sorted order."""
    if not migrations_dir.exists():
        return []
    return sorted(migrations_dir.glob("*.sql"))


def fetch_applied(dsn: str) -> set[str]:
    """Read the bookkeeping table; return the set of filenames already applied."""
    with borrow(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_MIGRATIONS_DDL)
            cur.execute("SELECT filename FROM schema_migrations")
            rows = cur.fetchall()
        conn.commit()
    return {row[0] for row in rows}


def apply_one(dsn: str, path: Path) -> None:
    """Apply a single migration file and record it as applied."""
    sql = path.read_text(encoding="utf-8")
    with borrow(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s) "
                "ON CONFLICT (filename) DO NOTHING",
                (path.name,),
            )
        conn.commit()


def _print(line: str) -> None:
    """Single print indirection so tests can capture or stub it."""
    print(line, flush=True)


def run(
    *,
    dsn: str | None = None,
    migrations_dir: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Execute the migration loop. Returns the desired process exit code."""
    dsn = dsn or settings.DATABASE_URL
    migrations_dir = migrations_dir or MIGRATIONS_DIR
    files = discover_migrations(migrations_dir)

    if not files:
        _print(f"[apply] no migrations found under {migrations_dir}")
        return 0

    if dry_run:
        # In --dry-run we still try to read the applied set so the operator
        # sees the realistic plan, but fall back to "everything is pending"
        # if the bookkeeping table can't be reached (e.g. no DB yet).
        try:
            applied = fetch_applied(dsn)
        except Exception as exc:  # noqa: BLE001 — diagnostic, not flow control
            _print(f"[warn] could not read schema_migrations: {exc!r}")
            applied = set()
        for path in files:
            if path.name in applied:
                _print(f"[skip] {path.name} (already applied)")
            else:
                _print(f"[apply] (dry-run) {path.name}")
        return 0

    try:
        applied = fetch_applied(dsn)
    except Exception as exc:  # noqa: BLE001 — surface and bail
        _print(f"[error] failed to read schema_migrations: {exc!r}")
        return 1

    errors = 0
    for path in files:
        if path.name in applied:
            _print(f"[skip] {path.name}")
            continue
        try:
            apply_one(dsn, path)
            _print(f"[apply] {path.name}")
        except Exception as exc:  # noqa: BLE001 — continue with remaining files
            errors += 1
            _print(f"[error] {path.name}: {exc!r}")

    return 1 if errors else 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply pending SQL migrations under scripts/migrations/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without executing.",
    )
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=MIGRATIONS_DIR,
        help="Override the migrations directory (default: scripts/migrations).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)
    return run(migrations_dir=args.migrations_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
