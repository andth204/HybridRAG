"""Process-wide, thread-safe psycopg2 connection pools keyed by DSN.

Repos previously opened a brand-new connection on every query
(`psycopg2.connect(dsn)` inside `_conn()`), which costs ~10-20ms per call.
This module hands out pooled connections via a `borrow()` context manager
so 1000 sequential queries reuse a small set of long-lived connections.

Notes on semantics — IMPORTANT to anyone refactoring callers:
- `with psycopg2.connect(dsn) as conn:` (the OLD pattern) auto-commits
  on normal exit and rolls back on exception. The pooled `borrow()`
  context manager below does NOT auto-commit. Callers that issue
  INSERT/UPDATE/DELETE must call `conn.commit()` explicitly inside the
  `with borrow(...) as conn:` block.
- On block exit, `borrow()` rolls back any in-flight transaction state
  before returning the connection to the pool, so we never leak dirty
  state to the next borrower.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from psycopg2.extensions import connection as PgConnection
from psycopg2.pool import ThreadedConnectionPool

from src.config.settings import settings

_pools: dict[str, ThreadedConnectionPool] = {}
_lock = threading.Lock()


def get_pool(
    dsn: str,
    minconn: Optional[int] = None,
    maxconn: Optional[int] = None,
) -> ThreadedConnectionPool:
    """Return the shared pool for `dsn`, creating it on first use.

    Pool size defaults to `settings.DB_POOL_MINCONN` / `DB_POOL_MAXCONN`
    when arguments are omitted. The first caller wins on sizing — subsequent
    calls with different sizing parameters get the already-built pool.
    """
    with _lock:
        pool = _pools.get(dsn)
        if pool is None:
            effective_min = minconn if minconn is not None else settings.DB_POOL_MINCONN
            effective_max = maxconn if maxconn is not None else settings.DB_POOL_MAXCONN
            pool = ThreadedConnectionPool(
                minconn=effective_min,
                maxconn=effective_max,
                dsn=dsn,
            )
            _pools[dsn] = pool
        return pool


@contextmanager
def borrow(dsn: str) -> Iterator[PgConnection]:
    """Borrow a pooled connection, returning it on block exit.

    Rolls back any open transaction before returning the connection to
    the pool so the next borrower starts clean. Callers performing
    writes MUST `conn.commit()` inside the block — there is no
    auto-commit on success.
    """
    pool = get_pool(dsn)
    conn = pool.getconn()
    try:
        yield conn
    finally:
        try:
            if conn.closed == 0:
                conn.rollback()
        except Exception:
            # Best-effort cleanup; never let cleanup raise out of `finally`.
            pass
        pool.putconn(conn)


def close_all() -> None:
    """Close every pool. Call once on process shutdown."""
    with _lock:
        for pool in _pools.values():
            try:
                pool.closeall()
            except Exception:
                # Don't let a misbehaving pool prevent the others from closing.
                pass
        _pools.clear()
