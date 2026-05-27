"""Unit tests for the shared psycopg2 connection pool helper.

These tests mock `ThreadedConnectionPool` so they require no live Postgres.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.hybridrag.utils import db_pool


@pytest.fixture(autouse=True)
def _reset_pool_registry():
    """Each test starts with an empty `_pools` registry."""
    db_pool._pools.clear()
    yield
    db_pool._pools.clear()


def test_get_pool_is_idempotent_for_same_dsn():
    """Calling get_pool twice with the same DSN must return the same instance,
    and the underlying ThreadedConnectionPool must only be constructed once."""
    fake_pool = MagicMock(name="ThreadedConnectionPool-instance-1")
    with patch.object(
        db_pool, "ThreadedConnectionPool", return_value=fake_pool
    ) as ctor:
        first = db_pool.get_pool("postgres://u:p@h/db")
        second = db_pool.get_pool("postgres://u:p@h/db")

    assert first is second, "get_pool must memoize per DSN"
    assert ctor.call_count == 1, "ThreadedConnectionPool should be built exactly once per DSN"


def test_get_pool_returns_distinct_pools_for_distinct_dsns():
    """Different DSNs must each get their own pool instance."""
    pool_a = MagicMock(name="pool-A")
    pool_b = MagicMock(name="pool-B")
    with patch.object(
        db_pool, "ThreadedConnectionPool", side_effect=[pool_a, pool_b]
    ) as ctor:
        result_a = db_pool.get_pool("postgres://u:p@host-a/db")
        result_b = db_pool.get_pool("postgres://u:p@host-b/db")

    assert result_a is pool_a
    assert result_b is pool_b
    assert result_a is not result_b
    assert ctor.call_count == 2


def test_get_pool_honors_settings_defaults():
    """When minconn/maxconn are not passed, the pool is sized from settings."""
    with patch.object(db_pool, "ThreadedConnectionPool") as ctor, patch.object(
        db_pool.settings, "DB_POOL_MINCONN", 3
    ), patch.object(db_pool.settings, "DB_POOL_MAXCONN", 25):
        db_pool.get_pool("postgres://u:p@h/db")

    ctor.assert_called_once_with(minconn=3, maxconn=25, dsn="postgres://u:p@h/db")


def test_borrow_yields_pooled_connection_and_returns_it():
    """borrow() must yield a connection from the pool and return it on exit."""
    fake_conn = MagicMock()
    fake_conn.closed = 0
    fake_pool = MagicMock()
    fake_pool.getconn.return_value = fake_conn

    with patch.object(db_pool, "ThreadedConnectionPool", return_value=fake_pool):
        with db_pool.borrow("postgres://u:p@h/db") as conn:
            assert conn is fake_conn

    fake_pool.getconn.assert_called_once()
    fake_pool.putconn.assert_called_once_with(fake_conn)
    # Block exited without an explicit commit/rollback by the caller, so
    # `borrow` must clean transactional state before returning the conn.
    fake_conn.rollback.assert_called_once()


def test_close_all_closes_every_registered_pool():
    """close_all() must close every pool and clear the registry."""
    pool_a = MagicMock(name="pool-A")
    pool_b = MagicMock(name="pool-B")
    with patch.object(
        db_pool, "ThreadedConnectionPool", side_effect=[pool_a, pool_b]
    ):
        db_pool.get_pool("postgres://u:p@host-a/db")
        db_pool.get_pool("postgres://u:p@host-b/db")

    db_pool.close_all()

    pool_a.closeall.assert_called_once()
    pool_b.closeall.assert_called_once()
    assert db_pool._pools == {}
