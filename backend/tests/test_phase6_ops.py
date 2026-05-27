"""Phase 6 ops tests — metrics, tracing, rate limit, cost tracker."""
from __future__ import annotations
import time
import pytest
from unittest.mock import MagicMock, patch


# ---------- Cost tracker ----------

from src.hybridrag.utils import cost_tracker


def test_estimate_cost_known_model():
    cost = cost_tracker.estimate_cost("gpt-4o-mini", 1000, 1000)
    assert abs(cost - (0.000150 + 0.000600)) < 1e-9


def test_estimate_cost_unknown_model():
    assert cost_tracker.estimate_cost("totally-unknown", 1000, 500) == 0.0


def test_estimate_cost_zero_tokens():
    assert cost_tracker.estimate_cost("gpt-4o-mini", 0, 0) == 0.0


def test_record_db_failure_does_not_raise():
    with patch("src.hybridrag.utils.cost_tracker.borrow") as mock_borrow:
        mock_borrow.side_effect = RuntimeError("db down")
        rec = cost_tracker.record(model="gpt-4o-mini", tokens_in=10, tokens_out=20)
        assert rec.cost_usd > 0
        assert rec.model == "gpt-4o-mini"


# ---------- Rate limiter ----------

from src.api.core.rate_limit import RateLimiter, key_from_authorization


def test_rate_limit_fallback_under_capacity():
    rl = RateLimiter(capacity=3, refill_per_sec=0.0)
    assert rl.allow("user1")[0] is True
    assert rl.allow("user1")[0] is True
    assert rl.allow("user1")[0] is True


def test_rate_limit_fallback_blocks_at_capacity():
    rl = RateLimiter(capacity=2, refill_per_sec=0.0)
    rl.allow("u")
    rl.allow("u")
    allowed, remaining = rl.allow("u")
    assert allowed is False
    assert remaining == 0


def test_rate_limit_fallback_refills():
    rl = RateLimiter(capacity=2, refill_per_sec=100.0)
    rl.allow("u")
    rl.allow("u")
    assert rl.allow("u")[0] is False
    time.sleep(0.05)
    assert rl.allow("u")[0] is True


def test_rate_limit_redis_unavailable_falls_back():
    rl = RateLimiter(capacity=5, refill_per_sec=0.0, redis_url="redis://invalid-host:1/0")
    assert rl.allow("u")[0] is True


def test_key_from_authorization_bearer():
    k = key_from_authorization("Bearer abc.def.ghi")
    assert k is not None and len(k) == 16


def test_key_from_authorization_none():
    assert key_from_authorization("") is None
    assert key_from_authorization("Basic xxx") is None


# ---------- Metrics ----------

from src.api.core.metrics import Metrics, get_metrics, reset_metrics_for_tests


def test_metrics_render_when_enabled():
    reset_metrics_for_tests()
    m = get_metrics()
    if not m.enabled:
        pytest.skip("prometheus_client not installed")
    m.http_requests.labels(route="/api/test", method="GET", status="200").inc()
    rendered = m.render()
    assert b"hybridrag_http_requests_total" in rendered


def test_metrics_disabled_returns_empty(monkeypatch):
    from src.config.settings import settings as s
    monkeypatch.setattr(s, "METRICS_ENABLED", False)
    reset_metrics_for_tests()
    m = get_metrics()
    assert m.render() == b""
    reset_metrics_for_tests()


# ---------- Tracing ----------

from src.api.core import tracing


def test_init_disabled_noop(monkeypatch):
    monkeypatch.setattr(tracing.settings, "OTEL_ENABLED", False)
    tracing.reset_for_tests()
    tracing.init_tracing()  # must not raise
    tracing.reset_for_tests()


def test_init_idempotent(monkeypatch):
    monkeypatch.setattr(tracing.settings, "OTEL_ENABLED", False)
    tracing.reset_for_tests()
    tracing.init_tracing()
    tracing.init_tracing()  # second call is no-op
    tracing.reset_for_tests()


def test_get_tracer_returns_object():
    t = tracing.get_tracer("test")
    with t.start_as_current_span("noop") as span:
        # span may be None in noop path; just confirm context manager works
        del span
