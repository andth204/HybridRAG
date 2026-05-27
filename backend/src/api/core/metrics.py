"""Phase 6.2 — Prometheus metrics registry.

This module owns a single lazily-built :class:`Metrics` singleton that
the API process can borrow from any thread. The registry is created on
first :func:`get_metrics` call so:

* unit tests can swap ``settings.METRICS_ENABLED`` before importing the
  router and get a fresh registry, and
* the import of this module never crashes a process that lacks
  ``prometheus_client`` installed — we degrade to a no-op shim.

Naming convention follows the Prometheus best practice
``<namespace>_<subsystem>_<unit>``: all series are prefixed
``hybridrag_`` and end in ``_total`` / ``_seconds`` / ``_usd`` to make
recording rules trivial.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

from src.config.settings import settings

log = logging.getLogger(__name__)

# Lazy import so prometheus_client is optional. The "_AVAILABLE" flag
# turns every method on Metrics into a no-op when the dependency isn't
# installed, so importing this module never breaks downstream code.
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dep missing
    _AVAILABLE = False
    CollectorRegistry = None  # type: ignore[assignment]
    Counter = None  # type: ignore[assignment]
    Gauge = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"  # type: ignore[assignment]

    def generate_latest(_registry: Any = None) -> bytes:  # type: ignore[no-redef]
        return b""


class Metrics:
    """Lazy-initialized metrics registry. No-op if prometheus_client not installed.

    Each instance owns its own :class:`CollectorRegistry` so test code can
    spin up fresh instances without polluting the default global registry
    (which would raise ``Duplicated timeseries`` on the second test run).
    """

    def __init__(self) -> None:
        self.enabled = bool(_AVAILABLE and getattr(settings, "METRICS_ENABLED", True))
        if not self.enabled:
            return

        self.registry = CollectorRegistry()

        # ----- HTTP plane -----
        self.http_requests = Counter(
            "hybridrag_http_requests_total",
            "HTTP requests",
            labelnames=("route", "method", "status"),
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "hybridrag_http_request_seconds",
            "HTTP request latency",
            labelnames=("route", "method"),
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self.registry,
        )

        # ----- LLM plane -----
        self.llm_tokens = Counter(
            "hybridrag_llm_tokens_total",
            "LLM tokens",
            labelnames=("model", "direction"),
            registry=self.registry,
        )
        self.llm_cost = Counter(
            "hybridrag_llm_cost_usd_total",
            "LLM cost in USD",
            labelnames=("model",),
            registry=self.registry,
        )

        # ----- Pipeline plane -----
        self.retrieval_seconds = Histogram(
            "hybridrag_retrieval_seconds",
            "Retrieval latency",
            labelnames=("backend",),
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
            registry=self.registry,
        )
        self.generation_seconds = Histogram(
            "hybridrag_generation_seconds",
            "Generation latency",
            labelnames=("model", "route"),
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self.registry,
        )
        self.intent_counter = Counter(
            "hybridrag_intent_total",
            "Classified intents",
            labelnames=("intent",),
            registry=self.registry,
        )
        self.tool_calls = Counter(
            "hybridrag_tool_calls_total",
            "Tool invocations",
            labelnames=("tool", "status"),
            registry=self.registry,
        )
        self.refusal_counter = Counter(
            "hybridrag_refusal_total",
            "Refusal responses",
            registry=self.registry,
        )
        self.clarification_counter = Counter(
            "hybridrag_clarification_total",
            "Clarification turns",
            labelnames=("reason",),
            registry=self.registry,
        )
        self.verification_failures = Counter(
            "hybridrag_verification_failures_total",
            "Answers with unverified claims",
            registry=self.registry,
        )

        # ----- Gauges -----
        self.active_sessions = Gauge(
            "hybridrag_active_sessions",
            "Active sessions (24h sliding)",
            registry=self.registry,
        )
        # Initialise to 0 so the series shows up immediately on scrape,
        # even before the first session-tracking call.
        try:
            self.active_sessions.set(0)
        except Exception:  # pragma: no cover - defensive
            pass

    @contextmanager
    def time_block(self, hist: Any, *labels: str) -> Iterator[None]:
        """Context manager that records `time.perf_counter()` delta into ``hist``.

        No-op when metrics are disabled so callers can wrap blocks
        unconditionally.
        """
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            try:
                hist.labels(*labels).observe(time.perf_counter() - start)
            except Exception:  # pragma: no cover - never let metrics raise
                log.debug("Metrics.time_block observe failed", exc_info=True)

    def render(self) -> bytes:
        """Render the registry in Prometheus exposition format.

        Returns empty bytes when metrics are disabled so the /metrics
        endpoint can still respond 200 with a benign payload.
        """
        if not self.enabled:
            return b""
        return generate_latest(self.registry)

    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST if self.enabled else "text/plain"


_singleton: Metrics | None = None


def get_metrics() -> Metrics:
    """Return the process-wide metrics registry (lazy)."""
    global _singleton
    if _singleton is None:
        _singleton = Metrics()
    return _singleton


def reset_metrics_for_tests() -> None:
    """Test-only helper to force a fresh registry on next get_metrics() call.

    Production code MUST NOT call this — Prometheus client raises
    ``Duplicated timeseries`` when the same metric name is registered
    twice in the default global registry. Each Metrics instance owns
    its own CollectorRegistry, so resetting the singleton is safe.
    """
    global _singleton
    _singleton = None


__all__ = ["Metrics", "get_metrics", "reset_metrics_for_tests"]
