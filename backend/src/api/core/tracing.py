"""Phase 6.1 — OpenTelemetry tracing (opt-in)."""
from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import Any, Iterator
from src.config.settings import settings

log = logging.getLogger(__name__)
_initialized = False


def init_tracing(service_name: str | None = None) -> None:
    """Idempotent. No-op if OTEL_ENABLED is False or libs missing."""
    global _initialized
    if _initialized:
        return
    _initialized = True
    if not getattr(settings, "OTEL_ENABLED", False):
        log.info("OTEL disabled")
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
        log.warning("OTEL libs missing (%s); skipping init", exc)
        return

    name = service_name or settings.OTEL_SERVICE_NAME
    provider = TracerProvider(resource=Resource.create({"service.name": name}))
    exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    log.info("OTEL initialized: service=%s endpoint=%s", name, settings.OTEL_EXPORTER_OTLP_ENDPOINT)


def instrument_app(app: Any) -> None:
    if not getattr(settings, "OTEL_ENABLED", False) or not _initialized:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass
    try:
        from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
        Psycopg2Instrumentor().instrument(enable_commenter=False)
    except ImportError:
        pass
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
    except ImportError:
        pass


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, *_a: Any, **_kw: Any) -> Iterator[None]:
        yield None


def get_tracer(name: str = "hybridrag") -> Any:
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()


def reset_for_tests() -> None:
    global _initialized
    _initialized = False
