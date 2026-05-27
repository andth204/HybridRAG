"""Phase 6.2 — Starlette middleware that records HTTP counters + latency histograms."""
from __future__ import annotations
import time
from typing import Awaitable, Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from src.api.core.metrics import get_metrics


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        m = get_metrics()
        start = time.perf_counter()
        response = await call_next(request)
        if not m.enabled:
            return response
        route = _route_template(request)
        elapsed = time.perf_counter() - start
        try:
            m.http_requests.labels(
                route=route, method=request.method, status=str(response.status_code)
            ).inc()
            m.http_latency.labels(route=route, method=request.method).observe(elapsed)
        except Exception:
            pass
        return response


def install_metrics_middleware(app) -> None:
    app.add_middleware(PrometheusMiddleware)
