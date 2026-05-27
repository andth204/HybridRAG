from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.core.dependencies import initialize_auth_storage
from src.api.core.metrics_middleware import install_metrics_middleware
from src.api.core.tracing import init_tracing, instrument_app
from src.api.routers.auth import router as auth_router
from src.api.routers.chat import router as chat_router
from src.api.routers.files import router as files_router
from src.api.routers.health import router as health_router
from src.api.routers.metrics import router as metrics_router
from src.api.routers.search import router as search_router
from src.api.routers.statistics import router as statistics_router
from src.api.routers.users import router as users_router
from src.api.core.runtime import warm_up_runtime
from src.config.settings import settings
from src.hybridrag.utils.db_pool import close_all as close_db_pools
log = logging.getLogger(__name__)


class _HealthAccessLogFilter(logging.Filter):
    _ignored_fragments = (
        "GET /api/v1/health/live",
        "GET /health/live",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not any(fragment in message for fragment in self._ignored_fragments)


def _install_access_log_filters() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if any(isinstance(filter_item, _HealthAccessLogFilter) for filter_item in access_logger.filters):
        return
    access_logger.addFilter(_HealthAccessLogFilter())


_install_access_log_filters()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Bump the asyncio default executor so long-lived SSE consumers and
    # concurrent DB-bound sync calls (psycopg2 via asyncio.to_thread) do
    # not exhaust the default ~32-worker pool. Default Python pool is
    # min(32, cpu+4); under SSE + chat load it depletes quickly.
    try:
        import concurrent.futures
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=128, thread_name_prefix="api-worker"
            )
        )
    except Exception:
        log.exception("Default executor bump failed (non-fatal)")

    try:
        await asyncio.to_thread(initialize_auth_storage)
        log.info("Auth storage initialization completed")
    except Exception:
        log.exception("Auth storage initialization failed")
        raise

    try:
        await asyncio.to_thread(warm_up_runtime)
        log.info("API runtime warm-up completed")
    except Exception:
        log.exception("API runtime warm-up failed")
    try:
        yield
    finally:
        try:
            await asyncio.to_thread(close_db_pools)
            log.info("Database connection pools closed")
        except Exception:
            log.exception("Failed to close database connection pools")

app = FastAPI(
    title="HybridRAG API",
    version="0.1.0",
    lifespan=lifespan,
)

# Phase 6 ops: tracing + metrics middleware (must register BEFORE routers).
init_tracing(service_name=settings.OTEL_SERVICE_NAME)
instrument_app(app)
if getattr(settings, "METRICS_ENABLED", True):
    install_metrics_middleware(app)
if getattr(settings, "RATE_LIMIT_ENABLED", False):
    from src.api.core.rate_limit import get_rate_limiter, key_from_authorization
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def _rate_limit_mw(request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/v1/chat/answer"):
            return await call_next(request)
        key = key_from_authorization(request.headers.get("authorization", "")) or (
            request.client.host if request.client else "anon"
        )
        limiter = get_rate_limiter()
        allowed, remaining = limiter.allow(key, cost=1)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests", "retry_after_seconds": 60},
                headers={"Retry-After": "60", "X-RateLimit-Remaining": str(remaining)},
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(search_router)
app.include_router(statistics_router)
app.include_router(chat_router)
app.include_router(files_router)


@app.get("/health/live", tags=["health"])
async def health_live() -> dict[str, str]:
    return {"status": "ok"}

# uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
