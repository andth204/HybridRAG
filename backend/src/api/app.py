from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.core.dependencies import initialize_auth_storage
from src.api.routers.auth import router as auth_router
from src.api.routers.chat import router as chat_router
from src.api.routers.files import router as files_router
from src.api.routers.health import router as health_router
from src.api.routers.search import router as search_router
from src.api.routers.statistics import router as statistics_router
from src.api.routers.users import router as users_router
from src.api.core.runtime import warm_up_runtime
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
    yield

app = FastAPI(
    title="HybridRAG API",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(health_router)
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
