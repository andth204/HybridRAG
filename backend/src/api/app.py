from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routers.auth import router as auth_router
from src.api.routers.chat import router as chat_router
from src.api.routers.files import router as files_router
from src.api.routers.health import router as health_router
from src.api.routers.search import router as search_router
from src.api.routers.users import router as users_router
from src.api.core.runtime import warm_up_runtime
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
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
app.include_router(chat_router)
app.include_router(files_router)


@app.get("/health/live", tags=["health"])
async def health_live() -> dict[str, str]:
    return {"status": "ok"}