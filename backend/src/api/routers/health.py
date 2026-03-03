from __future__ import annotations
import asyncio
from typing import Any
import psycopg2
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from src.api.core.runtime import get_hybrid_searcher
from src.api.core.storage import build_minio_client, resolve_bucket
from src.config.settings import settings

router = APIRouter(prefix="/api/v1/health", tags=["health"])


def _check_db() -> tuple[bool, str]:
    try:
        with psycopg2.connect(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                _ = cur.fetchone()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _check_minio() -> tuple[bool, str]:
    try:
        bucket = resolve_bucket(None)
        client = build_minio_client()
        exists = client.bucket_exists(bucket)
        if exists:
            return True, "ok"
        return False, f"bucket '{bucket}' not found"
    except Exception as exc:
        return False, str(exc)


def _check_retrieval() -> tuple[bool, str]:
    try:
        searcher = get_hybrid_searcher()
        searcher.load_indexes()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


@router.get("/live")
async def health_live_v1() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def health_ready() -> JSONResponse:
    db_task = asyncio.to_thread(_check_db)
    minio_task = asyncio.to_thread(_check_minio)
    retrieval_task = asyncio.to_thread(_check_retrieval)
    db_ok, db_detail = await db_task
    minio_ok, minio_detail = await minio_task
    retrieval_ok, retrieval_detail = await retrieval_task

    checks: dict[str, Any] = {
        "db": {"ok": db_ok, "detail": db_detail},
        "minio": {"ok": minio_ok, "detail": minio_detail},
        "retrieval": {"ok": retrieval_ok, "detail": retrieval_detail},
    }
    ready = db_ok and minio_ok and retrieval_ok
    payload = {"status": "ok" if ready else "degraded", "checks": checks}
    status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=payload)