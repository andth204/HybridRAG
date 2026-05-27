"""Phase 6.2 — Prometheus /metrics endpoint."""
from __future__ import annotations
import secrets
from fastapi import APIRouter, HTTPException, Request, Response, status
from src.api.core.metrics import get_metrics
from src.config.settings import settings

router = APIRouter(tags=["ops"])


def _check_basic_auth(request: Request) -> None:
    user = settings.METRICS_BASIC_AUTH_USER
    pwd = settings.METRICS_BASIC_AUTH_PASS
    if not user and not pwd:
        return
    import base64
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Basic auth required",
            headers={"WWW-Authenticate": 'Basic realm="metrics"'},
        )
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8", errors="ignore")
        u, _, p = decoded.partition(":")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid basic auth")
    if not (secrets.compare_digest(u, user) and secrets.compare_digest(p, pwd)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad credentials")


@router.get("/metrics", include_in_schema=False)
def metrics_endpoint(request: Request) -> Response:
    _check_basic_auth(request)
    m = get_metrics()
    return Response(content=m.render(), media_type=m.content_type())
