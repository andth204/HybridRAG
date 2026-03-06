from __future__ import annotations
import base64
import json
import time
from dataclasses import dataclass
import requests
from src.config.settings import settings


@dataclass(frozen=True)
class GoogleIdentity:
    sub: str
    email: str
    name: str | None
    email_verified: bool
    hd: str | None
    issued_at: int | None
    expires_at: int


class GoogleAuthError(ValueError):
    pass


def _decode_jwt_payload_unverified(id_token: str) -> dict:
    parts = id_token.split(".")
    if len(parts) != 3:
        return {}

    payload_segment = parts[1].strip()
    if not payload_segment:
        return {}

    padded = payload_segment + ("=" * (-len(payload_segment) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        payload = json.loads(decoded)
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def verify_google_id_token(id_token: str) -> GoogleIdentity:
    if not settings.GOOGLE_CLIENT_ID:
        raise GoogleAuthError("GOOGLE_CLIENT_ID is empty")
    token = id_token.strip()
    if not token:
        raise GoogleAuthError("id_token is empty")

    try:
        response = requests.get(
            settings.GOOGLE_TOKENINFO_URL,
            params={"id_token": token},
            timeout=10,
        )
    except Exception as exc:
        raise GoogleAuthError(f"Google verification request failed: {exc}") from exc

    if response.status_code != 200:
        raise GoogleAuthError("Invalid Google id_token")

    payload = response.json()
    audience = str(payload.get("aud", "")).strip()
    if audience != settings.GOOGLE_CLIENT_ID:
        raise GoogleAuthError("Google token audience mismatch")

    issuer = str(payload.get("iss", "")).strip()
    if issuer not in {"accounts.google.com", "https://accounts.google.com"}:
        raise GoogleAuthError("Invalid Google token issuer")

    sub = str(payload.get("sub", "")).strip()
    email = str(payload.get("email", "")).strip().lower()
    if not sub or not email:
        raise GoogleAuthError("Google token missing sub/email")

    email_verified = _to_bool(payload.get("email_verified"))
    if settings.GOOGLE_REQUIRE_VERIFIED_EMAIL and not email_verified:
        raise GoogleAuthError("Google email is not verified")

    hd = str(payload.get("hd", "")).strip() or None
    allowed_hd = settings.GOOGLE_ALLOWED_HD.strip()
    if allowed_hd and hd != allowed_hd:
        raise GoogleAuthError("Google hosted domain is not allowed")

    now_ts = int(time.time())
    try:
        exp = int(payload.get("exp", 0))
    except (TypeError, ValueError) as exc:
        raise GoogleAuthError("Google token exp is invalid") from exc
    if exp <= now_ts:
        raise GoogleAuthError("Google token expired")

    issued_at = None
    raw_iat = payload.get("iat")
    if raw_iat is not None:
        try:
            issued_at = int(raw_iat)
        except (TypeError, ValueError):
            issued_at = None

    name = str(payload.get("name", "")).strip() or None
    # tokeninfo may omit profile claims in some flows; decode the validated token payload as fallback.
    if not name:
        jwt_payload = _decode_jwt_payload_unverified(token)
        if not name:
            name = str(jwt_payload.get("name", "")).strip() or None
    return GoogleIdentity(
        sub=sub,
        email=email,
        name=name,
        email_verified=email_verified,
        hd=hd,
        issued_at=issued_at,
        expires_at=exp,
    )
