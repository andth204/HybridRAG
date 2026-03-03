from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from src.api.auth.tokens import AuthTokenRepo
from src.config.settings import settings
from src.hybridrag.chat.user import User


@dataclass(frozen=True)
class IssuedTokenPair:
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int


async def issue_token_pair(
    *,
    token_repo: AuthTokenRepo,
    user: User,
) -> IssuedTokenPair:
    now = datetime.now(timezone.utc)
    access_ttl = max(1, int(settings.AUTH_ACCESS_TOKEN_TTL_MINUTES))
    refresh_ttl = max(1, int(settings.AUTH_REFRESH_TOKEN_TTL_DAYS))

    access_expires_at = now + timedelta(minutes=access_ttl)
    refresh_expires_at = now + timedelta(days=refresh_ttl)

    access_token = await asyncio.to_thread(
        token_repo.issue,
        user_id=user.id,
        token_type="access",
        expires_at=access_expires_at,
        metadata={"email": user.email},
    )
    refresh_token = await asyncio.to_thread(
        token_repo.issue,
        user_id=user.id,
        token_type="refresh",
        expires_at=refresh_expires_at,
        metadata={"email": user.email},
    )
    return IssuedTokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=access_ttl * 60,
    )
