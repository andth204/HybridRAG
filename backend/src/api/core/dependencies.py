from __future__ import annotations
import asyncio
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from src.api.auth.tokens import AuthTokenRepo
from src.config.settings import settings
from src.hybridrag.chat.user import UserRepo


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    access_token: str
    user_role: str


def normalize_uuid_or_400(raw_value: str, field_name: str) -> str:
    try:
        return str(uuid.UUID(raw_value))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {field_name}: '{raw_value}'",
        ) from exc


@lru_cache(maxsize=1)
def get_user_repo() -> UserRepo:
    return UserRepo(settings.DATABASE_URL)


@lru_cache(maxsize=1)
def get_auth_token_repo() -> AuthTokenRepo:
    return AuthTokenRepo(
        settings.DATABASE_URL,
        schema=settings.AUTH_TOKEN_SCHEMA,
        table=settings.AUTH_TOKEN_TABLE,
    )


def initialize_auth_storage() -> None:
    get_user_repo().ensure_schema()
    get_auth_token_repo().ensure_schema()

bearer_scheme = HTTPBearer(auto_error=False)


async def get_auth_context(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Security(bearer_scheme),
    ] = None,
) -> AuthContext:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer access token",
        )

    raw_access_token = credentials.credentials.strip()
    token_repo = get_auth_token_repo()
    token_record = await asyncio.to_thread(token_repo.get_valid, raw_access_token, "access")
    if token_record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
        )

    user_id = normalize_uuid_or_400(token_record.user_id, "user_id")
    user_repo = get_user_repo()
    user = await asyncio.to_thread(user_repo.get_by_id_basic, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found",
        )
    if user.is_blocked:
        await asyncio.to_thread(token_repo.revoke, raw_access_token)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is blocked. Please contact a manager.",
        )
    return AuthContext(
        user_id=user_id,
        access_token=raw_access_token,
        user_role=user.role,
    )


async def get_manager_auth_context(auth: Annotated[AuthContext, Depends(get_auth_context)]) -> AuthContext:
    if auth.user_role != "manager":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Manager role is required for this action",
        )
    return auth
