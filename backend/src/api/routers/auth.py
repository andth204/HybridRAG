from __future__ import annotations
import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from src.api.auth.service import issue_token_pair
from src.api.core.dependencies import AuthContext, get_auth_context, get_auth_token_repo, get_user_repo
from src.api.auth.google import GoogleAuthError, verify_google_id_token
from src.config.settings import settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

class UserProfileResponse(BaseModel):
    id: str
    email: str
    username: str | None
    google_id: str | None
    role: str
    is_blocked: bool
    created_at: datetime
    updated_at: datetime


class GoogleLoginRequest(BaseModel):
    id_token: str = Field(..., min_length=1)

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserProfileResponse


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class RevokeRequest(BaseModel):
    refresh_token: str | None = None
    revoke_all_user_tokens: bool = False


def _to_user_profile(user) -> UserProfileResponse:
    return UserProfileResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        google_id=user.google_id,
        role=user.role,
        is_blocked=user.is_blocked,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.post("/google", response_model=TokenResponse)
async def google_login(payload: GoogleLoginRequest) -> TokenResponse:
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GOOGLE_CLIENT_ID is empty",
        )

    try:
        identity = await asyncio.to_thread(verify_google_id_token, payload.id_token)
    except GoogleAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    user_repo = get_user_repo()
    token_repo = get_auth_token_repo()
    user = await asyncio.to_thread(
        user_repo.upsert_google_user,
        google_id=identity.sub,
        email=identity.email,
        username=identity.name,
    )
    if user.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is blocked. Please contact a manager.",
        )
    tokens = await issue_token_pair(token_repo=token_repo, user=user)
    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
        user=_to_user_profile(user),
    )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(payload: RefreshTokenRequest) -> TokenResponse:
    raw_refresh = payload.refresh_token.strip()
    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="refresh_token must not be empty",
        )

    token_repo = get_auth_token_repo()
    consumed = await asyncio.to_thread(token_repo.consume_refresh, raw_refresh)
    if consumed is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user_repo = get_user_repo()
    user = await asyncio.to_thread(user_repo.get_by_id_basic, consumed.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found for refresh token",
        )
    if user.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is blocked. Please contact a manager.",
        )

    tokens = await issue_token_pair(token_repo=token_repo, user=user)
    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
        user=_to_user_profile(user),
    )


@router.get("/me", response_model=UserProfileResponse)
async def auth_me(auth: AuthContext = Depends(get_auth_context)) -> UserProfileResponse:
    user_repo = get_user_repo()
    user = await asyncio.to_thread(user_repo.get_by_id_basic, auth.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found",
        )
    return _to_user_profile(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    payload: RevokeRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    token_repo = get_auth_token_repo()
    await asyncio.to_thread(token_repo.revoke, auth.access_token)
    if payload.refresh_token:
        await asyncio.to_thread(token_repo.revoke, payload.refresh_token.strip())
    if payload.revoke_all_user_tokens:
        await asyncio.to_thread(token_repo.revoke_user_tokens, auth.user_id, None)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
