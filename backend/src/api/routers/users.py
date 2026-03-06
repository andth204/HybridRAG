from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
from src.api.core.dependencies import (
    AuthContext,
    get_auth_context,
    get_manager_auth_context,
    get_auth_token_repo,
    get_user_repo,
    normalize_uuid_or_400,
)
from src.hybridrag.chat.user import USER_ROLE_MANAGER

router = APIRouter(prefix="/api/v1/users", tags=["users"])

UserRole = Literal["manager", "user"]


class UserProfileResponse(BaseModel):
    id: str
    email: str
    username: str | None
    google_id: str | None
    role: UserRole
    is_blocked: bool
    session_count: int
    message_count: int
    created_at: datetime
    updated_at: datetime


class UsersListResponse(BaseModel):
    items: list[UserProfileResponse]


class UpdateUserRoleRequest(BaseModel):
    role: UserRole


class UpdateUserLoginAccessRequest(BaseModel):
    is_blocked: bool


def _to_user_profile(user) -> UserProfileResponse:
    return UserProfileResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        google_id=user.google_id,
        role=user.role,
        is_blocked=user.is_blocked,
        session_count=user.session_count,
        message_count=user.message_count,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.get("/me", response_model=UserProfileResponse)
async def users_me(auth: AuthContext = Depends(get_auth_context)) -> UserProfileResponse:
    user_repo = get_user_repo()
    user = await asyncio.to_thread(user_repo.get_by_id, auth.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found",
        )
    return _to_user_profile(user)


@router.get("", response_model=UsersListResponse)
async def list_users(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _: AuthContext = Depends(get_manager_auth_context),
) -> UsersListResponse:
    user_repo = get_user_repo()
    users = await asyncio.to_thread(user_repo.list_users, limit=limit, offset=offset)
    return UsersListResponse(items=[_to_user_profile(user) for user in users])


@router.patch("/{user_id}/role", response_model=UserProfileResponse)
async def update_user_role(
    user_id: str,
    payload: UpdateUserRoleRequest,
    auth: AuthContext = Depends(get_manager_auth_context),
) -> UserProfileResponse:
    normalized_user_id = normalize_uuid_or_400(user_id, "user_id")
    if normalized_user_id == auth.user_id and payload.role != USER_ROLE_MANAGER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot remove your own manager role",
        )

    user_repo = get_user_repo()
    updated_user = await asyncio.to_thread(user_repo.update_role, user_id=normalized_user_id, role=payload.role)
    if updated_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    refreshed_user = await asyncio.to_thread(user_repo.get_by_id, normalized_user_id)
    if refreshed_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found after role update",
        )
    return _to_user_profile(refreshed_user)


@router.patch("/{user_id}/login-access", response_model=UserProfileResponse)
async def update_user_login_access(
    user_id: str,
    payload: UpdateUserLoginAccessRequest,
    auth: AuthContext = Depends(get_manager_auth_context),
) -> UserProfileResponse:
    normalized_user_id = normalize_uuid_or_400(user_id, "user_id")
    if normalized_user_id == auth.user_id and payload.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot block your own account",
        )

    user_repo = get_user_repo()
    updated_user = await asyncio.to_thread(
        user_repo.update_login_access,
        user_id=normalized_user_id,
        is_blocked=payload.is_blocked,
    )
    if updated_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if payload.is_blocked:
        token_repo = get_auth_token_repo()
        await asyncio.to_thread(token_repo.revoke_user_tokens, normalized_user_id, None)

    refreshed_user = await asyncio.to_thread(user_repo.get_by_id, normalized_user_id)
    if refreshed_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found after access update",
        )
    return _to_user_profile(refreshed_user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    auth: AuthContext = Depends(get_manager_auth_context),
) -> Response:
    normalized_user_id = normalize_uuid_or_400(user_id, "user_id")
    if normalized_user_id == auth.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own account",
        )

    user_repo = get_user_repo()
    deleted = await asyncio.to_thread(user_repo.delete_user, user_id=normalized_user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
