from __future__ import annotations
import asyncio
from dataclasses import asdict
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from src.api.core.dependencies import AuthContext, get_auth_context, get_user_repo

router = APIRouter(prefix="/api/v1/users", tags=["users"])


class UserProfileResponse(BaseModel):
    id: str
    email: str
    username: str | None
    google_id: str | None
    created_at: datetime
    updated_at: datetime


@router.get("/me", response_model=UserProfileResponse)
async def users_me(auth: AuthContext = Depends(get_auth_context)) -> UserProfileResponse:
    user_repo = get_user_repo()
    user = await asyncio.to_thread(user_repo.get_by_id, auth.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found",
        )
    return UserProfileResponse(**asdict(user))