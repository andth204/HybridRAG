from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.core.dependencies import AuthContext, get_manager_auth_context
from src.config.settings import settings
from src.hybridrag.chat.statistics import StatisticsRepo

router = APIRouter(prefix="/api/v1/statistics", tags=["statistics"])


class StatisticsHourPointResponse(BaseModel):
    bucket_start: str
    label: str
    user_count: int


class StatisticsOverviewResponse(BaseModel):
    total_users: int
    manager_users: int
    standard_users: int
    new_users_7d: int
    active_users_24h: int
    retention_rate_7d: int
    avg_session_minutes: float
    total_sessions: int
    total_messages: int
    hourly_activity: list[StatisticsHourPointResponse]


@lru_cache(maxsize=1)
def get_statistics_repo() -> StatisticsRepo:
    return StatisticsRepo(settings.DATABASE_URL)


@router.get("/overview", response_model=StatisticsOverviewResponse)
async def get_statistics_overview(
    _: AuthContext = Depends(get_manager_auth_context),
) -> StatisticsOverviewResponse:
    repo = get_statistics_repo()
    overview = await asyncio.to_thread(repo.get_overview)
    hourly_activity = await asyncio.to_thread(repo.get_hourly_activity, hours=8)
    return StatisticsOverviewResponse(
        total_users=overview.total_users,
        manager_users=overview.manager_users,
        standard_users=overview.standard_users,
        new_users_7d=overview.new_users_7d,
        active_users_24h=overview.active_users_24h,
        retention_rate_7d=overview.retention_rate_7d,
        avg_session_minutes=overview.avg_session_minutes,
        total_sessions=overview.total_sessions,
        total_messages=overview.total_messages,
        hourly_activity=[
            StatisticsHourPointResponse(
                bucket_start=item.bucket_start,
                label=item.label,
                user_count=item.user_count,
            )
            for item in hourly_activity
        ],
    )
