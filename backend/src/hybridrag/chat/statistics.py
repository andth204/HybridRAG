from __future__ import annotations
from dataclasses import dataclass
from datetime import timezone
import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass(frozen=True)
class HourlyActivityPoint:
    bucket_start: str
    label: str
    user_count: int


@dataclass(frozen=True)
class OverviewStats:
    total_users: int
    manager_users: int
    standard_users: int
    new_users_7d: int
    active_users_24h: int
    retention_rate_7d: int
    avg_session_minutes: float
    total_sessions: int
    total_messages: int


class StatisticsRepo:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _conn(self):
        return psycopg2.connect(self.dsn)

    def get_overview(self) -> OverviewStats:
        sql = """
        WITH user_stats AS (
            SELECT
                COUNT(*)::BIGINT AS total_users,
                COUNT(*) FILTER (WHERE role = 'manager')::BIGINT AS manager_users,
                COUNT(*) FILTER (WHERE role = 'user')::BIGINT AS standard_users,
                COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days')::BIGINT AS new_users_7d
            FROM users
        ),
        session_stats AS (
            SELECT
                COUNT(*)::BIGINT AS total_sessions,
                COALESCE(
                    AVG(EXTRACT(EPOCH FROM (GREATEST(updated_at, created_at) - created_at)) / 60.0),
                    0
                )::DOUBLE PRECISION AS avg_session_minutes,
                COUNT(DISTINCT user_id) FILTER (
                    WHERE updated_at >= NOW() - INTERVAL '24 hours'
                )::BIGINT AS active_users_24h,
                COUNT(DISTINCT user_id) FILTER (
                    WHERE updated_at >= NOW() - INTERVAL '7 days'
                )::BIGINT AS retained_users_7d
            FROM chat_sessions
        ),
        message_stats AS (
            SELECT COUNT(*)::BIGINT AS total_messages
            FROM chat_messages
        )
        SELECT
            us.total_users,
            us.manager_users,
            us.standard_users,
            us.new_users_7d,
            ss.active_users_24h,
            CASE
                WHEN us.total_users = 0 THEN 0
                ELSE ROUND((ss.retained_users_7d::NUMERIC * 100.0) / us.total_users)
            END::INT AS retention_rate_7d,
            ROUND(ss.avg_session_minutes::NUMERIC, 1)::DOUBLE PRECISION AS avg_session_minutes,
            ss.total_sessions,
            ms.total_messages
        FROM user_stats us
        CROSS JOIN session_stats ss
        CROSS JOIN message_stats ms
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql)
                row = cur.fetchone() or {}
        return OverviewStats(
            total_users=int(row.get("total_users") or 0),
            manager_users=int(row.get("manager_users") or 0),
            standard_users=int(row.get("standard_users") or 0),
            new_users_7d=int(row.get("new_users_7d") or 0),
            active_users_24h=int(row.get("active_users_24h") or 0),
            retention_rate_7d=int(row.get("retention_rate_7d") or 0),
            avg_session_minutes=float(row.get("avg_session_minutes") or 0.0),
            total_sessions=int(row.get("total_sessions") or 0),
            total_messages=int(row.get("total_messages") or 0),
        )

    def get_hourly_activity(self, *, hours: int = 8) -> list[HourlyActivityPoint]:
        bounded_hours = max(2, min(hours, 24))
        sql = """
        WITH buckets AS (
            SELECT generate_series(
                date_trunc('hour', NOW()) - ((%s::INT) - 1) * INTERVAL '1 hour',
                date_trunc('hour', NOW()),
                INTERVAL '1 hour'
            ) AS bucket_start
        )
        SELECT
            b.bucket_start AS bucket_start,
            TO_CHAR(b.bucket_start, 'HH24:MI') AS label,
            COUNT(DISTINCT cs.user_id)::BIGINT AS user_count
        FROM buckets b
        LEFT JOIN chat_messages cm
            ON cm.created_at >= b.bucket_start
           AND cm.created_at < b.bucket_start + INTERVAL '1 hour'
        LEFT JOIN chat_sessions cs
            ON cs.id = cm.session_id
        GROUP BY b.bucket_start
        ORDER BY b.bucket_start ASC
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (bounded_hours,))
                rows = cur.fetchall()
        return [
            HourlyActivityPoint(
                bucket_start=(
                    row["bucket_start"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                    if row.get("bucket_start")
                    else ""
                ),
                label=str(row.get("label") or "--:--"),
                user_count=int(row.get("user_count") or 0),
            )
            for row in rows
        ]