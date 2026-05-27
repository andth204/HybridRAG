"""Phase 6.4 — Per-user token bucket rate limiter (Redis with in-process fallback)."""
from __future__ import annotations
import hashlib
import logging
import threading
import time
from typing import Any
from src.config.settings import settings

log = logging.getLogger(__name__)

_LUA_SCRIPT = """
local data = redis.call("HMGET", KEYS[1], "tokens", "ts")
local tokens = tonumber(data[1]) or tonumber(ARGV[1])
local ts     = tonumber(data[2]) or tonumber(ARGV[3])
local now    = tonumber(ARGV[3])
local cap    = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local cost   = tonumber(ARGV[4])
tokens = math.min(cap, tokens + ((now - ts) / 1000.0) * refill)
local allowed = tokens >= cost
if allowed then tokens = tokens - cost end
redis.call("HMSET", KEYS[1], "tokens", tokens, "ts", now)
redis.call("EXPIRE", KEYS[1], 300)
return {allowed and 1 or 0, math.floor(tokens)}
"""


class RateLimiter:
    """Token bucket per user. Backed by Redis if available, else in-process fallback."""

    def __init__(
        self,
        capacity: int | None = None,
        refill_per_sec: float | None = None,
        redis_url: str | None = None,
    ) -> None:
        self.capacity = int(capacity or settings.RATE_LIMIT_CAPACITY)
        self.refill_per_sec = float(refill_per_sec or settings.RATE_LIMIT_REFILL_PER_SEC)
        self.redis_url = redis_url or settings.REDIS_URL
        self._client: Any = None
        self._client_probed = False
        self._lock = threading.Lock()
        self._fallback: dict[str, tuple[float, float]] = {}
        self._script_sha: str | None = None

    def _redis(self) -> Any:
        if self._client_probed:
            return self._client
        self._client_probed = True
        try:
            import redis  # type: ignore
            client = redis.Redis.from_url(self.redis_url, socket_timeout=0.5)
            client.ping()
            self._script_sha = client.script_load(_LUA_SCRIPT)
            self._client = client
            log.info("RateLimiter: Redis at %s", self.redis_url)
        except Exception as exc:
            log.warning("RateLimiter: Redis unavailable (%s); using in-process fallback", exc)
            self._client = None
        return self._client

    def allow(self, key: str, cost: int = 1) -> tuple[bool, int]:
        client = self._redis()
        if client is not None and self._script_sha:
            try:
                now_ms = int(time.time() * 1000)
                result = client.evalsha(
                    self._script_sha, 1, f"rl:{key}",
                    self.capacity, self.refill_per_sec, now_ms, cost,
                )
                allowed = bool(int(result[0]))
                remaining = int(result[1])
                return allowed, remaining
            except Exception as exc:
                log.warning("RateLimiter Redis path failed (%s); using fallback", exc)
        return self._allow_fallback(key, cost)

    def _allow_fallback(self, key: str, cost: int) -> tuple[bool, int]:
        with self._lock:
            now = time.monotonic()
            tokens, last = self._fallback.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_sec)
            if tokens >= cost:
                tokens -= cost
                self._fallback[key] = (tokens, now)
                return True, int(tokens)
            self._fallback[key] = (tokens, now)
            return False, int(tokens)


_default: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _default
    if _default is None:
        _default = RateLimiter()
    return _default


def reset_for_tests() -> None:
    global _default
    _default = None


def key_from_authorization(header: str) -> str | None:
    if not header or not header.lower().startswith("bearer "):
        return None
    token = header.split(" ", 1)[1].strip()
    if not token:
        return None
    return hashlib.sha256(token.encode()).hexdigest()[:16]
