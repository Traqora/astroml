"""Token bucket rate limiting with per-endpoint configuration (issue #331).

Enhanced with:
- Sliding window algorithm support (issue #299)
- Redis-backed distributed rate limiting (issue #299)
- Admin override capabilities (issue #299)
- Rate limit headers (issue #299)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Optional, Tuple, Dict, Any, List

from api.auth.config import (
    ADMIN_BLACKLIST,
    ADMIN_OVERRIDE_ENABLED,
    ADMIN_WHITELIST,
    API_KEY_RATE_LIMIT_PER_MINUTE,
    JWT_RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_ALGORITHM,
    REDIS_RATE_LIMIT_ENABLED,
    REDIS_URL,
    SLIDING_WINDOW_SIZE_SECONDS,
)

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    tokens: float
    last_refill: float
    capacity: float
    refill_rate: float  # tokens per second


@dataclass
class RateLimitConfig:
    """Per-endpoint rate limit configuration."""

    requests_per_minute: int = 60
    burst_size: int = 10
    algorithm: str = "token_bucket"  # 'token_bucket' or 'sliding_window'


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    retry_after: Optional[int] = None
    remaining: int = 0
    limit: int = 0
    reset_time: Optional[int] = None
    algorithm: str = "token_bucket"


class SlidingWindowCounter:
    """
    Sliding window rate limiter using sorted set of timestamps.
    """

    def __init__(self, redis_client=None):
        self._buckets: dict[str, list[float]] = {}
        self._lock = Lock()
        self._redis_client = redis_client
        self._redis_enabled = REDIS_RATE_LIMIT_ENABLED and redis_client is not None

    def _get_redis_key(self, key: str) -> str:
        """Generate a Redis key for the rate limit bucket."""
        return f"rate_limit:{hashlib.sha256(key.encode()).hexdigest()}"

    def is_allowed(
        self,
        key: str,
        window_seconds: int,
        max_requests: int,
    ) -> Tuple[bool, int, int]:
        """
        Check if request is allowed using sliding window algorithm.

        Returns:
            Tuple of (allowed, remaining, retry_after_seconds)
        """
        now = time.time()
        window_start = now - window_seconds

        if self._redis_enabled:
            return self._is_allowed_redis(key, now, window_start, window_seconds, max_requests)

        return self._is_allowed_memory(key, now, window_start, window_seconds, max_requests)

    def _is_allowed_redis(
        self,
        key: str,
        now: float,
        window_start: float,
        window_seconds: int,
        max_requests: int,
    ) -> Tuple[bool, int, int]:
        """Redis-backed sliding window check."""
        try:
            redis_key = self._get_redis_key(key)
            redis = self._redis_client

            # Remove old entries
            redis.zremrangebyscore(redis_key, 0, window_start)

            # Get current count
            count = redis.zcard(redis_key)

            if count < max_requests:
                # Add current request
                redis.zadd(redis_key, {str(now): now})
                redis.expire(redis_key, window_seconds + 10)
                return True, max_requests - count - 1, 0

            # Calculate retry after
            oldest = redis.zrange(redis_key, 0, 0, withscores=True)
            retry_after = int((oldest[0][1] + window_seconds) - now) + 1 if oldest else 0
            return False, 0, max(0, retry_after)

        except Exception as e:
            logger.warning(f"Redis rate limit failed, falling back to memory: {e}")
            return self._is_allowed_memory(key, now, window_start, window_seconds, max_requests)

    def _is_allowed_memory(
        self,
        key: str,
        now: float,
        window_start: float,
        window_seconds: int,
        max_requests: int,
    ) -> Tuple[bool, int, int]:
        """In-memory sliding window check."""
        with self._lock:
            timestamps = self._buckets.get(key, [])
            # Remove old entries
            timestamps = [t for t in timestamps if t > window_start]
            self._buckets[key] = timestamps

            if len(timestamps) < max_requests:
                timestamps.append(now)
                self._buckets[key] = timestamps
                return True, max_requests - len(timestamps), 0

            # Calculate retry after
            oldest = timestamps[0] if timestamps else now
            retry_after = int((oldest + window_seconds) - now) + 1
            return False, 0, max(0, retry_after)


class RateLimiter:
    """Token bucket rate limiter with per-endpoint configuration and metrics."""

    def __init__(self, redis_client=None):
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()
        self._metrics: dict[str, int] = defaultdict(int)
        self._sliding_window = SlidingWindowCounter(redis_client)
        self._redis_client = redis_client
        self._endpoint_configs: dict[str, RateLimitConfig] = {
            "/api/v1/auth/login": RateLimitConfig(requests_per_minute=5, burst_size=2),
            "/api/v1/transactions": RateLimitConfig(requests_per_minute=100, burst_size=20),
            "/api/v1/fraud": RateLimitConfig(requests_per_minute=50, burst_size=10),
            "/api/v1/accounts": RateLimitConfig(requests_per_minute=30, burst_size=5),
            "/api/v1/monitoring": RateLimitConfig(requests_per_minute=60, burst_size=10),
            "/api/v1/llm": RateLimitConfig(requests_per_minute=100, burst_size=20, algorithm="sliding_window"),
            "/api/v1/llm/embedding": RateLimitConfig(requests_per_minute=200, burst_size=50, algorithm="sliding_window"),
            "/api/v1/llm/chat": RateLimitConfig(requests_per_minute=50, burst_size=10, algorithm="sliding_window"),
        }

    def _get_endpoint_config(self, path: str) -> RateLimitConfig:
        """Get rate limit config for an endpoint."""
        # Check for exact match first
        if path in self._endpoint_configs:
            return self._endpoint_configs[path]

        # Check for prefix match
        for endpoint_path, config in self._endpoint_configs.items():
            if path.startswith(endpoint_path):
                return config

        # Check for admin override
        if ADMIN_OVERRIDE_ENABLED and self._is_admin_path(path):
            return RateLimitConfig(requests_per_minute=1000, burst_size=200)

        # Default config
        return RateLimitConfig(requests_per_minute=60, burst_size=10)

    def _is_admin_path(self, path: str) -> bool:
        """Check if the path is an admin endpoint."""
        admin_paths = [
            "/api/v1/admin",
            "/api/v1/audit",
            "/api/v1/backup",
            "/api/v1/restore",
        ]
        return any(path.startswith(p) for p in admin_paths)

    def _is_whitelisted(self, key: str) -> bool:
        """Check if the key is whitelisted."""
        if not ADMIN_WHITELIST:
            return False
        return any(whitelisted in key for whitelisted in ADMIN_WHITELIST)

    def _is_blacklisted(self, key: str) -> bool:
        """Check if the key is blacklisted."""
        if not ADMIN_BLACKLIST:
            return False
        return any(blacklisted in key for blacklisted in ADMIN_BLACKLIST)

    def is_allowed(
        self,
        key: str,
        path: str,
        auth_type: str = "jwt",
    ) -> RateLimitResult:
        """Check if request is allowed using token bucket algorithm.

        Returns:
            RateLimitResult with allowed status and metadata
        """
        # Check blacklist first
        if self._is_blacklisted(key):
            logger.warning(f"Rate limit: Blacklisted key attempted: {key[:8]}...")
            return RateLimitResult(allowed=False, retry_after=86400, remaining=0, limit=0)

        # Check whitelist
        if self._is_whitelisted(key):
            return RateLimitResult(allowed=True, remaining=999999, limit=999999)

        config = self._get_endpoint_config(path)

        # Adjust limits based on auth type
        if auth_type == "api_key":
            requests_per_minute = API_KEY_RATE_LIMIT_PER_MINUTE
        else:
            requests_per_minute = JWT_RATE_LIMIT_PER_MINUTE

        # Use endpoint-specific limit if it's more restrictive
        requests_per_minute = min(requests_per_minute, config.requests_per_minute)

        # Determine algorithm
        algorithm = config.algorithm if hasattr(config, 'algorithm') else RATE_LIMIT_ALGORITHM

        # Use sliding window if configured
        if algorithm == "sliding_window":
            allowed, remaining, retry_after = self._sliding_window.is_allowed(
                key,
                SLIDING_WINDOW_SIZE_SECONDS,
                requests_per_minute,
            )
            self._metrics[f"rate_limit_{'allowed' if allowed else 'denied'}:{path}"] += 1
            return RateLimitResult(
                allowed=allowed,
                retry_after=retry_after if not allowed else None,
                remaining=remaining if allowed else 0,
                limit=requests_per_minute,
                algorithm="sliding_window",
            )

        # Token bucket algorithm
        capacity = config.burst_size
        refill_rate = requests_per_minute / 60.0  # tokens per second

        now = time.monotonic()

        with self._lock:
            bucket = self._buckets.get(key)

            if bucket is None:
                bucket = TokenBucket(
                    tokens=capacity,
                    last_refill=now,
                    capacity=capacity,
                    refill_rate=refill_rate,
                )
                self._buckets[key] = bucket

            # Refill tokens
            time_passed = now - bucket.last_refill
            bucket.tokens = min(
                bucket.capacity,
                bucket.tokens + time_passed * refill_rate,
            )
            bucket.last_refill = now

            # Check if request is allowed
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                self._metrics[f"rate_limit_allowed:{path}"] += 1
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket.tokens),
                    limit=requests_per_minute,
                    algorithm="token_bucket",
                )
            else:
                self._metrics[f"rate_limit_denied:{path}"] += 1
                # Calculate retry-after
                retry_after = int((1 - bucket.tokens) / refill_rate)
                return RateLimitResult(
                    allowed=False,
                    retry_after=retry_after,
                    remaining=0,
                    limit=requests_per_minute,
                    algorithm="token_bucket",
                )

    def get_metrics(self) -> dict[str, int]:
        """Get rate limiting metrics."""
        with self._lock:
            return dict(self._metrics)

    def reset_metrics(self) -> None:
        """Reset rate limiting metrics."""
        with self._lock:
            self._metrics.clear()

    def set_endpoint_config(self, path: str, config: RateLimitConfig) -> None:
        """Set rate limit config for a specific endpoint."""
        with self._lock:
            self._endpoint_configs[path] = config

    def add_to_whitelist(self, key: str) -> None:
        """Add a key to the whitelist."""
        global ADMIN_WHITELIST
        if key not in ADMIN_WHITELIST:
            ADMIN_WHITELIST.append(key)
            logger.info(f"Added {key} to rate limit whitelist")

    def remove_from_whitelist(self, key: str) -> None:
        """Remove a key from the whitelist."""
        global ADMIN_WHITELIST
        if key in ADMIN_WHITELIST:
            ADMIN_WHITELIST.remove(key)
            logger.info(f"Removed {key} from rate limit whitelist")

    def add_to_blacklist(self, key: str) -> None:
        """Add a key to the blacklist."""
        global ADMIN_BLACKLIST
        if key not in ADMIN_BLACKLIST:
            ADMIN_BLACKLIST.append(key)
            logger.info(f"Added {key} to rate limit blacklist")

    def remove_from_blacklist(self, key: str) -> None:
        """Remove a key from the blacklist."""
        global ADMIN_BLACKLIST
        if key in ADMIN_BLACKLIST:
            ADMIN_BLACKLIST.remove(key)
            logger.info(f"Removed {key} from rate limit blacklist")


# Global rate limiter instance (initialized without Redis by default)
rate_limiter = RateLimiter()


def init_rate_limiter(redis_client=None) -> None:
    """Initialize rate limiter with Redis client for distributed rate limiting."""
    global rate_limiter
    rate_limiter = RateLimiter(redis_client)
    logger.info("Rate limiter initialized with Redis support" if redis_client else "Rate limiter initialized without Redis")