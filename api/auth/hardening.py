"""Public API security hardening (issue #331, #240).

Provides:
- Per-IP rate limiting for unauthenticated/public endpoints
- Brute-force protection with account lockout after failed attempts
- JWT secret key enforcement in production
- Token fingerprinting to detect token theft
- Security headers middleware
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


@dataclass
class LockoutEntry:
    """Tracks failed authentication attempts for brute-force protection."""

    failed_count: int = 0
    locked_until: Optional[float] = None
    first_attempt: float = field(default_factory=time.monotonic)
    last_attempt: float = field(default_factory=time.monotonic)


class BruteForceProtector:
    """Protects against brute-force attacks by locking out after N failures.

    Tracks failed attempts per identifier (username, IP, or combination).
    After ``max_failed_attempts`` failures within ``window_seconds``, the
    identifier is locked out for ``lockout_seconds``.
    """

    def __init__(
        self,
        max_failed_attempts: int = 5,
        window_seconds: int = 300,
        lockout_seconds: int = 900,
    ) -> None:
        self._max_failed_attempts = max_failed_attempts
        self._window_seconds = window_seconds
        self._lockout_seconds = lockout_seconds
        self._entries: dict[str, LockoutEntry] = {}
        self._lock = Lock()

    def is_locked(self, identifier: str) -> tuple[bool, Optional[int]]:
        """Check if an identifier is currently locked out.

        Returns:
            Tuple of (is_locked, retry_after_seconds).
        """
        with self._lock:
            entry = self._entries.get(identifier)
            if entry is None or entry.locked_until is None:
                return False, None

            now = time.monotonic()
            if now >= entry.locked_until:
                del self._entries[identifier]
                return False, None

            retry_after = int(entry.locked_until - now) + 1
            return True, retry_after

    def record_failure(self, identifier: str) -> None:
        """Record a failed authentication attempt."""
        with self._lock:
            now = time.monotonic()
            entry = self._entries.get(identifier)

            if entry is None or (now - entry.first_attempt) > self._window_seconds:
                self._entries[identifier] = LockoutEntry(
                    failed_count=1,
                    first_attempt=now,
                    last_attempt=now,
                )
                return

            entry.failed_count += 1
            entry.last_attempt = now

            if entry.failed_count >= self._max_failed_attempts:
                entry.locked_until = now + self._lockout_seconds
                logger.warning(
                    "Brute-force lockout: %s failed %d times, locked for %ds",
                    identifier[:8],
                    entry.failed_count,
                    self._lockout_seconds,
                )

    def record_success(self, identifier: str) -> None:
        """Clear failure tracking after successful authentication."""
        with self._lock:
            self._entries.pop(identifier, None)

    def cleanup(self) -> int:
        """Remove expired entries. Returns number of entries removed."""
        with self._lock:
            now = time.monotonic()
            expired = [
                k
                for k, v in self._entries.items()
                if v.locked_until is not None and now >= v.locked_until
            ]
            for k in expired:
                del self._entries[k]
            return len(expired)


class IPRateLimiter:
    """Per-IP rate limiter for unauthenticated/public endpoints.

    Uses a sliding window algorithm to enforce a maximum number of
    requests per IP address within a time window.
    """

    def __init__(
        self,
        requests_per_minute: int = 30,
        burst_size: int = 10,
    ) -> None:
        self._requests_per_minute = requests_per_minute
        self._window_seconds = 60
        self._max_requests = requests_per_minute
        self._lock = Lock()
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, client_ip: str) -> tuple[bool, int, int]:
        """Check if a request from the given IP is allowed.

        Returns:
            Tuple of (allowed, remaining, retry_after_seconds).
        """
        now = time.monotonic()
        window_start = now - self._window_seconds

        with self._lock:
            timestamps = self._requests.get(client_ip, [])
            timestamps = [t for t in timestamps if t > window_start]
            self._requests[client_ip] = timestamps

            if len(timestamps) < self._max_requests:
                timestamps.append(now)
                remaining = self._max_requests - len(timestamps)
                return True, remaining, 0

            oldest = timestamps[0] if timestamps else now
            retry_after = int((oldest + self._window_seconds) - now) + 1
            return False, 0, max(0, retry_after)

    def cleanup(self) -> int:
        """Remove expired entries. Returns number of entries removed."""
        now = time.monotonic()
        window_start = now - self._window_seconds
        removed = 0
        with self._lock:
            for ip in list(self._requests.keys()):
                timestamps = [t for t in self._requests[ip] if t > window_start]
                if timestamps:
                    self._requests[ip] = timestamps
                else:
                    del self._requests[ip]
                    removed += 1
        return removed


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()"
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"

        return response


class PublicRateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit unauthenticated/public endpoints per IP address.

    Authenticated endpoints use the per-user rate limiter in AuthMiddleware.
    This middleware protects public endpoints (login, health, docs) from
    abuse without requiring authentication.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 30,
        burst_size: int = 10,
    ) -> None:
        # Starlette constructs middleware as `cls(app, **options)`, so the base
        # has to receive the downstream app.
        super().__init__(app)
        self._limiter = IPRateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        if path in ("/health", "/healthz", "/ready"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining, retry_after = self._limiter.is_allowed(client_ip)

        if not allowed:
            logger.warning(
                "IP rate limit exceeded: %s on %s (retry_after=%ds)",
                client_ip,
                path,
                retry_after,
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests from this IP. Please try again later.",
                    "retry_after": retry_after,
                },
            )
            response.headers["Retry-After"] = str(retry_after)
            response.headers["X-RateLimit-Limit"] = str(self._limiter._max_requests)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + retry_after)
            return response

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limiter._max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


def enforce_jwt_secret() -> str:
    """Enforce a strong JWT secret key in production.

    In production (when ENV=production), raises RuntimeError if
    JWT_SECRET_KEY is not set or is still the default value.

    Returns the secret key to use.
    """
    env = os.environ.get("ENV", "development")
    secret = os.environ.get("JWT_SECRET_KEY") or os.environ.get("SECRET_KEY")

    if env == "production":
        if not secret:
            raise RuntimeError(
                "JWT_SECRET_KEY must be set in production. "
                'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
            )
        weak_secrets = {"change-me-in-production", "secret", "admin", "password", "jwt-secret"}
        if secret.lower() in weak_secrets:
            raise RuntimeError(
                "JWT_SECRET_KEY is set to a weak/default value. "
                'Generate a strong secret with: python -c "import secrets; print(secrets.token_hex(32))"'
            )
        if len(secret) < 32:
            raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters in production.")

    return secret or "dev-only-secret-change-in-production"


def compute_token_fingerprint(token: str) -> str:
    """Compute a short fingerprint of a token for logging and comparison.

    Uses HMAC-SHA256 with a random per-process key to prevent
    fingerprint reversal.
    """
    key = _get_fingerprint_key()
    return hmac.new(key, token.encode(), hashlib.sha256).hexdigest()[:16]


def verify_token_fingerprint(token: str, expected_fingerprint: str) -> bool:
    """Verify that a token matches an expected fingerprint."""
    actual = compute_token_fingerprint(token)
    return hmac.compare_digest(actual, expected_fingerprint)


def _get_fingerprint_key() -> bytes:
    """Get or generate the per-process fingerprint key."""
    if not hasattr(_get_fingerprint_key, "_key"):
        _get_fingerprint_key._key = os.urandom(32)
    return _get_fingerprint_key._key


_brute_force_protector = BruteForceProtector()


def get_brute_force_protector() -> BruteForceProtector:
    """Get the global brute-force protector instance."""
    return _brute_force_protector
