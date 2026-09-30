"""HTTP auth and rate-limit middleware (issue #240, #331).

Enhanced with:
- Rate limit headers (X-RateLimit-*) (issue #299)
- Rate limit violation logging (issue #299)
- Whitelist/Blacklist support (issue #299)
- Brute-force protection with account lockout
- Token fingerprinting for theft detection
- JWT secret enforcement in production
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from api.auth.config import PUBLIC_PATHS, is_auth_enabled
from api.auth.dependencies import authenticate_token
from api.auth.hardening import (
    _brute_force_protector,
    compute_token_fingerprint,
    enforce_jwt_secret,
)
from api.auth.rate_limit import rate_limiter
from api.database import _sync_session_factory
from astroml.utils.logging import sanitize_log_value

logger = logging.getLogger(__name__)

# Enforce JWT secret on module load (production safety check).
# This fails fast at startup rather than at first request.
enforce_jwt_secret()


class AuthMiddleware(BaseHTTPMiddleware):
    """Require JWT/API-key auth on protected routes and enforce rate limits.

    Enhanced with brute-force protection: tracks failed auth attempts
    per user identity and temporarily locks out after threshold.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        if not is_auth_enabled() or path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Authentication required"})

        token = auth_header[7:]
        client_ip = request.client.host if request.client else "unknown"

        # Check brute-force lockout before attempting authentication
        lock_key = f"{client_ip}:{token[:16]}"
        is_locked, retry_after = _brute_force_protector.is_locked(lock_key)
        if is_locked:
            logger.warning(
                "Brute-force lockout active: ip=%s path=%s retry_after=%ds",
                sanitize_log_value(client_ip),
                sanitize_log_value(path),
                retry_after,
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many failed authentication attempts. Please try again later.",
                    "retry_after": retry_after,
                },
            )
            response.headers["Retry-After"] = str(retry_after)
            return response

        session = _sync_session_factory()()
        try:
            auth = authenticate_token(token, session)
        except Exception as e:
            _brute_force_protector.record_failure(lock_key)
            fingerprint = compute_token_fingerprint(token)
            logger.warning(
                "Authentication failed for %s (fp=%s): %s",
                sanitize_log_value(client_ip),
                fingerprint,
                sanitize_log_value(str(e)),
            )
            return JSONResponse(status_code=401, content={"detail": "Invalid or expired token"})
        finally:
            session.close()

        # Clear brute-force counter on success
        _brute_force_protector.record_success(lock_key)

        rate_key = f"{auth.auth_type}:{auth.subject}"
        rate_path = path

        # Check rate limit
        result = rate_limiter.is_allowed(rate_key, rate_path, auth.auth_type)

        # Log rate limit violations
        if not result.allowed:
            logger.warning(
                "Rate limit exceeded: %s | %s | %s | "
                "retry_after=%ss | limit=%s",
                sanitize_log_value(rate_key),
                sanitize_log_value(client_ip),
                sanitize_log_value(path),
                result.retry_after,
                result.limit,
            )

        # Build response with rate limit headers
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + (result.retry_after or 60))

        if not result.allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": result.retry_after,
                    "limit": result.limit,
                    "algorithm": result.algorithm,
                },
            )
            if result.retry_after is not None:
                response.headers["Retry-After"] = str(result.retry_after)

        request.state.auth = auth
        return response
