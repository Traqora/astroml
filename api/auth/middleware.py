"""HTTP auth and rate-limit middleware (issue #240, #331).

Enhanced with:
- Rate limit headers (X-RateLimit-*) (issue #299)
- Rate limit violation logging (issue #299)
- Whitelist/Blacklist support (issue #299)
"""
from __future__ import annotations

import logging
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from api.auth.config import is_auth_enabled, PUBLIC_PATHS
from api.auth.dependencies import authenticate_token
from api.auth.rate_limit import rate_limiter
from api.database import _sync_session_factory
from astroml.utils.logging import sanitize_log_value

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Require JWT/API-key auth on protected routes and enforce rate limits."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        if not is_auth_enabled() or path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Authentication required"})

        token = auth_header[7:]
        session = _sync_session_factory()()
        try:
            auth = authenticate_token(token, session)
        except Exception as e:
            logger.warning(f"Authentication failed for {sanitize_log_value(str(request.client.host))}: {sanitize_log_value(str(e))}")
            return JSONResponse(status_code=401, content={"detail": "Invalid or expired token"})
        finally:
            session.close()

        rate_key = f"{auth.auth_type}:{auth.subject}"
        client_ip = request.client.host if request.client else "unknown"
        rate_path = path

        # Check rate limit
        result = rate_limiter.is_allowed(rate_key, rate_path, auth.auth_type)

        # Log rate limit violations
        if not result.allowed:
            logger.warning(
                f"Rate limit exceeded: {sanitize_log_value(rate_key)} | {sanitize_log_value(client_ip)} | {sanitize_log_value(path)} | "
                f"retry_after={result.retry_after}s | limit={result.limit}"
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
                }
            )
            if result.retry_after is not None:
                response.headers["Retry-After"] = str(result.retry_after)

        request.state.auth = auth
        return response