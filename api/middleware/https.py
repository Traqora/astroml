"""HTTPS enforcement middleware for FastAPI.

Redirects HTTP requests to HTTPS and adds HSTS headers to enforce
secure connections in production environments.
"""
from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from starlette.types import ASGIApp


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Middleware to redirect HTTP requests to HTTPS.

    In production, this ensures all traffic uses HTTPS to prevent
    downgrade attacks. Can be disabled for development.

    Example:
        app.add_middleware(
            HTTPSRedirectMiddleware,
            enabled=True,
            allowed_hosts=["example.com", "api.example.com"]
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        enabled: bool = True,
        allowed_hosts: list[str] | None = None,
    ) -> None:
        """Initialize HTTPS redirect middleware.

        Args:
            app: ASGI application
            enabled: Whether to enable HTTPS redirects (False for development)
            allowed_hosts: List of allowed hostnames. If None, all hosts are allowed.
        """
        super().__init__(app)
        self._enabled = enabled
        self._allowed_hosts = set(allowed_hosts) if allowed_hosts else None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Redirect HTTP requests to HTTPS."""
        # Skip if HTTPS enforcement is disabled
        if not self._enabled:
            return await call_next(request)

        # Check if request is already HTTPS
        if request.url.scheme == "https":
            return await call_next(request)

        # Check if host is allowed (if whitelist is configured)
        if self._allowed_hosts:
            host = request.headers.get("host", "").split(":")[0]
            if host not in self._allowed_hosts:
                return await call_next(request)

        # Redirect to HTTPS
        https_url = str(request.url.replace(scheme="https"))
        return RedirectResponse(url=https_url, status_code=301)


class HSTSMiddleware(BaseHTTPMiddleware):
    """Middleware to add HTTP Strict Transport Security (HSTS) headers.

    HSTS tells browsers to always use HTTPS for future requests to this domain,
    preventing protocol downgrade attacks.

    Example:
        app.add_middleware(
            HSTSMiddleware,
            max_age=31536000,  # 1 year
            include_subdomains=True,
            preload=True
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        max_age: int = 31536000,
        include_subdomains: bool = True,
        preload: bool = False,
        enabled: bool = True,
    ) -> None:
        """Initialize HSTS middleware.

        Args:
            app: ASGI application
            max_age: Time in seconds that browsers should remember HSTS (default: 1 year)
            include_subdomains: Apply HSTS to all subdomains
            preload: Include in browser HSTS preload list
            enabled: Whether to enable HSTS (False for development)
        """
        super().__init__(app)
        self._max_age = max_age
        self._include_subdomains = include_subdomains
        self._preload = preload
        self._enabled = enabled

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add HSTS header to response."""
        response = await call_next(request)

        # Skip if HSTS is disabled
        if not self._enabled:
            return response

        # Only add HSTS header for HTTPS requests
        if request.url.scheme != "https":
            return response

        # Build HSTS header value
        hsts_parts = [f"max-age={self._max_age}"]
        if self._include_subdomains:
            hsts_parts.append("includeSubDomains")
        if self._preload:
            hsts_parts.append("preload")

        response.headers["Strict-Transport-Security"] = "; ".join(hsts_parts)
        return response
