"""Content Security Policy (CSP) middleware for FastAPI.

Adds CSP headers to prevent XSS attacks by controlling which resources
can be loaded by the browser. Implements a restrictive policy with
nonce-based script execution support.
"""
from __future__ import annotations

import base64
import hashlib
import os
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class CSPMiddleware(BaseHTTPMiddleware):
    """Add Content Security Policy headers to all responses.

    Implements a restrictive CSP policy to prevent XSS attacks:
    - default-src 'self': Only allow resources from same origin
    - script-src 'self' 'nonce-{nonce}': Only allow scripts from same origin
      or with valid nonce
    - style-src 'self' 'unsafe-inline': Allow inline styles for development
    - img-src 'self' data: https:: Allow images from same origin, data URIs,
      and HTTPS sources
    - connect-src 'self': Only allow fetch/XHR to same origin
    - font-src 'self': Only allow fonts from same origin
    - object-src 'none': Block plugins (Flash, etc.)
    - base-uri 'self': Restrict base tag
    - form-action 'self': Restrict form submissions
    - frame-ancestors 'none': Prevent clickjacking
    """

    def __init__(
        self,
        app: ASGIApp,
        report_only: bool = False,
        report_uri: str | None = None,
        enable_nonce: bool = True,
    ) -> None:
        """Initialize CSP middleware.

        Args:
            app: ASGI application
            report_only: If True, use CSP-Report-Only header (doesn't block)
            report_uri: URI to send CSP violation reports to
            enable_nonce: If True, generate nonce for script-src
        """
        super().__init__(app)
        self._report_only = report_only
        self._report_uri = report_uri
        self._enable_nonce = enable_nonce

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add CSP headers to response."""
        response = await call_next(request)

        # Generate nonce for script-src if enabled
        nonce = self._generate_nonce() if self._enable_nonce else None

        # Build CSP policy
        policy_parts = [
            "default-src 'self'",
            f"script-src 'self' 'nonce-{nonce}'" if nonce else "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "connect-src 'self'",
            "font-src 'self'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ]

        # Add report-uri if specified
        if self._report_uri:
            policy_parts.append(f"report-uri {self._report_uri}")

        policy = "; ".join(policy_parts)

        # Set CSP header
        header_name = "Content-Security-Policy-Report-Only" if self._report_only else "Content-Security-Policy"
        response.headers[header_name] = policy

        # Add nonce to response state for use in templates
        if nonce:
            response.headers["X-CSP-Nonce"] = nonce

        # Add other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response

    @staticmethod
    def _generate_nonce() -> str:
        """Generate a cryptographically secure nonce for CSP.

        Returns:
            Base64-encoded nonce string
        """
        # Generate 16 random bytes
        random_bytes = os.urandom(16)
        # Encode as base64
        return base64.b64encode(random_bytes).decode("utf-8")

    @staticmethod
    def hash_script(script_content: str) -> str:
        """Generate SHA-256 hash for script content.

        Can be used with script-src 'sha256-{hash}' instead of nonces.

        Args:
            script_content: JavaScript content to hash

        Returns:
            Base64-encoded SHA-256 hash
        """
        sha256_hash = hashlib.sha256(script_content.encode("utf-8")).digest()
        return base64.b64encode(sha256_hash).decode("utf-8")
