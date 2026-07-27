"""Input validation middleware for API requests (issue #333, #533)."""
from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from api.validation import (
    InputValidator,
    ValidationError,
)

# Maximum request body size: 1 MB.  Requests exceeding this limit are rejected
# with 413 before the body is read, preventing memory exhaustion from oversized
# payloads (issue #533).
MAX_REQUEST_SIZE_BYTES: int = 1 * 1024 * 1024  # 1 MB


class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate and sanitize incoming requests.

    Responsibilities (issue #533):
    - Enforce a maximum request body size (default 1 MB).
    - Reject query parameters containing SQL injection or XSS patterns.
    - Add defensive response headers (X-Content-Type-Options, etc.).
    """

    # Paths that skip validation
    SKIP_VALIDATION_PATHS = {
        "/health",
        "/api/v1",
        "/docs",
        "/openapi.json",
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and validate input."""
        path = request.url.path

        # Skip validation for certain paths
        if any(path.startswith(skip_path) for skip_path in self.SKIP_VALIDATION_PATHS):
            return await call_next(request)

        # ── Request size limit (issue #533) ───────────────────────────────────
        # Reject oversized payloads early using Content-Length when present.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_REQUEST_SIZE_BYTES:
                    return Response(
                        content='{"detail": "Request body too large. Maximum allowed size is 1 MB."}',
                        status_code=413,
                        media_type="application/json",
                    )
            except ValueError:
                # Malformed Content-Length header — let the app handle it.
                pass

        # ── Query parameter validation (issue #533) ───────────────────────────
        if request.query_params:
            try:
                self._validate_query_params(request)
            except ValidationError as e:
                return Response(
                    content=f'{{"detail": "{e.message}", "field": "{e.field}"}}',
                    status_code=400,
                    media_type="application/json",
                )

        # Process the request
        response = await call_next(request)

        # ── Defensive response headers (issue #533) ───────────────────────────
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        return response

    def _validate_query_params(self, request: Request) -> None:
        """Validate query parameters for injection attacks."""
        for key, value in request.query_params.items():
            if isinstance(value, str):
                # Check for SQL injection
                if InputValidator.check_sql_injection(value):
                    raise ValidationError(
                        f"Invalid query parameter '{key}': potential SQL injection",
                        field=key,
                    )

                # Check for XSS
                if InputValidator.check_xss(value):
                    raise ValidationError(
                        f"Invalid query parameter '{key}': potential XSS",
                        field=key,
                    )
