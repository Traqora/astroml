"""Input validation middleware for API requests (issue #333)."""
from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from api.validation import (
    InputValidator,
    ValidationError,
    XSSPreventionMiddleware,
)


class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate and sanitize incoming requests."""

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

        # Validate query parameters
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

        # Sanitize response to prevent XSS
        if response.headers.get("content-type", "").startswith("application/json"):
            response = await self._sanitize_json_response(response)

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

    async def _sanitize_json_response(self, response: Response) -> Response:
        """Sanitize JSON response to prevent XSS."""
        # Note: This is a simplified implementation
        # In production, you'd want to parse the JSON, sanitize, and re-serialize
        # For now, we'll just add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response
