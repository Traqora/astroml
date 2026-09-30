"""Audit logging middleware for sensitive API operations (issue #332, #535)."""

from __future__ import annotations

import json
from typing import Callable

from fastapi import Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from api.audit import audit_logger
from api.database import get_async_session_factory

SENSITIVE_ACTIONS = {
    "POST": "create",
    "PUT": "update",
    "PATCH": "update",
    "DELETE": "delete",
}

SENSITIVE_PATHS = {
    "/api/v1/auth/login": "login",
    "/api/v1/auth/logout": "logout",
    "/api/v1/users": "user_management",
    "/api/v1/api-keys": "api_key_management",
}

# Fields to sanitize from request parameters (issue #535)
SENSITIVE_FIELDS = {
    "password",
    "token",
    "api_key",
    "secret",
    "credit_card",
    "ssn",
    "social_security",
    "auth",
    "authorization",
}


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log sensitive API operations to the audit log.

    Enhanced for issue #535 with:
    - Request parameter sanitization
    - Tamper-resistant logging
    - Comprehensive audit trail
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log sensitive operations."""
        path = request.url.path
        method = request.method

        # Determine if this is a sensitive operation
        action = SENSITIVE_ACTIONS.get(method)
        resource_type = None

        # Check for specific sensitive paths
        for sensitive_path, resource in SENSITIVE_PATHS.items():
            if path.startswith(sensitive_path):
                resource_type = resource
                if path == "/api/v1/auth/login":
                    action = "login"
                elif path == "/api/v1/auth/logout":
                    action = "logout"
                break

        # Extract resource type from path if not already set
        if resource_type is None and action:
            parts = path.strip("/").split("/")
            if len(parts) >= 2:
                resource_type = parts[2]  # e.g., /api/v1/accounts -> accounts

        # Only log sensitive operations
        if action and resource_type:
            try:
                session_factory = get_async_session_factory()
                async with session_factory() as session:
                    # Get user info from request state if available
                    user_id = None
                    username = None
                    auth_type = None
                    api_key_id = None
                    if hasattr(request.state, "auth"):
                        user_id = request.state.auth.user_id
                        username = request.state.auth.username
                        auth_type = request.state.auth.auth_type
                        api_key_id = getattr(request.state.auth, "api_key_id", None)

                    # Get resource ID from path if available
                    resource_id = None
                    parts = path.strip("/").split("/")
                    if len(parts) >= 4:
                        resource_id = parts[3]

                    # Capture request parameters (sanitized)
                    request_params = await self._capture_request_params(request)

                    # Process the request
                    response = await call_next(request)

                    # Log the event with enhanced fields (issue #535)
                    await audit_logger.log_event(
                        session=session,
                        action=action,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        user_id=user_id,
                        username=username,
                        auth_type=auth_type,
                        api_key_id=api_key_id,
                        ip_address=self._get_client_ip(request),
                        user_agent=request.headers.get("user-agent"),
                        request_path=path,
                        request_method=method,
                        status_code=response.status_code,
                        request_params=request_params,
                    )

                    return response
            except Exception:  # noqa: BLE001
                # Don't break the request if audit logging fails
                return await call_next(request)

        return await call_next(request)

    async def _capture_request_params(self, request: Request) -> dict:
        """Capture and sanitize request parameters (issue #535).

        Args:
            request: FastAPI request

        Returns:
            Sanitized request parameters
        """
        params = {}

        # Query parameters
        if request.query_params:
            params["query"] = self._sanitize_params(dict(request.query_params))

        # Try to get body parameters for POST/PUT/PATCH
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                if body:
                    # Try to parse as JSON
                    try:
                        body_dict = json.loads(body.decode())
                        params["body"] = self._sanitize_params(body_dict)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # If not JSON, just note the size
                        params["body_size"] = len(body)
            except Exception:
                pass

        return params

    def _sanitize_params(self, params: dict) -> dict:
        """Sanitize sensitive parameters (issue #535).

        Args:
            params: Raw parameters

        Returns:
            Sanitized parameters with sensitive values redacted
        """
        sanitized = {}
        for key, value in params.items():
            if any(sensitive in key.lower() for sensitive in SENSITIVE_FIELDS):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_params(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_params(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized

    def _get_client_ip(self, request: Request) -> str | None:
        """Extract client IP address from request."""
        # Check for forwarded headers (proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct connection
        if request.client:
            return request.client.host

        return None
