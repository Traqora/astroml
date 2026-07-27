"""Audit logging middleware for sensitive API operations (issue #332)."""
from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

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


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log sensitive API operations to the audit log."""

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
                    if hasattr(request.state, "auth"):
                        user_id = request.state.auth.user_id
                        username = request.state.auth.username
                        auth_type = request.state.auth.auth_type

                    # Get resource ID from path if available
                    resource_id = None
                    parts = path.strip("/").split("/")
                    if len(parts) >= 4:
                        resource_id = parts[3]

                    # Process the request
                    response = await call_next(request)

                    # Log the event
                    await audit_logger.log_event(
                        session=session,
                        action=action,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        user_id=user_id,
                        username=username,
                        auth_type=auth_type,
                        ip_address=self._get_client_ip(request),
                        user_agent=request.headers.get("user-agent"),
                        request_path=path,
                        request_method=method,
                        status_code=response.status_code,
                    )

                    return response
            except Exception:  # noqa: BLE001
                # Don't break the request if audit logging fails
                return await call_next(request)

        return await call_next(request)

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
