"""API versioning middleware and utilities (issue #572)."""
from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

CURRENT_API_VERSION = "v1"
LATEST_API_VERSION = "1.0.0"
SUPPORTED_VERSIONS = ["v1"]
DEFAULT_VERSION = "v1"
DEPRECATED_VERSIONS: list[str] = []


class VersionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-API-Version"] = CURRENT_API_VERSION
        response.headers["X-API-Latest-Version"] = LATEST_API_VERSION
        for deprecated_version in DEPRECATED_VERSIONS:
            if request.url.path.startswith(f"/api/{deprecated_version}"):
                response.headers["X-API-Deprecated"] = deprecated_version
                response.headers["X-API-Sunset"] = "2026-12-31"
                response.headers["X-API-Deprecation-Info"] = (
                    f"Version {deprecated_version} is deprecated. "
                    f"Please migrate to {CURRENT_API_VERSION}."
                )
                break
        return response


def get_api_prefix(version: str = DEFAULT_VERSION) -> str:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported API version '{version}'. "
            f"Supported versions: {SUPPORTED_VERSIONS}"
        )
    return f"/api/{version}"
