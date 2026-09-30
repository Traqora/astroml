"""FastAPI exception handlers emitting RFC 7807 problem+json responses."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas.errors import PROBLEM_CONTENT_TYPE, problem_detail


def _http_error_code(status_code: int) -> str:
    return f"HTTP_{status_code}"


def _problem_response(status_code: int, **kwargs: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=problem_detail(status=status_code, **kwargs),
        media_type=PROBLEM_CONTENT_TYPE,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Normalize explicit HTTP failures raised by routers or dependencies."""
    detail: Any = exc.detail
    details: Any | None = None
    code = _http_error_code(exc.status_code)

    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("error") or exc.status_code)
        details = detail.get("details")
        if isinstance(detail.get("code"), str):
            code = detail["code"]
    else:
        message = str(detail or exc.status_code)

    response = _problem_response(
        exc.status_code,
        code=code,
        title=message,
        detail=message,
        instance=str(request.url.path),
        details=details,
    )
    for key, value in (getattr(exc, "headers", None) or {}).items():
        response.headers[key] = value
    return response


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return validation errors with stable top-level keys."""
    details = [
        {
            "loc": list(error.get("loc", [])),
            "message": error.get("msg", "Invalid request"),
            "type": error.get("type", "validation_error"),
        }
        for error in exc.errors()
    ]
    return _problem_response(
        422,
        code="VALIDATION_ERROR",
        title="Request validation failed",
        detail="Request validation failed",
        instance=str(request.url.path),
        details=details,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Avoid leaking internal exception details to API clients."""
    return _problem_response(
        500,
        code="INTERNAL_SERVER_ERROR",
        title="Internal server error",
        instance=str(request.url.path),
    )
