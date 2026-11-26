"""FastAPI exception handlers emitting RFC 7807 problem+json responses.

Validation error response shape (v1)
-------------------------------------
When a ``RequestValidationError`` is raised (HTTP 422), the ``details``
array contains one item per field error, each with this shape::

    {
        "field":   "address.zip",   # dot-separated field path string
        "message": "...",           # human-readable description
        "code":    "missing"        # Pydantic error type string
    }

``field`` construction rules:

- Pydantic provides a ``loc`` tuple of mixed ``str`` / ``int`` segments,
  e.g. ``("body", "address", "zip")`` or ``("body", "items", 0)``.
- Leading source-sentinel tokens (``"body"``, ``"query"``, ``"path"``,
  ``"header"``) are stripped; they describe the input source, not the
  field path.  ``("body", "email")`` → ``"email"``.
- If the ``loc`` is empty after stripping, ``"__root__"`` is used.
- Integer segments (list indices) are stringified: ``("body", "items", 0)``
  → ``"items.0"``.

``code`` values come directly from Pydantic's error ``type`` strings
(e.g. ``"string_too_short"``, ``"missing"``, ``"int_parsing"``).
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas.errors import PROBLEM_CONTENT_TYPE, problem_detail


_LOCATION_SENTINELS = {"body", "query", "path", "header"}


def _normalize_loc(loc: tuple | list) -> str:
    """Convert a Pydantic ``loc`` tuple into a dot-separated field path.

    The leading source-sentinel token (``"body"``, ``"query"``, ``"path"``,
    or ``"header"``), if present, is stripped.  Integer segments (list
    indices) are stringified.  Returns ``"__root__"`` when no segments
    remain after stripping.
    """
    segments = list(loc)
    if segments and segments[0] in _LOCATION_SENTINELS:
        segments = segments[1:]
    filtered = [str(s) for s in segments]
    return ".".join(filtered) if filtered else "__root__"


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
            "field": _normalize_loc(error.get("loc", ())),
            "message": error.get("msg", "Invalid request"),
            "code": error.get("type", "validation_error"),
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
