"""Frontend error logging endpoint — issue #292.

Provides:
  POST /api/v1/errors/report — receive browser error reports from ErrorBoundary

The endpoint is intentionally unauthenticated so it can be reached even when
a user's session has expired (which is often the condition that triggered the
error in the first place).  Rate-limiting and input size caps guard against
abuse.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from astroml.utils.logging import sanitize_log_value

logger = logging.getLogger("astroml.frontend_errors")

router = APIRouter(prefix="/api/v1/errors", tags=["errors"])

# Maximum characters we accept for free-text fields to prevent log flooding
_MAX_MSG_LEN = 2000
_MAX_STACK_LEN = 8000


# ── Schema ────────────────────────────────────────────────────────────────────

class FrontendErrorReport(BaseModel):
    """Payload sent by the browser ErrorBoundary / errorReporting module."""

    message: str = Field(..., max_length=_MAX_MSG_LEN)
    stack: Optional[str] = Field(None, max_length=_MAX_STACK_LEN)
    boundary: Optional[str] = Field(None, max_length=200)
    component_stack: Optional[str] = Field(None, max_length=_MAX_STACK_LEN)
    extra: Optional[Dict[str, Any]] = None
    user_agent: Optional[str] = Field(None, max_length=500)
    url: Optional[str] = Field(None, max_length=2000)
    timestamp: Optional[str] = Field(None, max_length=50)

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/report",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Receive a frontend error report",
    description=(
        "Called automatically by the browser ErrorBoundary when an unhandled "
        "React error is caught.  Always returns 204 so the client never retries "
        "on failure."
    ),
)
async def report_error(report: FrontendErrorReport, request: Request) -> JSONResponse:
    # Attach the client IP for triage (best-effort — may be a proxy)
    client_ip = request.client.host if request.client else "unknown"

    logger.error(
        "Frontend error reported",
        extra={
            "frontend_error": {
                "message": sanitize_log_value(report.message),
                "boundary": sanitize_log_value(report.boundary),
                "url": sanitize_log_value(report.url),
                "user_agent": sanitize_log_value(report.user_agent),
                "timestamp": report.timestamp or datetime.now(timezone.utc).isoformat(),
                "client_ip": sanitize_log_value(client_ip),
                "stack_preview": sanitize_log_value((report.stack or "")[:500] or ""),
                "extra": report.extra,
            }
        },
    )

    # Return 204 — no body, no retry signal
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
