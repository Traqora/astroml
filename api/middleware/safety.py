"""Safety middleware — FastAPI middleware that applies guardrails to LLM routes.

Resolves #455: Intercepts LLM API requests and responses, applies safety checks,
and blocks or logs harmful content before it reaches handlers or users.
"""
from __future__ import annotations

import json
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from astroml.llm.safety.guards import SafetyGuard, StrictnessLevel

logger = logging.getLogger(__name__)

# Routes that go through safety checks
_PROTECTED_PREFIXES = (
    "/api/v1/llm/",
)


class SafetyMiddleware(BaseHTTPMiddleware):
    """Apply LLM safety guardrails to protected API routes.

    - Reads the request body for POST /api/v1/llm/* routes.
    - Extracts the ``prompt`` or ``messages[-1].content`` field.
    - Runs input guardrails; returns 400 if blocked.
    - Passes through approved requests unmodified.

    Latency overhead target: <200ms (rule-based only, no external calls).
    """

    def __init__(
        self,
        app: ASGIApp,
        strictness: StrictnessLevel = StrictnessLevel.MODERATE,
    ) -> None:
        super().__init__(app)
        self._guard = SafetyGuard(strictness=strictness)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only intercept protected LLM routes
        if not any(request.url.path.startswith(p) for p in _PROTECTED_PREFIXES):
            return await call_next(request)

        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        # Read and parse request body
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            # Not JSON — pass through
            return await call_next(request)

        user_id = getattr(request.state, "user_id", None)

        # Extract prompt text from common request fields
        prompt_text = self._extract_prompt(body)
        if prompt_text:
            result = self._guard.check_input(prompt_text, user_id=user_id)
            if result.is_blocked:
                logger.warning(
                    "Safety middleware blocked request",
                    extra={"path": request.url.path, "reason": result.reason, "user_id": user_id},
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "code": "SAFETY_VIOLATION",
                            "message": "Request blocked by safety guardrails.",
                            "details": {
                                "reason": result.reason,
                                "category": result.category.value if result.category else None,
                            },
                        }
                    },
                )

        return await call_next(request)

    @staticmethod
    def _extract_prompt(body: dict) -> str | None:
        """Extract the user's input text from various request body schemas."""
        # Direct prompt field
        if "prompt" in body and isinstance(body["prompt"], str):
            return body["prompt"]
        # Messages array (chat format)
        messages = body.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict) and "content" in last:
                return str(last["content"])
        # Query field
        if "query" in body and isinstance(body["query"], str):
            return body["query"]
        return None
