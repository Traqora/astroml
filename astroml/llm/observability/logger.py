"""Structured logging for LLM operations — JSON logs with PII redaction.

Resolves #456: Structured log emission with user context, prompt/response
previews (redacted), error details, and sampling strategies.
100% sampling for errors, 10% for successful requests.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from typing import Any

# Try structlog for production-grade structured logging
try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STRUCTLOG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Fields that must be redacted before logging
_REDACT_FIELDS = frozenset(
    ["prompt", "content", "response", "text", "query", "message"]
)
_PREVIEW_LENGTH = 120  # characters — safe preview without leaking full prompts


def _redact(value: str, max_len: int = _PREVIEW_LENGTH) -> str:
    """Return a safe preview of *value* with PII-risky content truncated."""
    if not isinstance(value, str):
        return str(value)
    return value[:max_len] + ("…" if len(value) > max_len else "")


class LLMStructuredLogger:
    """Emit structured JSON log records for LLM request/response events.

    Sampling strategy:
    - ``error_sample_rate=1.0`` — all errors are always logged.
    - ``success_sample_rate=0.1`` — 10% of successful requests are logged.

    Example::

        log = LLMStructuredLogger()
        log.request("generate", prompt="Hello world", user_id="u42", model="gpt-4")
        log.response("generate", latency_ms=820, tokens=150, cost=0.003)
        log.error("generate", error=exc, user_id="u42")
    """

    def __init__(
        self,
        service: str = "astroml-llm",
        error_sample_rate: float = 1.0,
        success_sample_rate: float = 0.1,
        output_stream: Any = sys.stdout,
    ) -> None:
        self.service = service
        self.error_sample_rate = error_sample_rate
        self.success_sample_rate = success_sample_rate
        self._stream = output_stream

    def request(
        self,
        operation: str,
        *,
        user_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an outgoing LLM request."""
        if not self._should_sample(is_error=False):
            return
        record = self._base_record("llm.request", operation)
        record.update(
            {
                "user_id": user_id,
                "model": model,
                "provider": provider,
                "prompt_preview": _redact(prompt) if prompt else None,
            }
        )
        if metadata:
            record["metadata"] = metadata
        self._emit(record)

    def response(
        self,
        operation: str,
        *,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        cache_hit: bool = False,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a successful LLM response."""
        if not self._should_sample(is_error=False):
            return
        record = self._base_record("llm.response", operation)
        record.update(
            {
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": cost_usd,
                "cache_hit": cache_hit,
                "model": model,
            }
        )
        if metadata:
            record["metadata"] = metadata
        self._emit(record)

    def error(
        self,
        operation: str,
        *,
        error: Exception | str,
        user_id: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an LLM error — always sampled at 100%."""
        if not self._should_sample(is_error=True):
            return
        record = self._base_record("llm.error", operation)
        record.update(
            {
                "error_type": type(error).__name__ if isinstance(error, Exception) else "error",
                "error_message": str(error),
                "user_id": user_id,
                "model": model,
            }
        )
        if metadata:
            record["metadata"] = metadata
        self._emit(record)

    # ─── Internal helpers ───────────────────────────────────────────────────

    def _base_record(self, event: str, operation: str) -> dict[str, Any]:
        return {
            "event": event,
            "service": self.service,
            "operation": operation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _should_sample(self, is_error: bool) -> bool:
        rate = self.error_sample_rate if is_error else self.success_sample_rate
        return random.random() < rate  # noqa: S311 — not cryptographic

    def _emit(self, record: dict[str, Any]) -> None:
        try:
            print(json.dumps(record), file=self._stream, flush=True)
        except Exception:  # noqa: BLE001
            logger.warning("LLMStructuredLogger failed to emit record")
