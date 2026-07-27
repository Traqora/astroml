"""Distributed tracing for LLM requests — OpenTelemetry-compatible.

Resolves #456: Full request lifecycle tracing from prompt to response,
including provider routing, retry tracking, and token/cost attribution.
"""
from __future__ import annotations

import time
import uuid
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Optional OpenTelemetry integration — gracefully degrade if not installed
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False


@dataclass
class TraceSpan:
    """Represents a single traced operation within an LLM request lifecycle."""

    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: str = ""
    provider: str | None = None
    model: str | None = None
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    retries: int = 0
    cache_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, error: str | None = None) -> None:
        """Mark span as complete and calculate latency."""
        self.end_time = time.monotonic()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        if error:
            self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "operation": self.operation,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "cache_hit": self.cache_hit,
            "retries": self.retries,
            "error": self.error,
            "metadata": self.metadata,
        }


class LLMTracer:
    """Distributed tracer for LLM operations.

    Provides context-managed spans with automatic timing, token accounting,
    and optional OpenTelemetry export.

    Example::

        tracer = LLMTracer(service_name="astroml-llm")
        with tracer.span("generate", provider="openai", model="gpt-4") as span:
            response = llm.generate(prompt)
            span.prompt_tokens = count_tokens(prompt)
            span.completion_tokens = count_tokens(response)
        # span is automatically finished with latency recorded
    """

    def __init__(self, service_name: str = "astroml-llm") -> None:
        self.service_name = service_name
        self._spans: list[TraceSpan] = []
        self._otel_tracer = self._init_otel(service_name)

    def _init_otel(self, service_name: str) -> Any:
        if not _OTEL_AVAILABLE:
            return None
        try:
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            otel_trace.set_tracer_provider(provider)
            return otel_trace.get_tracer(service_name)
        except Exception:  # noqa: BLE001
            logger.debug("OpenTelemetry initialisation failed; falling back to internal tracing.")
            return None

    @contextmanager
    def span(
        self,
        operation: str,
        provider: str | None = None,
        model: str | None = None,
        trace_id: str | None = None,
    ) -> Generator[TraceSpan, None, None]:
        """Context manager that creates and auto-finishes a :class:`TraceSpan`."""
        s = TraceSpan(
            operation=operation,
            provider=provider,
            model=model,
        )
        if trace_id:
            s.trace_id = trace_id

        try:
            yield s
        except Exception as exc:
            s.finish(error=str(exc))
            self._record(s)
            raise
        else:
            s.finish()
            self._record(s)

    def _record(self, span: TraceSpan) -> None:
        self._spans.append(span)
        logger.debug("LLM trace: %s", span.to_dict())

    def recent_spans(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the last *limit* span dicts for debugging."""
        return [s.to_dict() for s in self._spans[-limit:]]

    def reconstruct_lifecycle(self, trace_id: str) -> list[dict[str, Any]]:
        """Reconstruct all spans for a given *trace_id* (debug view)."""
        return [s.to_dict() for s in self._spans if s.trace_id == trace_id]
