"""LLM Observability — distributed tracing, metrics, logging, and audit.

Resolves #456: Full observability stack for LLM services including
request tracing, performance metrics, structured logging, and audit trails.
"""
from astroml.llm.observability.tracer import LLMTracer, TraceSpan
from astroml.llm.observability.logger import LLMStructuredLogger
from astroml.llm.observability.metrics import LLMMetrics
from astroml.llm.observability.profiler import LLMProfiler, ProfileResult
from astroml.llm.observability.audit import LLMAuditLog, LLMAuditEntry
from astroml.llm.observability.debugger import LLMDebugger

__all__ = [
    "LLMTracer",
    "TraceSpan",
    "LLMStructuredLogger",
    "LLMMetrics",
    "LLMProfiler",
    "ProfileResult",
    "LLMAuditLog",
    "LLMAuditEntry",
    "LLMDebugger",
]
