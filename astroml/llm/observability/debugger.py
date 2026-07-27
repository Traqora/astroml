"""LLM Debugging utilities — reconstruct full request lifecycle.

Resolves #456: Debug view that assembles all spans, logs, and decisions
for a given trace ID to reconstruct the complete request flow.
"""
from __future__ import annotations

import json
import pprint
from typing import Any

from astroml.llm.observability.tracer import LLMTracer
from astroml.llm.observability.audit import LLMAuditLog


class LLMDebugger:
    """Debug utility for reconstructing LLM request lifecycles.

    Aggregates trace spans, audit entries, and safety decisions into a
    human-readable debug report for troubleshooting production issues.

    Example::

        debugger = LLMDebugger(tracer=tracer, audit=audit)
        report = debugger.debug_trace("trace-id-abc123")
        print(debugger.format_report(report))
    """

    def __init__(
        self,
        tracer: LLMTracer | None = None,
        audit: LLMAuditLog | None = None,
    ) -> None:
        self._tracer = tracer or LLMTracer()
        self._audit = audit or LLMAuditLog()

    def debug_trace(self, trace_id: str) -> dict[str, Any]:
        """Reconstruct the full lifecycle for *trace_id*.

        Returns a structured debug report containing all spans,
        audit entries, and aggregate metrics for the trace.
        """
        spans = self._tracer.reconstruct_lifecycle(trace_id)
        # Map audit entries that match this trace_id via metadata
        audit_entries = [
            e for e in self._audit.search(limit=1000)
            if e.metadata.get("trace_id") == trace_id
        ]

        total_latency = sum(s.get("latency_ms", 0) for s in spans)
        total_tokens = sum(s.get("total_tokens", 0) for s in spans)
        total_cost = sum(e.cost_usd for e in audit_entries)
        errors = [s for s in spans if s.get("error")]

        return {
            "trace_id": trace_id,
            "spans": spans,
            "audit_entries": [
                {
                    "audit_id": e.audit_id,
                    "operation": e.operation,
                    "provider": e.provider,
                    "model": e.model,
                    "latency_ms": e.latency_ms,
                    "tokens": e.prompt_tokens + e.completion_tokens,
                    "cost_usd": e.cost_usd,
                    "error": e.error,
                }
                for e in audit_entries
            ],
            "summary": {
                "total_spans": len(spans),
                "total_latency_ms": round(total_latency, 2),
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 6),
                "error_count": len(errors),
                "errors": [{"operation": e.get("operation"), "error": e.get("error")} for e in errors],
            },
        }

    def format_report(self, report: dict[str, Any]) -> str:
        """Return a human-readable formatted debug report."""
        lines = [
            f"=== LLM Debug Report: trace_id={report['trace_id']} ===",
            "",
            "--- Summary ---",
        ]
        summary = report.get("summary", {})
        for k, v in summary.items():
            lines.append(f"  {k}: {v}")

        lines.append("")
        lines.append("--- Spans ---")
        for i, span in enumerate(report.get("spans", []), 1):
            lines.append(
                f"  [{i}] {span.get('operation', '?')} "
                f"| {span.get('provider', '?')}/{span.get('model', '?')} "
                f"| {span.get('latency_ms', 0):.1f}ms "
                f"| {span.get('total_tokens', 0)} tok"
                + (f" | ERROR: {span.get('error')}" if span.get("error") else "")
            )

        return "\n".join(lines)

    def inspect_prompt(self, prompt: str, max_length: int = 500) -> dict[str, Any]:
        """Return an inspection dict for a prompt — useful in test debugging."""
        words = prompt.split()
        return {
            "char_count": len(prompt),
            "word_count": len(words),
            "estimated_tokens": len(prompt) // 4,  # ~4 chars/token heuristic
            "preview": prompt[:max_length] + ("…" if len(prompt) > max_length else ""),
            "has_system_override": any(
                kw in prompt.lower()
                for kw in ["ignore previous", "disregard", "you are now", "DAN"]
            ),
        }
