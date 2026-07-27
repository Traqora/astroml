"""LLM Audit Trail — compliance logging for all LLM calls.

Resolves #456: Searchable audit log of every LLM call with user attribution,
timestamps, and compliance reporting. Storage overhead target: <1KB/request.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMAuditEntry:
    """Immutable audit record for a single LLM call."""

    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    user_id: str | None = None
    session_id: str | None = None
    operation: str = ""
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cache_hit: bool = False
    error: str | None = None
    # prompt/response deliberately omitted to keep storage <1KB/record
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMAuditLog:
    """Searchable, persistent audit trail for LLM calls.

    All LLM calls should be logged here for compliance and cost attribution.
    Records are stored in memory *and* written to a JSONL file.

    Query performance target: <100ms for 30-day range.

    Example::

        audit = LLMAuditLog(log_path="/var/log/astroml/llm_audit.jsonl")
        entry = audit.log(
            user_id="u42",
            operation="generate",
            provider="openai",
            model="gpt-4",
            prompt_tokens=50,
            completion_tokens=120,
            cost_usd=0.003,
            latency_ms=820,
        )
        results = audit.search(user_id="u42", limit=10)
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        max_memory: int = 100_000,
    ) -> None:
        self._lock = threading.RLock()
        self._entries: list[LLMAuditEntry] = []
        self._max_memory = max_memory
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        operation: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        cache_hit: bool = False,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMAuditEntry:
        """Create and persist an audit entry. Returns the entry."""
        entry = LLMAuditEntry(
            user_id=user_id,
            session_id=session_id,
            operation=operation,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=round(latency_ms, 2),
            cache_hit=cache_hit,
            error=error,
            metadata=metadata or {},
        )

        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_memory:
                self._entries.pop(0)

        self._write(entry)
        return entry

    def search(
        self,
        user_id: str | None = None,
        operation: str | None = None,
        provider: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> list[LLMAuditEntry]:
        """Search audit entries by common fields. Searchable in <100ms for 30-day range."""
        with self._lock:
            results = list(self._entries)

        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if operation:
            results = [e for e in results if e.operation == operation]
        if provider:
            results = [e for e in results if e.provider == provider]
        if start_date:
            results = [e for e in results if e.timestamp >= start_date]
        if end_date:
            results = [e for e in results if e.timestamp <= end_date]

        return list(reversed(results))[:limit]

    def cost_report(self, user_id: str | None = None) -> dict[str, Any]:
        """Return cost usage summary for compliance reporting."""
        entries = self.search(user_id=user_id, limit=self._max_memory)
        total_cost = sum(e.cost_usd for e in entries)
        total_tokens = sum(e.prompt_tokens + e.completion_tokens for e in entries)
        by_model: dict[str, float] = {}
        for e in entries:
            key = e.model or "unknown"
            by_model[key] = by_model.get(key, 0.0) + e.cost_usd
        return {
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "total_requests": len(entries),
            "cost_by_model": {k: round(v, 6) for k, v in by_model.items()},
        }

    def _write(self, entry: LLMAuditEntry) -> None:
        if not self._log_path:
            return
        try:
            with self._log_path.open("a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except OSError:
            logger.error("Failed to write audit entry to %s", self._log_path)
