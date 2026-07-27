"""Safety incident audit log — persistent record of guardrail decisions.

Resolves #455: Structured logging of all safety incidents with user attribution,
category breakdown, and compliance reporting support.
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
class SafetyIncident:
    """A single safety incident record."""

    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    user_id: str | None = None
    is_output: bool = False
    decision: str = "block"
    category: str | None = None
    reason: str = ""
    confidence: float = 0.0
    text_preview: str = ""  # first 200 chars only — no full prompt storage
    metadata: dict[str, Any] = field(default_factory=dict)


class SafetyAuditLog:
    """Thread-safe safety incident audit log.

    Incidents are written to an in-memory buffer *and* optionally persisted
    to a JSONL file for compliance reporting and dashboard queries.

    Example::

        audit = SafetyAuditLog(log_path="/var/log/astroml/safety.jsonl")
        audit.log_incident(text, result, user_id="user_42")
        incidents = audit.recent_incidents(limit=50)
    """

    def __init__(self, log_path: str | Path | None = None, max_memory: int = 10_000) -> None:
        self._lock = threading.RLock()
        self._incidents: list[SafetyIncident] = []
        self._max_memory = max_memory
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_incident(
        self,
        text: str,
        result: Any,  # GuardrailResult — avoid circular import
        user_id: str | None = None,
        is_output: bool = False,
    ) -> str:
        """Record a safety incident and return the incident ID."""
        incident = SafetyIncident(
            user_id=user_id,
            is_output=is_output,
            decision=result.decision.value if hasattr(result.decision, "value") else str(result.decision),
            category=result.category.value if result.category and hasattr(result.category, "value") else str(result.category) if result.category else None,
            reason=result.reason,
            confidence=result.confidence,
            text_preview=text[:200],  # store only preview for privacy
        )

        with self._lock:
            self._incidents.append(incident)
            if len(self._incidents) > self._max_memory:
                self._incidents.pop(0)

        self._write_to_file(incident)
        logger.warning(
            "Safety incident logged",
            extra={
                "incident_id": incident.incident_id,
                "category": incident.category,
                "decision": incident.decision,
                "user_id": user_id,
            },
        )
        return incident.incident_id

    def recent_incidents(
        self,
        limit: int = 100,
        user_id: str | None = None,
        category: str | None = None,
    ) -> list[SafetyIncident]:
        """Return the most recent *limit* incidents, optionally filtered."""
        with self._lock:
            incidents = list(reversed(self._incidents))
        if user_id:
            incidents = [i for i in incidents if i.user_id == user_id]
        if category:
            incidents = [i for i in incidents if i.category == category]
        return incidents[:limit]

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics for monitoring dashboards."""
        with self._lock:
            total = len(self._incidents)
            by_category: dict[str, int] = {}
            by_decision: dict[str, int] = {}
            for inc in self._incidents:
                by_category[inc.category or "unknown"] = (
                    by_category.get(inc.category or "unknown", 0) + 1
                )
                by_decision[inc.decision] = by_decision.get(inc.decision, 0) + 1
        return {
            "total_incidents": total,
            "by_category": by_category,
            "by_decision": by_decision,
        }

    def _write_to_file(self, incident: SafetyIncident) -> None:
        if not self._log_path:
            return
        try:
            with self._log_path.open("a") as f:
                f.write(json.dumps(asdict(incident)) + "\n")
        except OSError:
            logger.error("Failed to write safety incident to %s", self._log_path)
