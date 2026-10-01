"""Audit logging for critical pipeline operations.

Issue #757: records an immutable (who/what/when/result) audit trail for
pipeline-critical operations — model activations, configuration changes, and
rollbacks — aligned with the existing audit trail conventions in
``docs/AUDIT_TRAIL.md`` and the model-operation events already emitted by
:class:`astroml.governance.audit_logger.ModelAuditLogger`.

Design notes:
    - Reuses :class:`~astroml.governance.audit_logger.AuditStore` backends
      (in-memory for tests, append-only NDJSON files in production) so audit
      records stay tamper-resistant without new infrastructure.
    - Every record answers who/what/when/result: ``actor``, ``operation`` +
      ``target``/``details``, ``timestamp``, and ``outcome``.
    - Consecutive configuration-change records are hash-chained: each entry
      stores the hash of the previous entry, so silent edits or deletions of
      history are detectable (``verify_chain``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from astroml.governance.audit_logger import AuditStore, InMemoryAuditStore

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineAuditLogger",
    "PipelineAuditRecord",
    "PipelineOperation",
]


class PipelineOperation(str, Enum):
    """Critical pipeline operations covered by the audit trail (#757)."""

    MODEL_ACTIVATED = "model_activated"
    MODEL_DEACTIVATED = "model_deactivated"
    MODEL_ROLLED_BACK = "model_rolled_back"
    CONFIG_CHANGED = "config_changed"
    CONFIG_ROLLED_BACK = "config_rolled_back"
    FEATURE_CONFIG_CHANGED = "feature_config_changed"


@dataclass(frozen=True)
class PipelineAuditRecord:
    """An immutable who/what/when/result record for a pipeline operation."""

    actor: str
    operation: PipelineOperation
    target: str
    timestamp: datetime
    outcome: str
    details: dict[str, Any] = field(default_factory=dict)
    record_id: str = ""
    previous_hash: str = ""
    record_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "record_id": self.record_id,
            "actor": self.actor,
            "operation": self.operation.value,
            "target": self.target,
            "timestamp": self.timestamp.isoformat(),
            "outcome": self.outcome,
            "details": self.details,
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
        }

    def compute_hash(self) -> str:
        """Content hash over everything except ``record_hash`` itself."""
        payload = {
            "actor": self.actor,
            "operation": self.operation.value,
            "target": self.target,
            "timestamp": self.timestamp.isoformat(),
            "outcome": self.outcome,
            "details": self.details,
            "previous_hash": self.previous_hash,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


class PipelineAuditLogger:
    """Records and queries the pipeline-critical audit trail (#757).

    Args:
        store: Any :class:`~astroml.governance.audit_logger.AuditStore`
            backend. Defaults to an in-memory store (tests/dev); wire
            :class:`~astroml.governance.audit_logger.FileAuditStore` for
            append-only production durability.
        actor: Default actor applied when a call does not pass one.
    """

    def __init__(self, store: AuditStore | None = None, actor: str = "system") -> None:
        self.store = store if store is not None else InMemoryAuditStore()
        self.default_actor = actor
        self._previous_hash = ""
        self._sequence = 0

    def _record(
        self,
        operation: PipelineOperation,
        target: str,
        details: dict[str, Any] | None,
        actor: str | None,
        outcome: str,
    ) -> PipelineAuditRecord:
        """Build, hash-chain, and persist one audit record."""
        self._sequence += 1
        record = PipelineAuditRecord(
            actor=actor or self.default_actor,
            operation=operation,
            target=target,
            timestamp=datetime.now(timezone.utc),
            outcome=outcome,
            details=dict(details or {}),
            record_id=f"pa_{self._sequence:08d}",
            previous_hash=self._previous_hash,
        )
        object.__setattr__(record, "record_hash", record.compute_hash())
        self._previous_hash = record.record_hash
        self.store.write(_to_governance_event(record))
        logger.info(
            "audit: %s actor=%s target=%s outcome=%s",
            operation.value,
            record.actor,
            record.target,
            record.outcome,
        )
        return record

    # -- critical operations -------------------------------------------------

    def log_model_activation(
        self,
        model_id: str,
        version: str,
        actor: str | None = None,
        *,
        activated_by: str | None = None,
        details: dict[str, Any] | None = None,
        outcome: str = "success",
    ) -> PipelineAuditRecord:
        """Record a model activation (who activated which version, when)."""
        merged = {"version": version, **(details or {})}
        if activated_by:
            merged["activated_by"] = activated_by
        return self._record(
            PipelineOperation.MODEL_ACTIVATED,
            target=model_id,
            details=merged,
            actor=actor,
            outcome=outcome,
        )

    def log_config_change(
        self,
        config_path: str,
        changes: dict[str, Any],
        actor: str | None = None,
        *,
        outcome: str = "success",
    ) -> PipelineAuditRecord:
        """Record a configuration change (path, before/after per key)."""
        return self._record(
            PipelineOperation.CONFIG_CHANGED,
            target=config_path,
            details={"changes": changes},
            actor=actor,
            outcome=outcome,
        )

    def log_rollback(
        self,
        target: str,
        from_version: str,
        to_version: str,
        actor: str | None = None,
        *,
        reason: str = "",
        outcome: str = "success",
    ) -> PipelineAuditRecord:
        """Record a model or config rollback with its reason."""
        return self._record(
            PipelineOperation.MODEL_ROLLED_BACK,
            target=target,
            details={"from_version": from_version, "to_version": to_version, "reason": reason},
            actor=actor,
            outcome=outcome,
        )

    # -- querying ------------------------------------------------------------

    def query(
        self,
        operation: PipelineOperation | None = None,
        target: str | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[PipelineAuditRecord]:
        """Query recorded audit records, newest first."""
        from astroml.governance.audit_logger import AuditEventType

        event_type = None
        if operation is not None:
            event_type = AuditEventType.CONFIGURATION_CHANGED
        events = self.store.query(
            model_id=target,
            event_type=event_type,
            actor=actor,
            limit=limit * 4,
        )
        records: list[PipelineAuditRecord] = []
        for event in events:
            record = _from_governance_event(event)
            if record is not None:
                records.append(record)
        if operation is not None:
            records = [r for r in records if r.operation == operation]
        return records[:limit]

    def verify_chain(self, records: list[PipelineAuditRecord] | None = None) -> bool:
        """Verify the hash chain over the stored history.

        Returns:
            True when every record's hash matches its content and links to its
            predecessor. False means history was edited or reordered.
        """
        if records is None:
            records = self.query(limit=10_000)
        # Sort by record_id (the write sequence) — timestamps can tie at
        # microsecond resolution and would scramble chain order.
        ordered = sorted(records, key=lambda r: r.record_id)
        previous = ""
        for record in ordered:
            if record.previous_hash != previous:
                return False
            if record.record_hash != record.compute_hash():
                return False
            previous = record.record_hash
        return True


# ---------------------------------------------------------------------------
# Bridging to the governance AuditStore backend
# ---------------------------------------------------------------------------


def _to_governance_event(record: PipelineAuditRecord) -> Any:
    """Wrap a pipeline record in the governance ``AuditEvent`` shape."""
    from astroml.governance.audit_logger import AuditEvent, AuditEventType

    return AuditEvent(
        event_type=AuditEventType.CONFIGURATION_CHANGED,
        actor=record.actor,
        model_id=record.target,
        timestamp=record.timestamp,
        details={
            "operation": record.operation.value,
            "record_id": record.record_id,
            "previous_hash": record.previous_hash,
            "record_hash": record.record_hash,
            **record.details,
        },
        outcome=record.outcome,
    )


def _from_governance_event(event: Any) -> PipelineAuditRecord | None:
    """Rebuild a pipeline record from a stored governance event, if it is one."""
    details = getattr(event, "details", {}) or {}
    operation_value = details.get("operation")
    try:
        operation = PipelineOperation(operation_value)
    except ValueError:
        return None
    nested = {
        k: v
        for k, v in details.items()
        if k not in {"operation", "record_id", "previous_hash", "record_hash"}
    }
    return PipelineAuditRecord(
        actor=getattr(event, "actor", "system"),
        operation=operation,
        target=getattr(event, "model_id", ""),
        timestamp=getattr(event, "timestamp", datetime.now(timezone.utc)),
        outcome=str(getattr(event, "outcome", "success")),
        details=nested,
        record_id=str(details.get("record_id", "")),
        previous_hash=str(details.get("previous_hash", "")),
        record_hash=str(details.get("record_hash", "")),
    )
