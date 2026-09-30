from astroml.utils.exceptions import AstroMLError
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from astroml.deployment.state_snapshot import StateSnapshotManager

logger = logging.getLogger(__name__)

RollbackStatus = Literal["requested", "approved", "executed", "cancelled", "failed"]
RollbackSeverity = Literal["low", "medium", "high", "critical"]


@dataclass
class RollbackTrigger:
    """Represents a rollback trigger event."""

    reason: str
    severity: RollbackSeverity
    target_version: str
    requested_by: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackRecord:
    """Tracks a requested or executed rollback."""

    rollback_id: str
    trigger: RollbackTrigger
    status: RollbackStatus
    snapshot_id: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    executed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RollbackManager:
    """Manages rollback requests, approval workflows, and recovery execution."""

    AUTO_APPROVE_SEVERITIES: list[RollbackSeverity] = ["critical"]
    VALID_SEVERITIES: list[RollbackSeverity] = ["low", "medium", "high", "critical"]

    def __init__(self, snapshot_manager: StateSnapshotManager | None = None) -> None:
        self._snapshot_manager = snapshot_manager
        self._records: dict[str, RollbackRecord] = {}

    def _generate_id(self) -> str:
        return uuid.uuid4().hex

    def trigger_rollback(
        self,
        reason: str,
        target_version: str,
        requested_by: str,
        severity: RollbackSeverity = "medium",
        snapshot_id: str | None = None,
        context: dict[str, Any] | None = None,
        auto_approve: bool = False,
    ) -> RollbackRecord:
        """Trigger a rollback request from monitored deployment telemetry."""
        if severity not in self.VALID_SEVERITIES:
            raise ValueError(f"Invalid rollback severity: {severity}")

        trigger = RollbackTrigger(
            reason=reason,
            severity=severity,
            target_version=target_version,
            requested_by=requested_by,
            context=context or {},
        )
        record = RollbackRecord(
            rollback_id=self._generate_id(),
            trigger=trigger,
            status=(
                "approved"
                if auto_approve or severity in self.AUTO_APPROVE_SEVERITIES
                else "requested"
            ),
            snapshot_id=snapshot_id,
        )
        self._records[record.rollback_id] = record

        logger.info(
            "Rollback triggered: id=%s target_version=%s severity=%s auto_approve=%s",
            record.rollback_id,
            target_version,
            severity,
            record.status == "approved",
        )
        return record

    def request_approval(self, rollback_id: str, approver: str) -> RollbackRecord:
        """Mark a rollback request as ready for approval."""
        record = self._get_record(rollback_id)
        if record.status != "requested":
            raise ValueError("Rollback request is not in a pending state")

        record.metadata["approval_requested_by"] = approver
        record.metadata["approval_requested_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Rollback approval requested: id=%s approver=%s", rollback_id, approver)
        return record

    def approve_rollback(self, rollback_id: str, approver: str) -> RollbackRecord:
        """Approve a rollback request before execution."""
        record = self._get_record(rollback_id)
        if record.status != "requested" and record.status != "failed":
            raise ValueError("Rollback request is not pending approval")

        record.status = "approved"
        record.approved_by = approver
        record.approved_at = datetime.now(timezone.utc)
        logger.info("Rollback approved: id=%s approver=%s", rollback_id, approver)
        return record

    def execute_rollback(
        self,
        rollback_id: str,
        apply_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> RollbackRecord:
        """Execute an approved rollback and optionally restore a state snapshot."""
        record = self._get_record(rollback_id)
        if record.status != "approved":
            raise ValueError("Rollback request must be approved before execution")

        try:
            if record.snapshot_id:
                if self._snapshot_manager is None:
                    raise RuntimeError("Snapshot manager is required to restore state snapshots")
                restored_state = self._snapshot_manager.restore_snapshot(
                    record.snapshot_id,
                    apply_fn=apply_fn,
                    restored_by=record.trigger.requested_by,
                )
                record.metadata["restored_state"] = restored_state

            record.status = "executed"
            record.executed_at = datetime.now(timezone.utc)
            logger.info(
                "Rollback executed: id=%s target_version=%s",
                rollback_id,
                record.trigger.target_version,
            )
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.executed_at = datetime.now(timezone.utc)
            logger.exception("Rollback execution failed: id=%s error=%s", rollback_id, exc)
        return record

    def get_rollback(self, rollback_id: str) -> RollbackRecord:
        return self._get_record(rollback_id)

    def list_rollbacks(self) -> list[RollbackRecord]:
        return list(self._records.values())

    def get_dashboard(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for record in self._records.values():
            statuses[record.status] = statuses.get(record.status, 0) + 1

        latest = max(
            self._records.values(), key=lambda record: record.trigger.created_at, default=None
        )
        dashboard = {
            "total_rollbacks": len(self._records),
            "rollbacks_by_status": statuses,
            "latest_rollback": (
                {
                    "rollback_id": latest.rollback_id,
                    "target_version": latest.trigger.target_version,
                    "status": latest.status,
                }
                if latest
                else {}
            ),
        }
        return dashboard

    def evaluate_health_metrics(
        self,
        failure_rate: float,
        latency_ms: float,
        error_budget: float = 0.1,
        latency_threshold_ms: float = 2500.0,
        **context: Any,
    ) -> RollbackRecord | None:
        """Evaluate deployment metrics to trigger an automated rollback."""
        if failure_rate >= error_budget:
            return self.trigger_rollback(
                reason=f"Failure rate exceeded threshold: {failure_rate:.2%}",
                target_version="unknown",
                requested_by="system",
                severity="critical",
                snapshot_id=None,
                context={"failure_rate": failure_rate, "latency_ms": latency_ms, **context},
                auto_approve=True,
            )

        if latency_ms >= latency_threshold_ms:
            return self.trigger_rollback(
                reason=f"Latency exceeded threshold: {latency_ms:.2f}ms",
                target_version="unknown",
                requested_by="system",
                severity="high",
                snapshot_id=None,
                context={"failure_rate": failure_rate, "latency_ms": latency_ms, **context},
                auto_approve=False,
            )

        return None

    def _get_record(self, rollback_id: str) -> RollbackRecord:
        record = self._records.get(rollback_id)
        if record is None:
            raise ValueError(f"Rollback record '{rollback_id}' not found")
        return record
