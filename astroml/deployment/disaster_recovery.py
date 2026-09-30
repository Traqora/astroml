from astroml.utils.exceptions import AstroMLError
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from astroml.deployment.recovery_procedures import RecoveryProcedure
from astroml.deployment.state_snapshot import Snapshot, StateSnapshotManager

logger = logging.getLogger(__name__)

RecoveryStatus = Literal["planned", "executing", "completed", "failed"]


@dataclass
class RecoveryPlan:
    """A plan that binds a snapshot to a recovery procedure."""

    plan_id: str
    incident_id: str
    snapshot_id: str
    procedure_name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "planned"
    executed_at: datetime | None = None
    result: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None


class DisasterRecoveryManager:
    """Manages disaster recovery procedures and recovery testing."""

    def __init__(self, snapshot_manager: StateSnapshotManager) -> None:
        self._snapshot_manager = snapshot_manager
        self._procedures: dict[str, RecoveryProcedure] = {}
        self._plans: dict[str, RecoveryPlan] = {}

    def register_procedure(self, procedure: RecoveryProcedure) -> None:
        if procedure.name in self._procedures:
            raise ValueError(f"Procedure '{procedure.name}' already registered")
        self._procedures[procedure.name] = procedure
        logger.info("Registered recovery procedure: %s", procedure.name)

    def plan_recovery(
        self, incident_id: str, procedure_name: str, snapshot_id: str
    ) -> RecoveryPlan:
        if procedure_name not in self._procedures:
            raise ValueError(f"Recovery procedure '{procedure_name}' not found")
        self._snapshot_manager.get_snapshot(snapshot_id)

        plan = RecoveryPlan(
            plan_id=uuid.uuid4().hex,
            incident_id=incident_id,
            snapshot_id=snapshot_id,
            procedure_name=procedure_name,
        )
        self._plans[plan.plan_id] = plan
        return plan

    def execute_recovery(
        self,
        plan_id: str,
        apply_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> RecoveryPlan:
        plan = self._plans.get(plan_id)
        if plan is None:
            raise ValueError(f"Recovery plan '{plan_id}' not found")
        procedure = self._procedures.get(plan.procedure_name)
        if procedure is None:
            raise ValueError(f"Recovery procedure '{plan.procedure_name}' not found")

        plan.status = "executing"
        try:
            snapshot = self._snapshot_manager.restore_snapshot(
                plan.snapshot_id,
                apply_fn=apply_fn,
                restored_by="disaster-recovery",
            )
            plan.result["snapshot_restored"] = True
            plan.result["snapshot_state"] = snapshot
            plan.result["procedure"] = procedure.execute()
            plan.status = "completed" if procedure.success else "failed"
            plan.executed_at = datetime.now(timezone.utc)
            if procedure.errors:
                plan.last_error = "; ".join(procedure.errors)
        except Exception as exc:
            plan.status = "failed"
            plan.executed_at = datetime.now(timezone.utc)
            plan.last_error = str(exc)
            plan.result["snapshot_restored"] = False
            logger.exception("Recovery execution failed for plan %s", plan_id)

        return plan

    def test_recovery_scenario(
        self,
        plan_id: str,
        validator: Callable[[Snapshot], bool],
    ) -> dict[str, Any]:
        plan = self._plans.get(plan_id)
        if plan is None:
            raise ValueError(f"Recovery plan '{plan_id}' not found")

        snapshot = self._snapshot_manager.get_snapshot(plan.snapshot_id)
        success = validator(snapshot)
        return {
            "plan_id": plan.plan_id,
            "incident_id": plan.incident_id,
            "snapshot_id": plan.snapshot_id,
            "validator_passed": success,
            "snapshot_available": snapshot is not None,
            "snapshot_id": snapshot.snapshot_id,
        }

    def get_recovery_summary(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for plan in self._plans.values():
            statuses[plan.status] = statuses.get(plan.status, 0) + 1
        return {
            "total_plans": len(self._plans),
            "plans_by_status": statuses,
            "registered_procedures": list(self._procedures.keys()),
        }
