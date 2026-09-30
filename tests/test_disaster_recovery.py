"""Tests for disaster recovery planning and execution."""

import pytest

from astroml.deployment.disaster_recovery import DisasterRecoveryManager
from astroml.deployment.recovery_procedures import RecoveryProcedure, RecoveryStep
from astroml.deployment.state_snapshot import StateSnapshotManager


def test_register_and_plan_recovery() -> None:
    snapshot_manager = StateSnapshotManager()
    snapshot = snapshot_manager.create_snapshot(
        name="dr-snap",
        environment="prod",
        model_version="1.1.0",
        state={"status": "ok"},
    )

    manager = DisasterRecoveryManager(snapshot_manager)
    procedure = RecoveryProcedure(
        name="restore-procedure",
        description="Restore state",
        steps=[RecoveryStep(name="step-1", description="noop", action=lambda: True)],
    )
    manager.register_procedure(procedure)

    plan = manager.plan_recovery("incident-1", "restore-procedure", snapshot.snapshot_id)
    assert plan.incident_id == "incident-1"
    assert plan.snapshot_id == snapshot.snapshot_id
    assert plan.status == "planned"


def test_execute_recovery_plan_success() -> None:
    snapshot_manager = StateSnapshotManager()
    snapshot = snapshot_manager.create_snapshot(
        name="dr-snap-2",
        environment="prod",
        model_version="1.1.1",
        state={"feature_flag": True},
    )

    manager = DisasterRecoveryManager(snapshot_manager)
    procedure = RecoveryProcedure(
        name="restore-procedure-2",
        description="Restore state",
        steps=[RecoveryStep(name="step-1", description="noop", action=lambda: True)],
    )
    manager.register_procedure(procedure)
    plan = manager.plan_recovery("incident-2", "restore-procedure-2", snapshot.snapshot_id)

    executed = manager.execute_recovery(
        plan.plan_id, apply_fn=lambda state: state["feature_flag"] is True
    )
    assert executed.status == "completed"
    assert executed.result["snapshot_restored"] is True
    assert executed.result["procedure"]["success"] is True


def test_execute_recovery_plan_fails_when_snapshot_missing() -> None:
    snapshot_manager = StateSnapshotManager()
    manager = DisasterRecoveryManager(snapshot_manager)
    procedure = RecoveryProcedure(
        name="restore-procedure-3",
        description="Restore state",
        steps=[RecoveryStep(name="step-1", description="noop", action=lambda: True)],
    )
    manager.register_procedure(procedure)

    with pytest.raises(ValueError, match="Snapshot 'nonexistent' not found"):
        manager.plan_recovery("incident-3", "restore-procedure-3", "nonexistent")


def test_test_recovery_scenario_validator() -> None:
    snapshot_manager = StateSnapshotManager()
    snapshot = snapshot_manager.create_snapshot(
        name="dr-snap-4",
        environment="prod",
        model_version="1.1.2",
        state={"enabled": True},
    )

    manager = DisasterRecoveryManager(snapshot_manager)
    procedure = RecoveryProcedure(
        name="restore-procedure-4",
        description="Restore state",
        steps=[RecoveryStep(name="step-1", description="noop", action=lambda: True)],
    )
    manager.register_procedure(procedure)
    plan = manager.plan_recovery("incident-4", "restore-procedure-4", snapshot.snapshot_id)

    result = manager.test_recovery_scenario(
        plan.plan_id, validator=lambda s: s.environment == "prod"
    )
    assert result["validator_passed"] is True
    assert result["snapshot_available"] is True


def test_get_recovery_summary_counts() -> None:
    snapshot_manager = StateSnapshotManager()
    snapshot = snapshot_manager.create_snapshot(
        name="dr-snap-5",
        environment="prod",
        model_version="1.1.3",
        state={"enabled": True},
    )

    manager = DisasterRecoveryManager(snapshot_manager)
    procedure = RecoveryProcedure(
        name="restore-procedure-5",
        description="Restore state",
        steps=[RecoveryStep(name="step-1", description="noop", action=lambda: True)],
    )
    manager.register_procedure(procedure)
    manager.plan_recovery("incident-5", "restore-procedure-5", snapshot.snapshot_id)

    summary = manager.get_recovery_summary()
    assert summary["total_plans"] == 1
    assert summary["registered_procedures"] == ["restore-procedure-5"]
