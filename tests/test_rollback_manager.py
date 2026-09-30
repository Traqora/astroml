"""Tests for rollback management and triggers."""

from astroml.deployment.rollback_manager import RollbackManager
from astroml.deployment.state_snapshot import StateSnapshotManager


def test_trigger_rollback_request() -> None:
    manager = RollbackManager()
    record = manager.trigger_rollback(
        reason="Degraded performance",
        target_version="1.0.0",
        requested_by="monitoring",
        severity="high",
        snapshot_id="snap-1",
    )

    assert record.status == "requested"
    assert record.trigger.severity == "high"
    assert record.snapshot_id == "snap-1"


def test_trigger_rollback_auto_approve_for_critical() -> None:
    manager = RollbackManager()
    record = manager.trigger_rollback(
        reason="Critical failure",
        target_version="1.0.0",
        requested_by="system",
        severity="critical",
    )

    assert record.status == "approved"
    assert record.trigger.requested_by == "system"


def test_request_and_approve_rollback() -> None:
    manager = RollbackManager()
    record = manager.trigger_rollback(
        reason="High latency",
        target_version="1.0.0",
        requested_by="monitor",
        severity="medium",
    )

    manager.request_approval(record.rollback_id, approver="lead")
    approved = manager.approve_rollback(record.rollback_id, approver="lead")

    assert approved.status == "approved"
    assert approved.approved_by == "lead"


def test_execute_rollback_restores_snapshot() -> None:
    snapshot_manager = StateSnapshotManager()
    snapshot = snapshot_manager.create_snapshot(
        name="rollback-snap",
        environment="prod",
        model_version="1.0.1",
        state={"route": "v2"},
    )
    manager = RollbackManager(snapshot_manager=snapshot_manager)
    record = manager.trigger_rollback(
        reason="Service regression",
        target_version="1.0.0",
        requested_by="ops",
        severity="critical",
        snapshot_id=snapshot.snapshot_id,
        auto_approve=True,
    )

    result = manager.execute_rollback(
        record.rollback_id, apply_fn=lambda state: state["route"] == "v2"
    )
    assert result.status == "executed"
    assert result.metadata["restored_state"] == {"route": "v2"}


def test_execute_rollback_fails_without_approval() -> None:
    manager = RollbackManager()
    record = manager.trigger_rollback(
        reason="Latency alert",
        target_version="1.0.2",
        requested_by="monitor",
        severity="high",
        auto_approve=False,
    )

    try:
        manager.execute_rollback(record.rollback_id)
    except ValueError as exc:
        assert "must be approved" in str(exc)
    else:
        assert False, "execute_rollback should raise ValueError for unapproved request"


def test_evaluate_health_metrics_triggers_critical_auto_approval() -> None:
    manager = RollbackManager()
    record = manager.evaluate_health_metrics(failure_rate=0.2, latency_ms=2000.0)

    assert record is not None
    assert record.status == "approved"
    assert record.trigger.severity == "critical"


def test_get_dashboard_counts() -> None:
    manager = RollbackManager()
    manager.trigger_rollback(
        reason="Minor issue",
        target_version="1.0.0",
        requested_by="sys",
        severity="low",
    )
    manager.trigger_rollback(
        reason="Critical outage",
        target_version="1.0.0",
        requested_by="sys",
        severity="critical",
    )

    dashboard = manager.get_dashboard()
    assert dashboard["total_rollbacks"] == 2
    assert dashboard["rollbacks_by_status"]["requested"] == 1
    assert dashboard["rollbacks_by_status"]["approved"] == 1
