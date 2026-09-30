"""Tests for state snapshot recovery management."""

from datetime import datetime, timedelta, timezone

import pytest

from astroml.deployment.state_snapshot import Snapshot, StateSnapshotManager


def test_create_and_get_snapshot() -> None:
    manager = StateSnapshotManager()
    snapshot = manager.create_snapshot(
        name="model-state",
        environment="production",
        model_version="1.0.0",
        state={"route": "v1"},
        recovery_objective_seconds=60,
    )

    assert snapshot.name == "model-state"
    assert snapshot.environment == "production"
    assert snapshot.model_version == "1.0.0"
    assert snapshot.within_objective is False

    loaded = manager.get_snapshot(snapshot.snapshot_id)
    assert loaded.snapshot_id == snapshot.snapshot_id


def test_restore_snapshot_applies_state() -> None:
    manager = StateSnapshotManager()
    snapshot = manager.create_snapshot(
        name="snapshot-restore",
        environment="staging",
        model_version="2.0.0",
        state={"config": "ok"},
        recovery_objective_seconds=300,
    )

    def apply_fn(state: dict[str, object]) -> bool:
        assert state == {"config": "ok"}
        return True

    restored = manager.restore_snapshot(
        snapshot.snapshot_id, apply_fn=apply_fn, restored_by="tester"
    )
    assert restored == {"config": "ok"}
    assert snapshot.restored_by == "tester"
    assert snapshot.recovery_metrics["applied_successfully"] is True
    assert snapshot.within_objective is True


def test_restore_snapshot_fails_when_already_restored() -> None:
    manager = StateSnapshotManager()
    snapshot = manager.create_snapshot(
        name="snapshot-single",
        environment="test",
        model_version="0.0.1",
        state={"x": 1},
        recovery_objective_seconds=1,
    )

    manager.restore_snapshot(snapshot.snapshot_id)
    with pytest.raises(ValueError, match="already been restored"):
        manager.restore_snapshot(snapshot.snapshot_id)


def test_cleanup_snapshot_removes_snapshot() -> None:
    manager = StateSnapshotManager()
    snapshot = manager.create_snapshot(
        name="snapshot-cleanup",
        environment="dev",
        model_version="0.9.0",
        state={"flag": True},
    )

    manager.cleanup_snapshot(snapshot.snapshot_id)
    with pytest.raises(ValueError, match="Snapshot 'snapshot-cleanup"):
        manager.get_snapshot(snapshot.snapshot_id)


def test_create_snapshot_invalid_rto() -> None:
    manager = StateSnapshotManager()
    with pytest.raises(ValueError, match="recovery_objective_seconds must be positive"):
        manager.create_snapshot(
            name="invalid",
            environment="dev",
            model_version="1.0.0",
            state={"x": 1},
            recovery_objective_seconds=0,
        )
