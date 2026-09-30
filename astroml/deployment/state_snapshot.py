from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Snapshot:
    """Represents a saved state snapshot for disaster recovery."""

    snapshot_id: str
    name: str
    environment: str
    model_version: str
    state: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recovery_objective_seconds: int = 300
    restored_at: datetime | None = None
    restored_by: str | None = None
    recovery_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def within_objective(self) -> bool:
        """Return True if the snapshot was restored within the RTO objective."""
        if self.restored_at is None:
            return False
        elapsed = (self.restored_at - self.created_at).total_seconds()
        return elapsed <= self.recovery_objective_seconds


class StateSnapshotManager:
    """Manages state snapshots and recovery operations."""

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self._snapshots: dict[str, Snapshot] = {}
        self._storage_path = Path(storage_path) if storage_path else None
        if self._storage_path is not None:
            self._storage_path.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        name: str,
        environment: str,
        model_version: str,
        state: dict[str, Any],
        recovery_objective_seconds: int = 300,
    ) -> Snapshot:
        """Create and register a new state snapshot."""
        if recovery_objective_seconds <= 0:
            raise ValueError("recovery_objective_seconds must be positive")

        snapshot_id = Path(name).stem + "-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            name=name,
            environment=environment,
            model_version=model_version,
            state=state.copy(),
            recovery_objective_seconds=recovery_objective_seconds,
        )
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> Snapshot:
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")
        return snapshot

    def list_snapshots(self) -> list[Snapshot]:
        return list(self._snapshots.values())

    def restore_snapshot(
        self,
        snapshot_id: str,
        apply_fn: Callable[[dict[str, Any]], bool] | None = None,
        restored_by: str = "system",
    ) -> dict[str, Any]:
        """Restore a snapshot and record recovery metrics."""
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot.restored_at is not None:
            raise ValueError(f"Snapshot '{snapshot_id}' has already been restored")

        snapshot.restored_at = datetime.now(timezone.utc)
        snapshot.restored_by = restored_by
        snapshot.recovery_metrics["rto_seconds"] = (
            snapshot.restored_at - snapshot.created_at
        ).total_seconds()

        if apply_fn is not None:
            success = apply_fn(snapshot.state)
            snapshot.recovery_metrics["applied_successfully"] = bool(success)
            if not success:
                raise RuntimeError(f"Snapshot '{snapshot_id}' failed to apply")

        return snapshot.state.copy()

    def cleanup_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot after recovery or archival."""
        if snapshot_id not in self._snapshots:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")
        del self._snapshots[snapshot_id]
