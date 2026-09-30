# Disaster Recovery and Rollback

## Overview

AstroML now includes automated rollback and disaster recovery management for deployed models. The new deployment package supports:

- Automated rollback triggers based on health, latency, or custom conditions.
- State snapshot creation and restoration for recovery scenarios.
- Structured recovery procedures with step validation.
- Recovery plan execution and automated testing.
- Rollback dashboards and recovery time objective (RTO) tracking.

## Key components

### `astroml.deployment.rollback_manager.RollbackManager`

Use this manager to:

- trigger rollback requests automatically or manually
- request and approve rollback workflows
- execute rollback actions using optional snapshot state restoration
- inspect rollback dashboard summaries

### `astroml.deployment.state_snapshot.StateSnapshotManager`

This manager handles snapshot lifecycle:

- creating snapshots with named state, environment, and version metadata
- restoring snapshots and capturing RTO metrics
- cleaning up snapshots after recovery

### `astroml.deployment.recovery_procedures.RecoveryProcedure`

Define recovery procedures as a sequence of atomic steps. Each step exposes:

- a name and description
- a callable action returning a boolean success indicator
- execution results and error capture

### `astroml.deployment.disaster_recovery.DisasterRecoveryManager`

Use this manager to:

- register recovery procedures
- plan recovery workflows for specific incidents and snapshots
- execute recovery plans with snapshot restoration and procedure orchestration
- test recovery scenarios before they are executed in production

## Example usage

```python
from astroml.deployment.rollback_manager import RollbackManager
from astroml.deployment.state_snapshot import StateSnapshotManager
from astroml.deployment.recovery_procedures import RecoveryProcedure, RecoveryStep
from astroml.deployment.disaster_recovery import DisasterRecoveryManager

snapshot_manager = StateSnapshotManager()
rollback_manager = RollbackManager(snapshot_manager=snapshot_manager)

snapshot = snapshot_manager.create_snapshot(
    name="model-state-1",
    environment="production",
    model_version="1.2.0",
    state={"route": "v1", "weights": "s3://..."},
    recovery_objective_seconds=300,
)

rollback = rollback_manager.trigger_rollback(
    reason="Deployment health degraded",
    target_version="1.1.0",
    requested_by="monitoring-system",
    severity="critical",
    snapshot_id=snapshot.snapshot_id,
    auto_approve=True,
)

procedure = RecoveryProcedure(
    name="basic-recovery",
    description="Restore state and validate service availability",
    steps=[
        RecoveryStep(name="verify-snapshot", description="Ensure snapshot exists", action=lambda: True),
        RecoveryStep(name="validate-dependencies", description="Confirm dependency health", action=lambda: True),
    ],
)

recovery_manager = DisasterRecoveryManager(snapshot_manager)
recovery_manager.register_procedure(procedure)
plan = recovery_manager.plan_recovery(
    incident_id="incident-123",
    procedure_name="basic-recovery",
    snapshot_id=snapshot.snapshot_id,
)

recovery_manager.execute_recovery(plan.plan_id)
```

## RTO Tracking

Snapshots record `recovery_objective_seconds` and automatically compute whether a restore completed within the objective. Use `Snapshot.within_objective` to validate recovery time.

## Testing recovery scenarios

The `DisasterRecoveryManager.test_recovery_scenario` method helps validate an incident plan against a snapshot before executing the recovery in production.
