"""GitOps deployment manager for AstroML model deployments.

Includes canary deployments, blue-green deployments, traffic routing,
rollback management, and state snapshots.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BlueGreenConfig",
    "BlueGreenDeployment",
    "BlueGreenManager",
    "CanaryConfig",
    "CanaryDeployment",
    "CanaryManager",
    "CanaryPhase",
    "DisasterRecoveryManager",
    "GitOpsManager",
    "RecoveryProcedure",
    "RollbackManager",
    "RouteTarget",
    "RoutingRule",
    "StateSnapshotManager",
    "TrafficRouter",
]

_LAZY: dict[str, tuple[str, str]] = {
    "BlueGreenConfig": ("blue_green", "BlueGreenConfig"),
    "BlueGreenDeployment": ("blue_green", "BlueGreenDeployment"),
    "BlueGreenManager": ("blue_green", "BlueGreenManager"),
    "CanaryConfig": ("canary", "CanaryConfig"),
    "CanaryDeployment": ("canary", "CanaryDeployment"),
    "CanaryManager": ("canary", "CanaryManager"),
    "CanaryPhase": ("canary", "CanaryPhase"),
    "DisasterRecoveryManager": ("disaster_recovery", "DisasterRecoveryManager"),
    "GitOpsManager": ("gitops_manager", "GitOpsManager"),
    "RecoveryProcedure": ("recovery_procedures", "RecoveryProcedure"),
    "RollbackManager": ("rollback_manager", "RollbackManager"),
    "RouteTarget": ("traffic_router", "RouteTarget"),
    "RoutingRule": ("traffic_router", "RoutingRule"),
    "StateSnapshotManager": ("state_snapshot", "StateSnapshotManager"),
    "TrafficRouter": ("traffic_router", "TrafficRouter"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY[name]
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
