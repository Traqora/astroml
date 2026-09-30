"""GitOps deployment manager for AstroML model deployments.

Includes canary deployments, blue-green deployments, traffic routing,
rollback management, and state snapshots.
"""

from __future__ import annotations

from importlib import import_module

from .blue_green import BlueGreenConfig, BlueGreenDeployment, BlueGreenManager
from .canary import CanaryConfig, CanaryDeployment, CanaryManager, CanaryPhase
from .disaster_recovery import DisasterRecoveryManager
from .gitops_manager import GitOpsManager
from .recovery_procedures import RecoveryProcedure
from .rollback_manager import RollbackManager
from .state_snapshot import StateSnapshotManager
from .traffic_router import RouteTarget, RoutingRule, TrafficRouter

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
