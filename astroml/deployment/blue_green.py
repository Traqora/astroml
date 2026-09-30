from astroml.utils.exceptions import AstroMLError
"""Blue-green deployment strategy for zero-downtime model rollouts.

Maintains two identical environments (blue = current, green = new).
Switches traffic atomically after green passes health checks.
Supports instant rollback by switching back to blue.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from astroml.deployment.rollback_manager import RollbackManager

logger = logging.getLogger(__name__)


class DeploymentEnv(str, Enum):
    """Environment labels."""

    BLUE = "blue"
    GREEN = "green"


class BGPhase(str, Enum):
    """Phases of a blue-green deployment."""

    IDLE = "idle"
    PREPARING = "preparing"
    TESTING = "testing"
    SWITCHING = "switching"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class BlueGreenConfig:
    """Configuration for a blue-green deployment.

    Attributes:
        health_check_timeout_seconds: Max time to wait for health checks.
        health_check_interval_seconds: Interval between health checks.
        stabilization_seconds: Wait time after switch before declaring success.
        auto_switch: Whether to automatically switch after tests pass.
        auto_rollback: Whether to auto-rollback on monitoring failure.
        max_retries: Maximum health check retries before marking failed.
    """

    health_check_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 5.0
    stabilization_seconds: float = 300.0  # 5 minutes
    auto_switch: bool = False
    auto_rollback: bool = True
    max_retries: int = 3


@dataclass
class BlueGreenDeployment:
    """Tracks one blue-green deployment cycle."""

    deployment_id: str
    model_name: str
    blue_version: str
    green_version: str
    active_env: DeploymentEnv = DeploymentEnv.BLUE
    config: BlueGreenConfig = field(default_factory=BlueGreenConfig)
    phase: BGPhase = BGPhase.IDLE
    switched_at: str | None = None
    completed_at: str | None = None
    rollback_count: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    health_check_results: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class BlueGreenManager:
    """Manages blue-green deployments for model serving.

    Usage::

        mgr = BlueGreenManager()
        dep = mgr.prepare("fraud-model", "v2.0.0", "v1.9.0")

        def check_health() -> dict:
            return {"healthy": True, "latency_ms": 45.0}

        dep = mgr.test_green(dep.deployment_id, check_health)
        dep = mgr.switch(dep.deployment_id)
    """

    def __init__(
        self,
        rollback_manager: RollbackManager | None = None,
    ) -> None:
        self._deployments: dict[str, BlueGreenDeployment] = {}
        self._rollback_manager = rollback_manager or RollbackManager()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(
        self,
        model_name: str,
        green_version: str,
        blue_version: str,
        config: BlueGreenConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BlueGreenDeployment:
        """Prepare the green environment with a new model version.

        Args:
            model_name: Model name.
            green_version: Version to deploy to green.
            blue_version: Current version on blue.
            config: Optional configuration overrides.
            metadata: Optional metadata.

        Returns:
            New BlueGreenDeployment in PREPARING phase.
        """
        dep = BlueGreenDeployment(
            deployment_id=uuid.uuid4().hex[:12],
            model_name=model_name,
            blue_version=blue_version,
            green_version=green_version,
            config=config or BlueGreenConfig(),
            metadata=metadata or {},
        )
        dep.phase = BGPhase.PREPARING
        self._deployments[dep.deployment_id] = dep

        logger.info(
            "Blue-green prepared: %s blue=%s green=%s",
            model_name,
            blue_version,
            green_version,
        )
        return dep

    def test_green(
        self,
        deployment_id: str,
        health_check: Callable[[], dict[str, Any]],
    ) -> BlueGreenDeployment:
        """Run health checks against the green environment.

        Args:
            deployment_id: Deployment to test.
            health_check: Callable returning a dict with at least a
                ``healthy`` key (bool). May include ``latency_ms``,
                ``error_rate``, etc.

        Returns:
            Updated deployment (phase = TESTING or FAILED).
        """
        dep = self._get(deployment_id)
        if dep.phase != BGPhase.PREPARING:
            raise ValueError(
                f"Deployment {deployment_id} is not in PREPARING phase (currently {dep.phase})"
            )

        dep.phase = BGPhase.TESTING

        for attempt in range(1, dep.config.max_retries + 1):
            try:
                result = health_check()
            except Exception as exc:
                result = {"healthy": False, "error": str(exc)}

            result["attempt"] = attempt
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            dep.health_check_results.append(result)

            if result.get("healthy", False):
                logger.info(
                    "Green health check passed (attempt %d/%d)",
                    attempt,
                    dep.config.max_retries,
                )
                break
            else:
                logger.warning(
                    "Green health check %d/%d failed: %s",
                    attempt,
                    dep.config.max_retries,
                    result.get("error", "unknown"),
                )
        else:
            dep.phase = BGPhase.FAILED
            dep.error = f"All {dep.config.max_retries} health checks failed"
            return dep

        if dep.config.auto_switch:
            return self.switch(deployment_id)

        return dep

    def switch(self, deployment_id: str) -> BlueGreenDeployment:
        """Perform the traffic switch from blue to green.

        Args:
            deployment_id: Deployment to switch.

        Returns:
            Updated deployment (phase = COMPLETED).
        """
        dep = self._get(deployment_id)
        if dep.phase not in (BGPhase.TESTING, BGPhase.PREPARING):
            raise ValueError(
                f"Deployment {deployment_id} cannot switch in phase {dep.phase}"
            )

        dep.phase = BGPhase.SWITCHING

        # Swap environments
        dep.blue_version, dep.green_version = dep.green_version, dep.blue_version
        dep.active_env = (
            DeploymentEnv.GREEN
            if dep.active_env == DeploymentEnv.BLUE
            else DeploymentEnv.BLUE
        )
        dep.switched_at = datetime.now(timezone.utc).isoformat()

        dep.phase = BGPhase.COMPLETED
        dep.completed_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Blue-green switch complete: %s now active on %s v=%s",
            dep.model_name,
            dep.active_env.value,
            dep.blue_version,
        )
        return dep

    def rollback(self, deployment_id: str) -> BlueGreenDeployment:
        """Roll back to the previous (blue) environment.

        Args:
            deployment_id: Deployment to roll back.

        Returns:
            Updated deployment.
        """
        dep = self._get(deployment_id)

        # Swap back
        dep.blue_version, dep.green_version = dep.green_version, dep.blue_version
        dep.active_env = (
            DeploymentEnv.GREEN
            if dep.active_env == DeploymentEnv.BLUE
            else DeploymentEnv.BLUE
        )
        dep.rollback_count += 1
        dep.phase = BGPhase.ROLLED_BACK
        dep.completed_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Blue-green rolled back: %s active=%s v=%s",
            dep.model_name,
            dep.active_env.value,
            dep.blue_version,
        )
        return dep

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def monitor(
        self,
        deployment_id: str,
        health_check: Callable[[], dict[str, Any]],
    ) -> BlueGreenDeployment:
        """Monitor the currently active environment post-switch.

        Triggers auto-rollback if health degrades.

        Args:
            deployment_id: Deployment to monitor.
            health_check: Health check callback.

        Returns:
            Updated deployment.
        """
        dep = self._get(deployment_id)
        if dep.phase not in (BGPhase.COMPLETED, BGPhase.MONITORING):
            raise ValueError(
                f"Deployment {deployment_id} not in monitorable phase: {dep.phase}"
            )

        dep.phase = BGPhase.MONITORING

        try:
            result = health_check()
        except Exception as exc:
            result = {"healthy": False, "error": str(exc)}

        dep.health_check_results.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        })

        if not result.get("healthy", False):
            if dep.config.auto_rollback:
                self._rollback_manager.trigger_rollback(
                    reason=(
                        f"Blue-green monitor failure: {dep.model_name} "
                        f"v={dep.blue_version}"
                    ),
                    target_version=dep.green_version,
                    requested_by="blue-green-monitor",
                    severity="high",
                    context={"error": result.get("error", "unknown")},
                    auto_approve=True,
                )
                return self.rollback(deployment_id)
            else:
                dep.error = f"Monitor failure: {result.get('error', 'unknown')}"

        return dep

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, deployment_id: str) -> BlueGreenDeployment | None:
        """Get a deployment by ID.

        Args:
            deployment_id: Deployment identifier.

        Returns:
            BlueGreenDeployment or None.
        """
        return self._deployments.get(deployment_id)

    def list_deployments(
        self, model_name: str | None = None
    ) -> list[BlueGreenDeployment]:
        """List deployments, optionally filtered by model name.

        Args:
            model_name: Optional filter.

        Returns:
            Filtered list.
        """
        if model_name:
            return [d for d in self._deployments.values() if d.model_name == model_name]
        return list(self._deployments.values())

    def status_summary(self) -> dict[str, Any]:
        """Return aggregate status summary.

        Returns:
            Dict with counts and active deployment info.
        """
        by_phase: dict[str, int] = {}
        for d in self._deployments.values():
            by_phase[d.phase.value] = by_phase.get(d.phase.value, 0) + 1

        active = [
            d
            for d in self._deployments.values()
            if d.phase not in (BGPhase.COMPLETED, BGPhase.FAILED, BGPhase.ROLLED_BACK)
        ]

        return {
            "total": len(self._deployments),
            "by_phase": by_phase,
            "active": [
                {
                    "deployment_id": d.deployment_id,
                    "model_name": d.model_name,
                    "active_version": d.blue_version,
                    "active_env": d.active_env.value,
                    "phase": d.phase.value,
                }
                for d in active
            ],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get(self, deployment_id: str) -> BlueGreenDeployment:
        d = self._deployments.get(deployment_id)
        if d is None:
            raise ValueError(f"Blue-green deployment '{deployment_id}' not found")
        return d