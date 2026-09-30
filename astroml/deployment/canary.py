from astroml.utils.exceptions import AstroMLError
"""Canary deployment strategy for safe model rollouts.

Gradually shifts traffic from the current (stable) model to a new (canary)
model while monitoring health metrics. Automatically increases traffic share
on success or triggers rollback on failure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from astroml.deployment.rollback_manager import RollbackManager, RollbackSeverity

logger = logging.getLogger(__name__)


class CanaryPhase(str, Enum):
    """Phases of a canary deployment."""

    INITIALIZED = "initialized"
    DEPLOYING = "deploying"
    RAMPING = "ramping"
    STABILIZING = "stabilizing"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class CanaryConfig:
    """Configuration for a canary deployment.

    Attributes:
        initial_weight: Starting traffic percentage for canary (0–100).
        increment_step: Percentage to increase traffic on each step.
        stabilization_seconds: Wait time between increments.
        max_canary_weight: Maximum traffic the canary can receive before promoting.
        failure_threshold: Error rate threshold that triggers rollback.
        latency_threshold_ms: Latency threshold in ms that triggers rollback.
        auto_promote: Whether to automatically promote the canary to stable.
        auto_rollback: Whether to automatically rollback on failure.
    """

    initial_weight: float = 5.0
    increment_step: float = 10.0
    stabilization_seconds: float = 60.0
    max_canary_weight: float = 50.0
    failure_threshold: float = 0.05
    latency_threshold_ms: float = 500.0
    auto_promote: bool = False
    auto_rollback: bool = True


@dataclass
class CanaryStep:
    """Records a single step in the canary ramp-up process."""

    step_number: int
    weight: float
    duration_seconds: float
    error_rate: float
    avg_latency_ms: float
    healthy: bool
    timestamp: str


@dataclass
class CanaryDeployment:
    """Tracks an active or completed canary deployment."""

    deployment_id: str
    model_name: str
    canary_version: str
    stable_version: str
    config: CanaryConfig = field(default_factory=CanaryConfig)
    phase: CanaryPhase = CanaryPhase.INITIALIZED
    current_weight: float = 0.0
    steps: list[CanaryStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CanaryManager:
    """Orchestrates canary deployments with traffic ramping and health checks.

    Usage::

        manager = CanaryManager()
        dep = manager.start_canary("fraud-model", "v2.0.0", "v1.9.0")
        dep = manager.step(dep.deployment_id, health_check_fn)
        if dep.phase == CanaryPhase.PROMOTED:
            # canary is now stable
            pass
    """

    def __init__(
        self,
        rollback_manager: RollbackManager | None = None,
    ) -> None:
        self._deployments: dict[str, CanaryDeployment] = {}
        self._rollback_manager = rollback_manager or RollbackManager()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_canary(
        self,
        model_name: str,
        canary_version: str,
        stable_version: str,
        config: CanaryConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CanaryDeployment:
        """Start a new canary deployment.

        Args:
            model_name: Name of the model.
            canary_version: Version of the new (canary) model.
            stable_version: Version of the current stable model.
            config: Canary configuration (uses defaults if None).
            metadata: Arbitrary deployment metadata.

        Returns:
            The created CanaryDeployment.
        """
        import uuid as _uuid

        dep = CanaryDeployment(
            deployment_id=_uuid.uuid4().hex[:12],
            model_name=model_name,
            canary_version=canary_version,
            stable_version=stable_version,
            config=config or CanaryConfig(),
            metadata=metadata or {},
        )
        dep.phase = CanaryPhase.DEPLOYING
        dep.current_weight = dep.config.initial_weight
        self._deployments[dep.deployment_id] = dep

        logger.info(
            "Canary started: %s %s->%s (initial weight=%.1f%%)",
            model_name,
            stable_version,
            canary_version,
            dep.current_weight,
        )
        return dep

    def step(
        self,
        deployment_id: str,
        health_check: Callable[[], dict[str, Any]] | None = None,
    ) -> CanaryDeployment:
        """Advance the canary by one ramp-up step.

        Checks health metrics, then either ramps up traffic, promotes the
        canary to stable, or triggers a rollback.

        Args:
            deployment_id: Canary deployment to advance.
            health_check: Optional callback returning health metrics dict
                with optional keys: ``error_rate``, ``latency_ms``.

        Returns:
            Updated CanaryDeployment.
        """
        dep = self._get(deployment_id)
        if dep.phase not in (CanaryPhase.DEPLOYING, CanaryPhase.RAMPING):
            logger.warning(
                "Cannot step canary in phase %s: %s",
                dep.phase,
                deployment_id,
            )
            return dep

        step_start = time.monotonic()

        # Evaluate health
        healthy = True
        error_rate = 0.0
        avg_latency_ms = 0.0

        if health_check:
            try:
                metrics = health_check()
                error_rate = float(metrics.get("error_rate", 0.0))
                avg_latency_ms = float(metrics.get("latency_ms", 0.0))
                healthy = (
                    error_rate <= dep.config.failure_threshold
                    and avg_latency_ms <= dep.config.latency_threshold_ms
                )
            except Exception as exc:
                logger.error("Health check raised: %s", exc)
                healthy = False

        duration = time.monotonic() - step_start

        step = CanaryStep(
            step_number=len(dep.steps) + 1,
            weight=dep.current_weight,
            duration_seconds=round(duration, 3),
            error_rate=round(error_rate, 6),
            avg_latency_ms=round(avg_latency_ms, 3),
            healthy=healthy,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        dep.steps.append(step)

        if not healthy:
            if dep.config.auto_rollback:
                self._trigger_rollback(dep, error_rate, avg_latency_ms)
                dep.phase = CanaryPhase.ROLLED_BACK
            else:
                dep.phase = CanaryPhase.FAILED
                dep.error = f"Health check failed at weight={dep.current_weight}%"
            dep.completed_at = datetime.now(timezone.utc).isoformat()
            return dep

        # Ramp up if not yet at max
        if dep.current_weight < dep.config.max_canary_weight:
            dep.phase = CanaryPhase.RAMPING
            dep.current_weight = min(
                dep.current_weight + dep.config.increment_step,
                dep.config.max_canary_weight,
            )
            logger.info(
                "Canary %s ramped to %.1f%% (step %d)",
                dep.deployment_id,
                dep.current_weight,
                step.step_number,
            )
        else:
            # At max weight – stabilize then promote
            dep.phase = CanaryPhase.STABILIZING
            if dep.config.auto_promote:
                dep.phase = CanaryPhase.PROMOTED
                dep.current_weight = 100.0
                dep.completed_at = datetime.now(timezone.utc).isoformat()
                logger.info("Canary %s promoted to stable", dep.deployment_id)

        return dep

    def promote(self, deployment_id: str) -> CanaryDeployment:
        """Manually promote a canary to stable.

        Args:
            deployment_id: Canary deployment to promote.

        Returns:
            Updated CanaryDeployment.
        """
        dep = self._get(deployment_id)
        dep.phase = CanaryPhase.PROMOTED
        dep.current_weight = 100.0
        dep.completed_at = datetime.now(timezone.utc).isoformat()
        logger.info("Canary %s manually promoted", dep.deployment_id)
        return dep

    def rollback(self, deployment_id: str) -> CanaryDeployment:
        """Manually roll back a canary deployment.

        Args:
            deployment_id: Canary deployment to roll back.

        Returns:
            Updated CanaryDeployment.
        """
        dep = self._get(deployment_id)
        dep.phase = CanaryPhase.ROLLED_BACK
        dep.current_weight = 0.0
        dep.completed_at = datetime.now(timezone.utc).isoformat()
        dep.error = "Manual rollback requested"
        logger.info("Canary %s manually rolled back", dep.deployment_id)
        return dep

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, deployment_id: str) -> CanaryDeployment | None:
        """Get a canary deployment by ID.

        Args:
            deployment_id: Deployment identifier.

        Returns:
            CanaryDeployment or None.
        """
        return self._deployments.get(deployment_id)

    def list_active(self) -> list[CanaryDeployment]:
        """List all active (non-terminated) canary deployments.

        Returns:
            List of active deployments.
        """
        active_phases = {
            CanaryPhase.INITIALIZED,
            CanaryPhase.DEPLOYING,
            CanaryPhase.RAMPING,
            CanaryPhase.STABILIZING,
        }
        return [d for d in self._deployments.values() if d.phase in active_phases]

    def status_summary(self) -> dict[str, Any]:
        """Return a summary of all canary deployment statuses.

        Returns:
            Dict with counts and active deployments.
        """
        by_phase: dict[str, int] = {}
        for d in self._deployments.values():
            by_phase[d.phase.value] = by_phase.get(d.phase.value, 0) + 1

        active = self.list_active()

        return {
            "total": len(self._deployments),
            "by_phase": by_phase,
            "active": [
                {
                    "deployment_id": d.deployment_id,
                    "model_name": d.model_name,
                    "canary_version": d.canary_version,
                    "current_weight": d.current_weight,
                    "phase": d.phase.value,
                }
                for d in active
            ],
        }

    # ------------------------------------------------------------------
    # Traffic weight API
    # ------------------------------------------------------------------

    def set_weight(self, deployment_id: str, weight: float) -> CanaryDeployment:
        """Manually set the canary traffic weight percentage.

        Args:
            deployment_id: Canary deployment.
            weight: Traffic percentage (0–100).

        Returns:
            Updated deployment.
        """
        dep = self._get(deployment_id)
        if weight < 0 or weight > 100:
            raise ValueError("Weight must be between 0 and 100")
        dep.current_weight = weight
        logger.info("Canary %s weight set to %.1f%%", dep.deployment_id, weight)
        return dep

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get(self, deployment_id: str) -> CanaryDeployment:
        d = self._deployments.get(deployment_id)
        if d is None:
            raise ValueError(f"Canary deployment '{deployment_id}' not found")
        return d

    def _trigger_rollback(
        self,
        dep: CanaryDeployment,
        error_rate: float,
        latency_ms: float,
    ) -> None:
        """Automatically trigger a rollback on failure."""
        severity: RollbackSeverity = "critical" if error_rate > 0.10 else "high"
        self._rollback_manager.trigger_rollback(
            reason=(
                f"Canary {dep.model_name} {dep.canary_version} failed: "
                f"error_rate={error_rate:.2%}, latency={latency_ms:.1f}ms"
            ),
            target_version=dep.stable_version,
            requested_by="canary-manager",
            severity=severity,
            context={
                "deployment_id": dep.deployment_id,
                "model_name": dep.model_name,
                "canary_version": dep.canary_version,
                "error_rate": error_rate,
                "latency_ms": latency_ms,
            },
            auto_approve=dep.config.auto_rollback,
        )
        dep.error = f"Auto-rollback: error_rate={error_rate:.2%}, latency={latency_ms:.1f}ms"
        # A rolled-back canary must stop receiving traffic.  Without this the
        # phase says ROLLED_BACK while the weight still says "send it X%", and
        # anything routing off ``current_weight`` keeps serving the failed
        # version.  ``rollback()`` already zeroes it for the manual path.
        dep.current_weight = 0.0
