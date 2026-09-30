from astroml.utils.exceptions import AstroMLError
"""Auto-scaling controller for model serving infrastructure.

Issue #639 Step 1 & 5: Implements metrics-based autoscaling and
predictive autoscaling for model inference pods.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scaling metrics
# ---------------------------------------------------------------------------


class ScalingMetric(Enum):
    """Metrics used for autoscaling decisions."""

    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    REQUEST_LATENCY_P50 = "request_latency_p50"
    REQUEST_LATENCY_P99 = "request_latency_p99"
    REQUEST_RATE = "request_rate"
    QUEUE_DEPTH = "queue_depth"
    ERROR_RATE = "error_rate"
    ACTIVE_CONNECTIONS = "active_connections"


@dataclass
class ScalingSnapshot:
    """Snapshot of all relevant metrics at a point in time."""

    timestamp: float = field(default_factory=time.time)
    metrics: dict[ScalingMetric, float] = field(default_factory=dict)
    current_replicas: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, metric: ScalingMetric) -> float:
        return self.metrics.get(metric, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "metrics": {k.name: v for k, v in self.metrics.items()},
            "current_replicas": self.current_replicas,
        }


# ---------------------------------------------------------------------------
# Scaling policy
# ---------------------------------------------------------------------------


@dataclass
class ScalingPolicy:
    """Policy governing how and when to scale.

    Attributes:
        metric: Primary metric to scale on.
        target_value: Target value for the metric.
        min_replicas: Minimum number of replicas.
        max_replicas: Maximum number of replicas.
        scale_up_threshold: Fraction above target to trigger scale-up.
        scale_down_threshold: Fraction below target to trigger scale-down.
        cooldown_seconds: Minimum time between scaling actions.
        scale_up_factor: Multiplicative factor for scale-up.
        scale_down_factor: Multiplicative factor for scale-down.
        max_scale_up_step: Maximum replicas to add in one step.
        max_scale_down_step: Maximum replicas to remove in one step.
    """

    metric: ScalingMetric = ScalingMetric.REQUEST_LATENCY_P99
    target_value: float = 100.0  # milliseconds
    min_replicas: int = 1
    max_replicas: int = 10
    scale_up_threshold: float = 1.2  # 20% above target
    scale_down_threshold: float = 0.8  # 20% below target
    cooldown_seconds: float = 60.0
    scale_up_factor: float = 2.0
    scale_down_factor: float = 0.5
    max_scale_up_step: int = 5
    max_scale_down_step: int = 2
    stabilization_window_seconds: float = 300.0  # Look back for stabilization


# ---------------------------------------------------------------------------
# Scaling history and prediction
# ---------------------------------------------------------------------------


@dataclass
class ScalingEvent:
    """Record of a scaling action."""

    timestamp: float
    previous_replicas: int
    new_replicas: int
    reason: str
    metrics_snapshot: ScalingSnapshot

    @property
    def direction(self) -> str:
        if self.new_replicas > self.previous_replicas:
            return "up"
        elif self.new_replicas < self.previous_replicas:
            return "down"
        return "noop"


# ---------------------------------------------------------------------------
# Autoscaler
# ---------------------------------------------------------------------------


class Autoscaler:
    """Metrics-based autoscaler for model serving infrastructure.

    Monitors inference metrics and automatically adjusts replica count
    based on configurable scaling policies. Supports both reactive
    (current metric-based) and predictive (trend-based) scaling.

    Example:
        autoscaler = Autoscaler(
            policy=ScalingPolicy(
                metric=ScalingMetric.REQUEST_LATENCY_P99,
                target_value=100.0,
                min_replicas=2,
                max_replicas=20,
            ),
            metrics_provider=get_prometheus_metrics,
        )

        # In a control loop:
        new_replicas = autoscaler.evaluate()
        if new_replicas != current_replicas:
            apply_scaling(new_replicas)
    """

    def __init__(
        self,
        policy: ScalingPolicy | None = None,
        metrics_provider: Callable[[], dict[ScalingMetric, float]] | None = None,
        scale_callback: Callable[[int], bool] | None = None,
    ) -> None:
        """Initialize the autoscaler.

        Args:
            policy: Scaling policy. Uses defaults if None.
            metrics_provider: Function that returns current metrics.
            scale_callback: Function called to apply scaling.
        """
        self.policy = policy or ScalingPolicy()
        self._metrics_provider = metrics_provider
        self._scale_callback = scale_callback
        self._history: deque[ScalingSnapshot] = deque(maxlen=1000)
        self._events: list[ScalingEvent] = []
        self._last_scale_time: float = 0.0
        self._current_replicas: int = self.policy.min_replicas
        self._lock = threading.Lock()
        self._enabled = True

    @property
    def current_replicas(self) -> int:
        return self._current_replicas

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def set_current_replicas(self, count: int) -> None:
        """Update the known current replica count (e.g., from K8s API)."""
        self._current_replicas = max(self.policy.min_replicas, min(count, self.policy.max_replicas))

    def evaluate(self) -> ScalingDecision:
        """Evaluate current metrics and decide whether to scale.

        Returns:
            ScalingDecision with the recommended replica count and reason.
        """
        with self._lock:
            return self._evaluate()

    def _evaluate(self) -> ScalingDecision:
        if not self._enabled:
            return ScalingDecision(
                replicas=self._current_replicas,
                action=ScalingAction.NOOP,
                reason="Autoscaler disabled",
            )

        # Collect current metrics
        snapshot = self._collect_metrics()
        self._history.append(snapshot)

        # Check cooldown
        if time.time() - self._last_scale_time < self.policy.cooldown_seconds:
            return ScalingDecision(
                replicas=self._current_replicas,
                action=ScalingAction.NOOP,
                reason="In cooldown period",
                metrics_snapshot=snapshot,
            )

        # Get the primary metric value
        metric_value = snapshot.get(self.policy.metric)

        # Calculate desired replicas
        desired = self._calculate_desired(metric_value, self._current_replicas)
        desired = max(self.policy.min_replicas, min(desired, self.policy.max_replicas))

        # Apply stabilization
        desired = self._stabilize(desired)

        if desired > self._current_replicas:
            # Check scale-up threshold
            ratio = metric_value / self.policy.target_value if self.policy.target_value > 0 else 1.0
            if ratio > self.policy.scale_up_threshold:
                step = min(desired - self._current_replicas, self.policy.max_scale_up_step)
                new_replicas = self._current_replicas + step

                return self._apply_scaling(
                    new_replicas,
                    ScalingAction.SCALE_UP,
                    f"Metric {self.policy.metric.name}={metric_value:.1f} above target {self.policy.target_value} (ratio={ratio:.2f})",
                    snapshot,
                )

        elif desired < self._current_replicas:
            # Check scale-down threshold
            ratio = metric_value / self.policy.target_value if self.policy.target_value > 0 else 1.0
            if ratio < self.policy.scale_down_threshold:
                step = min(self._current_replicas - desired, self.policy.max_scale_down_step)
                new_replicas = self._current_replicas - step

                return self._apply_scaling(
                    new_replicas,
                    ScalingAction.SCALE_DOWN,
                    f"Metric {self.policy.metric.name}={metric_value:.1f} below target {self.policy.target_value} (ratio={ratio:.2f})",
                    snapshot,
                )

        return ScalingDecision(
            replicas=self._current_replicas,
            action=ScalingAction.NOOP,
            reason=f"Within threshold range",
            metrics_snapshot=snapshot,
        )

    def _collect_metrics(self) -> ScalingSnapshot:
        """Collect current metrics from the provider or use defaults."""
        if self._metrics_provider:
            try:
                metrics = self._metrics_provider()
            except AstroMLError as e:
                logger.warning(f"Metrics provider failed: {e}")
                metrics = {}
        else:
            metrics = {}

        return ScalingSnapshot(
            timestamp=time.time(),
            metrics=metrics,
            current_replicas=self._current_replicas,
        )

    def _calculate_desired(self, metric_value: float, current: int) -> int:
        """Calculate desired replica count based on metric and current count."""
        if metric_value <= 0 or self.policy.target_value <= 0:
            return current

        ratio = metric_value / self.policy.target_value
        desired = int(np.ceil(current * ratio))
        return desired

    def _stabilize(self, desired: int) -> int:
        """Apply stabilization: don't change unless consistently needed."""
        if len(self._history) < 3:
            return desired

        # Check recent trends
        recent = list(self._history)[-5:]
        recent_vals = [s.get(self.policy.metric) for s in recent]

        # If recent values are noisy, delay scaling
        if len(recent_vals) >= 2:
            std = np.std(recent_vals)
            mean = np.mean(recent_vals) if np.mean(recent_vals) > 0 else 1.0
            if std / mean > 0.5:  # High variability
                logger.debug(f"Stabilization: high metric variability ({std / mean:.2f}), delaying scale")
                return self._current_replicas

        return desired

    def predict_future_load(self, horizon_seconds: float = 300.0) -> float | None:
        """Predict future load using simple linear regression on recent history.

        Args:
            horizon_seconds: How far ahead to predict.

        Returns:
            Predicted metric value, or None if insufficient data.
        """
        if len(self._history) < 10:
            return None

        recent = sorted(list(self._history)[-50:], key=lambda s: s.timestamp)

        times = np.array([s.timestamp for s in recent])
        values = np.array([s.get(self.policy.metric) for s in recent])

        # Normalize times relative to earliest
        times = times - times[0]

        # Simple linear regression
        coeffs = np.polyfit(times, values, 1)
        future_time = times[-1] + horizon_seconds
        predicted = np.polyval(coeffs, future_time)

        return float(max(0.0, predicted))

    def _apply_scaling(
        self,
        new_replicas: int,
        action: ScalingAction,
        reason: str,
        snapshot: ScalingSnapshot,
    ) -> ScalingDecision:
        """Apply and record a scaling action."""
        event = ScalingEvent(
            timestamp=time.time(),
            previous_replicas=self._current_replicas,
            new_replicas=new_replicas,
            reason=reason,
            metrics_snapshot=snapshot,
        )
        self._events.append(event)
        self._last_scale_time = time.time()
        self._current_replicas = new_replicas

        if self._scale_callback:
            try:
                success = self._scale_callback(new_replicas)
                if not success:
                    logger.error(f"Scale callback failed for {new_replicas} replicas")
            except AstroMLError as e:
                logger.error(f"Scale callback error: {e}")

        logger.info(
            f"SCALING {action.name}: {event.previous_replicas} -> {new_replicas} replicas "
            f"({reason})"
        )

        return ScalingDecision(
            replicas=new_replicas,
            action=action,
            reason=reason,
            metrics_snapshot=snapshot,
            event=event,
        )

    # ── Predictive scaling ──────────────────────────────────────────────

    def evaluate_predictive(self, horizon_seconds: float = 300.0) -> ScalingDecision:
        """Evaluate scaling using predicted future load.

        Combines reactive metrics with trend prediction to make
        proactive scaling decisions.

        Args:
            horizon_seconds: Prediction horizon in seconds.

        Returns:
            ScalingDecision.
        """
        predicted = self.predict_future_load(horizon_seconds)

        if predicted is not None:
            # Check if we should pre-scale
            ratio = predicted / self.policy.target_value if self.policy.target_value > 0 else 1.0

            if ratio > self.policy.scale_up_threshold:
                # Pre-emptively scale up
                desired = self._calculate_desired(predicted, self._current_replicas)
                desired = min(desired, self.policy.max_replicas)

                if desired > self._current_replicas:
                    # Override current snapshot
                    snapshot = self._collect_metrics()
                    return self._apply_scaling(
                        desired,
                        ScalingAction.SCALE_UP,
                        f"Predictive: metric predicted at {predicted:.1f} in {horizon_seconds}s (ratio={ratio:.2f})",
                        snapshot,
                    )

        # Fall back to reactive
        return self.evaluate()

    # ── History and stats ───────────────────────────────────────────────

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent scaling events."""
        return [
            {
                "timestamp": e.timestamp,
                "previous_replicas": e.previous_replicas,
                "new_replicas": e.new_replicas,
                "direction": e.direction,
                "reason": e.reason,
            }
            for e in self._events[-limit:]
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get autoscaler statistics."""
        total_events = len(self._events)
        scale_ups = sum(1 for e in self._events if e.direction == "up")
        scale_downs = sum(1 for e in self._events if e.direction == "down")

        return {
            "current_replicas": self._current_replicas,
            "enabled": self._enabled,
            "policy_metric": self.policy.metric.name,
            "policy_target": self.policy.target_value,
            "total_events": total_events,
            "scale_ups": scale_ups,
            "scale_downs": scale_downs,
            "last_scale_time": self._last_scale_time if self._last_scale_time else None,
        }


# ---------------------------------------------------------------------------
# Scaling decision
# ---------------------------------------------------------------------------


class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NOOP = "noop"


@dataclass
class ScalingDecision:
    """Result of an autoscaling evaluation."""

    replicas: int
    action: ScalingAction = ScalingAction.NOOP
    reason: str = ""
    metrics_snapshot: ScalingSnapshot | None = None
    event: ScalingEvent | None = None

    @property
    def should_scale(self) -> bool:
        return self.action != ScalingAction.NOOP

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicas": self.replicas,
            "action": self.action.value,
            "reason": self.reason,
            "should_scale": self.should_scale,
            "metrics": self.metrics_snapshot.to_dict() if self.metrics_snapshot else {},
        }


__all__ = [
    "ScalingMetric",
    "ScalingSnapshot",
    "ScalingPolicy",
    "ScalingEvent",
    "ScalingAction",
    "ScalingDecision",
    "Autoscaler",
]