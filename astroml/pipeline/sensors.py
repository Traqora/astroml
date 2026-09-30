"""Pipeline Sensors for event-driven and schedule-driven pipeline triggers.

Includes sensors for data arrival, model performance degradation, data drift,
file existence, and interval schedule triggers.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SensorResult:
    """Outcome of a sensor check."""

    triggered: bool
    message: str = ""
    payload: dict[str, Any] | None = None
    checked_at: datetime = datetime.now(timezone.utc)


class BaseSensor(ABC):
    """Abstract base class for pipeline sensors."""

    def __init__(
        self,
        name: str,
        poke_interval_seconds: float = 5.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize base sensor."""
        self.name = name
        self.poke_interval = poke_interval_seconds
        self.timeout = timeout_seconds

    @abstractmethod
    def poke(self, context: dict[str, Any] | None = None) -> SensorResult:
        """Check if trigger condition is met (return SensorResult)."""

    def wait(self, context: dict[str, Any] | None = None) -> SensorResult:
        """Poll the sensor until condition is met or timeout is reached."""
        start = time.time()
        attempts = 0
        while (time.time() - start) < self.timeout:
            attempts += 1
            result = self.poke(context)
            if result.triggered:
                logger.info(
                    "Sensor '%s' triggered after %d attempts (%.1fs)",
                    self.name,
                    attempts,
                    time.time() - start,
                )
                return result
            time.sleep(self.poke_interval)

        return SensorResult(
            triggered=False,
            message=f"Sensor '{self.name}' timed out after {self.timeout}s",
        )


class DataArrivalSensor(BaseSensor):
    """Sensor that triggers when new transaction data or records arrive."""

    def __init__(
        self,
        name: str = "data_arrival_sensor",
        check_fn: Callable[[], int | bool] | None = None,
        min_records: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize data arrival sensor."""
        super().__init__(name=name, **kwargs)
        self.check_fn = check_fn
        self.min_records = min_records

    def poke(self, context: dict[str, Any] | None = None) -> SensorResult:
        """Check if incoming record threshold is reached."""
        if self.check_fn is not None:
            val = self.check_fn()
            if isinstance(val, bool) and val:
                return SensorResult(triggered=True, message="Data arrived (boolean trigger)")
            if isinstance(val, (int, float)) and val >= self.min_records:
                return SensorResult(
                    triggered=True,
                    message=f"Found {val} records (min: {self.min_records})",
                    payload={"record_count": val},
                )
            return SensorResult(
                triggered=False, message=f"Insufficient records: {val} < {self.min_records}"
            )
        return SensorResult(triggered=True, message="No check_fn provided, triggered immediately")


class ModelPerformanceSensor(BaseSensor):
    """Sensor that triggers when model metric drops below target threshold."""

    def __init__(
        self,
        name: str = "model_performance_sensor",
        metric_eval_fn: Callable[[], float] | None = None,
        min_threshold: float = 0.85,
        metric_name: str = "f1_score",
        **kwargs: Any,
    ) -> None:
        """Initialize model performance sensor."""
        super().__init__(name=name, **kwargs)
        self.metric_eval_fn = metric_eval_fn
        self.min_threshold = min_threshold
        self.metric_name = metric_name

    def poke(self, context: dict[str, Any] | None = None) -> SensorResult:
        """Check if model metric indicates need for retraining."""
        if self.metric_eval_fn is not None:
            current_metric = self.metric_eval_fn()
            needs_retrain = current_metric < self.min_threshold
            return SensorResult(
                triggered=needs_retrain,
                message=f"Metric {self.metric_name}={current_metric:.4f} (threshold: {self.min_threshold})",
                payload={"metric_name": self.metric_name, "value": current_metric},
            )
        return SensorResult(triggered=False, message="No metric_eval_fn provided")


class DriftSensor(BaseSensor):
    """Sensor that triggers when feature or prediction drift is detected."""

    def __init__(
        self,
        name: str = "drift_sensor",
        drift_check_fn: Callable[[], tuple[bool, float]] | None = None,
        p_value_threshold: float = 0.05,
        **kwargs: Any,
    ) -> None:
        """Initialize drift sensor."""
        super().__init__(name=name, **kwargs)
        self.drift_check_fn = drift_check_fn
        self.p_value_threshold = p_value_threshold

    def poke(self, context: dict[str, Any] | None = None) -> SensorResult:
        """Check for statistical drift."""
        if self.drift_check_fn is not None:
            has_drift, p_val = self.drift_check_fn()
            return SensorResult(
                triggered=has_drift or (p_val < self.p_value_threshold),
                message=f"Drift check result: has_drift={has_drift}, p_value={p_val:.4f}",
                payload={"has_drift": has_drift, "p_value": p_val},
            )
        return SensorResult(triggered=False, message="No drift_check_fn provided")


class FileArrivalSensor(BaseSensor):
    """Sensor that triggers when a file or directory path exists."""

    def __init__(
        self,
        filepath: str | Path,
        name: str = "file_arrival_sensor",
        min_size_bytes: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize file arrival sensor."""
        super().__init__(name=name, **kwargs)
        self.filepath = Path(filepath)
        self.min_size_bytes = min_size_bytes

    def poke(self, context: dict[str, Any] | None = None) -> SensorResult:
        """Check file existence and size."""
        if self.filepath.exists():
            size = self.filepath.stat().st_size if self.filepath.is_file() else 1
            if size >= self.min_size_bytes:
                return SensorResult(
                    triggered=True,
                    message=f"File {self.filepath} arrived (size: {size} bytes)",
                    payload={"path": str(self.filepath), "size": size},
                )
        return SensorResult(
            triggered=False,
            message=f"File {self.filepath} not found or size < {self.min_size_bytes}",
        )
