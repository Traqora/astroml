"""Early stopping for the supported v1 training paths (issue #1208).

A single stopper is shared by the Hydra entry point and the documented
baseline GCN loop. It watches one of four canonical metrics, applies a
``min_delta`` threshold, and stops after ``patience`` consecutive evaluations
that fail to improve. ``patience=0`` disables stopping.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

EARLY_STOPPING_MONITORS: tuple[str, ...] = (
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
)
EarlyStoppingMonitor = Literal["train_loss", "train_accuracy", "val_loss", "val_accuracy"]

_ALIASES = {
    "loss": "train_loss",
    "accuracy": "train_accuracy",
    "acc": "train_accuracy",
    "train_acc": "train_accuracy",
    "val_acc": "val_accuracy",
}

__all__ = [
    "EARLY_STOPPING_MONITORS",
    "EarlyStopper",
    "EarlyStoppingMonitor",
    "default_mode",
    "normalize_monitor",
    "monitor_value",
]


def normalize_monitor(monitor: str) -> str:
    """Return the canonical name for a supported monitor."""
    if not isinstance(monitor, str):
        raise ValueError(f"Monitor must be a string, got {monitor!r}.")
    name = _ALIASES.get(monitor.strip().lower(), monitor.strip().lower())
    if name not in EARLY_STOPPING_MONITORS:
        choices = ", ".join(EARLY_STOPPING_MONITORS)
        raise ValueError(f"Unsupported early-stopping monitor {monitor!r}; choose from {choices}.")
    return name


def default_mode(monitor: str) -> str:
    """Return ``min`` for loss monitors and ``max`` for accuracy monitors."""
    canonical = normalize_monitor(monitor)
    return "min" if canonical.endswith("_loss") else "max"


def monitor_value(metrics: Mapping[str, Any], monitor: str) -> float:
    """Read a monitor from flat or train/val-nested metric mappings."""
    canonical = normalize_monitor(monitor)
    if canonical in metrics:
        raw = metrics[canonical]
    elif monitor in metrics:
        raw = metrics[monitor]
    else:
        split_name, metric_name = canonical.split("_", 1)
        nested = metrics.get(split_name)
        if not isinstance(nested, Mapping) or metric_name not in nested:
            raise ValueError(
                f"Metrics do not contain early-stopping monitor {canonical!r}: "
                f"{sorted(str(key) for key in metrics)}"
            )
        raw = nested[metric_name]

    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Early-stopping monitor {canonical!r} is not numeric: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Early-stopping monitor {canonical!r} is not finite: {raw!r}")
    return value


@dataclass
class EarlyStopper:
    """Stateful early-stopping decision maker.

    Args:
        monitor: One of :data:`EARLY_STOPPING_MONITORS`, or a legacy alias such
            as ``val_acc``.
        mode: ``min`` or ``max``. When omitted, loss monitors use ``min`` and
            accuracy monitors use ``max``.
        patience: Consecutive non-improving evaluations before stopping. Zero
            disables stopping.
        min_delta: Minimum decrease (``min``) or increase (``max``) that counts
            as an improvement.

    Attributes:
        best_value: Best finite value seen so far, or ``None`` before the first
            call to :meth:`step`.
        best_epoch: Epoch of the best value, when an epoch was supplied.
        bad_epochs: Consecutive evaluations without a sufficient improvement.
        last_improved: Whether the most recent step improved the monitor.
        should_stop: Whether training should stop after the most recent step.
    """

    monitor: str = "val_loss"
    mode: str | None = None
    patience: int = 50
    min_delta: float = 1e-4

    best_value: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    last_improved: bool = False
    should_stop: bool = False

    def __post_init__(self) -> None:
        self.monitor = normalize_monitor(self.monitor)
        self.mode = default_mode(self.monitor) if self.mode is None else str(self.mode).lower()
        if self.mode not in ("min", "max"):
            raise ValueError(f"Early-stopping mode must be 'min' or 'max', got {self.mode!r}.")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int):
            raise ValueError(f"Early-stopping patience must be an integer, got {self.patience!r}.")
        if self.patience < 0:
            raise ValueError(f"Early-stopping patience must be >= 0, got {self.patience!r}.")
        if isinstance(self.min_delta, bool) or not isinstance(self.min_delta, (int, float)):
            raise ValueError(f"Early-stopping min_delta must be numeric, got {self.min_delta!r}.")
        self.min_delta = float(self.min_delta)
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError(
                f"Early-stopping min_delta must be finite and >= 0, got {self.min_delta!r}."
            )

    @classmethod
    def from_config(cls, config: Any) -> EarlyStopper:
        """Build a stopper from a typed early-stopping config."""
        return cls(
            monitor=config.monitor,
            mode=getattr(config, "mode", None),
            patience=config.patience,
            min_delta=config.min_delta,
        )

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def step(self, metrics: Mapping[str, Any], epoch: int | None = None) -> bool:
        """Record one evaluation and return whether training should stop."""
        value = monitor_value(metrics, self.monitor)
        self.last_improved = self._improved(value)

        if self.last_improved:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        self.should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return self.should_stop

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot for logs and results files."""
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "bad_epochs": self.bad_epochs,
            "stopped_early": self.should_stop,
        }
