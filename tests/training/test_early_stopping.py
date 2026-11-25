"""Tests for the shared early-stopping rule (issue #1208)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from astroml.training.config import EarlyStoppingConfig, TrainingConfig
from astroml.training.early_stopping import (
    EARLY_STOPPING_MONITORS,
    EarlyStopper,
    default_mode,
    monitor_value,
    normalize_monitor,
)


def test_supported_monitors_and_default_directions() -> None:
    """Aliases resolve to the four canonical names with sensible directions."""
    assert EARLY_STOPPING_MONITORS == (
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    )
    assert normalize_monitor("val_acc") == "val_accuracy"
    assert normalize_monitor("loss") == "train_loss"
    assert default_mode("val_loss") == "min"
    assert default_mode("val_accuracy") == "max"


@pytest.mark.parametrize("monitor", ["macro_f1", "auc", "", 42])
def test_unsupported_monitor_is_rejected(monitor: object) -> None:
    """Only the documented monitor set is accepted."""
    with pytest.raises(ValueError, match="Monitor must be a string|Unsupported early-stopping"):
        EarlyStopper(monitor=monitor)  # type: ignore[arg-type]


def test_min_mode_stops_after_configured_patience() -> None:
    """Loss stopping counts consecutive evaluations without a decrease."""
    stopper = EarlyStopper(monitor="val_loss", patience=2, min_delta=0.0)

    assert stopper.step({"val_loss": 0.5}, epoch=1) is False
    assert stopper.best_value == 0.5
    assert stopper.last_improved is True

    assert stopper.step({"val_loss": 0.6}, epoch=2) is False
    assert stopper.bad_epochs == 1
    assert stopper.step({"val_loss": 0.55}, epoch=3) is True
    assert stopper.best_epoch == 1
    assert stopper.best_value == 0.5
    assert stopper.bad_epochs == 2


def test_max_mode_uses_the_opposite_direction() -> None:
    """Accuracy stopping counts consecutive evaluations without an increase."""
    stopper = EarlyStopper(monitor="val_accuracy", patience=1, min_delta=0.0)

    assert stopper.step({"val_accuracy": 0.8}, epoch=1) is False
    assert stopper.step({"val_accuracy": 0.81}, epoch=2) is False
    assert stopper.best_value == 0.81
    assert stopper.step({"val_accuracy": 0.79}, epoch=3) is True
    assert stopper.best_epoch == 2


def test_min_delta_suppresses_noise_improvements() -> None:
    """Changes smaller than min_delta do not reset the patience counter."""
    stopper = EarlyStopper(monitor="val_loss", patience=1, min_delta=0.01)

    stopper.step({"val_loss": 0.500}, epoch=1)
    assert stopper.step({"val_loss": 0.499}, epoch=2) is True
    assert stopper.best_value == 0.500
    assert stopper.best_epoch == 1

    improving = EarlyStopper(monitor="val_loss", patience=1, min_delta=0.01)
    improving.step({"val_loss": 0.500}, epoch=1)
    assert improving.step({"val_loss": 0.480}, epoch=2) is False
    assert improving.best_value == 0.480


def test_patience_zero_disables_stopping() -> None:
    """patience=0 runs every configured epoch."""
    stopper = EarlyStopper(monitor="val_loss", patience=0)

    for epoch in range(1, 20):
        assert stopper.step({"val_loss": float(epoch)}, epoch=epoch) is False

    assert stopper.bad_epochs == 18


def test_missing_monitor_has_an_actionable_error() -> None:
    """A missing monitor names the expected canonical key."""
    stopper = EarlyStopper(monitor="val_accuracy")

    with pytest.raises(ValueError, match="do not contain early-stopping monitor 'val_accuracy'"):
        stopper.step({"val_loss": 0.2})


@pytest.mark.parametrize("value", ["high", float("nan"), float("inf"), None])
def test_non_numeric_or_non_finite_values_are_rejected(value: object) -> None:
    """Bad monitor values fail loudly instead of poisoning the best value."""
    stopper = EarlyStopper(monitor="val_loss")

    with pytest.raises(ValueError, match="not finite|not numeric"):
        stopper.step({"val_loss": value})


def test_nested_metrics_are_supported_for_the_baseline_loop() -> None:
    """The baseline's train/val dictionaries can be passed directly."""
    assert monitor_value({"val": {"loss": 0.3}}, "val_loss") == 0.3
    assert monitor_value({"train": {"loss": 0.8}}, "train_loss") == 0.8


def test_invalid_threshold_settings_are_rejected() -> None:
    """Patience, min_delta and mode have strict failure modes."""
    for kwargs in (
        {"patience": -1},
        {"patience": True},
        {"min_delta": -0.1},
        {"min_delta": float("nan")},
        {"mode": "both"},
    ):
        with pytest.raises(ValueError):
            EarlyStopper(**kwargs)  # type: ignore[arg-type]


def test_from_typed_config() -> None:
    """The shared stopper consumes the typed training config directly."""
    config = EarlyStoppingConfig(monitor="val_accuracy", patience=7, min_delta=0.02)

    stopper = EarlyStopper.from_config(config)

    assert stopper.monitor == "val_accuracy"
    assert stopper.mode == "max"
    assert stopper.patience == 7
    assert stopper.min_delta == 0.02


def test_training_config_limits_monitor_and_threshold() -> None:
    """Hydra config validation rejects unsupported monitors and thresholds."""
    from tests.test_training_config_schema import _base_training_dict

    data = _base_training_dict()
    data["early_stopping"]["monitor"] = "macro_f1"
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(data)

    data = _base_training_dict()
    data["early_stopping"]["min_delta"] = -1.0
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(data)

    data = _base_training_dict()
    data["early_stopping"]["mode"] = None
    data["early_stopping"]["monitor"] = "val_accuracy"
    assert TrainingConfig.model_validate(data).early_stopping.mode == "max"


def test_both_supported_training_paths_are_wired_to_the_stopper() -> None:
    """The Hydra entry point and baseline loop both use the shared rule."""
    root = Path(__file__).resolve().parents[2]

    hydra_source = (root / "train.py").read_text(encoding="utf-8")
    assert "EarlyStopper.from_config(cfg.training.early_stopping)" in hydra_source
    assert '"val_accuracy": val_metrics["accuracy"]' in hydra_source

    baseline_source = (root / "astroml" / "training" / "train_gcn.py").read_text(encoding="utf-8")
    assert "stopper.step(stopper_metrics, epoch=epoch)" in baseline_source
    assert 'val_metrics["loss"] = val_loss' in baseline_source
