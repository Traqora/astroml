from __future__ import annotations

import pytest

from astroml.training.config import TrainingConfig, validate_training_config_data


def _base_training_dict() -> dict:
    return {
        "epochs": 200,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "optimizer": "adam",
        "scheduler": None,
        "early_stopping": {
            "patience": 50,
            "min_delta": 1e-4,
            "monitor": "val_loss",
            "mode": "min",
        },
        "batch_size": None,
        "val_split": 0.1,
        "test_split": 0.1,
        "shuffle": True,
        "temporal_split": {
            "enabled": False,
            "time_col": "timestamp",
            "train_ratio": 0.8,
            "cutoff": None,
        },
        "log_interval": 20,
        "save_best_only": True,
        "save_last": True,
        "optimizer_configs": {
            "adam": {"betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False},
            "sgd": {"momentum": 0.9, "nesterov": True},
            "adamw": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-2},
        },
    }


def test_training_config_accepts_valid_defaults() -> None:
    cfg = TrainingConfig.model_validate(_base_training_dict())
    assert cfg.epochs == 200
    assert cfg.optimizer == "adam"


def test_training_config_rejects_non_positive_epochs() -> None:
    data = _base_training_dict()
    data["epochs"] = 0
    with pytest.raises(Exception):
        TrainingConfig.model_validate(data)


def test_training_config_rejects_invalid_split_sum() -> None:
    data = _base_training_dict()
    data["val_split"] = 0.6
    data["test_split"] = 0.4
    with pytest.raises(Exception, match=r"val_split \+ test_split must be < 1.0"):
        TrainingConfig.model_validate(data)


def test_training_config_rejects_shuffle_with_temporal_split() -> None:
    data = _base_training_dict()
    data["temporal_split"]["enabled"] = True
    data["shuffle"] = True
    with pytest.raises(
        Exception,
        match="shuffle must be false when temporal_split.enabled is true",
    ):
        TrainingConfig.model_validate(data)


def test_validate_training_config_startup_hook_rejects_invalid_cfg() -> None:
    data = {
        **_base_training_dict(),
        "epochs": -1,
    }

    with pytest.raises(ValueError, match="Invalid training configuration"):
        validate_training_config_data(data)


def test_training_config_rejects_unknown_fields() -> None:
    data = _base_training_dict()
    data["unknown_option"] = True

    with pytest.raises(Exception):
        TrainingConfig.model_validate(data)
