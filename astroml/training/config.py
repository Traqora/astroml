
"""Typed configuration for training using pydantic."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
import yaml

from astroml.storage import ArtifactStorageConfig


class EarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patience: int = Field(default=50, ge=0, description="Number of epochs with no improvement after which training will stop.")
    min_delta: float = Field(default=1e-4, description="Minimum change in monitored quantity to qualify as an improvement.")
    monitor: str = Field(default="val_loss", description="Quantity to be monitored.")
    mode: Literal["min", "max"] = Field(default="min", description="One of `min`, `max`. In `min` mode, training will stop when the quantity monitored has stopped decreasing; in `max` mode it will stop when the quantity monitored has stopped increasing.")


class TemporalSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Whether to use temporal split instead of random split.")
    time_col: str = Field(default="timestamp", description="Column to use for temporal ordering.")
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0, description="Fraction of data to use for training when using temporal split.")
    cutoff: Optional[float] = Field(default=None, description="Optional explicit cutoff value for temporal split (overrides train_ratio).")


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adam: Dict[str, Any] = Field(default={"betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False})
    sgd: Dict[str, Any] = Field(default={"momentum": 0.9, "nesterov": True})
    adamw: Dict[str, Any] = Field(default={"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-2})


class TrainingConfig(BaseModel):
    """Typed configuration for training models."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=200, gt=0, description="Number of training epochs.")
    lr: float = Field(default=0.01, gt=0.0, description="Learning rate.")
    weight_decay: float = Field(default=5e-4, ge=0.0, description="Weight decay.")
    optimizer: Literal["adam", "sgd", "adamw"] = Field(default="adam", description="Optimizer to use.")
    scheduler: Optional[str] = Field(default=None, description="Learning rate scheduler to use (if any).")
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    batch_size: Optional[int] = Field(default=None, gt=0, description="Batch size (None for full batch, which is common for graph data).")
    val_split: float = Field(default=0.1, ge=0.0, le=1.0, description="Validation split fraction.")
    test_split: float = Field(default=0.1, ge=0.0, le=1.0, description="Test split fraction.")
    shuffle: bool = Field(default=True, description="Whether to shuffle data before splitting (set to False when using temporal split to prevent leakage).")
    temporal_split: TemporalSplitConfig = Field(default_factory=TemporalSplitConfig)
    log_interval: int = Field(default=20, gt=0, description="Logging interval (in epochs).")
    save_best_only: bool = Field(default=True, description="Whether to save only the best model.")
    save_last: bool = Field(default=True, description="Whether to save the last model.")
    optimizer_configs: OptimizerConfig = Field(default_factory=OptimizerConfig)
    artifact_storage: ArtifactStorageConfig = Field(
        default_factory=ArtifactStorageConfig,
        description="Configuration for artifact storage (local, S3, or GCS)"
    )

    @model_validator(mode="after")
    def _validate_split_and_temporal_flags(self) -> "TrainingConfig":
        if self.val_split + self.test_split >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")

        if self.temporal_split.enabled and self.shuffle:
            raise ValueError(
                "shuffle must be false when temporal_split.enabled is true to prevent leakage"
            )

        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


def validate_training_config_data(data: Mapping[str, Any]) -> TrainingConfig:
    """Validate a raw training config mapping and return typed config.

    Raises:
        ValueError: If the provided mapping fails schema validation.
    """
    try:
        return TrainingConfig.model_validate(dict(data))
    except ValidationError as exc:
        raise ValueError(f"Invalid training configuration: {exc}") from exc
