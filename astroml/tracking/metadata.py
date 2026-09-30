"""Model metadata container and schema specifications for the AstroML model registry.

Provides structured metadata tracking for ML framework types, training parameters,
input/output schemas, evaluation metrics, and arbitrary custom annotations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelFramework(str, Enum):
    """Supported machine learning frameworks."""

    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ONNX = "onnx"
    CUSTOM = "custom"


class TaskType(str, Enum):
    """Model task categories."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASSIFICATION = "multi_classification"
    REGRESSION = "regression"
    GRAPH_NODE_CLASSIFICATION = "graph_node_classification"
    LINK_PREDICTION = "link_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    EMBEDDING = "embedding"


@dataclass
class ModelMetadata:
    """Comprehensive metadata associated with a registered ML model version."""

    model_name: str
    framework: str = ModelFramework.CUSTOM.value
    task_type: str = TaskType.BINARY_CLASSIFICATION.value
    description: str = ""
    author: str = ""
    tags: list[str] = dc_field(default_factory=list)
    hyperparameters: dict[str, Any] = dc_field(default_factory=dict)
    metrics: dict[str, float] = dc_field(default_factory=dict)
    input_schema: dict[str, str] = dc_field(default_factory=dict)
    output_schema: dict[str, str] = dc_field(default_factory=dict)
    custom_properties: dict[str, Any] = dc_field(default_factory=dict)
    created_at: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Instantiate ModelMetadata from a dictionary."""
        known_fields = {
            "model_name",
            "framework",
            "task_type",
            "description",
            "author",
            "tags",
            "hyperparameters",
            "metrics",
            "input_schema",
            "output_schema",
            "custom_properties",
            "created_at",
            "updated_at",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def add_tag(self, tag: str) -> None:
        """Add a unique tag to the metadata."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the metadata."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now(timezone.utc).isoformat()
            return True
        return False

    def update_metrics(self, new_metrics: dict[str, float]) -> None:
        """Update or insert performance evaluation metrics."""
        self.metrics.update(new_metrics)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def set_property(self, key: str, value: Any) -> None:
        """Set a custom metadata key-value pair."""
        self.custom_properties[key] = value
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def validate(self) -> list[str]:
        """Validate metadata completeness and structure, returning error messages."""
        errors: list[str] = []
        if not self.model_name:
            errors.append("model_name must not be empty.")
        if not self.framework:
            errors.append("framework must be specified.")
        if not self.task_type:
            errors.append("task_type must be specified.")
        return errors
