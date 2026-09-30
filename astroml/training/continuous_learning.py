"""Continuous learning orchestration, model versioning, and automated rollback."""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .incremental.adaptive_model import AdaptiveModel, AdaptiveModelConfig
from .incremental.online_learner import OnlineLearnerConfig
from .incremental.stream_trainer import StreamTrainer, StreamTrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelVersionMetadata:
    """Metadata for an incremental model version."""

    version: str
    created_at: float
    samples_trained: int
    metrics: dict[str, float]
    state_snapshot: dict[str, Any]
    description: str = ""
    is_healthy: bool = True


class ModelVersionManager:
    """Manager for versioning and storing model checkpoints with rollback support."""

    def __init__(self) -> None:
        self.versions: dict[str, ModelVersionMetadata] = {}
        self.version_history: list[str] = []
        self.active_version: str | None = None
        self._major = 1
        self._minor = 0
        self._patch = 0

    def _next_version_tag(self, bump: str = "patch") -> str:
        if bump == "major":
            self._major += 1
            self._minor = 0
            self._patch = 0
        elif bump == "minor":
            self._minor += 1
            self._patch = 0
        else:
            self._patch += 1
        return f"v{self._major}.{self._minor}.{self._patch}"

    def create_version(
        self,
        model: AdaptiveModel,
        metrics: dict[str, float],
        samples_trained: int,
        description: str = "",
        bump: str = "patch",
        is_healthy: bool = True,
    ) -> ModelVersionMetadata:
        """Create and store a new model version snapshot."""
        tag = self._next_version_tag(bump)
        metadata = ModelVersionMetadata(
            version=tag,
            created_at=time.time(),
            samples_trained=samples_trained,
            metrics=dict(metrics),
            state_snapshot=copy.deepcopy(model.get_state()),
            description=description,
            is_healthy=is_healthy,
        )
        self.versions[tag] = metadata
        self.version_history.append(tag)
        self.active_version = tag
        logger.info(f"Created model version {tag} (samples: {samples_trained})")
        return metadata

    def get_latest_healthy_version(self) -> ModelVersionMetadata | None:
        """Return the most recent version marked healthy."""
        for tag in reversed(self.version_history):
            ver = self.versions.get(tag)
            if ver and ver.is_healthy:
                return ver
        return None

    def rollback(self, model: AdaptiveModel, target_version: str | None = None) -> ModelVersionMetadata:
        """Rollback model state to a target version or latest healthy version."""
        if target_version is not None:
            if target_version not in self.versions:
                raise ValueError(f"Version {target_version} not found.")
            target_meta = self.versions[target_version]
        else:
            target_meta = self.get_latest_healthy_version()
            if target_meta is None:
                raise RuntimeError("No healthy version available for rollback.")

        model.load_state(copy.deepcopy(target_meta.state_snapshot))
        self.active_version = target_meta.version
        logger.warning(f"Rolled back model to version {target_meta.version}")
        return target_meta


@dataclass
class ContinuousLearningPipelineConfig:
    """Configuration for continuous learning pipeline."""

    adaptive_config: AdaptiveModelConfig = field(default_factory=AdaptiveModelConfig)
    trainer_config: StreamTrainerConfig = field(default_factory=StreamTrainerConfig)
    snapshot_interval_samples: int = 100
    degradation_threshold: float = 0.20  # Metric drop percentage triggering rollback
    auto_rollback: bool = True
    metric_to_monitor: str = "accuracy"


class ContinuousLearningPipeline:
    """End-to-end continuous learning coordinator with auto-checkpointing and rollback."""

    def __init__(
        self,
        config: ContinuousLearningPipelineConfig | None = None,
        model: AdaptiveModel | None = None,
    ) -> None:
        self.config = config or ContinuousLearningPipelineConfig()
        self.model = model or AdaptiveModel(self.config.adaptive_config)
        self.version_manager = ModelVersionManager()
        self.trainer = StreamTrainer(self.model, self.config.trainer_config)
        self.last_snapshot_sample: int = 0
        self.rollback_count: int = 0

        # Initial baseline version
        self.version_manager.create_version(
            self.model,
            metrics={self.config.metric_to_monitor: 1.0},
            samples_trained=0,
            description="Initial baseline model",
            is_healthy=True,
        )

    def process_stream(
        self,
        data_stream: Iterable[tuple[Sequence[float], Any]],
        on_batch_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Process streaming data with automated checkpointing and degradation rollback."""
        def batch_callback(record: dict[str, Any]) -> None:
            samples = record["samples_processed"]
            metrics = record["metrics"]
            current_metric = metrics.get(self.config.metric_to_monitor, 0.0)

            # Check for degradation compared to latest healthy checkpoint
            latest_healthy = self.version_manager.get_latest_healthy_version()
            if latest_healthy and self.config.auto_rollback:
                base_metric = latest_healthy.metrics.get(self.config.metric_to_monitor, 1.0)
                if base_metric > 0:
                    rel_drop = (base_metric - current_metric) / base_metric
                    if rel_drop > self.config.degradation_threshold and samples > 50:
                        logger.warning(
                            f"Degradation detected ({self.config.metric_to_monitor} dropped {rel_drop:.2%}). Initiating rollback."
                        )
                        self.version_manager.rollback(self.model)
                        self.rollback_count += 1
                        return

            # Checkpoint if interval reached
            if samples - self.last_snapshot_sample >= self.config.snapshot_interval_samples:
                self.version_manager.create_version(
                    self.model,
                    metrics=metrics,
                    samples_trained=samples,
                    description=f"Snapshot at {samples} samples",
                    is_healthy=(current_metric >= 0.5),
                )
                self.last_snapshot_sample = samples

            if on_batch_complete:
                on_batch_complete(record)

        result = self.trainer.train_stream(data_stream, on_batch_complete=batch_callback)
        result["rollback_count"] = self.rollback_count
        result["active_version"] = self.version_manager.active_version
        result["total_versions"] = len(self.version_manager.version_history)
        return result
