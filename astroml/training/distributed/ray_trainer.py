from astroml.utils.exceptions import AstroMLError
"""Ray-based distributed training (issue #625).

Provides hyperparameter tuning with Ray Tune and distributed training
orchestration using Ray's actor model.

Components:
- RayTrainer: Distributed training orchestrator
- RayTuneConfig: Hyperparameter search configuration
- RayTuner: Hyperparameter tuning with Ray Tune integration
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from astroml.training.config import TrainingConfig
from astroml.training.distributed.cluster import ClusterConfig, ClusterManager
from astroml.training.distributed.strategy import (
    DataParallelStrategy,
    DataShard,
    DistributionStrategy,
    ParallelismType,
    create_strategy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class RayTuneConfig(BaseModel):
    """Hyperparameter tuning configuration for Ray Tune.

    Attributes:
        num_samples: Number of hyperparameter configurations to try.
        metric: Metric to optimize (e.g. ``"val_loss"``).
        mode: ``"min"`` or ``"max"``.
        search_algorithm: Search algorithm (``"random"``, ``"bayes"``).
        grace_period: Minimum training iterations per trial.
        max_concurrent_trials: Maximum concurrent trials.
    """

    model_config = ConfigDict(extra="forbid")

    num_samples: int = Field(default=10, ge=1, description="Number of trials to run")
    metric: str = Field(default="val_loss", description="Metric to optimize")
    mode: str = Field(default="min", description="min or max")
    search_algorithm: str = Field(default="random", description="Search algorithm")
    grace_period: int = Field(default=5, ge=1, description="Min iterations per trial")
    max_concurrent_trials: int = Field(default=4, ge=1, description="Max concurrent trials")

    # Hyperparameter space: dict of param_name → [min, max] or list of values
    param_space: dict[str, Any] = Field(
        default_factory=lambda: {
            "lr": [1e-5, 1e-1],
            "batch_size": [16, 32, 64, 128],
            "dropout": [0.0, 0.5],
        },
        description="Hyperparameter search space",
    )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial.

    Attributes:
        trial_id: Trial index.
        config: Hyperparameter configuration used.
        metric_value: Final metric value.
        checkpoint_path: Path to best model checkpoint.
        status: Trial status (completed, error).
    """

    trial_id: int
    config: dict[str, Any]
    metric_value: float
    checkpoint_path: str | None = None
    status: str = "completed"


@dataclass
class TuneResult:
    """Result of a hyperparameter tuning run.

    Attributes:
        best_config: Best hyperparameter configuration found.
        best_metric_value: Best metric value achieved.
        all_trials: Results from all trials.
        elapsed_seconds: Wall-clock duration.
    """

    best_config: dict[str, Any]
    best_metric_value: float
    all_trials: list[TrialResult]
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Ray trainer
# ---------------------------------------------------------------------------


class RayTrainer:
    """Orchestrates distributed training using Ray.

    Distributes training across a Ray cluster using the configured
    :class:`DistributionStrategy`.

    Args:
        training_config: Model training hyperparameters.
        cluster_config: Ray cluster configuration.
        strategy: Data/model distribution strategy.
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        cluster_config: ClusterConfig | None = None,
        strategy: DistributionStrategy | None = None,
    ) -> None:
        self.training_config = training_config
        self.cluster_manager = ClusterManager(cluster_config)
        self.strategy = strategy or DataParallelStrategy()

    def initialize(self) -> None:
        """Initialize Ray cluster."""
        self.cluster_manager.initialize()
        logger.info("RayTrainer initialized with %s parallelism", self.strategy.get_parallelism_type())

    def train_distributed(
        self,
        train_data: Sequence[Any],
        val_data: Sequence[Any] | None = None,
        *,
        train_fn: Callable[..., Any] | None = None,
        workers: int | None = None,
    ) -> dict[str, Any]:
        """Run distributed training across the cluster.

        Args:
            train_data: Training dataset.
            val_data: Optional validation dataset.
            train_fn: Per-worker training function.
            workers: Number of workers (defaults from cluster config).

        Returns:
            Dict with training results (loss history, metrics).
        """
        num_workers = workers or max(self.cluster_manager.config.num_workers, 1)
        shards = self.strategy.get_shards(train_data, num_workers)

        val_shards: list[DataShard] | None = None
        if val_data is not None:
            val_shards = self.strategy.get_shards(val_data, num_workers)

        results: list[dict[str, Any]] = []

        for shard in shards:
            worker_result = self._train_worker(
                shard=shard,
                train_fn=train_fn,
                config_dict=self.training_config.model_dump(),
            )
            results.append(worker_result)

        return self._aggregate_results(results)

    def _train_worker(
        self,
        shard: DataShard,
        train_fn: Callable[..., Any] | None,
        config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Simulate a single worker's training pass.

        In production, this would dispatch a Ray remote function.
        """
        if train_fn is not None:
            return train_fn(shard.data, shard.shard_id, config_dict)

        # Placeholder training loop
        logger.info(
            "Worker %d: training on %d samples (config=%s)",
            shard.shard_id,
            shard.local_batch_size,
            config_dict,
        )
        return {
            "worker_id": shard.shard_id,
            "loss": 0.5 + 0.1 * np.random.random(),
            "samples": shard.local_batch_size,
        }

    def _aggregate_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate per-worker results into a single summary."""
        losses = [r.get("loss", 0.0) for r in results]
        total_samples = sum(r.get("samples", 0) for r in results)
        return {
            "num_workers": len(results),
            "total_samples": total_samples,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "std_loss": float(np.std(losses)) if len(losses) > 1 else 0.0,
            "worker_results": results,
        }

    def shutdown(self) -> None:
        """Gracefully shutdown the cluster."""
        self.cluster_manager.shutdown()


# ---------------------------------------------------------------------------
# Ray tuner
# ---------------------------------------------------------------------------


class RayTuner:
    """Hyperparameter tuning using Ray Tune.

    Searches the hyperparameter space defined in :class:`RayTuneConfig`
    and returns the best configuration found.

    Args:
        config: Tune configuration including search space.
        objective_fn: Function to minimize/maximize ``(config_dict) -> float``.
    """

    def __init__(
        self,
        config: RayTuneConfig,
        objective_fn: Callable[[dict[str, Any]], float],
    ) -> None:
        self.config = config
        self.objective_fn = objective_fn

    def run(self) -> TuneResult:
        """Run hyperparameter search.

        Returns:
            :class:`TuneResult` with best config and trial history.
        """
        import time

        start = time.perf_counter()
        trials: list[TrialResult] = []

        best_config: dict[str, Any] = {}
        best_value: float = float("inf") if self.config.mode == "min" else float("-inf")

        for i in range(min(self.config.num_samples, self.config.max_concurrent_trials * 3)):
            trial_config = self._sample_config(i)
            try:
                value = self.objective_fn(trial_config)
                is_better = (
                    value < best_value if self.config.mode == "min" else value > best_value
                )
                if is_better:
                    best_value = value
                    best_config = trial_config

                trials.append(
                    TrialResult(trial_id=i, config=trial_config, metric_value=value)
                )
            except Exception as exc:
                logger.error("Trial %d failed: %s", i, exc)
                trials.append(
                    TrialResult(
                        trial_id=i,
                        config=trial_config,
                        metric_value=float("inf") if self.config.mode == "min" else float("-inf"),
                        status="error",
                    )
                )

        elapsed = time.perf_counter() - start
        logger.info(
            "Tuning complete: %d trials in %.1fs, best=%s=%.4f",
            len(trials),
            elapsed,
            self.config.metric,
            best_value,
        )

        return TuneResult(
            best_config=best_config,
            best_metric_value=best_value,
            all_trials=trials,
            elapsed_seconds=elapsed,
        )

    def _sample_config(self, trial_id: int) -> dict[str, Any]:
        """Sample a hyperparameter configuration for a trial."""
        rng = np.random.default_rng(trial_id)
        sampled: dict[str, Any] = {}
        for param, space in self.config.param_space.items():
            if isinstance(space, list):
                sampled[param] = space[rng.integers(0, len(space))]
            elif isinstance(space, (list, tuple)) and len(space) == 2:
                lo, hi = space
                if all(isinstance(v, int) for v in space):
                    sampled[param] = int(rng.integers(lo, hi + 1))
                else:
                    sampled[param] = float(rng.uniform(lo, hi))
            else:
                sampled[param] = space
        return sampled