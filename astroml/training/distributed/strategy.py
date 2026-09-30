"""Distribution strategies for training (issue #625).

Implements data-parallel, model-parallel, and hybrid distribution strategies
that can be composed with Ray or Horovod backends.

Components:
- DistributionStrategy: Abstract base for distribution approaches
- DataParallelStrategy: Replicate model, split data across workers
- ModelParallelStrategy: Shard large models across devices
- HybridParallelStrategy: Combined data + model parallelism
- PipelineParallelStrategy: Sequential stage-based pipeline parallelism
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np

from astroml.training.distributed.cluster import ClusterBackend, ClusterConfig

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel")
TData = TypeVar("TData")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ParallelismType(str, Enum):
    """Supported parallelism strategies."""

    DATA = "data"
    MODEL = "model"
    HYBRID = "hybrid"
    PIPELINE = "pipeline"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DataShard(Generic[TData]):
    """A shard of the global dataset assigned to one worker.

    Attributes:
        shard_id: Zero-based shard index.
        data: The shard data.
        global_batch_size: Total batch size across all shards.
        local_batch_size: Per-shard batch size.
    """

    shard_id: int
    data: TData
    global_batch_size: int
    local_batch_size: int


@dataclass
class ModelShard:
    """A portion of a large model assigned to one device.

    Attributes:
        shard_id: Zero-based shard index.
        layer_range: (start, end) layer indices (inclusive-exclusive).
        device: Device identifier (e.g. ``"cuda:0"``).
    """

    shard_id: int
    layer_range: tuple[int, int]
    device: str


# ---------------------------------------------------------------------------
# Distribution strategy base
# ---------------------------------------------------------------------------


class DistributionStrategy(ABC, Generic[TData]):
    """Abstract base for distribution strategies."""

    @abstractmethod
    def get_shards(self, data: TData, num_workers: int) -> list[DataShard[TData]]:
        """Split data into shards for parallel processing.

        Args:
            data: Full training dataset.
            num_workers: Number of parallel workers.

        Returns:
            List of :class:`DataShard` instances, one per worker.
        """
        ...

    @abstractmethod
    def get_parallelism_type(self) -> ParallelismType:
        """Return the parallelism type implemented by this strategy."""
        ...

    def validate(self, num_workers: int, data_size: int | None = None) -> list[str]:
        """Validate that the strategy is applicable.

        Returns:
            List of warning / error messages (empty = valid).
        """
        issues: list[str] = []
        if num_workers < 1:
            issues.append("num_workers must be >= 1")
        if data_size is not None and data_size < num_workers:
            issues.append(f"Data size ({data_size}) smaller than workers ({num_workers})")
        return issues


# ---------------------------------------------------------------------------
# Data parallelism
# ---------------------------------------------------------------------------


class DataParallelStrategy(DistributionStrategy[Sequence[Any]]):
    """Splits data evenly across workers; each holds a full model replica.

    Args:
        shuffle: Whether to shuffle data before sharding.
        seed: Random seed for reproducibility.
    """

    def __init__(self, shuffle: bool = True, seed: int = 42) -> None:
        self.shuffle = shuffle
        self.seed = seed

    def get_shards(
        self,
        data: Sequence[Any],
        num_workers: int,
    ) -> list[DataShard[Sequence[Any]]]:
        n = len(data)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        base_size = n // num_workers
        remainder = n % num_workers

        shards: list[DataShard[Sequence[Any]]] = []
        offset = 0
        for i in range(num_workers):
            shard_size = base_size + (1 if i < remainder else 0)
            shard_indices = indices[offset : offset + shard_size]
            offset += shard_size
            shard_data = [data[idx] for idx in shard_indices]
            shards.append(
                DataShard(
                    shard_id=i,
                    data=shard_data,
                    global_batch_size=n,
                    local_batch_size=shard_size,
                )
            )

        logger.info(
            "Data-parallel sharding: %d samples → %d workers (min=%d, max=%d)",
            n,
            num_workers,
            min(s.local_batch_size for s in shards),
            max(s.local_batch_size for s in shards),
        )
        return shards

    def get_parallelism_type(self) -> ParallelismType:
        return ParallelismType.DATA


# ---------------------------------------------------------------------------
# Model parallelism
# ---------------------------------------------------------------------------


class ModelParallelStrategy(DistributionStrategy[Sequence[Any]]):
    """Shards a large model across multiple devices.

    Args:
        num_layers: Total number of model layers to shard.
        devices: List of device identifiers (e.g. ``["cuda:0", "cuda:1"]``).
    """

    def __init__(
        self,
        num_layers: int,
        devices: list[str] | None = None,
    ) -> None:
        self.num_layers = num_layers
        self.devices = devices or _auto_devices()

    def get_shards(
        self,
        data: Sequence[Any],
        num_workers: int,
    ) -> list[DataShard[Sequence[Any]]]:
        # Model parallelism doesn't shard data — all workers see all data
        n = len(data)
        return [
            DataShard(
                shard_id=i,
                data=data,
                global_batch_size=n,
                local_batch_size=n,
            )
            for i in range(num_workers)
        ]

    def get_model_shards(self, num_workers: int | None = None) -> list[ModelShard]:
        """Return model layer assignment across devices.

        Args:
            num_workers: Number of workers (defaults to len(devices)).

        Returns:
            List of :class:`ModelShard` with layer ranges.
        """
        n = num_workers if num_workers is not None else len(self.devices)
        base = self.num_layers // n
        remainder = self.num_layers % n
        shards: list[ModelShard] = []
        start = 0
        for i in range(n):
            size = base + (1 if i < remainder else 0)
            end = start + size
            device = self.devices[i % len(self.devices)]
            shards.append(ModelShard(shard_id=i, layer_range=(start, end), device=device))
            start = end
        return shards

    def get_parallelism_type(self) -> ParallelismType:
        return ParallelismType.MODEL

    def validate(self, num_workers: int, data_size: int | None = None) -> list[str]:
        issues = super().validate(num_workers, data_size)
        if self.num_layers < num_workers:
            issues.append(f"Layers ({self.num_layers}) < workers ({num_workers})")
        return issues


# ---------------------------------------------------------------------------
# Hybrid parallelism
# ---------------------------------------------------------------------------


class HybridParallelStrategy(DistributionStrategy[Sequence[Any]]):
    """Combines data and model parallelism.

    Data is sharded across *data_workers*; the model is sharded across
    *model_workers* within each data shard.

    Args:
        data_parallel: Data-parallel sharding strategy.
        model_parallel: Model-parallel sharding strategy.
    """

    def __init__(
        self,
        data_parallel: DataParallelStrategy,
        model_parallel: ModelParallelStrategy,
    ) -> None:
        self.data_parallel = data_parallel
        self.model_parallel = model_parallel

    def get_shards(
        self,
        data: Sequence[Any],
        num_workers: int,
    ) -> list[DataShard[Sequence[Any]]]:
        return self.data_parallel.get_shards(data, num_workers)

    def get_model_shards(self, num_workers: int | None = None) -> list[ModelShard]:
        return self.model_parallel.get_model_shards(num_workers)

    def get_parallelism_type(self) -> ParallelismType:
        return ParallelismType.HYBRID

    def validate(self, num_workers: int, data_size: int | None = None) -> list[str]:
        return self.data_parallel.validate(num_workers, data_size) + self.model_parallel.validate(
            num_workers, data_size
        )


# ---------------------------------------------------------------------------
# Pipeline parallelism
# ---------------------------------------------------------------------------


class PipelineParallelStrategy(DistributionStrategy[Sequence[Any]]):
    """Sequential stage-based pipeline parallelism.

    Each worker owns a contiguous chunk of model layers and processes
    micro-batches in a pipelined fashion.

    Args:
        num_microbatches: Number of micro-batches to split each batch into.
        num_stages: Number of pipeline stages (defaults to num_workers).
    """

    def __init__(
        self,
        num_microbatches: int = 4,
        num_stages: int | None = None,
    ) -> None:
        self.num_microbatches = num_microbatches
        self.num_stages = num_stages

    def get_shards(
        self,
        data: Sequence[Any],
        num_workers: int,
    ) -> list[DataShard[Sequence[Any]]]:
        n = len(data)
        micro = min(self.num_microbatches, n)
        micro_size = n // micro

        base_size = micro_size // num_workers
        shards: list[DataShard[Sequence[Any]]] = []
        for i in range(num_workers):
            shards.append(
                DataShard(
                    shard_id=i,
                    data=data,
                    global_batch_size=n,
                    local_batch_size=base_size,
                )
            )
        return shards

    def get_parallelism_type(self) -> ParallelismType:
        return ParallelismType.PIPELINE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_devices() -> list[str]:
    """Auto-detect available devices."""
    try:
        import torch

        n = torch.cuda.device_count()
        if n > 0:
            return [f"cuda:{i}" for i in range(n)]
    except ImportError:
        pass
    return ["cpu"]


def create_strategy(
    parallelism_type: ParallelismType,
    **kwargs: Any,
) -> DistributionStrategy:
    """Factory for creating a distribution strategy.

    Args:
        parallelism_type: The type of parallelism desired.
        **kwargs: Forwarded to the strategy constructor.

    Returns:
        A concrete :class:`DistributionStrategy` instance.
    """
    strategies: dict[ParallelismType, type[DistributionStrategy]] = {
        ParallelismType.DATA: DataParallelStrategy,
        ParallelismType.MODEL: ModelParallelStrategy,
        ParallelismType.HYBRID: HybridParallelStrategy,
        ParallelismType.PIPELINE: PipelineParallelStrategy,
    }
    cls = strategies.get(parallelism_type)
    if cls is None:
        raise ValueError(f"Unknown distribution strategy: {parallelism_type}")
    return cls(**kwargs)