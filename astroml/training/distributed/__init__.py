"""Distributed training with Ray and Horovod (issue #625).

Provides distributed training capabilities using Ray for hyperparameter
tuning and Horovod for efficient multi-GPU training with ring-allreduce
gradient synchronization.

Components:
- ClusterManager: Ray cluster lifecycle management
- RayTrainer: Distributed training orchestrator
- RayTuner: Hyperparameter tuning with Ray Tune
- HorovodTrainer: Multi-GPU training with Horovod
- DistributionStrategy: Data/model/hybrid/pipeline parallelism
"""

from __future__ import annotations

from .cluster import ClusterBackend, ClusterConfig, ClusterManager, NodeInfo, NodeStatus
from .horovod_trainer import (
    GradientReduction,
    GradientSynchronizer,
    HorovodBackend,
    HorovodConfig,
    HorovodTrainer,
)
from .ray_trainer import RayTrainer, RayTuneConfig, RayTuner, TrialResult, TuneResult
from .strategy import (
    DataParallelStrategy,
    DataShard,
    DistributionStrategy,
    HybridParallelStrategy,
    ModelParallelStrategy,
    ModelShard,
    ParallelismType,
    PipelineParallelStrategy,
    create_strategy,
)

__all__ = [
    # Cluster
    "ClusterBackend",
    "ClusterConfig",
    "ClusterManager",
    "NodeInfo",
    "NodeStatus",
    # Ray
    "RayTrainer",
    "RayTuneConfig",
    "RayTuner",
    "TrialResult",
    "TuneResult",
    # Horovod
    "GradientReduction",
    "GradientSynchronizer",
    "HorovodBackend",
    "HorovodConfig",
    "HorovodTrainer",
    # Strategies
    "DataParallelStrategy",
    "DataShard",
    "DistributionStrategy",
    "HybridParallelStrategy",
    "ModelParallelStrategy",
    "ModelShard",
    "ParallelismType",
    "PipelineParallelStrategy",
    "create_strategy",
]