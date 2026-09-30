"""Tests for distributed training with Ray and Horovod (#625)."""

from __future__ import annotations

import numpy as np
import pytest

from astroml.training import TrainingConfig
from astroml.training.distributed import (
    ClusterBackend,
    ClusterConfig,
    ClusterManager,
    DataParallelStrategy,
    DataShard,
    GradientReduction,
    GradientSynchronizer,
    HorovodConfig,
    HorovodTrainer,
    HybridParallelStrategy,
    ModelParallelStrategy,
    ModelShard,
    ParallelismType,
    PipelineParallelStrategy,
    RayTrainer,
    RayTuneConfig,
    RayTuner,
    create_strategy,
)

# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


class TestCluster:
    def test_cluster_initialization(self) -> None:
        config = ClusterConfig(num_workers=2, num_cpus_per_worker=4)
        manager = ClusterManager(config)
        manager.initialize()
        info = manager.get_cluster_info()
        assert info["backend"] == "ray"
        assert info["num_workers"] == 2
        # 1 head + 2 workers = 3 nodes
        assert info["resources"]["node_count"] == 3

    def test_health_check(self) -> None:
        manager = ClusterManager(ClusterConfig(num_workers=1))
        manager.initialize()
        statuses = manager.health_check()
        assert len(statuses) == 2  # head + 1 worker
        assert all(s.value == "healthy" for s in statuses.values())

    def test_shutdown(self) -> None:
        manager = ClusterManager(ClusterConfig())
        manager.initialize()
        manager.shutdown()
        assert manager.tracker.nodes == {}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class TestStrategies:
    def test_data_parallel_shards(self) -> None:
        strategy = DataParallelStrategy(shuffle=False)
        data = list(range(100))
        shards = strategy.get_shards(data, 4)
        assert len(shards) == 4
        # All samples present, shard sizes are balanced
        total = sum(s.local_batch_size for s in shards)
        assert total == 100
        assert all(abs(s.local_batch_size - 25) <= 1 for s in shards)

    def test_data_parallel_shuffle(self) -> None:
        strategy = DataParallelStrategy(shuffle=True, seed=42)
        data = list(range(100))
        shards = strategy.get_shards(data, 4)
        # With shuffle, first shard should not equal range(0, 25)
        first_shard = shards[0].data
        assert first_shard != list(range(25))

    def test_model_parallel_shards(self) -> None:
        strategy = ModelParallelStrategy(num_layers=12, devices=["cuda:0", "cuda:1"])
        model_shards = strategy.get_model_shards(4)
        assert len(model_shards) == 4
        total_layers = sum(
            s.layer_range[1] - s.layer_range[0] for s in model_shards
        )
        assert total_layers == 12
        # Layer ranges should be contiguous
        for i in range(3):
            assert model_shards[i].layer_range[1] == model_shards[i + 1].layer_range[0]

    def test_hybrid_parallel(self) -> None:
        dp = DataParallelStrategy()
        mp = ModelParallelStrategy(num_layers=8)
        hybrid = HybridParallelStrategy(dp, mp)
        assert hybrid.get_parallelism_type() == ParallelismType.HYBRID

    def test_pipeline_parallel(self) -> None:
        strategy = PipelineParallelStrategy(num_microbatches=4)
        assert strategy.get_parallelism_type() == ParallelismType.PIPELINE

    def test_create_strategy_factory(self) -> None:
        s = create_strategy(ParallelismType.DATA)
        assert isinstance(s, DataParallelStrategy)
        s = create_strategy(ParallelismType.MODEL, num_layers=6)
        assert isinstance(s, ModelParallelStrategy)

    def test_validation(self) -> None:
        strategy = DataParallelStrategy()
        issues = strategy.validate(num_workers=-1)
        assert len(issues) == 1
        issues = strategy.validate(num_workers=10, data_size=5)
        assert len(issues) == 1  # data smaller than workers


# ---------------------------------------------------------------------------
# Ray trainer / tuner
# ---------------------------------------------------------------------------


class TestRay:
    def test_ray_trainer_initialization(self) -> None:
        tcfg = TrainingConfig(epochs=10, lr=0.01)
        trainer = RayTrainer(tcfg, ClusterConfig(num_workers=1))
        trainer.initialize()
        info = trainer.cluster_manager.get_cluster_info()
        assert info["resources"]["node_count"] == 2  # head + 1 worker

    def test_ray_trainer_train(self) -> None:
        tcfg = TrainingConfig(epochs=5, lr=0.01)
        trainer = RayTrainer(tcfg, ClusterConfig(num_workers=2))
        trainer.initialize()
        data = list(range(100))
        result = trainer.train_distributed(data)
        assert "avg_loss" in result
        assert "worker_results" in result
        assert result["num_workers"] == 2

    def test_ray_tuner(self) -> None:
        def objective(cfg: dict) -> float:
            return (cfg["lr"] - 0.01) ** 2 + np.random.random() * 0.01

        tune_cfg = RayTuneConfig(
            num_samples=5,
            param_space={"lr": [1e-5, 1e-1]},
        )
        tuner = RayTuner(tune_cfg, objective)
        result = tuner.run()
        assert result.best_metric_value is not None
        assert len(result.all_trials) == 5
        assert result.best_config is not None


# ---------------------------------------------------------------------------
# Horovod
# ---------------------------------------------------------------------------


class TestHorovod:
    def test_horovod_trainer_setup(self) -> None:
        tcfg = TrainingConfig(epochs=5, lr=0.01)
        hcfg = HorovodConfig(num_processes=4, backend="nccl")
        trainer = HorovodTrainer(tcfg, hcfg)
        trainer.initialize(rank=0, size=4)
        assert trainer.rank == 0
        assert trainer.size == 4

    def test_gradient_synchronizer_ring_allreduce(self) -> None:
        sync = GradientSynchronizer(HorovodConfig(gradient_reduction="ring_allreduce"))
        grads = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        synced = sync.allreduce(grads, worker_id=0, num_workers=4)
        assert synced[0].shape == (3,)

    def test_gradient_synchronizer_simple(self) -> None:
        sync = GradientSynchronizer(HorovodConfig(gradient_reduction="allreduce"))
        grads = [np.array([4.0, 4.0], dtype=np.float32)]
        synced = sync.allreduce(grads, num_workers=2)
        np.testing.assert_array_almost_equal(synced[0], np.array([2.0, 2.0]))

    def test_horovod_train(self) -> None:
        tcfg = TrainingConfig(epochs=3, lr=0.01)
        trainer = HorovodTrainer(tcfg, HorovodConfig(num_processes=2))
        trainer.initialize(rank=0, size=2)
        data = list(range(50))
        result = trainer.train(data)
        assert "loss" in result
        assert "loss_history" in result
        assert len(result["loss_history"]) == 3

    def test_horovod_all_ranks(self) -> None:
        tcfg = TrainingConfig(epochs=2, lr=0.01)
        trainer = HorovodTrainer(tcfg, HorovodConfig(num_processes=3))
        trainer.initialize(rank=0, size=3)
        data = list(range(60))
        result = trainer.run_all_ranks(data)
        assert result["num_ranks"] == 3
        assert "avg_loss" in result

    def test_horovod_benchmark(self) -> None:
        trainer = HorovodTrainer(
            TrainingConfig(),
            HorovodConfig(num_processes=2),
        )
        bm = trainer.benchmark(num_steps=10, data_size_per_gpu=512)
        assert bm["num_gpus"] == 1  # baseline before init
        assert bm["throughput_samples_per_sec"] > 0