"""Tests for ONNXOptimizer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroml.training.optimization.onnx_optimizer import (
    BenchmarkResult,
    ONNXOptimizer,
    OptimizationResult,
)


@pytest.fixture
def mock_onnx():
    with patch("astroml.training.optimization.onnx_optimizer.onnx") as mock:
        yield mock


@pytest.fixture
def mock_onnxoptimizer():
    with patch("astroml.training.optimization.onnx_optimizer.onnxoptimizer") as mock:
        yield mock


@pytest.fixture
def mock_session():
    with patch("astroml.training.optimization.onnx_optimizer.InferenceSession") as mock:
        yield mock


@pytest.fixture
def dummy_onnx_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"dummy onnx content")
    return path


@pytest.fixture
def mock_model_proto():
    model = MagicMock()
    model.graph.node = [MagicMock() for _ in range(10)]
    return model


class TestOptimizationResult:
    def test_size_reduction(self):
        result = OptimizationResult(
            original_size=1000,
            optimized_size=400,
            original_nodes=10,
            optimized_nodes=6,
            optimizations_applied=["fuse"],
        )
        assert result.size_reduction == pytest.approx(0.6)
        assert result.node_reduction == pytest.approx(0.4)

    def test_size_reduction_zero_original(self):
        result = OptimizationResult(
            original_size=0,
            optimized_size=0,
            original_nodes=0,
            optimized_nodes=0,
            optimizations_applied=[],
        )
        assert result.size_reduction == 0.0
        assert result.node_reduction == 0.0

    def test_to_dict(self):
        result = OptimizationResult(100, 50, 10, 5, ["opt1"])
        d = result.to_dict()
        assert d["original_size"] == 100
        assert d["optimized_size"] == 50
        assert d["optimizations_applied"] == ["opt1"]


class TestBenchmarkResult:
    def test_to_dict(self):
        result = BenchmarkResult(
            mean_latency=0.01, std_latency=0.001, min_latency=0.005, max_latency=0.02, n_runs=100
        )
        d = result.to_dict()
        assert d["mean_latency"] == 0.01
        assert d["n_runs"] == 100


class TestOptimize:
    def test_optimize_basic(
        self, mock_onnx, mock_onnxoptimizer, mock_model_proto, dummy_onnx_path: Path
    ):
        mock_onnx.load.return_value = mock_model_proto
        mock_onnxoptimizer.optimize.return_value = mock_model_proto

        result = ONNXOptimizer.optimize(dummy_onnx_path, optimization_level="basic")

        assert isinstance(result, OptimizationResult)
        mock_onnxoptimizer.optimize.assert_called_once()

    def test_optimize_extended(
        self, mock_onnx, mock_onnxoptimizer, mock_model_proto, dummy_onnx_path: Path
    ):
        mock_onnx.load.return_value = mock_model_proto
        mock_onnxoptimizer.optimize.return_value = mock_model_proto

        result = ONNXOptimizer.optimize(dummy_onnx_path, optimization_level="extended")

        assert isinstance(result, OptimizationResult)

    def test_optimize_all(
        self, mock_onnx, mock_onnxoptimizer, mock_model_proto, dummy_onnx_path: Path
    ):
        mock_onnx.load.return_value = mock_model_proto
        mock_onnxoptimizer.optimize.return_value = mock_model_proto

        result = ONNXOptimizer.optimize(dummy_onnx_path, optimization_level="all")

        assert isinstance(result, OptimizationResult)

    def test_optimize_invalid_level(self, dummy_onnx_path: Path):
        with pytest.raises(ValueError, match="Invalid optimization level"):
            ONNXOptimizer.optimize(dummy_onnx_path, optimization_level="invalid")

    def test_optimize_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            ONNXOptimizer.optimize("/nonexistent/model.onnx")

    def test_optimize_with_output_path(
        self, mock_onnx, mock_onnxoptimizer, mock_model_proto, tmp_path: Path
    ):
        input_path = tmp_path / "input.onnx"
        input_path.write_bytes(b"dummy")
        output_path = tmp_path / "output.onnx"

        def _save_side_effect(model, path):
            Path(path).write_bytes(b"optimized")

        mock_onnx.load.return_value = mock_model_proto
        mock_onnxoptimizer.optimize.return_value = mock_model_proto
        mock_onnx.save.side_effect = _save_side_effect

        ONNXOptimizer.optimize(input_path, output_path=output_path)

        mock_onnx.save.assert_called_once()
        saved_path = mock_onnx.save.call_args[0][1]
        assert Path(saved_path) == output_path


class TestFuseOps:
    def test_fuse_ops(self, mock_onnx, mock_onnxoptimizer, mock_model_proto, dummy_onnx_path: Path):
        mock_onnx.load.return_value = mock_model_proto
        mock_onnxoptimizer.optimize.return_value = mock_model_proto

        result = ONNXOptimizer.fuse_ops(dummy_onnx_path)

        assert isinstance(result, OptimizationResult)
        assert "fuse_matmul_add_bias_into_gemm" in result.optimizations_applied


class TestEliminateDeadNodes:
    def test_eliminate_dead_nodes(
        self, mock_onnx, mock_onnxoptimizer, mock_model_proto, dummy_onnx_path: Path
    ):
        mock_onnx.load.return_value = mock_model_proto
        mock_onnxoptimizer.optimize.return_value = mock_model_proto

        result = ONNXOptimizer.eliminate_dead_nodes(dummy_onnx_path)

        assert isinstance(result, OptimizationResult)
        assert "eliminate_deadend" in result.optimizations_applied


class TestConstantFolding:
    def test_constant_folding(
        self, mock_onnx, mock_onnxoptimizer, mock_model_proto, dummy_onnx_path: Path
    ):
        mock_onnx.load.return_value = mock_model_proto
        mock_onnxoptimizer.optimize.return_value = mock_model_proto

        result = ONNXOptimizer.constant_folding(dummy_onnx_path)

        assert isinstance(result, OptimizationResult)
        assert "extract_constant_to_initializer" in result.optimizations_applied


class TestOptimizeForCPU:
    def test_optimize_for_cpu(
        self, mock_onnx, mock_onnxoptimizer, mock_model_proto, dummy_onnx_path: Path
    ):
        mock_onnx.load.return_value = mock_model_proto
        mock_onnxoptimizer.optimize.return_value = mock_model_proto

        result = ONNXOptimizer.optimize_for_cpu(dummy_onnx_path)

        assert isinstance(result, OptimizationResult)
        assert "fuse_consecutive_squeezes" in result.optimizations_applied


class TestOptimizeForGPU:
    def test_optimize_for_gpu(
        self, mock_onnx, mock_onnxoptimizer, mock_model_proto, dummy_onnx_path: Path
    ):
        mock_onnx.load.return_value = mock_model_proto
        mock_onnxoptimizer.optimize.return_value = mock_model_proto

        result = ONNXOptimizer.optimize_for_gpu(dummy_onnx_path)

        assert isinstance(result, OptimizationResult)
        assert "fuse_bn_into_conv" in result.optimizations_applied


class TestBenchmark:
    def test_benchmark(self, mock_session, dummy_onnx_path: Path, tmp_path: Path):
        dummy_onnx_path.write_bytes(b"onnx")
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        input_data = {"input": np.array([[1.0, 2.0]], dtype=np.float32)}
        result = ONNXOptimizer.benchmark(dummy_onnx_path, input_data, n_runs=5, warmup_runs=2)

        assert isinstance(result, BenchmarkResult)
        assert result.n_runs == 5
        assert mock_session_instance.run.call_count == 7  # 2 warmup + 5 runs

    def test_benchmark_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            ONNXOptimizer.benchmark("/nonexistent/model.onnx", {"input": np.array([[1.0]])})
