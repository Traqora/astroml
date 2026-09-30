"""ONNX model optimizer using onnxruntime and onnxoptimizer."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

try:
    import onnx
except ImportError:
    onnx = None  # type: ignore[assignment]

try:
    import onnxoptimizer
except ImportError:
    onnxoptimizer = None  # type: ignore[assignment]

try:
    from onnxruntime import InferenceSession
except ImportError:
    InferenceSession = None  # type: ignore[assignment,misc]

OptimizationLevel = Literal["basic", "extended", "all"]


class OptimizationResult:
    """Result of an ONNX optimization."""

    def __init__(
        self,
        original_size: int,
        optimized_size: int,
        original_nodes: int,
        optimized_nodes: int,
        optimizations_applied: list[str],
    ) -> None:
        self.original_size = original_size
        self.optimized_size = optimized_size
        self.original_nodes = original_nodes
        self.optimized_nodes = optimized_nodes
        self.optimizations_applied = optimizations_applied

    @property
    def size_reduction(self) -> float:
        """Return the size reduction as a fraction (0.0 to 1.0)."""
        if self.original_size == 0:
            return 0.0
        return 1.0 - (self.optimized_size / self.original_size)

    @property
    def node_reduction(self) -> float:
        """Return the node count reduction as a fraction (0.0 to 1.0)."""
        if self.original_nodes == 0:
            return 0.0
        return 1.0 - (self.optimized_nodes / self.original_nodes)

    def to_dict(self) -> dict[str, Any]:
        """Return result as a dictionary."""
        return {
            "original_size": self.original_size,
            "optimized_size": self.optimized_size,
            "original_nodes": self.original_nodes,
            "optimized_nodes": self.optimized_nodes,
            "size_reduction": self.size_reduction,
            "node_reduction": self.node_reduction,
            "optimizations_applied": self.optimizations_applied,
        }


class BenchmarkResult:
    """Result of a benchmark run."""

    def __init__(
        self,
        mean_latency: float,
        std_latency: float,
        min_latency: float,
        max_latency: float,
        n_runs: int,
    ) -> None:
        self.mean_latency = mean_latency
        self.std_latency = std_latency
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.n_runs = n_runs

    def to_dict(self) -> dict[str, Any]:
        """Return result as a dictionary."""
        return {
            "mean_latency": self.mean_latency,
            "std_latency": self.std_latency,
            "min_latency": self.min_latency,
            "max_latency": self.max_latency,
            "n_runs": self.n_runs,
        }


_OPTIMIZATION_LEVELS: dict[OptimizationLevel, list[str]] = {
    "basic": [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_transpose",
        "eliminate_nop_pad",
        "eliminate_unused_initializer",
        "extract_constant_to_initializer",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "fuse_transpose_into_gemm",
    ],
    "extended": [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_transpose",
        "eliminate_nop_pad",
        "eliminate_unused_initializer",
        "extract_constant_to_initializer",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "fuse_transpose_into_gemm",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_relu",
        "fuse_bn_into_conv",
    ],
    "all": [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_transpose",
        "eliminate_nop_pad",
        "eliminate_unused_initializer",
        "extract_constant_to_initializer",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "fuse_transpose_into_gemm",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_relu",
        "fuse_bn_into_conv",
        "fuse_bn_add_bn_into_conv",
        "fuse_consecutive_log_softmax",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_reshape",
        "eliminate_duplicate_initializer",
    ],
}


_ONNX_PATH_RE = re.compile(r"^[^\x00]+\.onnx$", re.IGNORECASE)


def _validate_onnx_path(path: Path) -> None:
    """Validate that a path refers to an ONNX file without suspicious components.

    Args:
        path: Resolved absolute path to validate.

    Raises:
        ValueError: If the path fails validation.
    """
    if not _ONNX_PATH_RE.match(path.name):
        msg = f"Invalid ONNX filename: {path.name!r}"
        raise ValueError(msg)


class ONNXOptimizer:
    """Optimize ONNX models using graph transformations."""

    @staticmethod
    def _load_model(onnx_path: str | Path) -> Any:
        onnx_path = Path(onnx_path).resolve()
        _validate_onnx_path(onnx_path)
        if not onnx_path.exists():
            msg = f"ONNX file not found: {onnx_path}"
            raise FileNotFoundError(msg)
        return onnx.load(str(onnx_path))

    @staticmethod
    def _save_model(model: Any, onnx_path: str | Path) -> None:
        onnx_path = Path(onnx_path).resolve()
        _validate_onnx_path(onnx_path)
        onnx.save(model, str(onnx_path))

    @staticmethod
    def optimize(
        onnx_path: str | Path,
        optimization_level: OptimizationLevel = "extended",
        output_path: str | Path | None = None,
    ) -> OptimizationResult:
        """Apply graph optimizations to an ONNX model.

        Args:
            onnx_path: Path to the ONNX model file.
            optimization_level: Level of optimization: "basic", "extended", or "all".
            output_path: Optional output path; if None, overwrites the input.

        Returns:
            OptimizationResult with before/after metrics.

        Raises:
            FileNotFoundError: If the ONNX file does not exist.
            ValueError: If the optimization level is invalid.
        """
        if optimization_level not in _OPTIMIZATION_LEVELS:
            msg = f"Invalid optimization level: {optimization_level}. Choose from: basic, extended, all."
            raise ValueError(msg)

        onnx_path = Path(onnx_path).resolve()
        output_path = Path(output_path).resolve() if output_path else onnx_path

        model = ONNXOptimizer._load_model(onnx_path)
        original_nodes = len(model.graph.node)
        original_size = onnx_path.stat().st_size

        passes = _OPTIMIZATION_LEVELS[optimization_level]
        optimized_model = onnxoptimizer.optimize(model, passes)

        ONNXOptimizer._save_model(optimized_model, output_path)
        optimized_size = output_path.stat().st_size
        optimized_nodes = len(optimized_model.graph.node)

        logger.info(
            "Applied %d optimizations at level %r: %d -> %d nodes, %d -> %d bytes",
            len(passes),
            optimization_level,
            original_nodes,
            optimized_nodes,
            original_size,
            optimized_size,
        )

        return OptimizationResult(
            original_size=original_size,
            optimized_size=optimized_size,
            original_nodes=original_nodes,
            optimized_nodes=optimized_nodes,
            optimizations_applied=passes,
        )

    @staticmethod
    def fuse_ops(
        onnx_path: str | Path,
        output_path: str | Path | None = None,
    ) -> OptimizationResult:
        """Fuse operations in the ONNX model (e.g., Conv+BN, MatMul+Add).

        Args:
            onnx_path: Path to the ONNX model file.
            output_path: Optional output path; if None, overwrites the input.

        Returns:
            OptimizationResult with before/after metrics.
        """
        fusions = [
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_relu",
            "fuse_bn_into_conv",
            "fuse_bn_add_bn_into_conv",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
        ]

        onnx_path = Path(onnx_path).resolve()
        output_path = Path(output_path).resolve() if output_path else onnx_path

        model = ONNXOptimizer._load_model(onnx_path)
        original_nodes = len(model.graph.node)
        original_size = onnx_path.stat().st_size

        optimized_model = onnxoptimizer.optimize(model, fusions)

        ONNXOptimizer._save_model(optimized_model, output_path)
        optimized_size = output_path.stat().st_size
        optimized_nodes = len(optimized_model.graph.node)

        return OptimizationResult(
            original_size=original_size,
            optimized_size=optimized_size,
            original_nodes=original_nodes,
            optimized_nodes=optimized_nodes,
            optimizations_applied=fusions,
        )

    @staticmethod
    def eliminate_dead_nodes(
        onnx_path: str | Path,
        output_path: str | Path | None = None,
    ) -> OptimizationResult:
        """Remove unused nodes from the ONNX model.

        Args:
            onnx_path: Path to the ONNX model file.
            output_path: Optional output path; if None, overwrites the input.

        Returns:
            OptimizationResult with before/after metrics.
        """
        dead_passes = [
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
            "eliminate_unused_initializer",
        ]

        onnx_path = Path(onnx_path).resolve()
        output_path = Path(output_path).resolve() if output_path else onnx_path

        model = ONNXOptimizer._load_model(onnx_path)
        original_nodes = len(model.graph.node)
        original_size = onnx_path.stat().st_size

        optimized_model = onnxoptimizer.optimize(model, dead_passes)

        ONNXOptimizer._save_model(optimized_model, output_path)
        optimized_size = output_path.stat().st_size
        optimized_nodes = len(optimized_model.graph.node)

        return OptimizationResult(
            original_size=original_size,
            optimized_size=optimized_size,
            original_nodes=original_nodes,
            optimized_nodes=optimized_nodes,
            optimizations_applied=dead_passes,
        )

    @staticmethod
    def constant_folding(
        onnx_path: str | Path,
        output_path: str | Path | None = None,
    ) -> OptimizationResult:
        """Fold constant computations in the ONNX model.

        Args:
            onnx_path: Path to the ONNX model file.
            output_path: Optional output path; if None, overwrites the input.

        Returns:
            OptimizationResult with before/after metrics.
        """
        fold_passes = [
            "extract_constant_to_initializer",
            "eliminate_duplicate_initializer",
            "eliminate_unused_initializer",
        ]

        onnx_path = Path(onnx_path).resolve()
        output_path = Path(output_path).resolve() if output_path else onnx_path

        model = ONNXOptimizer._load_model(onnx_path)
        original_nodes = len(model.graph.node)
        original_size = onnx_path.stat().st_size

        optimized_model = onnxoptimizer.optimize(model, fold_passes)

        ONNXOptimizer._save_model(optimized_model, output_path)
        optimized_size = output_path.stat().st_size
        optimized_nodes = len(optimized_model.graph.node)

        return OptimizationResult(
            original_size=original_size,
            optimized_size=optimized_size,
            original_nodes=original_nodes,
            optimized_nodes=optimized_nodes,
            optimizations_applied=fold_passes,
        )

    @staticmethod
    def optimize_for_cpu(
        onnx_path: str | Path,
        output_path: str | Path | None = None,
    ) -> OptimizationResult:
        """Apply CPU-specific optimizations to an ONNX model.

        Args:
            onnx_path: Path to the ONNX model file.
            output_path: Optional output path; if None, overwrites the input.

        Returns:
            OptimizationResult with before/after metrics.
        """
        cpu_passes = [
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_relu",
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
        ]

        onnx_path = Path(onnx_path).resolve()
        output_path = Path(output_path).resolve() if output_path else onnx_path

        model = ONNXOptimizer._load_model(onnx_path)
        original_nodes = len(model.graph.node)
        original_size = onnx_path.stat().st_size

        optimized_model = onnxoptimizer.optimize(model, cpu_passes)

        ONNXOptimizer._save_model(optimized_model, output_path)
        optimized_size = output_path.stat().st_size
        optimized_nodes = len(optimized_model.graph.node)

        return OptimizationResult(
            original_size=original_size,
            optimized_size=optimized_size,
            original_nodes=original_nodes,
            optimized_nodes=optimized_nodes,
            optimizations_applied=cpu_passes,
        )

    @staticmethod
    def optimize_for_gpu(
        onnx_path: str | Path,
        output_path: str | Path | None = None,
    ) -> OptimizationResult:
        """Apply GPU-specific optimizations to an ONNX model.

        Args:
            onnx_path: Path to the ONNX model file.
            output_path: Optional output path; if None, overwrites the input.

        Returns:
            OptimizationResult with before/after metrics.
        """
        gpu_passes = [
            "fuse_bn_into_conv",
            "fuse_bn_add_bn_into_conv",
            "fuse_pad_into_conv",
            "fuse_relu",
            "eliminate_deadend",
            "eliminate_identity",
        ]

        onnx_path = Path(onnx_path).resolve()
        output_path = Path(output_path).resolve() if output_path else onnx_path

        model = ONNXOptimizer._load_model(onnx_path)
        original_nodes = len(model.graph.node)
        original_size = onnx_path.stat().st_size

        optimized_model = onnxoptimizer.optimize(model, gpu_passes)

        ONNXOptimizer._save_model(optimized_model, output_path)
        optimized_size = output_path.stat().st_size
        optimized_nodes = len(optimized_model.graph.node)

        return OptimizationResult(
            original_size=original_size,
            optimized_size=optimized_size,
            original_nodes=original_nodes,
            optimized_nodes=optimized_nodes,
            optimizations_applied=gpu_passes,
        )

    @staticmethod
    def benchmark(
        onnx_path: str | Path,
        input_data: dict[str, Any],
        n_runs: int = 100,
        warmup_runs: int = 10,
    ) -> BenchmarkResult:
        """Benchmark ONNX model inference time.

        Args:
            onnx_path: Path to the ONNX model file.
            input_data: Dict mapping input names to numpy arrays.
            n_runs: Number of inference runs for timing.
            warmup_runs: Number of warmup runs before timing.

        Returns:
            BenchmarkResult with latency statistics.

        Raises:
            FileNotFoundError: If the ONNX file does not exist.
        """
        import numpy as np

        onnx_path = Path(onnx_path).resolve()
        _validate_onnx_path(onnx_path)
        if not onnx_path.exists():
            msg = f"ONNX file not found: {onnx_path}"
            raise FileNotFoundError(msg)

        session = InferenceSession(str(onnx_path))

        for _ in range(warmup_runs):
            session.run(None, input_data)

        latencies: list[float] = []
        for _ in range(n_runs):
            start = time.perf_counter()
            session.run(None, input_data)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        mean_latency = float(np.mean(latencies))
        std_latency = float(np.std(latencies))
        min_latency = float(np.min(latencies))
        max_latency = float(np.max(latencies))

        return BenchmarkResult(
            mean_latency=mean_latency,
            std_latency=std_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            n_runs=n_runs,
        )
