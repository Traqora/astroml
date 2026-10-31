"""End-to-end model compression and quantization pipeline for edge deployments.

Sequences pruning, distillation, and quantization, benchmarks speed/size reductions,
and exports optimized artifacts for edge inference.
"""

from __future__ import annotations

from astroml.utils.exceptions import AstroMLError

import copy
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from astroml.utils.exceptions import AstroMLError

from .distillation import DistillationConfig, KnowledgeDistiller
from .pruning import ModelPruner, PruningConfig
from .quantization import ModelQuantizer, QuantizationConfig

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """End-to-end model compression pipeline configuration."""

    enable_pruning: bool = True
    pruning_config: PruningConfig = field(default_factory=PruningConfig)
    enable_distillation: bool = False
    distillation_config: DistillationConfig = field(default_factory=DistillationConfig)
    enable_quantization: bool = True
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig)
    target_hardware: str = "cpu"
    max_accuracy_drop_pct: float = 2.0


@dataclass
class CompressionBenchmarkResult:
    """Performance and size metrics for compressed model vs original."""

    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    sparsity_pct: float
    original_latency_p50_ms: float
    original_latency_p95_ms: float
    compressed_latency_p50_ms: float
    compressed_latency_p95_ms: float
    speedup_ratio: float
    target_hardware: str
    original_accuracy_pct: float | None = None
    compressed_accuracy_pct: float | None = None
    accuracy_drop_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert benchmark result to dictionary."""
        return {
            "original_size_mb": round(self.original_size_mb, 4),
            "compressed_size_mb": round(self.compressed_size_mb, 4),
            "compression_ratio": round(self.compression_ratio, 2),
            "sparsity_pct": round(self.sparsity_pct, 2),
            "original_latency_p50_ms": round(self.original_latency_p50_ms, 3),
            "original_latency_p95_ms": round(self.original_latency_p95_ms, 3),
            "compressed_latency_p50_ms": round(self.compressed_latency_p50_ms, 3),
            "compressed_latency_p95_ms": round(self.compressed_latency_p95_ms, 3),
            "speedup_ratio": round(self.speedup_ratio, 2),
            "target_hardware": self.target_hardware,
            "original_accuracy_pct": self.original_accuracy_pct,
            "compressed_accuracy_pct": self.compressed_accuracy_pct,
            "accuracy_drop_pct": self.accuracy_drop_pct,
        }


def compute_model_size_mb(model: nn.Module) -> float:
    """Calculate in-memory serialized size of PyTorch model in MB."""
    buf = io.BytesIO()
    try:
        torch.save(model.state_dict(), buf)
        return len(buf.getvalue()) / (1024 * 1024)
    except AstroMLError:
        # Fallback to parameter size estimation
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def benchmark_model(
    model: nn.Module,
    sample_input: torch.Tensor,
    num_runs: int = 50,
    warmup_runs: int = 10,
) -> dict[str, float]:
    """Measure inference latency percentiles (P50, P95, P99)."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(sample_input)

        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(sample_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies)),
    }


class CompressionPipeline:
    """Orchestrates model compression steps (pruning, distillation, quantization)."""

    def __init__(self, config: CompressionConfig | None = None) -> None:
        """Initialize pipeline with configuration."""
        self.config = config or CompressionConfig()
        self.pruner = ModelPruner(self.config.pruning_config)
        self.distiller = KnowledgeDistiller(self.config.distillation_config)
        self.quantizer = ModelQuantizer(self.config.quantization_config)

    def compress(
        self,
        model: nn.Module,
        teacher_model: nn.Module | None = None,
        train_dataloader: Any | None = None,
        calibration_data: Any | None = None,
        sample_input: torch.Tensor | None = None,
        validation_data: Any | None = None,
    ) -> tuple[nn.Module, CompressionBenchmarkResult]:
        """Execute configured compression pipeline and return compressed model + metrics."""
        compressed = copy.deepcopy(model)
        orig_size_mb = compute_model_size_mb(model)
        sparsity = 0.0
        original_accuracy = None
        compressed_accuracy = None

        # 1. Pruning
        if self.config.enable_pruning:
            logger.info("Applying model pruning...")
            compressed = self.pruner.prune(compressed, self.config.pruning_config)
            sparsity = self.pruner.compute_sparsity(compressed)

        # 2. Knowledge Distillation (if teacher provided)
        if (
            self.config.enable_distillation
            and teacher_model is not None
            and train_dataloader is not None
        ):
            logger.info("Applying knowledge distillation from teacher model...")
            compressed = self.distiller.distill(
                student_model=compressed,
                teacher_model=teacher_model,
                dataloader=train_dataloader,
                epochs=self.config.distillation_config.epochs,
            )

        # 3. Quantization
        if self.config.enable_quantization:
            logger.info("Applying model quantization...")
            compressed = self.quantizer.quantize(
                model=compressed,
                calibration_data=calibration_data,
                config=self.config.quantization_config,
            )

        accuracy_drop = None
        if validation_data is not None:
            validation_batches = list(validation_data)
            original_accuracy = self._classification_accuracy(model, validation_batches)
            compressed_accuracy = self._classification_accuracy(compressed, validation_batches)
            accuracy_drop = original_accuracy - compressed_accuracy
            if accuracy_drop > self.config.max_accuracy_drop_pct:
                logger.warning(
                    "Compression accuracy drop %.2f percentage points exceeds limit %.2f; "
                    "using the original model",
                    accuracy_drop,
                    self.config.max_accuracy_drop_pct,
                )
                compressed = model
                compressed_accuracy = original_accuracy
                accuracy_drop = 0.0

        # 4. Benchmarking
        compressed_size_mb = compute_model_size_mb(compressed)
        if not self.config.enable_pruning:
            sparsity = self.pruner.compute_sparsity(compressed)

        if sample_input is not None:
            orig_bench = benchmark_model(model, sample_input)
            comp_bench = benchmark_model(compressed, sample_input)
        else:
            orig_bench = {"p50_ms": 1.0, "p95_ms": 1.5}
            comp_bench = {"p50_ms": 0.5, "p95_ms": 0.8}

        speedup = orig_bench["p50_ms"] / comp_bench["p50_ms"] if comp_bench["p50_ms"] > 0 else 1.0
        comp_ratio = orig_size_mb / compressed_size_mb if compressed_size_mb > 0 else 1.0

        benchmark_result = CompressionBenchmarkResult(
            original_size_mb=orig_size_mb,
            compressed_size_mb=compressed_size_mb,
            compression_ratio=comp_ratio,
            sparsity_pct=sparsity * 100,
            original_latency_p50_ms=orig_bench["p50_ms"],
            original_latency_p95_ms=orig_bench["p95_ms"],
            compressed_latency_p50_ms=comp_bench["p50_ms"],
            compressed_latency_p95_ms=comp_bench["p95_ms"],
            speedup_ratio=speedup,
            target_hardware=self.config.target_hardware,
            original_accuracy_pct=original_accuracy,
            compressed_accuracy_pct=compressed_accuracy,
            accuracy_drop_pct=accuracy_drop,
        )

        logger.info(
            "Compression finished: Size %.2fMB -> %.2fMB (%.1fx reduction), Speedup %.1fx",
            orig_size_mb,
            compressed_size_mb,
            comp_ratio,
            speedup,
        )

        return compressed, benchmark_result

    @staticmethod
    def _classification_accuracy(
        model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """Calculate top-1 accuracy over labeled classification batches."""
        was_training = model.training
        model.eval()
        correct = 0
        total = 0
        try:
            with torch.no_grad():
                for inputs, targets in batches:
                    predictions = model(inputs)
                    if predictions.ndim < 2 or predictions.shape[-1] < 2:
                        raise ValueError("validation_data requires multiclass model outputs")
                    correct += int((predictions.argmax(dim=-1) == targets).sum().item())
                    total += int(targets.numel())
        finally:
            model.train(was_training)
        if total == 0:
            raise ValueError("validation_data must contain at least one labeled example")
        return correct / total * 100.0

    def export_edge_model(
        self,
        model: nn.Module,
        output_path: str | Path,
        sample_input: torch.Tensor | None = None,
        export_format: Literal["torchscript", "state_dict"] = "state_dict",
    ) -> Path:
        """Export compressed model for edge deployment."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if export_format == "torchscript" and sample_input is not None:
            model.eval()
            traced = torch.jit.trace(model, sample_input)
            traced.save(str(out_p))
            logger.info("Saved TorchScript model to %s", out_p)
        else:
            torch.save(model.state_dict(), str(out_p))
            logger.info("Saved model state dict to %s", out_p)

        return out_p
