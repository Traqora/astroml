"""Unit and integration tests for model compression, pruning, quantization, and distillation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader, TensorDataset

from astroml.api.app import app
from astroml.training.compression.distillation import (
    DistillationConfig,
    DistillationLoss,
    KnowledgeDistiller,
)
from astroml.training.compression.pipeline import (
    CompressionConfig,
    CompressionPipeline,
    benchmark_model,
)
from astroml.training.compression.pruning import (
    ModelPruner,
    PruningConfig,
    PruningMethod,
)
from astroml.training.compression.quantization import (
    ModelQuantizer,
    QuantizationConfig,
    QuantizationType,
)


class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 32, hidden: int = 64, out_features: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class TestQuantization:
    def test_dynamic_quantization(self):
        model = SimpleMLP()
        quantizer = ModelQuantizer(QuantizationConfig(quant_type=QuantizationType.DYNAMIC_INT8))
        quantized = quantizer.quantize(model)

        x = torch.randn(4, 32)
        out = quantized(x)
        assert out.shape == (4, 2)

    def test_fp16_quantization(self):
        model = SimpleMLP()
        quantizer = ModelQuantizer(QuantizationConfig(quant_type=QuantizationType.FP16))
        quantized = quantizer.quantize(model)

        x = torch.randn(4, 32).half()
        out = quantized(x)
        assert out.dtype == torch.float16

    def test_qat_preparation_and_conversion(self):
        model = SimpleMLP()
        quantizer = ModelQuantizer()
        qat_prepared = quantizer.prepare_qat(model)
        assert qat_prepared is not None

        converted = quantizer.convert_qat(qat_prepared)
        assert converted is not None


class TestPruning:
    def test_l1_unstructured_pruning(self):
        model = SimpleMLP()
        pruner = ModelPruner()
        pruned = pruner.prune_unstructured(model, amount=0.5)

        sparsity = pruner.compute_sparsity(pruned)
        assert sparsity >= 0.4  # At least ~40-50% weights should be zero

        x = torch.randn(4, 32)
        out = pruned(x)
        assert out.shape == (4, 2)

    def test_ln_structured_pruning(self):
        model = SimpleMLP()
        pruner = ModelPruner()
        pruned = pruner.prune_structured(model, amount=0.25, dim=0)

        sparsity = pruner.compute_sparsity(pruned)
        assert sparsity > 0.0

    def test_global_pruning(self):
        model = SimpleMLP()
        pruner = ModelPruner()
        pruned = pruner.prune_global(model, amount=0.3)

        sparsity = pruner.compute_sparsity(pruned)
        assert sparsity > 0.2


class TestDistillation:
    def test_distillation_loss(self):
        loss_fn = DistillationLoss(temperature=4.0, alpha=0.5)
        student_logits = torch.randn(8, 2)
        teacher_logits = torch.randn(8, 2)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

        loss = loss_fn(student_logits, teacher_logits, targets)
        assert loss.item() > 0.0

    def test_distillation_training_loop(self):
        teacher = SimpleMLP(hidden=128)
        student = SimpleMLP(hidden=32)

        # Synthetic data
        x = torch.randn(64, 32)
        y = torch.randint(0, 2, (64,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=16)

        distiller = KnowledgeDistiller(DistillationConfig(epochs=2, learning_rate=1e-2))
        distilled = distiller.distill(student, teacher, loader)

        out = distilled(x[:4])
        assert out.shape == (4, 2)


class TestCompressionPipeline:
    def test_end_to_end_compression(self):
        model = SimpleMLP(hidden=128)
        config = CompressionConfig(
            enable_pruning=True,
            pruning_config=PruningConfig(amount=0.4),
            enable_quantization=True,
            quantization_config=QuantizationConfig(quant_type=QuantizationType.DYNAMIC_INT8),
            target_hardware="edge_cpu",
        )
        pipeline = CompressionPipeline(config)
        sample_input = torch.randn(1, 32)

        compressed, bench = pipeline.compress(model, sample_input=sample_input)

        assert bench.sparsity_pct > 30.0
        assert bench.compression_ratio >= 1.0
        assert bench.target_hardware == "edge_cpu"

        # Edge export
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_f:
            tmp_path = tmp_f.name

        out_path = pipeline.export_edge_model(compressed, tmp_path)
        assert Path(out_path).exists()
        Path(tmp_path).unlink(missing_ok=True)


class TestCompressionRouter:
    @pytest.fixture
    def client(self):
        from astroml.api.routers import compression

        app.include_router(compression.router)
        return TestClient(app)

    def test_compression_api_endpoints(self, client):
        # 1. Quantize
        q_resp = client.post(
            "/api/v1/compression/quantize",
            json={"model_id": "test_model_1", "quantization_type": "dynamic_int8"},
        )
        assert q_resp.status_code == 200
        assert q_resp.json()["status"] == "quantized"

        # 2. Prune
        p_resp = client.post(
            "/api/v1/compression/prune",
            json={"model_id": "test_model_1", "amount": 0.4, "method": "l1_unstructured"},
        )
        assert p_resp.status_code == 200
        assert "pruned" in p_resp.json()["status"]

        # 3. Pipeline
        pipe_resp = client.post(
            "/api/v1/compression/pipeline",
            json={
                "model_id": "test_model_1",
                "enable_pruning": True,
                "prune_amount": 0.3,
                "enable_quantization": True,
                "quantization_type": "dynamic_int8",
                "target_hardware": "edge_cpu",
            },
        )
        assert pipe_resp.status_code == 200
        assert "benchmark" in pipe_resp.json()
