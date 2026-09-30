from astroml.utils.exceptions import AstroMLError
"""Model compression and quantization API endpoints for edge deployment workflows."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from astroml.training.compression.pipeline import (
    CompressionBenchmarkResult,
    CompressionConfig,
    CompressionPipeline,
)
from astroml.training.compression.pruning import ModelPruner, PruningConfig, PruningMethod
from astroml.training.compression.quantization import (
    ModelQuantizer,
    QuantizationConfig,
    QuantizationType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/compression", tags=["model-compression"])

# In-memory mock model store for API demonstrations and testing
_MODELS_CACHE: dict[str, nn.Module] = {}


class QuantizeRequest(BaseModel):
    """Request model for standalone model quantization."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="ID or name of model in store")
    quantization_type: Literal["dynamic_int8", "static_int8", "fp16", "qat"] = "dynamic_int8"
    backend: str = "fbgemm"


class PruneRequest(BaseModel):
    """Request model for model pruning."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="ID or name of model in store")
    amount: float = Field(0.3, ge=0.0, le=0.99, description="Fraction of weights to prune")
    method: Literal["l1_unstructured", "ln_structured", "global_unstructured"] = "l1_unstructured"


class PipelineCompressionRequest(BaseModel):
    """Request model for full compression pipeline execution."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="Model ID")
    enable_pruning: bool = True
    prune_amount: float = 0.3
    enable_quantization: bool = True
    quantization_type: Literal["dynamic_int8", "static_int8", "fp16"] = "dynamic_int8"
    target_hardware: str = "edge_cpu"


class BenchmarkRequest(BaseModel):
    """Request model for model latency and size benchmarking."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    num_runs: int = 50
    input_dim: int = 64


class CompressionJobResponse(BaseModel):
    """Response model with compression job results."""

    job_id: str
    model_id: str
    status: str
    benchmark: dict[str, Any] | None = None


def _get_or_create_model(model_id: str, input_dim: int = 64, output_dim: int = 2) -> nn.Module:
    """Helper to fetch or instantiate a simple model for compression testing."""
    if model_id not in _MODELS_CACHE:
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        _MODELS_CACHE[model_id] = model
    return _MODELS_CACHE[model_id]


@router.post("/quantize", response_model=CompressionJobResponse)
async def quantize_endpoint(req: QuantizeRequest) -> dict[str, Any]:
    """Apply quantization to a PyTorch model."""
    try:
        model = _get_or_create_model(req.model_id)
        config = QuantizationConfig(
            quant_type=QuantizationType(req.quantization_type),
            backend=req.backend,
        )
        quantizer = ModelQuantizer(config)
        quantized = quantizer.quantize(model)
        new_model_id = f"{req.model_id}_quantized_{uuid.uuid4().hex[:6]}"
        _MODELS_CACHE[new_model_id] = quantized

        return {
            "job_id": f"job_{uuid.uuid4().hex[:8]}",
            "model_id": new_model_id,
            "status": "quantized",
        }
    except Exception as exc:
        logger.error("Quantization failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/prune", response_model=CompressionJobResponse)
async def prune_endpoint(req: PruneRequest) -> dict[str, Any]:
    """Apply structured or unstructured pruning to a PyTorch model."""
    try:
        model = _get_or_create_model(req.model_id)
        config = PruningConfig(
            method=PruningMethod(req.method),
            amount=req.amount,
        )
        pruner = ModelPruner(config)
        pruned = pruner.prune(model, config)
        sparsity = pruner.compute_sparsity(pruned)
        new_model_id = f"{req.model_id}_pruned_{uuid.uuid4().hex[:6]}"
        _MODELS_CACHE[new_model_id] = pruned

        return {
            "job_id": f"job_{uuid.uuid4().hex[:8]}",
            "model_id": new_model_id,
            "status": f"pruned (sparsity: {sparsity * 100:.1f}%)",
        }
    except Exception as exc:
        logger.error("Pruning failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/pipeline", response_model=CompressionJobResponse)
async def pipeline_endpoint(req: PipelineCompressionRequest) -> dict[str, Any]:
    """Execute combined compression pipeline (Pruning -> Quantization -> Benchmarking)."""
    try:
        model = _get_or_create_model(req.model_id)
        config = CompressionConfig(
            enable_pruning=req.enable_pruning,
            pruning_config=PruningConfig(amount=req.prune_amount),
            enable_quantization=req.enable_quantization,
            quantization_config=QuantizationConfig(
                quant_type=QuantizationType(req.quantization_type)
            ),
            target_hardware=req.target_hardware,
        )
        pipeline = CompressionPipeline(config)
        sample_input = torch.randn(1, 64)
        compressed, benchmark = pipeline.compress(model, sample_input=sample_input)

        new_model_id = f"{req.model_id}_compressed_{uuid.uuid4().hex[:6]}"
        _MODELS_CACHE[new_model_id] = compressed

        return {
            "job_id": f"job_{uuid.uuid4().hex[:8]}",
            "model_id": new_model_id,
            "status": "completed",
            "benchmark": benchmark.to_dict(),
        }
    except Exception as exc:
        logger.error("Compression pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
