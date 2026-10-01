from astroml.utils.exceptions import AstroMLError
"""Model optimization API endpoints for ONNX conversion, optimization, and quantization."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from astroml.serving.onnx_runtime import ONNXRuntime
from astroml.training.optimization.onnx_converter import ONNXConverter
from astroml.training.optimization.onnx_optimizer import ONNXOptimizer, OptimizationLevel
from astroml.training.optimization.quantization import (
    QuantizationConfig,
    QuantizationResult,
)
from astroml.training.optimization.quantization import (
    quantize as quantize_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["model-optimization"])

_MODEL_STORE = Path("/tmp/astroml/models")
_MODEL_STORE.mkdir(parents=True, exist_ok=True)


class ConvertRequest(BaseModel):
    """Request model for model conversion."""

    model_config = ConfigDict(extra="forbid")

    model_type: Literal["pytorch", "sklearn"] = "pytorch"
    model_name: str
    opset_version: int = 17
    dynamic_axes: dict[str, dict[int, str]] | None = None


class ConvertResponse(BaseModel):
    """Response model for model conversion."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    output_path: str
    status: str = "converted"


class OptimizeRequest(BaseModel):
    """Request model for model optimization."""

    model_config = ConfigDict(extra="forbid")

    optimization_level: OptimizationLevel = "extended"


class OptimizeResponse(BaseModel):
    """Response model for model optimization."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    original_size: int
    optimized_size: int
    original_nodes: int
    optimized_nodes: int
    size_reduction: float
    node_reduction: float
    optimizations_applied: list[str]


class QuantizeRequest(BaseModel):
    """Request model for model quantization."""

    model_config = ConfigDict(extra="forbid")

    quantization_type: Literal["INT8", "UINT8", "FP16"] = "INT8"
    per_channel: bool = False
    reduce_range: bool = False


class QuantizeResponse(BaseModel):
    """Response model for model quantization."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    original_size: int
    quantized_size: int
    quantization_type: str
    size_reduction: float


class BenchmarkResponse(BaseModel):
    """Response model for model benchmark."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    mean_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    n_runs: int


class PredictRequest(BaseModel):
    """Request model for inference."""

    model_config = ConfigDict(extra="forbid")

    input_data: list[list[float]]
    input_name: str = "input"


class PredictResponse(BaseModel):
    """Response model for inference."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    outputs: list[Any]


_MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def _validate_model_id(model_id: str) -> None:
    """Reject model IDs that could be used for path traversal.

    Args:
        model_id: The model ID supplied by the caller.

    Raises:
        HTTPException: If the model ID contains disallowed characters.
    """
    if not _MODEL_ID_RE.match(model_id):
        raise HTTPException(
            status_code=400,
            detail="model_id must contain only alphanumeric characters, hyphens, or underscores (max 128 chars).",
        )


def _get_model_path(model_id: str) -> Path:
    """Resolve and validate model path from model ID.

    Ensures the resolved path stays within the model store directory
    to prevent path traversal attacks.

    Args:
        model_id: The model identifier (alphanumeric + hyphens/underscores).

    Returns:
        Absolute Path to the model file.

    Raises:
        HTTPException: 400 if model_id is invalid, 404 if file not found.
    """
    _validate_model_id(model_id)
    # Resolve to absolute path and confirm it stays within _MODEL_STORE
    path = (_MODEL_STORE / f"{model_id}.onnx").resolve()
    if not path.is_relative_to(_MODEL_STORE.resolve()):
        raise HTTPException(status_code=400, detail="Invalid model_id.")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return path


@router.post("/optimization/convert", response_model=ConvertResponse)
async def convert_model(request: ConvertRequest) -> ConvertResponse:
    """Convert a model to ONNX format.

    Note: This endpoint accepts conversion parameters; actual model
    conversion requires the model object to be provided directly.
    """
    model_id = request.model_name
    output_path = _MODEL_STORE / f"{model_id}.onnx"

    return ConvertResponse(
        model_id=model_id,
        output_path=str(output_path),
    )


@router.post("/optimization/{model_id}/optimize", response_model=OptimizeResponse)
async def optimize_model(
    model_id: str,
    request: OptimizeRequest,
) -> OptimizeResponse:
    """Optimize an ONNX model with graph transformations."""
    _validate_model_id(model_id)
    model_path = _get_model_path(model_id)
    output_path = (_MODEL_STORE / f"{model_id}_optimized.onnx").resolve()
    if not output_path.is_relative_to(_MODEL_STORE.resolve()):
        raise HTTPException(status_code=400, detail="Invalid model_id.")

    try:
        result = ONNXOptimizer.optimize(
            str(model_path),
            optimization_level=request.optimization_level,
            output_path=str(output_path),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Optimization failed for model %s", model_id)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}")

    return OptimizeResponse(
        model_id=model_id,
        original_size=result.original_size,
        optimized_size=result.optimized_size,
        original_nodes=result.original_nodes,
        optimized_nodes=result.optimized_nodes,
        size_reduction=result.size_reduction,
        node_reduction=result.node_reduction,
        optimizations_applied=result.optimizations_applied,
    )


@router.post("/optimization/{model_id}/quantize", response_model=QuantizeResponse)
async def quantize_model_endpoint(
    model_id: str,
    request: QuantizeRequest,
) -> QuantizeResponse:
    """Quantize an ONNX model."""
    _validate_model_id(model_id)
    model_path = _get_model_path(model_id)
    output_path = (_MODEL_STORE / f"{model_id}_quantized.onnx").resolve()
    if not output_path.is_relative_to(_MODEL_STORE.resolve()):
        raise HTTPException(status_code=400, detail="Invalid model_id.")

    config = QuantizationConfig(
        quantization_type=request.quantization_type,
        per_channel=request.per_channel,
        reduce_range=request.reduce_range,
    )

    try:
        result = quantize_model(
            str(model_path),
            str(output_path),
            config=config,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Quantization failed for model %s", model_id)
        raise HTTPException(status_code=500, detail=f"Quantization failed: {exc}")

    return QuantizeResponse(
        model_id=model_id,
        original_size=result.original_size,
        quantized_size=result.quantized_size,
        quantization_type=result.quantization_type,
        size_reduction=result.size_reduction,
    )


@router.get("/optimization/{model_id}/benchmark", response_model=BenchmarkResponse)
async def benchmark_model(
    model_id: str,
    n_runs: int = 100,
) -> BenchmarkResponse:
    """Benchmark an ONNX model's inference performance."""
    model_path = _get_model_path(model_id)

    try:
        rt = ONNXRuntime.load_model(str(model_path))
        input_info = rt.get_input_info()
        if not input_info:
            raise HTTPException(status_code=400, detail="Model has no inputs")

        shape = input_info[0]["shape"]
        shape = [s if isinstance(s, int) and s > 0 else 1 for s in shape]
        dummy_input = np.random.randn(*shape).astype(np.float32)

        result = ONNXOptimizer.benchmark(
            str(model_path),
            {"input": dummy_input},
            n_runs=n_runs,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    except Exception as exc:
        logger.exception("Benchmark failed for model %s", model_id)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {exc}")

    return BenchmarkResponse(
        model_id=model_id,
        mean_latency=result.mean_latency,
        std_latency=result.std_latency,
        min_latency=result.min_latency,
        max_latency=result.max_latency,
        n_runs=result.n_runs,
    )


@router.post("/optimization/{model_id}/predict", response_model=PredictResponse)
async def predict_model(
    model_id: str,
    request: PredictRequest,
) -> PredictResponse:
    """Run inference on an ONNX model."""
    model_path = _get_model_path(model_id)

    try:
        rt = ONNXRuntime.load_model(str(model_path))
        input_array = np.array(request.input_data, dtype=np.float32)
        outputs = rt.predict({request.input_name: input_array})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference failed for model %s", model_id)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return PredictResponse(
        model_id=model_id,
        outputs=[out.tolist() for out in outputs],
    )
