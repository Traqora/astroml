"""ONNX model quantization utilities."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

try:
    import onnx
except ImportError:
    onnx = None  # type: ignore[assignment]

try:
    from onnxruntime import InferenceSession
    from onnxruntime import quantization as ort_quant
except ImportError:
    InferenceSession = None  # type: ignore[assignment,misc]
    ort_quant = None  # type: ignore[assignment]

QuantizationType = Literal["INT8", "UINT8", "FP16"]


class QuantizationConfig(BaseModel):
    """Configuration for ONNX model quantization."""

    model_config = ConfigDict(extra="forbid")

    quantization_type: QuantizationType = "INT8"
    calibrate: bool = True
    per_channel: bool = False
    reduce_range: bool = False


class QuantizationResult(BaseModel):
    """Result of a quantization operation."""

    model_config = ConfigDict(extra="forbid")

    original_size: int
    quantized_size: int
    original_accuracy: float | None = None
    quantized_accuracy: float | None = None
    original_latency: float | None = None
    quantized_latency: float | None = None
    quantization_type: QuantizationType
    size_reduction: float = 0.0
    accuracy_change: float | None = None
    latency_change: float | None = None


_ONNX_PATH_RE = re.compile(r"^[^\x00]+\.onnx$", re.IGNORECASE)


def _validate_onnx_path(path: Path) -> None:
    """Validate that a resolved path refers to an ONNX file.

    Args:
        path: Resolved absolute path to check.

    Raises:
        ValueError: If the filename is invalid.
    """
    if not _ONNX_PATH_RE.match(path.name):
        msg = f"Invalid ONNX filename: {path.name!r}"
        raise ValueError(msg)


def _get_model_size(path: Path) -> int:
    return path.stat().st_size


def _run_inference(
    session: Any,
    input_data: np.ndarray,
    n_runs: int = 10,
) -> float:
    latencies: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {"input": input_data})
        latencies.append(time.perf_counter() - start)
    return float(np.mean(latencies))


def quantize(
    model_path: str | Path,
    output_path: str | Path,
    config: QuantizationConfig | None = None,
) -> QuantizationResult:
    """Quantize an ONNX model with the given configuration.

    Args:
        model_path: Path to the ONNX model file.
        output_path: Path to save the quantized model.
        config: QuantizationConfig specifying quantization parameters.

    Returns:
        QuantizationResult with before/after metrics.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the quantization type is unsupported.
    """
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    _validate_onnx_path(model_path)
    _validate_onnx_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        msg = f"ONNX model not found: {model_path}"
        raise FileNotFoundError(msg)

    if config is None:
        config = QuantizationConfig()

    original_size = _get_model_size(model_path)

    if config.quantization_type in ("INT8", "UINT8"):
        weight_type = (
            ort_quant.QuantType.QUInt8
            if config.quantization_type == "UINT8"
            else ort_quant.QuantType.QInt8
        )
        ort_quant.quantize_dynamic(
            str(model_path),
            str(output_path),
            weight_type=weight_type,
            per_channel=config.per_channel,
            reduce_range=config.reduce_range,
        )
    elif config.quantization_type == "FP16":
        ort_quant.quantize_fp16(str(model_path), str(output_path))
    else:
        msg = f"Unsupported quantization type: {config.quantization_type}"
        raise ValueError(msg)

    quantized_size = _get_model_size(output_path)
    size_reduction = 1.0 - (quantized_size / original_size) if original_size else 0.0

    logger.info(
        "Quantized model (%s): %d -> %d bytes (%.1f%% reduction)",
        config.quantization_type,
        original_size,
        quantized_size,
        size_reduction * 100,
    )

    return QuantizationResult(
        original_size=original_size,
        quantized_size=quantized_size,
        quantization_type=config.quantization_type,
        size_reduction=size_reduction,
    )


def quantize_int8(
    model_path: str | Path,
    output_path: str | Path,
    calibration_data: np.ndarray | None = None,
    per_channel: bool = False,
) -> QuantizationResult:
    """Quantize an ONNX model to INT8.

    Args:
        model_path: Path to the ONNX model file.
        output_path: Path to save the quantized model.
        calibration_data: Optional calibration data for static quantization.
        per_channel: Whether to quantize per channel.

    Returns:
        QuantizationResult with before/after metrics.
    """
    config = QuantizationConfig(
        quantization_type="INT8",
        calibrate=calibration_data is not None,
        per_channel=per_channel,
    )
    return quantize(model_path, output_path, config)


def quantize_fp16(
    model_path: str | Path,
    output_path: str | Path,
) -> QuantizationResult:
    """Quantize an ONNX model to FP16.

    Args:
        model_path: Path to the ONNX model file.
        output_path: Path to save the quantized model.

    Returns:
        QuantizationResult with before/after metrics.
    """
    config = QuantizationConfig(quantization_type="FP16")
    return quantize(model_path, output_path, config)


def calibrate(
    model_path: str | Path,
    calibration_data_loader: Any,
    num_calib_samples: int = 100,
) -> dict[str, Any]:
    """Calibrate an ONNX model for INT8 quantization.

    Args:
        model_path: Path to the ONNX model file.
        calibration_data_loader: Iterable that yields calibration data batches.
        num_calib_samples: Maximum number of samples to use for calibration.

    Returns:
        Dict with calibration results including range information.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(model_path).resolve()
    _validate_onnx_path(model_path)
    if not model_path.exists():
        msg = f"ONNX model not found: {model_path}"
        raise FileNotFoundError(msg)

    model = onnx.load(str(model_path))
    calib_result: dict[str, Any] = {
        "num_samples": 0,
        "tensor_ranges": {},
    }

    samples_collected = 0
    for batch in calibration_data_loader:
        if samples_collected >= num_calib_samples:
            break
        if isinstance(batch, np.ndarray):
            samples_collected += batch.shape[0] if batch.ndim > 0 else 1
        else:
            samples_collected += 1

    calib_result["num_samples"] = min(samples_collected, num_calib_samples)

    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        calib_result["tensor_ranges"][init.name] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    logger.info(
        "Calibration complete: %d samples, %d tensors",
        calib_result["num_samples"],
        len(calib_result["tensor_ranges"]),
    )
    return calib_result


def evaluate_quantized(
    model_path: str | Path,
    original_model_path: str | Path,
    test_data: np.ndarray,
    test_labels: np.ndarray | None = None,
) -> QuantizationResult:
    """Compare original vs quantized model performance.

    Args:
        model_path: Path to the quantized ONNX model file.
        original_model_path: Path to the original ONNX model file.
        test_data: Input test data as numpy array.
        test_labels: Optional ground truth labels for accuracy comparison.

    Returns:
        QuantizationResult with accuracy and latency comparisons.

    Raises:
        FileNotFoundError: If either model file does not exist.
    """
    model_path = Path(model_path).resolve()
    original_model_path = Path(original_model_path).resolve()
    _validate_onnx_path(model_path)
    _validate_onnx_path(original_model_path)

    if not model_path.exists():
        msg = f"Quantized model not found: {model_path}"
        raise FileNotFoundError(msg)
    if not original_model_path.exists():
        msg = f"Original model not found: {original_model_path}"
        raise FileNotFoundError(msg)

    original_size = _get_model_size(original_model_path)
    quantized_size = _get_model_size(model_path)
    size_reduction = 1.0 - (quantized_size / original_size) if original_size else 0.0

    original_session = InferenceSession(str(original_model_path))
    quantized_session = InferenceSession(str(model_path))

    original_latency = _run_inference(original_session, test_data)
    quantized_latency = _run_inference(quantized_session, test_data)
    latency_change = (
        (quantized_latency - original_latency) / original_latency if original_latency else 0.0
    )

    original_accuracy: float | None = None
    quantized_accuracy: float | None = None
    accuracy_change: float | None = None

    if test_labels is not None:
        original_preds = original_session.run(None, {"input": test_data})[0]
        quantized_preds = quantized_session.run(None, {"input": test_data})[0]

        if original_preds.ndim > 1 and original_preds.shape[1] > 1:
            original_preds = np.argmax(original_preds, axis=1)
        if quantized_preds.ndim > 1 and quantized_preds.shape[1] > 1:
            quantized_preds = np.argmax(quantized_preds, axis=1)

        original_accuracy = float(np.mean(original_preds == test_labels))
        quantized_accuracy = float(np.mean(quantized_preds == test_labels))
        accuracy_change = quantized_accuracy - original_accuracy

    return QuantizationResult(
        original_size=original_size,
        quantized_size=quantized_size,
        original_accuracy=original_accuracy,
        quantized_accuracy=quantized_accuracy,
        original_latency=original_latency,
        quantized_latency=quantized_latency,
        quantization_type="INT8",
        size_reduction=size_reduction,
        accuracy_change=accuracy_change,
        latency_change=latency_change,
    )
