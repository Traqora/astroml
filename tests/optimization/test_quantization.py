"""Tests for ONNX quantization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroml.training.optimization.quantization import (
    QuantizationConfig,
    QuantizationResult,
    calibrate,
    evaluate_quantized,
    quantize,
    quantize_fp16,
    quantize_int8,
)


@pytest.fixture
def mock_ort_quant():
    with patch("astroml.training.optimization.quantization.ort_quant") as mock:

        def _quantize_dynamic(
            model_input, model_output, weight_type=None, per_channel=False, reduce_range=False
        ):
            Path(model_output).write_bytes(b"quantized")

        mock.quantize_dynamic.side_effect = _quantize_dynamic

        def _quantize_fp16(model_input, model_output):
            Path(model_output).write_bytes(b"quantized_fp16")

        mock.quantize_fp16.side_effect = _quantize_fp16
        yield mock


@pytest.fixture
def mock_onnx():
    with patch("astroml.training.optimization.quantization.onnx") as mock:
        yield mock


@pytest.fixture
def mock_session():
    with patch("astroml.training.optimization.quantization.InferenceSession") as mock:
        yield mock


@pytest.fixture
def dummy_onnx_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"x" * 1000)
    return path


class TestQuantizationConfig:
    def test_default_config(self):
        config = QuantizationConfig()
        assert config.quantization_type == "INT8"
        assert config.calibrate is True
        assert config.per_channel is False
        assert config.reduce_range is False

    def test_custom_config(self):
        config = QuantizationConfig(quantization_type="FP16", per_channel=True, reduce_range=True)
        assert config.quantization_type == "FP16"
        assert config.per_channel is True
        assert config.reduce_range is True

    def test_extra_forbidden(self):
        with pytest.raises(ValueError):
            QuantizationConfig(unknown_field=True)  # type: ignore


class TestQuantize:
    def test_quantize_int8(self, mock_ort_quant, dummy_onnx_path: Path, tmp_path: Path):
        output_path = tmp_path / "quantized.onnx"
        config = QuantizationConfig(quantization_type="INT8")

        result = quantize(dummy_onnx_path, output_path, config)

        assert isinstance(result, QuantizationResult)
        assert result.original_size == 1000
        assert result.quantization_type == "INT8"
        mock_ort_quant.quantize_dynamic.assert_called_once()

    def test_quantize_uint8(self, mock_ort_quant, dummy_onnx_path: Path, tmp_path: Path):
        output_path = tmp_path / "quantized.onnx"
        config = QuantizationConfig(quantization_type="UINT8")

        result = quantize(dummy_onnx_path, output_path, config)

        assert result.quantization_type == "UINT8"
        mock_ort_quant.quantize_dynamic.assert_called_once()

    def test_quantize_fp16(self, mock_ort_quant, dummy_onnx_path: Path, tmp_path: Path):
        output_path = tmp_path / "quantized.onnx"
        config = QuantizationConfig(quantization_type="FP16")

        result = quantize(dummy_onnx_path, output_path, config)

        assert result.quantization_type == "FP16"
        mock_ort_quant.quantize_fp16.assert_called_once()

    def test_quantize_default_config(self, mock_ort_quant, dummy_onnx_path: Path, tmp_path: Path):
        output_path = tmp_path / "quantized.onnx"

        result = quantize(dummy_onnx_path, output_path)

        assert result.quantization_type == "INT8"

    def test_quantize_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            quantize("/nonexistent/model.onnx", tmp_path / "out.onnx")

    def test_quantize_unsupported_type(self, dummy_onnx_path: Path, tmp_path: Path):
        with pytest.raises((ValueError, RuntimeError), match="Unsupported quantization type"):
            # Bypass Pydantic validation by constructing config dict directly
            config = QuantizationConfig.model_construct(quantization_type="BF16")
            quantize(dummy_onnx_path, tmp_path / "out.onnx", config)

    def test_quantize_creates_parent_dir(
        self, mock_ort_quant, dummy_onnx_path: Path, tmp_path: Path
    ):
        nested = tmp_path / "sub" / "dir" / "quantized.onnx"
        quantize(dummy_onnx_path, nested)
        assert nested.parent.exists()


class TestQuantizeInt8:
    def test_quantize_int8_with_data(self, mock_ort_quant, dummy_onnx_path: Path, tmp_path: Path):
        calib_data = np.random.randn(10, 5).astype(np.float32)
        result = quantize_int8(dummy_onnx_path, tmp_path / "q.onnx", calibration_data=calib_data)
        assert result.quantization_type == "INT8"


class TestQuantizeFP16:
    def test_quantize_fp16(self, mock_ort_quant, dummy_onnx_path: Path, tmp_path: Path):
        result = quantize_fp16(dummy_onnx_path, tmp_path / "q.onnx")
        assert result.quantization_type == "FP16"


class TestCalibrate:
    def test_calibrate(self, mock_onnx, dummy_onnx_path: Path):
        mock_model = MagicMock()
        mock_model.graph.initializer = []
        mock_onnx.load.return_value = mock_model

        loader = [np.random.randn(5, 10).astype(np.float32) for _ in range(3)]
        result = calibrate(dummy_onnx_path, loader, num_calib_samples=10)

        assert result["num_samples"] > 0
        assert "tensor_ranges" in result

    def test_calibrate_with_tensors(self, mock_onnx, dummy_onnx_path: Path):
        mock_init = MagicMock()
        mock_init.name = "weight"
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_model = MagicMock()
        mock_model.graph.initializer = [mock_init]
        mock_onnx.load.return_value = mock_model

        with patch(
            "astroml.training.optimization.quantization.onnx.numpy_helper.to_array",
            return_value=arr,
        ):
            result = calibrate(dummy_onnx_path, [np.array([1.0])], num_calib_samples=1)

        assert "weight" in result["tensor_ranges"]
        assert result["tensor_ranges"]["weight"]["min"] == 1.0
        assert result["tensor_ranges"]["weight"]["max"] == 4.0

    def test_calibrate_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            calibrate("/nonexistent/model.onnx", [])


class TestEvaluateQuantized:
    def test_evaluate_quantized(self, mock_session, tmp_path: Path):
        original_path = tmp_path / "original.onnx"
        quantized_path = tmp_path / "quantized.onnx"
        original_path.write_bytes(b"x" * 2000)
        quantized_path.write_bytes(b"x" * 500)

        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.run.return_value = [np.array([[0.1, 0.9], [0.8, 0.2]])]

        test_data = np.random.randn(2, 4).astype(np.float32)
        test_labels = np.array([1, 0])

        result = evaluate_quantized(quantized_path, original_path, test_data, test_labels)

        assert isinstance(result, QuantizationResult)
        assert result.original_size == 2000
        assert result.quantized_size == 500
        assert result.original_accuracy is not None
        assert result.quantized_accuracy is not None

    def test_evaluate_quantized_no_labels(self, mock_session, tmp_path: Path):
        original_path = tmp_path / "original.onnx"
        quantized_path = tmp_path / "quantized.onnx"
        original_path.write_bytes(b"x" * 2000)
        quantized_path.write_bytes(b"x" * 500)

        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.run.return_value = [np.array([[0.1, 0.9], [0.8, 0.2]])]

        result = evaluate_quantized(
            quantized_path, original_path, np.random.randn(2, 4).astype(np.float32)
        )

        assert result.original_accuracy is None
        assert result.quantized_accuracy is None

    def test_evaluate_quantized_original_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            evaluate_quantized(tmp_path / "q.onnx", tmp_path / "o.onnx", np.array([1.0]))

    def test_evaluate_quantized_quantized_not_found(self, tmp_path: Path):
        original = tmp_path / "original.onnx"
        original.write_bytes(b"data")
        with pytest.raises(FileNotFoundError, match="not found"):
            evaluate_quantized(tmp_path / "q.onnx", original, np.array([1.0]))
