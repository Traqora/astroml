"""Tests for ONNXConverter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astroml.training.optimization.onnx_converter import ONNXConverter


@pytest.fixture
def mock_torch():
    with patch("astroml.training.optimization.onnx_converter.torch") as mock:
        yield mock


@pytest.fixture
def mock_onnx():
    with patch("astroml.training.optimization.onnx_converter.onnx") as mock:
        mock.checker = MagicMock()
        yield mock


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.eval = MagicMock()
    return model


@pytest.fixture
def temp_output(tmp_path: Path) -> Path:
    return tmp_path / "models"


class TestConvert:
    def test_convert_pytorch_model(self, mock_torch, mock_model, temp_output: Path):
        output_path = temp_output / "model.onnx"
        input_sample = mock_torch.Tensor()

        result = ONNXConverter.convert(mock_model, input_sample, output_path)

        assert result == output_path
        mock_model.eval.assert_called_once()
        mock_torch.onnx.export.assert_called_once_with(
            mock_model,
            input_sample,
            str(output_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    def test_convert_with_custom_opset(self, mock_torch, mock_model, temp_output: Path):
        output_path = temp_output / "model.onnx"

        ONNXConverter.convert(mock_model, mock_torch.Tensor(), output_path, opset_version=15)

        mock_torch.onnx.export.assert_called_once()
        assert mock_torch.onnx.export.call_args[1]["opset_version"] == 15

    def test_convert_with_dynamic_axes(self, mock_torch, mock_model, temp_output: Path):
        output_path = temp_output / "model.onnx"
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

        ONNXConverter.convert(
            mock_model, mock_torch.Tensor(), output_path, dynamic_axes=dynamic_axes
        )

        assert mock_torch.onnx.export.call_args[1]["dynamic_axes"] == dynamic_axes

    def test_convert_unsupported_opset(self, mock_torch, mock_model, temp_output: Path):
        with pytest.raises(ValueError, match="Unsupported opset"):
            ONNXConverter.convert(
                mock_model, mock_torch.Tensor(), temp_output / "m.onnx", opset_version=99
            )

    def test_convert_runtime_error(self, mock_torch, mock_model, temp_output: Path):
        mock_torch.onnx.export.side_effect = RuntimeError("export failed")

        with pytest.raises(RuntimeError, match="export failed"):
            ONNXConverter.convert(mock_model, mock_torch.Tensor(), temp_output / "m.onnx")

    def test_convert_creates_parent_dir(self, mock_torch, mock_model, temp_output: Path):
        nested = temp_output / "sub" / "nested" / "model.onnx"

        ONNXConverter.convert(mock_model, mock_torch.Tensor(), nested)

        assert nested.parent.exists()


class TestConvertFromSklearn:
    def test_convert_sklearn_model(self, temp_output: Path):
        mock_estimator = MagicMock()
        output_path = temp_output / "sklearn_model.onnx"

        with patch("astroml.training.optimization.onnx_converter.skl2onnx") as mock_skl:
            mock_onnx_model = MagicMock()
            mock_onnx_model.SerializeToString.return_value = b"model_content"
            mock_skl.convert_sklearn.return_value = mock_onnx_model

            result = ONNXConverter.convert_from_sklearn(mock_estimator, output_path)

        assert result == output_path
        mock_skl.convert_sklearn.assert_called_once_with(mock_estimator)
        mock_onnx_model.SerializeToString.assert_called_once()

    def test_convert_sklearn_runtime_error(self, temp_output: Path):
        with patch("astroml.training.optimization.onnx_converter.skl2onnx") as mock_skl:
            mock_skl.convert_sklearn.side_effect = ValueError("unknown type")

            with pytest.raises(RuntimeError, match="unknown type"):
                ONNXConverter.convert_from_sklearn(MagicMock(), temp_output / "m.onnx")

    def test_convert_sklearn_creates_dir(self, temp_output: Path):
        nested = temp_output / "deep" / "dir" / "model.onnx"
        with patch("astroml.training.optimization.onnx_converter.skl2onnx") as mock_skl:
            mock_model = MagicMock()
            mock_model.SerializeToString.return_value = b"data"
            mock_skl.convert_sklearn.return_value = mock_model

            ONNXConverter.convert_from_sklearn(MagicMock(), nested)

        assert nested.parent.exists()


class TestValidateOnnx:
    def test_validate_valid_model(self, mock_onnx, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"dummy")
        mock_onnx.load.return_value = MagicMock()

        result = ONNXConverter.validate_onnx(onnx_path)

        assert result is True
        mock_onnx.load.assert_called_once_with(str(onnx_path))
        mock_onnx.checker.check_model.assert_called_once()

    def test_validate_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            ONNXConverter.validate_onnx("/nonexistent/model.onnx")

    def test_validate_invalid_model(self, mock_onnx, tmp_path: Path):
        onnx_path = tmp_path / "bad.onnx"
        onnx_path.write_bytes(b"garbage")
        mock_onnx.checker.check_model.side_effect = Exception("check failed")

        with pytest.raises(RuntimeError, match="check failed"):
            ONNXConverter.validate_onnx(onnx_path)


class TestGetModelMetadata:
    def test_get_metadata(self, mock_onnx, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"dummy")

        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.type.tensor_type.shape.dim = [MagicMock(), MagicMock()]
        mock_input.type.tensor_type.shape.dim[0].dim_param = ""
        mock_input.type.tensor_type.shape.dim[0].dim_value = 1
        mock_input.type.tensor_type.shape.dim[1].dim_param = "batch"
        mock_input.type.tensor_type.shape.dim[1].dim_value = 0
        mock_input.type.tensor_type.elem_type = 1

        mock_output = MagicMock()
        mock_output.name = "output"
        mock_output.type.tensor_type.shape.dim = []
        mock_output.type.tensor_type.elem_type = 1

        mock_model = MagicMock()
        mock_model.graph.input = [mock_input]
        mock_model.graph.output = [mock_output]
        mock_model.opset_import = [MagicMock()]
        mock_model.opset_import[0].domain = ""
        mock_model.opset_import[0].version = 17
        mock_model.producer_name = "test"
        mock_model.producer_version = "1.0"
        mock_onnx.load.return_value = mock_model

        metadata = ONNXConverter.get_model_metadata(onnx_path)

        assert "inputs" in metadata
        assert "outputs" in metadata
        assert "opset" in metadata
        assert metadata["producer_name"] == "test"
        assert metadata["inputs"][0]["name"] == "input"
        assert metadata["inputs"][0]["shape"] == [1, "batch"]

    def test_get_metadata_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            ONNXConverter.get_model_metadata("/nonexistent/model.onnx")


class TestConvertBatch:
    def test_convert_batch_pytorch(self, mock_torch, mock_model, temp_output: Path):
        models = {"model_a": mock_model, "model_b": mock_model}
        samples = {"model_a": mock_torch.Tensor(), "model_b": mock_torch.Tensor()}

        results = ONNXConverter.convert_batch(models, samples, temp_output)

        assert len(results) == 2
        assert mock_torch.onnx.export.call_count == 2

    def test_convert_batch_sklearn(self, temp_output: Path):
        models = {"sk_a": MagicMock(), "sk_b": MagicMock()}
        samples = {}

        with patch("astroml.training.optimization.onnx_converter.skl2onnx") as mock_skl:
            mock_model = MagicMock()
            mock_model.SerializeToString.return_value = b"data"
            mock_skl.convert_sklearn.return_value = mock_model

            results = ONNXConverter.convert_batch(
                models, samples, temp_output, model_type="sklearn"
            )

        assert len(results) == 2
        assert mock_skl.convert_sklearn.call_count == 2

    def test_convert_batch_missing_input(self, mock_torch, mock_model, temp_output: Path):
        models = {"a": mock_model}
        results = ONNXConverter.convert_batch(models, {}, temp_output, model_type="pytorch")
        assert len(results) == 0

    def test_convert_batch_unsupported_type(self, mock_model, temp_output: Path):
        with pytest.raises(ValueError, match="Unsupported model type"):
            ONNXConverter.convert_batch({"a": mock_model}, {}, temp_output, model_type="tensorflow")
