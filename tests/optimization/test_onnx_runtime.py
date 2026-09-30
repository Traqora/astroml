"""Tests for ONNXRuntime."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroml.serving.onnx_runtime import ONNXRuntime


@pytest.fixture
def mock_ort():
    with patch("astroml.serving.onnx_runtime.ort") as mock:
        mock.get_available_providers.return_value = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ]
        yield mock


@pytest.fixture
def mock_session():
    session = MagicMock()

    mock_inp = MagicMock()
    mock_inp.name = "input"
    mock_inp.shape = [1, 3, 224, 224]
    mock_inp.type = "tensor(float)"
    session.get_inputs.return_value = [mock_inp]

    mock_out = MagicMock()
    mock_out.name = "output"
    mock_out.shape = [1, 1000]
    mock_out.type = "tensor(float)"
    session.get_outputs.return_value = [mock_out]

    return session


@pytest.fixture
def dummy_onnx_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"dummy onnx content")
    return path


class TestLoadModel:
    def test_load_model_with_providers(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session

        rt = ONNXRuntime.load_model(dummy_onnx_path, providers=["CPUExecutionProvider"])

        assert rt._session is not None
        mock_ort.InferenceSession.assert_called_once_with(
            str(dummy_onnx_path), providers=["CPUExecutionProvider"]
        )

    def test_load_model_no_providers(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session

        rt = ONNXRuntime.load_model(dummy_onnx_path)

        assert rt._session is not None
        mock_ort.InferenceSession.assert_called_once_with(str(dummy_onnx_path))

    def test_load_model_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            ONNXRuntime.load_model("/nonexistent/model.onnx")

    def test_load_model_runtime_error(self, mock_ort, dummy_onnx_path: Path):
        mock_ort.InferenceSession.side_effect = Exception("load failed")

        with pytest.raises(RuntimeError, match="load failed"):
            ONNXRuntime.load_model(dummy_onnx_path)


class TestPredict:
    def test_predict_single_input(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session
        mock_session.run.return_value = [np.array([[0.1, 0.9]])]

        rt = ONNXRuntime.load_model(dummy_onnx_path)
        result = rt.predict(np.array([[1.0, 2.0, 3.0]]))

        assert len(result) == 1
        assert result[0][0][1] == pytest.approx(0.9)

    def test_predict_dict_input(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session
        mock_session.run.return_value = [np.array([[0.5]])]

        rt = ONNXRuntime.load_model(dummy_onnx_path)
        result = rt.predict({"input": np.array([[1.0]])})

        assert len(result) == 1

    def test_predict_no_model(self):
        rt = ONNXRuntime()
        with pytest.raises(RuntimeError, match="No model loaded"):
            rt.predict(np.array([[1.0]]))


class TestPredictBatch:
    def test_predict_batch(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session
        mock_session.run.return_value = [np.array([[0.5]])]

        rt = ONNXRuntime.load_model(dummy_onnx_path)
        inputs = [np.array([[1.0]]), np.array([[2.0]])]
        results = rt.predict_batch(inputs)

        assert len(results) == 2
        assert mock_session.run.call_count == 2

    def test_predict_batch_no_model(self):
        rt = ONNXRuntime()
        with pytest.raises(RuntimeError, match="No model loaded"):
            rt.predict_batch([np.array([[1.0]])])


class TestInputOutputInfo:
    def test_get_input_info(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session

        rt = ONNXRuntime.load_model(dummy_onnx_path)
        info = rt.get_input_info()

        assert len(info) == 1
        assert info[0]["name"] == "input"

    def test_get_output_info(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session

        rt = ONNXRuntime.load_model(dummy_onnx_path)
        info = rt.get_output_info()

        assert len(info) == 1
        assert info[0]["name"] == "output"

    def test_get_input_info_no_model(self):
        rt = ONNXRuntime()
        with pytest.raises(RuntimeError, match="No model loaded"):
            rt.get_input_info()

    def test_get_output_info_no_model(self):
        rt = ONNXRuntime()
        with pytest.raises(RuntimeError, match="No model loaded"):
            rt.get_output_info()


class TestProviders:
    def test_get_providers(self, mock_ort):
        rt = ONNXRuntime()
        providers = rt.get_providers()
        assert "CPUExecutionProvider" in providers
        assert "CUDAExecutionProvider" in providers
        mock_ort.get_available_providers.assert_called_once()

    def test_set_provider(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session

        rt = ONNXRuntime.load_model(dummy_onnx_path, providers=["CPUExecutionProvider"])
        mock_ort.InferenceSession.reset_mock()

        rt.set_provider("CUDAExecutionProvider")

        mock_ort.InferenceSession.assert_called_once_with(
            str(dummy_onnx_path), providers=["CUDAExecutionProvider"]
        )

    def test_set_provider_unavailable(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session

        rt = ONNXRuntime.load_model(dummy_onnx_path)

        with pytest.raises(ValueError, match="not available"):
            rt.set_provider("TensorRTExecutionProvider")

    def test_set_provider_no_model(self):
        rt = ONNXRuntime()
        with pytest.raises(RuntimeError, match="No model loaded"):
            rt.set_provider("CPUExecutionProvider")


class TestTransformHooks:
    def test_transform_input_default(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session

        rt = ONNXRuntime.load_model(dummy_onnx_path)
        data = np.array([[1.0]])
        assert rt.transform_input(data) is data

    def test_transform_output_default(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session

        rt = ONNXRuntime.load_model(dummy_onnx_path)
        data = np.array([[1.0]])
        assert rt.transform_output(data) is data


class TestContextManager:
    def test_context_manager(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session

        with ONNXRuntime.load_model(dummy_onnx_path) as rt:
            assert rt._session is not None

        assert rt._session is None
        assert rt._model_path is None

    def test_close(self, mock_ort, mock_session, dummy_onnx_path: Path):
        mock_ort.InferenceSession.return_value = mock_session

        rt = ONNXRuntime.load_model(dummy_onnx_path)
        rt.close()

        assert rt._session is None
        assert rt._model_path is None
        assert rt._providers == []
