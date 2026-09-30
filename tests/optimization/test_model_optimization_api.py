"""Tests for model optimization API endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.model_optimization import router

app = FastAPI()
app.include_router(router)


@pytest.fixture
def client():
    return TestClient(app)


class TestConvertEndpoint:
    def test_convert_pytorch(self, client):
        response = client.post(
            "/api/v1/optimization/convert",
            json={
                "model_type": "pytorch",
                "model_name": "my_model",
                "opset_version": 17,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "my_model"
        assert data["status"] == "converted"

    def test_convert_sklearn(self, client):
        response = client.post(
            "/api/v1/optimization/convert",
            json={
                "model_type": "sklearn",
                "model_name": "sk_model",
            },
        )
        assert response.status_code == 200
        assert response.json()["model_id"] == "sk_model"


class TestOptimizeEndpoint:
    def test_optimize_model_not_found(self, client):
        response = client.post(
            "/api/v1/optimization/nonexistent/optimize",
            json={"optimization_level": "basic"},
        )
        assert response.status_code == 404

    @patch("astroml.api.routers.model_optimization.ONNXOptimizer")
    def test_optimize_success(self, mock_optimizer, client, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True)
        (model_dir / "test_model.onnx").write_bytes(b"dummy")

        mock_result = MagicMock()
        mock_result.original_size = 1000
        mock_result.optimized_size = 600
        mock_result.original_nodes = 20
        mock_result.optimized_nodes = 12
        mock_result.size_reduction = 0.4
        mock_result.node_reduction = 0.4
        mock_result.optimizations_applied = ["fuse"]
        mock_optimizer.optimize.return_value = mock_result

        with patch("astroml.api.routers.model_optimization._MODEL_STORE", model_dir):
            response = client.post(
                "/api/v1/optimization/test_model/optimize",
                json={"optimization_level": "extended"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["original_size"] == 1000
        assert data["optimized_size"] == 600

    @patch("astroml.api.routers.model_optimization.ONNXOptimizer")
    def test_optimize_invalid_level(self, mock_optimizer, client, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True)
        (model_dir / "test_model.onnx").write_bytes(b"dummy")

        mock_optimizer.optimize.side_effect = ValueError("Invalid optimization level")

        with patch("astroml.api.routers.model_optimization._MODEL_STORE", model_dir):
            response = client.post(
                "/api/v1/optimization/test_model/optimize",
                json={"optimization_level": "invalid"},
            )

        assert response.status_code == 422


class TestQuantizeEndpoint:
    def test_quantize_model_not_found(self, client):
        response = client.post(
            "/api/v1/optimization/nonexistent/quantize",
            json={"quantization_type": "INT8"},
        )
        assert response.status_code == 404

    @patch("astroml.api.routers.model_optimization.quantize_model")
    def test_quantize_success(self, mock_quantize, client, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True)
        (model_dir / "test_model.onnx").write_bytes(b"dummy")

        mock_result = MagicMock()
        mock_result.original_size = 1000
        mock_result.quantized_size = 300
        mock_result.quantization_type = "INT8"
        mock_result.size_reduction = 0.7
        mock_quantize.return_value = mock_result

        with patch("astroml.api.routers.model_optimization._MODEL_STORE", model_dir):
            response = client.post(
                "/api/v1/optimization/test_model/quantize",
                json={"quantization_type": "INT8", "per_channel": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["original_size"] == 1000
        assert data["quantized_size"] == 300
        assert data["quantization_type"] == "INT8"


class TestBenchmarkEndpoint:
    def test_benchmark_model_not_found(self, client):
        response = client.get("/api/v1/optimization/nonexistent/benchmark?n_runs=10")
        assert response.status_code == 404

    @patch("astroml.api.routers.model_optimization.ONNXRuntime")
    @patch("astroml.api.routers.model_optimization.ONNXOptimizer")
    def test_benchmark_success(self, mock_optimizer, mock_runtime, client, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True)
        (model_dir / "test_model.onnx").write_bytes(b"dummy")

        mock_rt_instance = MagicMock()
        mock_input = {"shape": [1, 3, 224, 224]}
        mock_rt_instance.get_input_info.return_value = [mock_input]
        mock_runtime.load_model.return_value = mock_rt_instance

        mock_result = MagicMock()
        mock_result.mean_latency = 0.015
        mock_result.std_latency = 0.002
        mock_result.min_latency = 0.010
        mock_result.max_latency = 0.020
        mock_result.n_runs = 50
        mock_optimizer.benchmark.return_value = mock_result

        with patch("astroml.api.routers.model_optimization._MODEL_STORE", model_dir):
            response = client.get("/api/v1/optimization/test_model/benchmark?n_runs=50")

        assert response.status_code == 200
        data = response.json()
        assert data["mean_latency"] == 0.015
        assert data["n_runs"] == 50


class TestPredictEndpoint:
    def test_predict_model_not_found(self, client):
        response = client.post(
            "/api/v1/optimization/nonexistent/predict",
            json={"input_data": [[1.0, 2.0]]},
        )
        assert response.status_code == 404

    @patch("astroml.api.routers.model_optimization.ONNXRuntime")
    def test_predict_success(self, mock_runtime, client, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True)
        (model_dir / "test_model.onnx").write_bytes(b"dummy")

        mock_rt_instance = MagicMock()
        mock_runtime.load_model.return_value = mock_rt_instance
        mock_rt_instance.predict.return_value = [np.array([[0.1, 0.9], [0.8, 0.2]])]

        with patch("astroml.api.routers.model_optimization._MODEL_STORE", model_dir):
            response = client.post(
                "/api/v1/optimization/test_model/predict",
                json={"input_data": [[1.0, 2.0], [3.0, 4.0]]},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["outputs"]) == 1
        assert data["outputs"][0] == [[0.1, 0.9], [0.8, 0.2]]
