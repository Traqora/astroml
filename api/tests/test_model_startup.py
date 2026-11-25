from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

import astroml.api.app as api_app


def test_configured_model_is_loaded_once_before_scheduler_start(monkeypatch, tmp_path):
    artifact_path = tmp_path / "model.pkl"
    artifact_path.write_bytes(b"trusted artifact")
    loader = Mock()
    loader.load_model.return_value = object()
    start_scheduler = Mock()
    stop_scheduler = AsyncMock()
    monkeypatch.setattr(api_app, "_model_loader", loader)
    monkeypatch.setattr(api_app, "start_scheduler", start_scheduler)
    monkeypatch.setattr(api_app, "stop_scheduler", stop_scheduler)
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(artifact_path))
    monkeypatch.setenv("MODEL_NAME", "api-model")

    with TestClient(api_app.app):
        loader.load_model.assert_called_once_with("api-model", str(artifact_path))
        start_scheduler.assert_called_once()

    stop_scheduler.assert_awaited_once()


def test_invalid_model_path_stops_startup_before_scheduler(monkeypatch):
    loader = Mock()
    loader.load_model.side_effect = FileNotFoundError("missing artifact")
    start_scheduler = Mock()
    monkeypatch.setattr(api_app, "_model_loader", loader)
    monkeypatch.setattr(api_app, "start_scheduler", start_scheduler)
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", "/missing/model.pkl")

    with pytest.raises(FileNotFoundError, match="missing artifact"):
        with TestClient(api_app.app):
            pass

    start_scheduler.assert_not_called()