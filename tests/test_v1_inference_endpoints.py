"""HTTP-level tests for the supported v1 inference surface (issue #1224)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "v1-inference-api.md"


@pytest.fixture(scope="module")
def readiness_module() -> Any:
    """Load the readiness router without importing the full application graph."""
    import importlib.util

    path = ROOT / "api" / "routers" / "v1" / "readiness.py"
    spec = importlib.util.spec_from_file_location("astroml_v1_readiness", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.skip(f"Cannot load readiness router from {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"API dependencies unavailable: {exc}")

    return module


def _client(readiness_module: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Mount the router on a bare FastAPI app with a known checkpoint path."""
    app = FastAPI()
    app.include_router(readiness_module.router)
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "models/ready.pt")
    return TestClient(app)


def test_readiness_reports_loaded_model(
    readiness_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loadable scorer returns the stable 200 response."""
    monkeypatch.setattr(readiness_module, "_get_scorer", lambda: object())
    client = _client(readiness_module, monkeypatch)

    response = client.get("/api/v1/fraud/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "fraud",
        "checkpoint": "models/ready.pt",
        "ready": True,
    }


def test_readiness_fails_closed_when_checkpoint_is_missing(
    readiness_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing scorer returns a stable 503 with remediation guidance."""
    monkeypatch.setattr(readiness_module, "_get_scorer", lambda: None)
    client = _client(readiness_module, monkeypatch)

    response = client.get("/api/v1/fraud/ready")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "fail"
    assert detail["reason"] == "model_unavailable"
    assert detail["checkpoint"] == "models/ready.pt"
    assert "MODEL_CHECKPOINT_PATH" in detail["remediation"]


def test_readiness_surfaces_loader_errors_as_503(
    readiness_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loader exceptions do not leak internals or become a 500."""

    def _boom() -> None:
        raise RuntimeError("checkpoint corrupt")

    monkeypatch.setattr(readiness_module, "_get_scorer", _boom)
    client = _client(readiness_module, monkeypatch)

    response = client.get("/api/v1/fraud/ready")

    assert response.status_code == 503
    assert "checkpoint corrupt" not in response.text


def test_docs_define_the_supported_v1_surface() -> None:
    """The user-facing contract lists every supported endpoint and auth."""
    text = DOC.read_text(encoding="utf-8")

    for needle in (
        "GET /api/v1",
        "GET /api/v1/fraud/ready",
        "POST /api/v1/fraud/score",
        "Authorization: Bearer",
        "Not part of the v1 inference contract",
    ):
        assert needle in text
