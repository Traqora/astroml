"""Tests for the feature selection API router (#635).

Covers every endpoint, request validation, the bounded selection store, the
evaluation scores, and a high-dimensional dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers import feature_selection as feature_selection_module
from astroml.api.routers.feature_selection import router


@pytest.fixture
def client() -> TestClient:
    """TestClient over an app that mounts only the feature-selection router."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _classification_data(
    n_samples: int = 60, n_features: int = 8
) -> tuple[list[list[float]], list[float]]:
    """Build a matrix whose first three columns drive a binary target."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n_samples, n_features))
    logits = X[:, 0] * 2.5 - X[:, 1] * 2.0 + X[:, 2] * 1.5
    y = (rng.random(n_samples) < 1 / (1 + np.exp(-logits))).astype(np.float64)
    return X.tolist(), y.tolist()


def _regression_data(
    n_samples: int = 60, n_features: int = 8
) -> tuple[list[list[float]], list[float]]:
    """Build a matrix with a continuous target."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((n_samples, n_features))
    y = X[:, 0] * 3.0 + X[:, 1] * 1.5 - X[:, 2] * 2.0
    return X.tolist(), y.tolist()


# ---------------------------------------------------------------------------
# Selection endpoints
# ---------------------------------------------------------------------------


def test_filter_selection_returns_requested_count(client: TestClient) -> None:
    data, target = _classification_data()
    response = client.post(
        "/api/v1/feature-selection/filter",
        json={"data": data, "target": target, "method": "mutual_info", "k": 3},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["method"] == "filter-mutual_info"
    assert body["num_features_selected"] == 3
    assert body["num_features_total"] == 8
    assert len(body["selected_indices"]) == 3
    assert len(body["scores"]) == 3
    assert body["selection_id"]


def test_filter_selection_with_feature_names(client: TestClient) -> None:
    data, target = _classification_data(n_features=6)
    names = [f"f{i}" for i in range(6)]
    response = client.post(
        "/api/v1/feature-selection/filter",
        json={"data": data, "target": target, "method": "variance", "k": 2, "feature_names": names},
    )

    assert response.status_code == 201
    body = response.json()
    assert len(body["feature_names"]) == 2
    assert set(body["feature_names"]).issubset(set(names))


def test_embedded_selection_reports_importances(client: TestClient) -> None:
    data, target = _classification_data()
    response = client.post(
        "/api/v1/feature-selection/embedded",
        json={"data": data, "target": target, "method": "tree", "k": 4},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["method"] == "embedded-tree"
    assert body["num_features_selected"] == 4


def test_embedded_selection_falls_back_to_tree(client: TestClient) -> None:
    """An unknown embedded method must not 500 — it defaults to ``tree``."""
    data, target = _classification_data()
    response = client.post(
        "/api/v1/feature-selection/embedded",
        json={"data": data, "target": target, "method": "not-a-method", "k": 2},
    )

    assert response.status_code == 201
    assert response.json()["method"] == "embedded-not-a-method"


def test_hybrid_selection_votes_across_filters(client: TestClient) -> None:
    data, target = _classification_data()
    response = client.post(
        "/api/v1/feature-selection/hybrid",
        json={
            "data": data,
            "target": target,
            "strategy": "vote",
            "methods": ["mutual_info", "correlation", "variance"],
            "min_votes": 2,
            "k": 4,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["method"] == "hybrid-vote"
    assert body["metadata"]["min_votes"] == 2
    assert 0 < body["num_features_selected"] <= 4


def test_wrapper_rfe_selects_requested_count(client: TestClient) -> None:
    data, target = _classification_data()
    response = client.post(
        "/api/v1/feature-selection/wrapper",
        json={
            "data": data,
            "target": target,
            "method": "rfe",
            "estimator": "decision_tree",
            "n_features_to_select": 3,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["method"] == "wrapper-rfe"
    assert body["num_features_selected"] == 3
    assert set(body["selected_indices"]).issubset(set(range(8)))


def test_wrapper_forward_selection_with_regression_target(client: TestClient) -> None:
    """Wrapper search needs a regressor when the target is continuous."""
    data, target = _regression_data()
    response = client.post(
        "/api/v1/feature-selection/wrapper",
        json={
            "data": data,
            "target": target,
            "method": "forward",
            "estimator": "decision_tree",
            "task": "regression",
            "n_features_to_select": 2,
            "cv": 2,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["method"] == "wrapper-forward"
    assert body["num_features_selected"] == 2


def test_wrapper_rejects_unknown_estimator(client: TestClient) -> None:
    data, target = _classification_data()
    response = client.post(
        "/api/v1/feature-selection/wrapper",
        json={"data": data, "target": target, "estimator": "xgboost"},
    )

    assert response.status_code == 422
    assert "xgboost" in response.json()["detail"]


def test_wrapper_n_features_capped_at_column_count(client: TestClient) -> None:
    """Asking for more features than exist must not blow up sklearn."""
    data, target = _classification_data(n_features=5)
    response = client.post(
        "/api/v1/feature-selection/wrapper",
        json={"data": data, "target": target, "method": "rfe", "n_features_to_select": 50},
    )

    assert response.status_code == 201
    assert response.json()["num_features_selected"] == 5


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_maps_indices_back_to_original_columns(client: TestClient) -> None:
    data, target = _classification_data(n_features=10)
    response = client.post(
        "/api/v1/feature-selection/pipeline",
        json={
            "data": data,
            "target": target,
            "feature_names": [f"feat_{i}" for i in range(10)],
            "steps": [
                {"type": "filter", "method": "mutual_info", "k": 6},
                {"type": "embedded", "method": "tree", "k": 3},
            ],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["num_features_total"] == 10
    assert body["num_features_selected"] == 3
    # Step 2 reports indices into step 1's reduced matrix; the API must resolve
    # them against the original columns instead.
    assert max(body["selected_indices"]) < 10
    assert body["feature_names"] == [f"feat_{i}" for i in body["selected_indices"]]
    assert [s["num_features_selected"] for s in body["metadata"]["steps"]] == [6, 3]
    assert "2 steps" in body["metadata"]["pipeline"]


def test_pipeline_supports_wrapper_step(client: TestClient) -> None:
    data, target = _classification_data(n_features=8)
    response = client.post(
        "/api/v1/feature-selection/pipeline",
        json={
            "data": data,
            "target": target,
            "steps": [
                {"type": "filter", "method": "variance", "k": 5},
                {"type": "wrapper", "method": "rfe", "estimator": "random_forest", "k": 2},
            ],
        },
    )

    assert response.status_code == 201
    assert response.json()["num_features_selected"] == 2


def test_pipeline_rejects_unknown_step_type(client: TestClient) -> None:
    data, target = _classification_data()
    response = client.post(
        "/api/v1/feature-selection/pipeline",
        json={"data": data, "target": target, "steps": [{"type": "magic", "method": "x"}]},
    )

    assert response.status_code == 422


def test_pipeline_rejects_empty_steps(client: TestClient) -> None:
    data, target = _classification_data()
    response = client.post(
        "/api/v1/feature-selection/pipeline",
        json={"data": data, "target": target, "steps": []},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluate_reports_held_out_test_score(client: TestClient) -> None:
    data, target = _classification_data(n_samples=90)
    selection = client.post(
        "/api/v1/feature-selection/filter",
        json={"data": data, "target": target, "method": "mutual_info", "k": 4},
    ).json()

    response = client.post(f"/api/v1/feature-selection/evaluate/{selection['selection_id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["selection_id"] == selection["selection_id"]
    # Every score the response advertises must actually be computed.
    for field in ("train_score", "test_score", "cv_score", "cv_std", "run_time_ms"):
        assert body[field] is not None, field
    assert 0.0 <= body["test_score"] <= 1.0
    assert body["run_time_ms"] >= 0.0


def test_evaluate_handles_continuous_target(client: TestClient) -> None:
    """A regression target is scored by a regressor, not rejected outright."""
    data, target = _regression_data(n_samples=60)
    selection = client.post(
        "/api/v1/feature-selection/filter",
        json={"data": data, "target": target, "method": "correlation", "k": 3},
    ).json()

    response = client.post(
        f"/api/v1/feature-selection/evaluate/{selection['selection_id']}", json=None
    )

    assert response.status_code == 200
    assert response.json()["test_score"] is not None


def test_evaluate_unknown_selection_is_404(client: TestClient) -> None:
    response = client.post("/api/v1/feature-selection/evaluate/doesnotexist")
    assert response.status_code == 404
    assert response.json()["detail"] == "Selection not found"


def test_evaluate_needs_enough_rows(client: TestClient) -> None:
    selection = client.post(
        "/api/v1/feature-selection/filter",
        json={"data": [[1.0, 2.0], [3.0, 4.0]], "target": [0.0, 1.0], "method": "variance"},
    ).json()

    response = client.post(f"/api/v1/feature-selection/evaluate/{selection['selection_id']}")

    assert response.status_code == 422
    assert "at least 4 rows" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"data": [], "target": []}, "non-empty matrix"),
        ({"data": [[1.0, 2.0], [3.0]], "target": [0.0, 1.0]}, "same width"),
        ({"data": [[1.0, 2.0]], "target": [0.0, 1.0]}, "target has 2 values"),
        (
            {"data": [[1.0, 2.0], [3.0, 4.0]], "target": [0.0, 1.0], "feature_names": ["a"]},
            "feature_names has 1 entries",
        ),
        (
            {"data": [[1.0, 2.0], [3.0, 4.0]], "target": [0.0, 1.0], "feature_names": ["a"]},
            "feature_names has 1 entries",
        ),
        ({"data": [[1.0]], "target": [0.0], "unknown_field": 1}, None),
    ],
)
def test_malformed_payloads_are_422_not_500(
    client: TestClient, payload: dict[str, object], expected: str | None
) -> None:
    """Bad input is a client error; it must not reach numpy as a 500."""
    response = client.post("/api/v1/feature-selection/filter", json=payload)

    assert response.status_code == 422
    if expected is not None:
        assert expected in response.json()["detail"]


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity"])
def test_non_finite_payload_is_422(client: TestClient, literal: bytes) -> None:
    """``NaN``/``Infinity`` parse from JSON but must not reach sklearn.

    They are sent as raw bytes because the client-side JSON encoder refuses to
    emit them in the first place.
    """
    response = client.post(
        "/api/v1/feature-selection/filter",
        content=b'{"data": [[' + literal + b', 2.0], [3.0, 4.0]], "target": [0.0, 1.0]}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert "finite" in response.json()["detail"]


def test_high_dimensional_selection_keeps_informative_features(
    client: TestClient,
) -> None:
    """Step 8 of #635: selection behaves on a wide, mostly-noise matrix."""
    rng = np.random.default_rng(3)
    n_samples, n_features = 120, 200
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 5] * 3.0 + X[:, 77] * -2.5 + X[:, 199] * 2.0 > 0).astype(np.float64)

    response = client.post(
        "/api/v1/feature-selection/filter",
        json={
            "data": X.tolist(),
            "target": y.tolist(),
            "feature_names": [f"f{i}" for i in range(n_features)],
            "method": "mutual_info",
            "k": 10,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["num_features_total"] == 200
    assert body["num_features_selected"] == 10
    assert {5, 77, 199}.issubset(set(body["selected_indices"]))
    assert body["feature_names"] == [f"f{i}" for i in body["selected_indices"]]


# ---------------------------------------------------------------------------
# Store and registration
# ---------------------------------------------------------------------------


def test_selection_store_is_bounded(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-memory store must evict rather than grow without limit."""
    monkeypatch.setattr(feature_selection_module, "MAX_STORED_SELECTIONS", 2)
    feature_selection_module._selections.clear()
    data, target = _classification_data(n_samples=20, n_features=4)
    payload = {"data": data, "target": target, "method": "variance"}

    first = client.post("/api/v1/feature-selection/filter", json=payload).json()["selection_id"]
    for _ in range(3):
        client.post("/api/v1/feature-selection/filter", json=payload)

    assert len(feature_selection_module._selections) == 2
    assert first not in feature_selection_module._selections
    assert client.post(f"/api/v1/feature-selection/evaluate/{first}").status_code == 404


def test_response_models_are_consistent(client: TestClient) -> None:
    """Generating the OpenAPI schema validates every response model."""
    schema = client.get("/openapi.json").json()
    paths = [p for p in schema["paths"] if p.startswith("/api/v1/feature-selection")]
    assert len(paths) == 6


def test_router_is_exposed_by_the_routers_package() -> None:
    """``routers.__getattr__`` only serves names listed in ``__all__``."""
    from astroml.api import routers

    assert "feature_selection" in routers.__all__
    assert routers.feature_selection.router is router


def test_router_is_mounted_on_the_api_app() -> None:
    """#635 step 6 is only delivered if the routes exist on the real app."""
    pytest.importorskip("greenlet", reason="astroml.api.app needs sqlalchemy asyncio support")
    from astroml.api.app import app

    paths = {route.path for route in app.routes}
    assert "/api/v1/feature-selection/filter" in paths
    assert "/api/v1/feature-selection/wrapper" in paths
    assert "/api/v1/feature-selection/pipeline" in paths
    assert "/api/v1/feature-selection/evaluate/{selection_id}" in paths
