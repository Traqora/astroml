"""API tests for the cost, documentation, security and validation routers.

Covers the API surface added by #644, #645, #646 and #647.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers import cost_management, data_validation, model_security, pipeline_docs


@pytest.fixture
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    """A client over the four routers with isolated per-test state."""
    monkeypatch.setattr(cost_management, "_tracker", cost_management.CostTracker())
    monkeypatch.setattr(
        cost_management, "_budgets", cost_management.BudgetManager(cost_management._tracker)
    )
    monkeypatch.setattr(
        cost_management,
        "_optimizer",
        cost_management.ResourceOptimizer(cost_management._tracker),
    )
    monkeypatch.setattr(
        pipeline_docs, "_generator", pipeline_docs.PipelineDocGenerator(tmp_path / "docs")
    )
    monkeypatch.setattr(model_security, "_MODELS", {})
    monkeypatch.setattr(
        model_security, "_extraction_detector", model_security.ModelExtractionDetector()
    )
    monkeypatch.setattr(data_validation, "_suites", {})
    monkeypatch.setattr(
        data_validation, "_store", data_validation.ValidationStore(tmp_path / "results")
    )
    monkeypatch.setattr(
        data_validation, "_docs", data_validation.DataDocsBuilder(tmp_path / "data_docs")
    )
    monkeypatch.setattr(data_validation, "_SUITE_DIR", tmp_path / "suites")

    app = FastAPI()
    app.include_router(cost_management.router)
    app.include_router(pipeline_docs.router)
    app.include_router(model_security.router)
    app.include_router(data_validation.router)
    with TestClient(app) as test_client:
        yield test_client


# ─── Cost management (#647) ──────────────────────────────────────────────────


class TestCostRouter:
    """/api/v1/cost."""

    def test_record_compute_and_summarise(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/cost/compute",
            json={"resource": "gpu_a100", "hours": 2.0, "project": "fraud"},
        )
        assert response.status_code == 201
        assert response.json()["cost_usd"] > 0

        summary = client.get("/api/v1/cost/summary").json()
        assert summary["record_count"] == 1
        assert summary["by_project"]["fraud"] > 0

    def test_record_api_usage(self, client: TestClient) -> None:
        response = client.post("/api/v1/cost/api-usage", json={"service": "openai", "calls": 1000})
        assert response.status_code == 201
        assert response.json()["unit"] == "calls"

    def test_negative_hours_rejected(self, client: TestClient) -> None:
        response = client.post("/api/v1/cost/compute", json={"resource": "cpu", "hours": -1.0})
        assert response.status_code == 422

    def test_allocation_and_breakdown(self, client: TestClient) -> None:
        client.post(
            "/api/v1/cost/compute",
            json={"resource": "cpu", "hours": 1.0, "project": "fraud", "team": "ml"},
        )
        allocation = client.get("/api/v1/cost/allocation").json()
        assert "fraud" in allocation["by_project"]
        assert "ml" in client.get("/api/v1/cost/breakdown/team").json()

    def test_unknown_breakdown_dimension_rejected(self, client: TestClient) -> None:
        assert client.get("/api/v1/cost/breakdown/nonsense").status_code == 422

    def test_forecast(self, client: TestClient) -> None:
        client.post("/api/v1/cost/compute", json={"resource": "gpu_a100", "hours": 1.0})
        forecast = client.get("/api/v1/cost/forecast?horizon_days=30").json()
        assert forecast["horizon_days"] == 30
        assert forecast["projected_cost_usd"] >= 0

    def test_budget_lifecycle(self, client: TestClient) -> None:
        created = client.post("/api/v1/cost/budgets", json={"name": "monthly", "limit_usd": 10.0})
        assert created.status_code == 201

        client.post("/api/v1/cost/compute", json={"resource": "gpu_a100", "hours": 2.0})
        status = client.get("/api/v1/cost/budgets/monthly").json()
        assert status["spent_usd"] > 0

        alerts = client.post("/api/v1/cost/budgets/evaluate").json()
        assert alerts["alert_count"] >= 1
        assert client.get("/api/v1/cost/alerts").json()["alert_count"] >= 1

        assert client.delete("/api/v1/cost/budgets/monthly").status_code == 204
        assert client.delete("/api/v1/cost/budgets/monthly").status_code == 404

    def test_unknown_budget_returns_404(self, client: TestClient) -> None:
        assert client.get("/api/v1/cost/budgets/ghost").status_code == 404

    def test_invalid_budget_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/cost/budgets",
            json={"name": "b", "limit_usd": 10.0, "thresholds": [-1.0]},
        )
        assert response.status_code == 422

    def test_hard_limit_blocks_spend(self, client: TestClient) -> None:
        client.post(
            "/api/v1/cost/budgets",
            json={"name": "hard", "limit_usd": 1.0, "hard_limit": True},
        )
        allowed = client.post("/api/v1/cost/budgets/check", json={"additional_usd": 0.5})
        assert allowed.json()["allowed"] is True

        blocked = client.post("/api/v1/cost/budgets/check", json={"additional_usd": 50.0})
        assert blocked.status_code == 402

    def test_utilization_and_recommendations(self, client: TestClient) -> None:
        for _ in range(5):
            response = client.post(
                "/api/v1/cost/utilization",
                json={
                    "resource_id": "gpu-0",
                    "resource_type": "gpu_a100",
                    "utilization": 0.01,
                },
            )
            assert response.status_code == 202

        report = client.get("/api/v1/cost/recommendations").json()
        assert report["recommendation_count"] == 1
        assert report["recommendations"][0]["kind"] == "shutdown_idle"


# ─── Pipeline documentation (#646) ───────────────────────────────────────────


def _pipeline_payload(**overrides: Any) -> dict[str, Any]:
    """Return a valid pipeline documentation request body."""
    payload: dict[str, Any] = {
        "name": "fraud-scoring",
        "description": "Nightly scoring.",
        "owner": "ml@example.com",
        "stages": [
            {"name": "ingest", "kind": "source"},
            {"name": "score", "kind": "model", "depends_on": ["ingest"]},
            {"name": "publish", "kind": "sink", "depends_on": ["score"]},
        ],
        "inputs": [{"name": "ledger"}],
        "outputs": [{"name": "alerts"}],
    }
    payload.update(overrides)
    return payload


class TestPipelineDocsRouter:
    """/api/v1/pipeline-docs."""

    def test_generate_returns_markdown(self, client: TestClient) -> None:
        response = client.post("/api/v1/pipeline-docs/generate", json=_pipeline_payload())
        assert response.status_code == 201
        body = response.json()
        assert body["is_complete"] is True
        assert "# Pipeline — fraud-scoring" in body["markdown"]

    def test_generate_and_publish_records_a_version(self, client: TestClient) -> None:
        client.post("/api/v1/pipeline-docs/generate", json=_pipeline_payload(publish=True))
        versions = client.get("/api/v1/pipeline-docs/versions/fraud-scoring").json()
        assert versions["version_count"] == 1

    def test_unknown_pipeline_versions_returns_404(self, client: TestClient) -> None:
        assert client.get("/api/v1/pipeline-docs/versions/ghost").status_code == 404

    def test_incomplete_pipeline_reports_problems(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/pipeline-docs/generate",
            json={"name": "bare"},
        )
        assert response.status_code == 201
        assert response.json()["is_complete"] is False

    def test_unknown_stage_dependency_is_rejected(self, client: TestClient) -> None:
        payload = _pipeline_payload(
            stages=[{"name": "a", "kind": "source", "depends_on": ["ghost"]}]
        )
        assert client.post("/api/v1/pipeline-docs/generate", json=payload).status_code == 422

    @pytest.mark.parametrize("fmt", ["mermaid", "dot", "json"])
    def test_diagram_formats(self, client: TestClient, fmt: str) -> None:
        response = client.post(f"/api/v1/pipeline-docs/diagram?fmt={fmt}", json=_pipeline_payload())
        assert response.status_code == 200
        assert response.json()["format"] == fmt

    def test_model_card_endpoint(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/pipeline-docs/model-card",
            json={
                "name": "fraud-gnn",
                "overview": "GraphSAGE scorer.",
                "owners": ["ml@example.com"],
                "primary_uses": ["Flag accounts"],
                "metrics": {"roc_auc": 0.94},
                "ethical_considerations": ["No auto-freeze."],
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["is_valid"] is True
        assert "# Model Card — fraud-gnn" in body["markdown"]

    def test_index_rebuild(self, client: TestClient) -> None:
        client.post("/api/v1/pipeline-docs/generate", json=_pipeline_payload(publish=True))
        assert client.post("/api/v1/pipeline-docs/index").status_code == 201


# ─── Model security (#645) ───────────────────────────────────────────────────


def _threshold_model(x: Any) -> Any:
    """A simple two-class model used by the security endpoint tests."""
    array = np.asarray(x, dtype=float)
    positive = 1.0 / (1.0 + np.exp(-(array.sum(axis=1) - 1.0) * 8.0))
    return np.column_stack([1.0 - positive, positive])


class TestModelSecurityRouter:
    """/api/v1/model-security."""

    @pytest.fixture(autouse=True)
    def _register(self, client: TestClient) -> None:
        model_security.register_model("fraud", _threshold_model)

    def test_models_are_listed(self, client: TestClient) -> None:
        assert client.get("/api/v1/model-security/models").json()["models"] == ["fraud"]

    def test_unknown_model_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/model-security/attack",
            json={"model_name": "ghost", "features": [[0.1, 0.1]], "labels": [0]},
        )
        assert response.status_code == 404

    def test_attack_endpoint(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/model-security/attack",
            json={
                "model_name": "fraud",
                "attack": "fgsm",
                "features": [[0.4, 0.4], [0.8, 0.8]],
                "labels": [0, 1],
                "config": {"epsilon": 0.5},
                "include_examples": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["attack"] == "fgsm"
        assert len(body["adversarial_examples"]) == 2

    def test_mismatched_labels_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/model-security/attack",
            json={"model_name": "fraud", "features": [[0.1, 0.1]], "labels": [0, 1]},
        )
        assert response.status_code == 422

    def test_robustness_endpoint(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/model-security/robustness",
            json={
                "model_name": "fraud",
                "features": [[0.2, 0.2], [0.9, 0.9]],
                "labels": [0, 1],
                "config": {"epsilon": 0.1, "max_iterations": 5},
            },
        )
        assert response.status_code == 200
        assert set(response.json()["attacks"]) == {"fgsm", "pgd"}

    def test_adversarial_detection_endpoint(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/model-security/detect-adversarial",
            json={"model_name": "fraud", "features": [[0.5, 0.5], [0.1, 0.1]]},
        )
        assert response.status_code == 200
        assert "flagged_indices" in response.json()

    def test_extraction_monitoring(self, client: TestClient) -> None:
        for index in range(25):
            accepted = client.post(
                "/api/v1/model-security/extraction/observe",
                json={
                    "client_id": "prober",
                    "features": [index / 25.0, 1 - index / 25.0],
                    "confidence": 0.5,
                },
            )
            assert accepted.status_code == 202

        verdict = client.get("/api/v1/model-security/extraction/prober").json()
        assert verdict["client_id"] == "prober"
        assert client.get("/api/v1/model-security/extraction").json()["client_count"] == 1

    def test_poisoning_endpoint(self, client: TestClient) -> None:
        features = [[0.0, 0.0]] * 10 + [[3.0, 3.0]] * 10 + [[50.0, 50.0]]
        labels = [0] * 10 + [1] * 10 + [1]
        response = client.post(
            "/api/v1/model-security/poisoning",
            json={"features": features, "labels": labels},
        )
        assert response.status_code == 200
        assert response.json()["suspicious_count"] >= 1

    def test_full_scan(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/model-security/scan",
            json={
                "model_name": "fraud",
                "eval_features": [[0.2, 0.2], [0.9, 0.9]],
                "eval_labels": [0, 1],
                "config": {"epsilon": 0.05, "max_iterations": 5},
                "include_markdown": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["score"]["grade"] in {"A", "B", "C", "D", "F"}
        assert "# Model Security Report" in body["markdown"]


# ─── Data validation (#644) ──────────────────────────────────────────────────


_DATA = {
    "transaction_id": ["t1", "t2", "t3"],
    "amount": [1.0, 2.0, 3.0],
    "status": ["confirmed", "pending", "confirmed"],
}


class TestDataValidationRouter:
    """/api/v1/data-validation."""

    def test_status_reports_the_backend(self, client: TestClient) -> None:
        body = client.get("/api/v1/data-validation/status").json()
        assert "great_expectations_installed" in body
        assert body["suite_count"] == 0

    def test_profile_then_validate(self, client: TestClient) -> None:
        profiled = client.post(
            "/api/v1/data-validation/suites/profile",
            json={"suite_name": "transactions", "data": _DATA},
        )
        assert profiled.status_code == 201
        assert profiled.json()["expectation_count"] > 0

        validated = client.post(
            "/api/v1/data-validation/validate",
            json={"suite_name": "transactions", "data": _DATA},
        )
        assert validated.status_code == 200
        assert validated.json()["success"] is True

    def test_validation_failure_is_reported(self, client: TestClient) -> None:
        client.post(
            "/api/v1/data-validation/suites/profile",
            json={"suite_name": "transactions", "data": _DATA},
        )
        broken = dict(_DATA, amount=[1_000_000.0, 2.0, 3.0])
        response = client.post(
            "/api/v1/data-validation/validate",
            json={"suite_name": "transactions", "data": broken},
        )
        assert response.json()["success"] is False

    def test_register_and_fetch_suite(self, client: TestClient) -> None:
        suite = {
            "expectation_suite_name": "manual",
            "expectations": [
                {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "a"}}
            ],
        }
        created = client.post("/api/v1/data-validation/suites", json={"suite": suite})
        assert created.status_code == 201

        assert (
            client.get("/api/v1/data-validation/suites/manual").json()["expectation_suite_name"]
            == "manual"
        )
        assert len(client.get("/api/v1/data-validation/suites").json()["suites"]) == 1

    def test_invalid_suite_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/data-validation/suites",
            json={"suite": {"expectation_suite_name": "bad", "expectations": [{}]}},
        )
        assert response.status_code == 422

    def test_unknown_suite_returns_404(self, client: TestClient) -> None:
        assert client.get("/api/v1/data-validation/suites/ghost").status_code == 404
        assert client.delete("/api/v1/data-validation/suites/ghost").status_code == 404

    def test_persisted_suite_is_reloaded_from_disk(self, client: TestClient, monkeypatch) -> None:
        client.post(
            "/api/v1/data-validation/suites/profile",
            json={"suite_name": "persisted", "data": _DATA, "persist": True},
        )
        monkeypatch.setattr(data_validation, "_suites", {})
        assert client.get("/api/v1/data-validation/suites/persisted").status_code == 200

    def test_results_history_and_dashboard(self, client: TestClient) -> None:
        client.post(
            "/api/v1/data-validation/suites/profile",
            json={"suite_name": "transactions", "data": _DATA},
        )
        client.post(
            "/api/v1/data-validation/validate",
            json={"suite_name": "transactions", "data": _DATA},
        )

        history = client.get("/api/v1/data-validation/results/transactions").json()
        assert history["run_count"] == 1
        run_id = history["latest"]["run_id"]
        assert (
            client.get(f"/api/v1/data-validation/results/transactions/{run_id}").json()["run_id"]
            == run_id
        )

        dashboard = client.get("/api/v1/data-validation/dashboard").json()
        assert dashboard["all_passing"] is True

    def test_missing_history_returns_404(self, client: TestClient) -> None:
        assert client.get("/api/v1/data-validation/results/ghost").status_code == 404
        assert client.get("/api/v1/data-validation/results/ghost/abc").status_code == 404

    def test_data_docs_build(self, client: TestClient) -> None:
        client.post(
            "/api/v1/data-validation/suites/profile",
            json={"suite_name": "transactions", "data": _DATA},
        )
        client.post(
            "/api/v1/data-validation/validate",
            json={"suite_name": "transactions", "data": _DATA, "build_docs": True},
        )
        assert client.post("/api/v1/data-validation/data-docs").status_code == 201
