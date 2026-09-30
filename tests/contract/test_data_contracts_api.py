from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from astroml.api.routers.data_contracts import _verifier
from astroml.pipeline.contracts.schema_contract import SchemaContract

# We need a FastAPI app to test against
try:
    from astroml.api.app import app
except ImportError:
    # Fallback: create a minimal app for testing
    from fastapi import FastAPI

    app = FastAPI()
    from astroml.api.routers.data_contracts import router

    app.include_router(router)

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_verifier() -> None:
    """Reset the verifier state before each test."""
    _verifier.contracts.clear()
    _verifier.breach_history.clear()
    _verifier._failure_callbacks.clear()


class TestValidateEndpoint:
    def test_validate_valid_data(self) -> None:
        response = client.post(
            "/api/v1/contracts/validate",
            json={
                "data": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                "contract_type": "schema",
                "schema_def": {
                    "columns": {
                        "id": {"dtype": "int64", "nullable": False},
                        "name": {"dtype": "object", "nullable": True},
                    }
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"]
        assert data["contract_type"] == "schema"

    def test_validate_invalid_data(self) -> None:
        response = client.post(
            "/api/v1/contracts/validate",
            json={
                "data": [{"id": 1}, {"id": None}],
                "contract_type": "schema",
                "schema_def": {
                    "columns": {
                        "id": {"dtype": "int64", "nullable": False},
                    }
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert not data["is_valid"]

    def test_validate_empty_data(self) -> None:
        response = client.post(
            "/api/v1/contracts/validate",
            json={
                "data": [],
                "contract_type": "schema",
                "schema_def": {"columns": {"id": {"dtype": "int64"}}},
            },
        )
        assert response.status_code == 400
        assert "non-empty" in response.json()["detail"].lower()

    def test_validate_invalid_json(self) -> None:
        response = client.post(
            "/api/v1/contracts/validate",
            json={
                "data": [{"id": "not_a_number"}],
                "contract_type": "schema",
                "schema_def": {
                    "columns": {
                        "id": {"dtype": "int64"},
                    }
                },
            },
        )
        assert response.status_code == 200

    def test_validate_quality_contract(self) -> None:
        response = client.post(
            "/api/v1/contracts/validate",
            json={
                "data": [{"age": 25}, {"age": 30}],
                "contract_type": "quality",
                "schema_def": {
                    "columns": {
                        "age": {"dtype": "numeric", "nullable": True},
                    }
                },
            },
        )
        assert response.status_code == 200


class TestInferEndpoint:
    def test_infer_from_data(self) -> None:
        response = client.post(
            "/api/v1/contracts/infer",
            json={
                "data": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
                "contract_type": "schema",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["contract_type"] == "schema"
        assert "a" in data["columns"]
        assert "b" in data["columns"]
        assert data["row_count"] == 2

    def test_infer_empty_data(self) -> None:
        response = client.post(
            "/api/v1/contracts/infer",
            json={
                "data": [],
                "contract_type": "schema",
            },
        )
        assert response.status_code == 400


class TestBreachesEndpoint:
    def test_get_breaches_empty(self) -> None:
        response = client.get("/api/v1/contracts/breaches")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["breaches"] == []

    def test_get_breaches_after_validation(self) -> None:
        # Trigger a breach by validating bad data
        client.post(
            "/api/v1/contracts/validate",
            json={
                "data": [{"id": None}],
                "contract_type": "schema",
                "schema_def": {
                    "columns": {
                        "id": {"dtype": "int64", "nullable": False},
                    }
                },
            },
        )
        response = client.get("/api/v1/contracts/breaches")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert data["breaches"][0]["contract_type"] == "schema"


class TestPipelineVerifyEndpoint:
    def test_pipeline_verify(self) -> None:
        schema = SchemaContract(name="pipeline_schema")
        _verifier.add_contract(schema, "stage1_schema")

        response = client.post(
            "/api/v1/contracts/pipeline/verify",
            json={
                "data": [{"id": 1}, {"id": 2}],
                "stages": {
                    "ingest": {"contracts": ["stage1_schema"]},
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "passed" in data
        assert "ingest" in data["stages"]

    def test_pipeline_verify_empty_data(self) -> None:
        response = client.post(
            "/api/v1/contracts/pipeline/verify",
            json={
                "data": [],
                "stages": {},
            },
        )
        assert response.status_code == 400

    def test_pipeline_verify_unknown_contract_passes_silently_by_default(self) -> None:
        # Reproduces the CI-validation gap issue #968 targets: a stage
        # naming a contract that was never registered (e.g. a typo, or a
        # contract removed without updating the pipeline config) reports a
        # 200 with passed=True rather than surfacing the misconfiguration.
        response = client.post(
            "/api/v1/contracts/pipeline/verify",
            json={
                "data": [{"id": 1}],
                "stages": {"ingest": {"contracts": ["this_contract_does_not_exist"]}},
            },
        )
        assert response.status_code == 200
        assert response.json()["passed"] is True

    def test_pipeline_verify_unknown_contract_rejected_in_strict_mode(self) -> None:
        response = client.post(
            "/api/v1/contracts/pipeline/verify",
            json={
                "data": [{"id": 1}],
                "stages": {"ingest": {"contracts": ["this_contract_does_not_exist"]}},
                "strict": True,
            },
        )
        assert response.status_code == 422

    def test_pipeline_verify_strict_mode_still_passes_for_known_contracts(self) -> None:
        schema = SchemaContract(name="strict_ok_schema")
        _verifier.add_contract(schema, "strict_ok_schema")

        response = client.post(
            "/api/v1/contracts/pipeline/verify",
            json={
                "data": [{"id": 1}],
                "stages": {"ingest": {"contracts": ["strict_ok_schema"]}},
                "strict": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["passed"] is True

    def test_validate_quality_nullable(self) -> None:
        response = client.post(
            "/api/v1/contracts/validate",
            json={
                "data": [{"age": 25}, {"age": None}],
                "contract_type": "quality",
                "schema_def": {
                    "columns": {
                        "age": {"dtype": "numeric", "nullable": False},
                    }
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert not data["is_valid"]

    def test_validate_quality_range(self) -> None:
        response = client.post(
            "/api/v1/contracts/validate",
            json={
                "data": [{"age": 25}, {"age": 150}],
                "contract_type": "quality",
                "schema_def": {
                    "columns": {
                        "age": {"dtype": "numeric", "nullable": True},
                    }
                },
            },
        )
        assert response.status_code == 200

    def test_validate_unknown_contract_type(self) -> None:
        response = client.post(
            "/api/v1/contracts/validate",
            json={
                "data": [{"x": 1}],
                "contract_type": "unknown_type",
                "schema_def": {"columns": {"x": {"dtype": "int64"}}},
            },
        )
        assert response.status_code == 422  # FastAPI validation rejects invalid pattern


class TestApiResponseModels:
    def test_validate_response_structure(self) -> None:
        response = client.post(
            "/api/v1/contracts/validate",
            json={
                "data": [{"x": 1}],
                "contract_type": "schema",
                "schema_def": {"columns": {"x": {"dtype": "int64"}}},
            },
        )
        body = response.json()
        assert "is_valid" in body
        assert "contract_type" in body
        assert "details" in body

    def test_infer_response_structure(self) -> None:
        response = client.post(
            "/api/v1/contracts/infer",
            json={
                "data": [{"x": 1}],
                "contract_type": "schema",
            },
        )
        body = response.json()
        assert "contract_type" in body
        assert "columns" in body
        assert "row_count" in body
