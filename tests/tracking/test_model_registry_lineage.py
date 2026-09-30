"""Tests for ModelMetadata, TrainingLineage, ModelStage, and Model Registry API."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.model_registry import router as model_registry_router
from astroml.tracking.lineage import (
    DataLineageTracker,
    ModelLineage,
    TrainingLineage,
)
from astroml.tracking.metadata import (
    ModelFramework,
    ModelMetadata,
    TaskType,
)
from astroml.tracking.model_registry import (
    DeploymentEnvironment,
    ModelRegistry,
    ModelStage,
    SemanticVersion,
)


class TestMetadataAndLineage:
    def test_model_metadata_crud(self):
        meta = ModelMetadata(
            model_name="fraud_detector_v1",
            framework=ModelFramework.PYTORCH.value,
            task_type=TaskType.BINARY_CLASSIFICATION.value,
            description="Production fraud detector",
            author="AstroML Team",
            tags=["production", "v1"],
            hyperparameters={"learning_rate": 0.001, "hidden_dim": 64},
            metrics={"f1_score": 0.94, "precision": 0.96},
        )
        assert len(meta.validate()) == 0
        meta.add_tag("fraud")
        assert "fraud" in meta.tags
        assert meta.remove_tag("v1")

        meta.update_metrics({"recall": 0.92})
        assert meta.metrics["recall"] == 0.92

        data = meta.to_dict()
        restored = ModelMetadata.from_dict(data)
        assert restored.model_name == meta.model_name
        assert restored.metrics["f1_score"] == 0.94

    def test_training_lineage_and_dag(self):
        lineage = TrainingLineage(
            dataset_id="stellar_tx_2026_q1",
            dataset_version="v2.1",
            commit_hash="a1b2c3d4e5",
            hyperparameters={"batch_size": 64},
        )
        assert lineage.dataset_id == "stellar_tx_2026_q1"
        assert lineage.dataset_version == "v2.1"

        model_lin = ModelLineage(
            model_name="stellar_classifier",
            version="1.0.0",
            training_lineage=lineage,
        )
        lin_dict = model_lin.to_dict()
        assert lin_dict["model_name"] == "stellar_classifier"
        assert "stellar_tx_2026_q1" in lin_dict["upstream_nodes"]

    def test_semantic_versioning(self):
        v1 = SemanticVersion("1.0.0")
        v2 = SemanticVersion("1.1.0")
        v3 = SemanticVersion("2.0.0")
        assert v1 < v2 < v3
        assert SemanticVersion("1.0.0") == v1


class TestModelRegistryAPI:
    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(model_registry_router)
        return TestClient(app)

    def test_full_registry_api_lifecycle(self, client):
        # 1. Register Model
        res_reg = client.post(
            "/api/v1/models",
            json={
                "name": "stellar_gnn_model",
                "framework": "pytorch",
                "task_type": "graph_node_classification",
                "description": "GNN model for Stellar graph analysis",
                "tags": ["graph", "production"],
            },
        )
        assert res_reg.status_code == 200
        assert res_reg.json()["model"]["name"] == "stellar_gnn_model"

        # 2. List Models
        res_list = client.get("/api/v1/models?framework=pytorch")
        assert res_list.status_code == 200
        assert any(m["name"] == "stellar_gnn_model" for m in res_list.json()["models"])

        # 3. Create Version 1.0.0
        res_v1 = client.post(
            "/api/v1/models/stellar_gnn_model/versions",
            json={
                "version": "1.0.0",
                "stage": "staging",
                "parameters": {"lr": 0.01},
                "metrics": {"auc": 0.89},
                "dataset_id": "dataset_stellar_graphs_v1",
            },
        )
        assert res_v1.status_code == 200
        assert res_v1.json()["model_version"]["version"] == "1.0.0"

        # 4. Create Version 1.1.0 (auto-incremented or specified)
        res_v2 = client.post(
            "/api/v1/models/stellar_gnn_model/versions",
            json={
                "version": "1.1.0",
                "stage": "development",
                "parameters": {"lr": 0.005},
                "metrics": {"auc": 0.93},
                "dataset_id": "dataset_stellar_graphs_v2",
            },
        )
        assert res_v2.status_code == 200

        # 5. List Versions
        res_versions = client.get("/api/v1/models/stellar_gnn_model/versions")
        assert res_versions.status_code == 200
        assert res_versions.json()["total_versions"] == 2

        # 6. Promote 1.1.0 to Production
        res_stage = client.post(
            "/api/v1/models/stellar_gnn_model/versions/1.1.0/stage",
            json={"stage": "production", "reason": "Higher AUC on test benchmark"},
        )
        assert res_stage.status_code == 200
        assert res_stage.json()["stage"] == "production"

        # 7. Get Production Version
        res_prod = client.get("/api/v1/models/stellar_gnn_model/production")
        assert res_prod.status_code == 200
        assert res_prod.json()["production_version"]["version"] == "1.1.0"

        # 8. Update Metrics
        res_up_m = client.post(
            "/api/v1/models/stellar_gnn_model/versions/1.1.0/metrics",
            json={"metrics": {"latency_ms": 12.5}},
        )
        assert res_up_m.status_code == 200
        assert res_up_m.json()["metrics"]["latency_ms"] == 12.5

        # 9. Get Lineage
        res_lin = client.get("/api/v1/models/stellar_gnn_model/versions/1.1.0/lineage")
        assert res_lin.status_code == 200
        assert res_lin.json()["model_name"] == "stellar_gnn_model"

        # 10. Delete Version 1.0.0
        res_del = client.delete("/api/v1/models/stellar_gnn_model/versions/1.0.0")
        assert res_del.status_code == 200
