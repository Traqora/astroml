"""Integration tests — model registry (issue #237)."""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.mark.xdist_group("api_models")
class TestModelRegistry:

    def test_list_models_empty(self, client):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_register_model(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")

        resp = client.post("/api/v1/models", json={
            "name": "gcn",
            "path": str(src),
            "metrics": {"auc": 0.95},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "gcn"
        assert data["status"] == "inactive"
        assert "gcn_v" in data["version"]
        assert data["metrics"]["auc"] == pytest.approx(0.95)
        assert data["parent_id"] is None

    def test_register_model_with_parent(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")

        parent = client.post("/api/v1/models", json={
            "name": "gcn",
            "version": "v1.0.0",
            "path": str(src),
        }).json()
        
        child = client.post("/api/v1/models", json={
            "name": "gcn",
            "version": "v1.1.0",
            "path": str(src),
            "parent_id": parent["id"],
        }).json()
        
        assert child["parent_id"] == parent["id"]

    def test_register_model_with_invalid_parent(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")
        
        resp = client.post("/api/v1/models", json={
            "name": "gcn",
            "path": str(src),
            "parent_id": 9999,
        })
        assert resp.status_code == 404

    def test_register_model_with_custom_version(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")

        resp = client.post("/api/v1/models", json={
            "name": "gcn",
            "version": "v1.0.0",
            "path": str(src),
            "metrics": {"auc": 0.95},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["version"] == "v1.0.0"

    def test_register_model_nonexistent_path(self, client):
        resp = client.post("/api/v1/models", json={
            "name": "gcn",
            "path": "/nonexistent/path/model.pth",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["path"] == "/nonexistent/path/model.pth"

    def test_activate_model(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")

        created = client.post("/api/v1/models", json={
            "name": "gcn",
            "path": str(src),
        }).json()

        resp = client.post(f"/api/v1/models/{created['id']}/activate")
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_activate_model_deactivates_others(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")

        # Create two versions
        v1 = client.post("/api/v1/models", json={
            "name": "gcn",
            "version": "v1.0.0",
            "path": str(src),
        }).json()
        v2 = client.post("/api/v1/models", json={
            "name": "gcn",
            "version": "v2.0.0",
            "path": str(src),
        }).json()

        # Activate v1
        client.post(f"/api/v1/models/{v1['id']}/activate")
        # Now activate v2
        client.post(f"/api/v1/models/{v2['id']}/activate")

        # List all models and check statuses
        models = client.get("/api/v1/models").json()
        v1_status = next(m["status"] for m in models if m["version"] == "v1.0.0")
        v2_status = next(m["status"] for m in models if m["version"] == "v2.0.0")

        assert v1_status == "inactive"
        assert v2_status == "active"

    def test_model_metrics(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")

        created = client.post("/api/v1/models", json={
            "name": "gcn",
            "path": str(src),
            "metrics": {"f1": 0.88},
        }).json()

        resp = client.get(f"/api/v1/models/{created['id']}/metrics")
        assert resp.status_code == 200
        assert resp.json()["metrics"]["f1"] == pytest.approx(0.88)

    def test_model_metrics_empty(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")

        created = client.post("/api/v1/models", json={
            "name": "gcn",
            "path": str(src),
        }).json()

        resp = client.get(f"/api/v1/models/{created['id']}/metrics")
        assert resp.status_code == 200
        assert resp.json()["metrics"] == {}

    def test_activate_not_found(self, client):
        resp = client.post("/api/v1/models/9999/activate")
        assert resp.status_code == 404

    def test_get_metrics_not_found(self, client):
        resp = client.get("/api/v1/models/9999/metrics")
        assert resp.status_code == 404

    def test_list_models_ordered_by_created_desc(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")

        # Create 3 models
        client.post("/api/v1/models", json={"name": "m1", "path": str(src)})
        client.post("/api/v1/models", json={"name": "m2", "path": str(src)})
        client.post("/api/v1/models", json={"name": "m3", "path": str(src)})

        models = client.get("/api/v1/models").json()
        assert len(models) == 3
        assert [m["name"] for m in models] == ["m3", "m2", "m1"]

    def test_compare_versions(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")

        v1 = client.post("/api/v1/models", json={
            "name": "gcn",
            "version": "v1.0.0",
            "path": str(src),
            "metrics": {"auc": 0.90, "precision": 0.80},
        }).json()
        v2 = client.post("/api/v1/models", json={
            "name": "gcn",
            "version": "v1.1.0",
            "path": str(src),
            "metrics": {"auc": 0.92, "precision": 0.85},
        }).json()
        
        resp = client.post("/api/v1/models/compare", json={
            "version_ids": [v1["id"], v2["id"]],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["versions"]) == 2
        assert len(data["metric_deltas"]) == 2
        
        auc_delta = next(m for m in data["metric_deltas"] if m["metric"] == "auc")
        assert auc_delta["delta"] == 0.02
        assert auc_delta["best"] == v2["id"]
        assert auc_delta["worst"] == v1["id"]
        
        precision_delta = next(m for m in data["metric_deltas"] if m["metric"] == "precision")
        assert precision_delta["delta"] == 0.05
        assert precision_delta["best"] == v2["id"]
        assert precision_delta["worst"] == v1["id"]

    def test_compare_versions_insufficient_ids(self, client):
        resp = client.post("/api/v1/models/compare", json={"version_ids": [1]})
        assert resp.status_code == 400

    def test_compare_versions_missing_id(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")
        v1 = client.post("/api/v1/models", json={
            "name": "gcn",
            "path": str(src),
        }).json()
        resp = client.post("/api/v1/models/compare", json={"version_ids": [v1["id"], 9999]})
        assert resp.status_code == 404

    def test_get_lineage(self, client, tmp_path):
        src = tmp_path / "model.pth"
        src.write_bytes(b"fake-checkpoint")

        v1 = client.post("/api/v1/models", json={
            "name": "gcn",
            "version": "v1.0.0",
            "path": str(src),
        }).json()
        v2 = client.post("/api/v1/models", json={
            "name": "gcn",
            "version": "v1.1.0",
            "path": str(src),
            "parent_id": v1["id"],
        }).json()
        v3 = client.post("/api/v1/models", json={
            "name": "gcn",
            "version": "v1.2.0",
            "path": str(src),
            "parent_id": v2["id"],
        }).json()
        
        resp = client.get(f"/api/v1/models/{v3['id']}/lineage")
        assert resp.status_code == 200
        chain = resp.json()["chain"]
        assert len(chain) == 3
        assert chain[0]["id"] == v3["id"]
        assert chain[1]["id"] == v2["id"]
        assert chain[2]["id"] == v1["id"]

    def test_get_lineage_not_found(self, client):
        resp = client.get("/api/v1/models/9999/lineage")
        assert resp.status_code == 404
