"""Tests for the data lineage API router."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from astroml.api.routers.data_lineage import (
    _lineage_tracker,
    _provenance_tracker,
    router,
)
from astroml.tracking.lineage.provenance import StageRecord

app = FastAPI()
app.include_router(router)

_test_transport = ASGITransport(app=app)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _clear_state() -> None:
    """Reset tracker state between tests."""
    _lineage_tracker._store._records.clear()
    _lineage_tracker._store._by_type["dataset"].clear()
    _lineage_tracker._store._by_type["transformation"].clear()
    _lineage_tracker._store._by_type["model"].clear()
    _provenance_tracker._chains.clear()


@pytest.mark.asyncio
async def test_get_lineage_found() -> None:
    _lineage_tracker.record_source("ds1", {"format": "csv"})

    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/dataset/ds1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["entity"]["id"] == "ds1"
    assert "upstream" in data
    assert "downstream" in data


@pytest.mark.asyncio
async def test_get_lineage_not_found() -> None:
    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/dataset/nonexistent")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_upstream_found() -> None:
    _lineage_tracker.record_source("src1")
    _lineage_tracker.record_transformation(["src1"], "tx1", "clean")
    _lineage_tracker.record_model_training("m1", "tx1")

    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/model/m1/upstream")

    assert resp.status_code == 200
    data = resp.json()
    ids = [r["id"] for r in data]
    assert "src1" in ids
    assert "tx1" in ids


@pytest.mark.asyncio
async def test_get_upstream_not_found() -> None:
    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/dataset/nonexistent/upstream")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_downstream_found() -> None:
    _lineage_tracker.record_source("src1")
    _lineage_tracker.record_transformation(["src1"], "tx1", "clean")
    _lineage_tracker.record_model_training("m1", "tx1")
    _lineage_tracker.record_prediction("m1", "tx1", "pred1")

    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/dataset/src1/downstream")

    assert resp.status_code == 200
    data = resp.json()
    ids = [r["id"] for r in data]
    assert "tx1" in ids
    assert "m1" in ids
    assert "pred1" in ids


@pytest.mark.asyncio
async def test_get_downstream_not_found() -> None:
    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/dataset/nonexistent/downstream")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_provenance_found() -> None:
    stages = [StageRecord(name="ingest", row_count_input=0, row_count_output=100)]
    _provenance_tracker.finalize_run("run_001", stages=stages)

    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/run/run_001/provenance")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "run_001"
    assert len(data["stages"]) == 1


@pytest.mark.asyncio
async def test_get_provenance_not_found() -> None:
    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/run/nonexistent/provenance")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_lineage_json() -> None:
    _lineage_tracker.record_source("ds1")

    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/dataset/ds1/export?format=json")

    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert isinstance(data["data"], str)


@pytest.mark.asyncio
async def test_export_lineage_dict() -> None:
    _lineage_tracker.record_source("ds1")

    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/dataset/ds1/export?fmt=dict")

    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert isinstance(data["data"], dict)


@pytest.mark.asyncio
async def test_export_lineage_not_found() -> None:
    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/lineage/dataset/nonexistent/export")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_record_lineage_source() -> None:
    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/lineage/record",
            json={
                "event_type": "source",
                "entity_id": "ds1",
                "parent_ids": [],
                "metadata": {"format": "csv"},
            },
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["success"] is True
    assert data["entity_id"] == "ds1"


@pytest.mark.asyncio
async def test_record_lineage_transformation() -> None:
    _lineage_tracker.record_source("ds1")

    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/lineage/record",
            json={
                "event_type": "transformation",
                "entity_id": "tx1",
                "parent_ids": ["ds1"],
                "metadata": {
                    "transform_name": "normalize",
                    "params": {"method": "zscore"},
                },
            },
        )

    assert resp.status_code == 201
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_record_lineage_training() -> None:
    _lineage_tracker.record_source("ds1")

    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/lineage/record",
            json={
                "event_type": "training",
                "entity_id": "m1",
                "parent_ids": ["ds1"],
                "metadata": {"epochs": 10},
            },
        )

    assert resp.status_code == 201
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_record_lineage_prediction() -> None:
    _lineage_tracker.record_source("ds1")
    _lineage_tracker.record_model_training("m1", "ds1")

    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/lineage/record",
            json={
                "event_type": "prediction",
                "entity_id": "pred1",
                "parent_ids": ["m1", "ds1"],
                "metadata": {"confidence": 0.95},
            },
        )

    assert resp.status_code == 201
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_record_lineage_invalid_event_type() -> None:
    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/lineage/record",
            json={
                "event_type": "invalid_type",
                "entity_id": "x",
                "parent_ids": [],
                "metadata": {},
            },
        )

    assert resp.status_code == 422  # FastAPI validation error


@pytest.mark.asyncio
async def test_record_lineage_unknown_event_type() -> None:
    async with AsyncClient(transport=_test_transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/lineage/record",
            json={
                "event_type": "source",
                "entity_id": "test",
                "parent_ids": [],
                "metadata": {},
            },
        )

    assert resp.status_code == 201
    assert resp.json()["success"] is True
