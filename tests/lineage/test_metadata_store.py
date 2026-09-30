"""Tests for MetadataStore."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from astroml.tracking.lineage.metadata_store import MetadataRecord, MetadataStore


class TestMetadataRecord:
    """Tests for MetadataRecord model."""

    def test_valid_records(self) -> None:
        record = MetadataRecord(id="ds1", type="dataset")
        assert record.id == "ds1"
        assert record.type == "dataset"
        assert record.parent_ids == []
        assert record.child_ids == []

    def test_invalid_type(self) -> None:
        with pytest.raises(ValueError):
            MetadataRecord(id="bad", type="invalid_type")

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValueError):
            MetadataRecord(id="x", type="dataset", unknown_field="test")  # type: ignore[call-arg]


class TestMetadataStore:
    """Tests for MetadataStore."""

    def setup_method(self) -> None:
        self.store = MetadataStore()

    def test_store_and_get_run(self) -> None:
        record = self.store.store_run("run1", {"key": "val"})
        assert record.id == "run1"
        assert record.metadata == {"key": "val"}

        fetched = self.store.get_run("run1")
        assert fetched is not None
        assert fetched.id == "run1"

    def test_get_nonexistent_run(self) -> None:
        assert self.store.get_run("nonexistent") is None

    def test_store_and_get_dataset(self) -> None:
        record = self.store.store_dataset("ds1", {"source": "s3"})
        assert record.id == "ds1"
        assert record.type == "dataset"

        fetched = self.store.get_dataset("ds1")
        assert fetched is not None
        assert fetched.metadata == {"source": "s3"}

    def test_get_nonexistent_dataset(self) -> None:
        assert self.store.get_dataset("nonexistent") is None

    def test_store_transformation(self) -> None:
        record = self.store.store_transformation(
            "tx1",
            metadata={"fn": "normalize"},
            parent_ids=["ds1"],
            child_ids=["ds2"],
        )
        assert record.id == "tx1"
        assert record.type == "transformation"
        assert record.parent_ids == ["ds1"]
        assert record.child_ids == ["ds2"]

    def test_store_transformation_updates_relationships(self) -> None:
        self.store.store_dataset("ds1")
        self.store.store_dataset("ds2")
        self.store.store_transformation("tx1", parent_ids=["ds1"], child_ids=["ds2"])

        ds1 = self.store.get_dataset("ds1")
        assert ds1 is not None
        assert "tx1" in ds1.child_ids

        ds2 = self.store.get_dataset("ds2")
        assert ds2 is not None
        assert "tx1" in ds2.parent_ids

    def test_get_transformations_filtered(self) -> None:
        self.store.store_transformation("tx1", parent_ids=["ds1"])
        self.store.store_transformation("tx2", parent_ids=["ds2"])
        self.store.store_transformation("tx3", parent_ids=["ds1"])

        result = self.store.get_transformations("ds1")
        assert len(result) == 2
        assert {r.id for r in result} == {"tx1", "tx3"}

    def test_get_transformations_all(self) -> None:
        self.store.store_transformation("tx1")
        self.store.store_transformation("tx2")

        result = self.store.get_transformations()
        assert len(result) == 2

    def test_store_and_get_model(self) -> None:
        record = self.store.store_model(
            "model1",
            metadata={"accuracy": 0.95},
            parent_ids=["ds1"],
        )
        assert record.id == "model1"
        assert record.type == "model"
        assert record.metadata == {"accuracy": 0.95}

        fetched = self.store.get_model("model1")
        assert fetched is not None
        assert fetched.parent_ids == ["ds1"]

    def test_get_nonexistent_model(self) -> None:
        assert self.store.get_model("nonexistent") is None

    def test_query_lineage_by_id(self) -> None:
        self.store.store_dataset("ds1")
        results = self.store.query_lineage(entity_id="ds1")
        assert len(results) == 1
        assert results[0].id == "ds1"

    def test_query_lineage_by_type(self) -> None:
        self.store.store_dataset("ds1")
        self.store.store_model("m1")
        self.store.store_transformation("tx1")

        datasets = self.store.query_lineage(entity_type="dataset")
        assert len(datasets) == 1
        assert datasets[0].id == "ds1"

        models = self.store.query_lineage(entity_type="model")
        assert len(models) == 1
        assert models[0].id == "m1"

    def test_query_lineage_empty_id_returns_all(self) -> None:
        self.store.store_dataset("ds1")
        self.store.store_model("m1")

        results = self.store.query_lineage()
        assert len(results) == 2

    def test_query_lineage_by_time_range(self) -> None:
        now = datetime.utcnow()
        past = now - timedelta(hours=2)

        ds1 = self.store.store_dataset("ds1")
        ds1.timestamp = past

        ds2 = self.store.store_dataset("ds2")
        ds2.timestamp = now

        results = self.store.query_lineage(time_start=now - timedelta(hours=1))
        assert len(results) == 1
        assert results[0].id == "ds2"

        results = self.store.query_lineage(time_end=now - timedelta(hours=1))
        assert len(results) == 1
        assert results[0].id == "ds1"

    def test_cycle_prevention(self) -> None:
        self.store.store_dataset("a")
        self.store.store_dataset("b")
        # Create a -> tx -> b
        self.store.store_transformation("tx", parent_ids=["a"], child_ids=["b"])
        # Manually create a cycle by updating a's parent_ids
        a = self.store.get_dataset("a")
        assert a is not None
        a.parent_ids.append("b")

        # This should not cause infinite loops
        results = self.store.query_lineage(entity_id="a")
        assert len(results) == 1
