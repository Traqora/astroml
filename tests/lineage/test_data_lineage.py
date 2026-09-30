"""Tests for DataLineageTracker."""

from __future__ import annotations

import pytest

from astroml.tracking.lineage.data_lineage import DataLineageTracker
from astroml.tracking.lineage.metadata_store import MetadataStore


class TestDataLineageTracker:
    """Tests for DataLineageTracker."""

    def setup_method(self) -> None:
        self.store = MetadataStore()
        self.tracker = DataLineageTracker(self.store)

    def test_record_source(self) -> None:
        record = self.tracker.record_source("src1", {"format": "csv"})
        assert record.id == "src1"
        assert record.metadata == {"format": "csv"}

        fetched = self.store.get_dataset("src1")
        assert fetched is not None

    def test_record_transformation(self) -> None:
        self.tracker.record_source("src1")
        record = self.tracker.record_transformation(
            input_ids=["src1"],
            output_id="tx1",
            transform_name="normalize",
            params={"method": "zscore"},
        )
        assert record.id == "tx1"
        assert record.type == "transformation"
        assert record.metadata["transform_name"] == "normalize"

    def test_record_model_training(self) -> None:
        self.tracker.record_source("ds1")
        record = self.tracker.record_model_training(
            model_id="m1",
            dataset_id="ds1",
            model_metadata={"epochs": 10},
        )
        assert record.id == "m1"
        assert record.type == "model"
        assert record.metadata["training_dataset_id"] == "ds1"
        assert record.parent_ids == ["ds1"]

    def test_record_prediction(self) -> None:
        self.tracker.record_source("ds1")
        self.tracker.record_model_training("m1", "ds1")
        record = self.tracker.record_prediction(
            model_id="m1",
            input_id="ds1",
            prediction_id="pred1",
            prediction_metadata={"batch_size": 32},
        )
        assert record.id == "pred1"
        assert record.type == "dataset"
        assert record.metadata["prediction"] is True
        assert record.metadata["model_id"] == "m1"
        assert "ds1" in record.parent_ids
        assert "m1" in record.parent_ids

    def test_get_lineage_full(self) -> None:
        self.tracker.record_source("raw_data")
        self.tracker.record_transformation(
            input_ids=["raw_data"],
            output_id="processed_data",
            transform_name="clean",
        )
        self.tracker.record_model_training("model_v1", "processed_data")
        self.tracker.record_prediction("model_v1", "processed_data", "pred_001")

        lineage = self.tracker.get_lineage("model_v1")
        assert "entity" in lineage
        assert "upstream" in lineage
        assert "downstream" in lineage
        assert "full_dag" in lineage

        assert lineage["entity"]["id"] == "model_v1"
        # Upstream should include raw_data and processed_data
        upstream_ids = {r["id"] for r in lineage["upstream"]}
        assert "raw_data" in upstream_ids
        assert "processed_data" in upstream_ids

        # Downstream should include pred_001
        downstream_ids = {r["id"] for r in lineage["downstream"]}
        assert "pred_001" in downstream_ids

    def test_get_lineage_nonexistent(self) -> None:
        lineage = self.tracker.get_lineage("nonexistent")
        assert lineage == {}

    def test_get_upstream(self) -> None:
        self.tracker.record_source("src1")
        self.tracker.record_transformation(["src1"], "tx1", "clean")
        self.tracker.record_model_training("m1", "tx1")

        upstream = self.tracker.get_upstream("m1")
        upstream_ids = {r.id for r in upstream}
        assert "src1" in upstream_ids
        assert "tx1" in upstream_ids

    def test_get_upstream_empty(self) -> None:
        self.tracker.record_source("src1")
        upstream = self.tracker.get_upstream("src1")
        assert upstream == []

    def test_get_downstream(self) -> None:
        self.tracker.record_source("src1")
        self.tracker.record_transformation(["src1"], "tx1", "clean")
        self.tracker.record_model_training("m1", "tx1")
        self.tracker.record_prediction("m1", "tx1", "pred1")

        downstream = self.tracker.get_downstream("src1")
        downstream_ids = {r.id for r in downstream}
        assert "tx1" in downstream_ids
        assert "m1" in downstream_ids
        assert "pred1" in downstream_ids

    def test_get_downstream_empty(self) -> None:
        self.tracker.record_source("src1")
        downstream = self.tracker.get_downstream("src1")
        assert downstream == []

    def test_cycle_handling(self) -> None:
        self.tracker.record_source("a")
        self.tracker.record_source("b")
        self.tracker.record_transformation(["a", "b"], "c", "merge")
        self.tracker.record_transformation(["c"], "d", "split")
        self.tracker.record_transformation(["d"], "a", "feedback")

        lineage = self.tracker.get_lineage("a")
        assert lineage != {}

    def test_tracker_without_store(self) -> None:
        tracker = DataLineageTracker()
        record = tracker.record_source("test")
        assert record.id == "test"

    def test_record_transformation_no_params(self) -> None:
        self.tracker.record_source("src1")
        record = self.tracker.record_transformation(
            input_ids=["src1"],
            output_id="tx1",
            transform_name="clean",
        )
        assert record.metadata["params"] == {}
