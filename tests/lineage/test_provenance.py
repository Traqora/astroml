"""Tests for ProvenanceTracker."""

from __future__ import annotations

import json

import pytest

from astroml.tracking.lineage.provenance import ProvenanceTracker, StageRecord


class TestStageRecord:
    """Tests for StageRecord model."""

    def test_duration(self) -> None:
        record = StageRecord(name="test")
        assert record.duration_seconds() is None

        record.end_time = record.start_time
        assert record.duration_seconds() == 0.0

    def test_to_dict(self) -> None:
        record = StageRecord(
            name="stage1",
            row_count_input=100,
            row_count_output=90,
        )
        d = record.to_dict()
        assert d["name"] == "stage1"
        assert d["row_count_input"] == 100
        assert d["duration_seconds"] is None

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValueError):
            StageRecord(name="x", unknown="val")  # type: ignore[call-arg]


class TestProvenanceTracker:
    """Tests for ProvenanceTracker."""

    def setup_method(self) -> None:
        self.tracker = ProvenanceTracker()

    def test_stage_context_manager(self) -> None:
        with self.tracker.stage("preprocessing", {"key": "val"}) as stage:
            stage.row_count_input = 100
            stage.row_count_output = 95

        assert stage.name == "preprocessing"
        assert stage.metadata == {"key": "val"}
        assert stage.row_count_input == 100
        assert stage.row_count_output == 95
        assert stage.end_time is not None

    def test_nested_stages(self) -> None:
        with self.tracker.stage("pipeline") as outer:
            outer.row_count_input = 1000
            with self.tracker.stage("step1") as inner1:
                inner1.row_count_input = 1000
                inner1.row_count_output = 800
            with self.tracker.stage("step2") as inner2:
                inner2.row_count_input = 800
                inner2.row_count_output = 750

        assert len(outer.nested_stages) == 2
        assert outer.nested_stages[0].name == "step1"
        assert outer.nested_stages[1].name == "step2"

    def test_finalize_run(self) -> None:
        with self.tracker.stage("ingest") as s:
            s.row_count_input = 0
            s.row_count_output = 100

        chain = self.tracker.finalize_run("run_001")
        assert chain.run_id == "run_001"
        assert len(chain.stages) == 0  # stages passed via context are not auto-added

    def test_finalize_run_with_stages(self) -> None:
        stages = [
            StageRecord(name="ingest", row_count_input=0, row_count_output=100),
            StageRecord(name="transform", row_count_input=100, row_count_output=90),
        ]
        chain = self.tracker.finalize_run("run_002", stages=stages)
        assert len(chain.stages) == 2

    def test_verify_provenance_valid(self) -> None:
        import datetime

        now = datetime.datetime.utcnow()
        stages = [
            StageRecord(
                name="ingest",
                start_time=now,
                end_time=now,
                row_count_input=0,
                row_count_output=100,
                checksum_input=None,
                checksum_output="abc",
            ),
            StageRecord(
                name="transform",
                start_time=now,
                end_time=now,
                row_count_input=100,
                row_count_output=90,
                checksum_input="abc",
                checksum_output="def",
            ),
        ]
        self.tracker.finalize_run("run_valid", stages=stages)
        result = self.tracker.verify_provenance("run_valid")
        assert result["valid"] is True

    def test_verify_provenance_missing_run(self) -> None:
        result = self.tracker.verify_provenance("nonexistent")
        assert result["valid"] is False
        assert len(result["errors"]) == 1

    def test_verify_provenance_checksum_mismatch(self) -> None:
        stages = [
            StageRecord(
                name="ingest",
                row_count_input=0,
                row_count_output=100,
                checksum_output="abc",
            ),
            StageRecord(
                name="transform",
                row_count_input=100,
                row_count_output=90,
                checksum_input="xyz",
                checksum_output="def",
            ),
        ]
        self.tracker.finalize_run("run_bad", stages=stages)
        result = self.tracker.verify_provenance("run_bad")
        assert result["valid"] is False
        assert any("checksum mismatch" in e.lower() for e in result["errors"])

    def test_verify_provenance_row_count_mismatch(self) -> None:
        stages = [
            StageRecord(
                name="ingest",
                row_count_input=0,
                row_count_output=100,
                checksum_output="abc",
            ),
            StageRecord(
                name="transform",
                row_count_input=200,
                row_count_output=90,
                checksum_input="abc",
                checksum_output="def",
            ),
        ]
        self.tracker.finalize_run("run_row_bad", stages=stages)
        result = self.tracker.verify_provenance("run_row_bad")
        assert result["valid"] is False
        assert any("row count mismatch" in e.lower() for e in result["errors"])

    def test_verify_provenance_no_end_time(self) -> None:
        stage = StageRecord(name="incomplete")
        stage.end_time = None
        stages = [stage]
        self.tracker.finalize_run("run_no_end", stages=stages)
        result = self.tracker.verify_provenance("run_no_end")
        assert result["valid"] is False
        assert any("no end time" in e.lower() for e in result["errors"])

    def test_verify_provenance_warning_same_checksum_diff_rows(self) -> None:
        import datetime

        now = datetime.datetime.utcnow()
        stages = [
            StageRecord(
                name="dedup",
                start_time=now,
                end_time=now,
                row_count_input=100,
                row_count_output=50,
                checksum_input="abc",
                checksum_output="abc",
            ),
        ]
        self.tracker.finalize_run("run_warn", stages=stages)
        result = self.tracker.verify_provenance("run_warn")
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_export_provenance_dict(self) -> None:
        stages = [StageRecord(name="ingest")]
        self.tracker.finalize_run("run_export", stages=stages)
        exported = self.tracker.export_provenance("run_export", fmt="dict")
        assert isinstance(exported, dict)
        assert exported["run_id"] == "run_export"

    def test_export_provenance_json(self) -> None:
        stages = [StageRecord(name="ingest")]
        self.tracker.finalize_run("run_json", stages=stages)
        exported = self.tracker.export_provenance("run_json", fmt="json")
        assert isinstance(exported, str)
        parsed = json.loads(exported)
        assert parsed["run_id"] == "run_json"

    def test_export_provenance_not_found(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            self.tracker.export_provenance("nonexistent")

    def test_export_provenance_invalid_format(self) -> None:
        self.tracker.finalize_run("run_fmt", stages=[])
        with pytest.raises(ValueError, match="Unsupported format"):
            self.tracker.export_provenance("run_fmt", fmt="xml")

    def test_compare_provenance(self) -> None:
        stages1 = [StageRecord(name="ingest", row_count_input=0, row_count_output=100)]
        stages2 = [StageRecord(name="ingest", row_count_input=0, row_count_output=200)]

        self.tracker.finalize_run("run_a", stages=stages1)
        self.tracker.finalize_run("run_b", stages=stages2)

        result = self.tracker.compare_provenance("run_a", "run_b")
        assert result["stage_count_match"] is True
        assert result["run_id_1"] == "run_a"
        assert result["run_id_2"] == "run_b"

    def test_compare_provenance_different_lengths(self) -> None:
        stages1 = [StageRecord(name="a")]
        stages2 = [StageRecord(name="a"), StageRecord(name="b")]

        self.tracker.finalize_run("run_short", stages=stages1)
        self.tracker.finalize_run("run_long", stages=stages2)

        result = self.tracker.compare_provenance("run_short", "run_long")
        assert result["stage_count_match"] is False

    def test_compare_provenance_mismatched_names(self) -> None:
        stages1 = [StageRecord(name="ingest")]
        stages2 = [StageRecord(name="load")]

        self.tracker.finalize_run("run_x", stages=stages1)
        self.tracker.finalize_run("run_y", stages=stages2)

        result = self.tracker.compare_provenance("run_x", "run_y")
        assert len(result["stage_differences"]) > 0
        assert result["stage_differences"][0]["difference"] == "stage name mismatch"

    def test_compare_provenance_not_found(self) -> None:
        self.tracker.finalize_run("run_a", stages=[])
        with pytest.raises(ValueError, match="not found"):
            self.tracker.compare_provenance("run_a", "nonexistent")
        with pytest.raises(ValueError, match="not found"):
            self.tracker.compare_provenance("nonexistent", "run_a")

    def test_compute_checksum_with_list(self) -> None:
        data = [{"a": 1}, {"b": 2}]
        checksum = ProvenanceTracker.compute_checksum(data)
        assert isinstance(checksum, str)
        assert len(checksum) == 32

    def test_compute_checksum_with_string(self) -> None:
        data = "hello world"
        checksum = ProvenanceTracker.compute_checksum(data)
        assert isinstance(checksum, str)
        assert len(checksum) == 32

    def test_provenance_tracker_with_store(self) -> None:
        from astroml.tracking.lineage.metadata_store import MetadataStore

        store = MetadataStore()
        tracker = ProvenanceTracker(store)
        stages = [StageRecord(name="test")]
        tracker.finalize_run("run_store", stages=stages)
        assert store.get_run("run_store") is not None
