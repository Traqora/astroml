"""Tests for pipeline audit logging (#757)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from astroml.governance.audit_logger import AuditEventType, FileAuditStore
from astroml.tracking.pipeline_audit import (
    PipelineAuditLogger,
    PipelineAuditRecord,
    PipelineOperation,
)


class TestRecording:
    """Who/what/when/result capture for critical operations."""

    def test_model_activation_recorded(self):
        logger = PipelineAuditLogger()
        record = logger.log_model_activation(
            "fraud-detector", "v3", actor="alice", activated_by="alice@example.com"
        )
        assert record.actor == "alice"
        assert record.operation == PipelineOperation.MODEL_ACTIVATED
        assert record.target == "fraud-detector"
        assert record.details["version"] == "v3"
        assert record.details["activated_by"] == "alice@example.com"
        assert record.outcome == "success"
        assert record.timestamp.tzinfo is not None

    def test_default_actor_applied(self):
        logger = PipelineAuditLogger(actor="pipeline-service")
        record = logger.log_model_activation("m", "v1")
        assert record.actor == "pipeline-service"

    def test_config_change_records_diff(self):
        logger = PipelineAuditLogger()
        record = logger.log_config_change(
            "configs/model/thresholds.yaml",
            {"false_positive_weight": {"before": 0.2, "after": 0.5}},
            actor="bob",
        )
        assert record.operation == PipelineOperation.CONFIG_CHANGED
        assert record.target.endswith("thresholds.yaml")
        assert record.details["changes"]["false_positive_weight"]["after"] == 0.5

    def test_rollback_records_versions_and_reason(self):
        logger = PipelineAuditLogger()
        record = logger.log_rollback(
            "fraud-detector",
            from_version="v3",
            to_version="v2",
            actor="carol",
            reason="regression in precision",
        )
        assert record.operation == PipelineOperation.MODEL_ROLLED_BACK
        assert record.details["from_version"] == "v3"
        assert record.details["to_version"] == "v2"
        assert record.details["reason"] == "regression in precision"

    def test_failure_outcome_captured(self):
        logger = PipelineAuditLogger()
        record = logger.log_model_activation("m", "vX", outcome="failure")
        assert record.outcome == "failure"


class TestImmutability:
    """Records are append-only and tamper-evident."""

    def test_record_is_frozen(self):
        logger = PipelineAuditLogger()
        record = logger.log_model_activation("m", "v1")
        try:
            record.actor = "mallory"  # type: ignore[misc]
        except AttributeError:
            pass
        else:  # pragma: no cover
            raise AssertionError("PipelineAuditRecord should be immutable")

    def test_chain_links_every_record(self):
        logger = PipelineAuditLogger()
        for i in range(5):
            logger.log_config_change(f"cfg_{i}.yaml", {"seq": i})
        records = logger.query(limit=10)
        assert len(records) == 5
        assert logger.verify_chain(records) is True

    def test_tampered_detail_breaks_chain(self):
        logger = PipelineAuditLogger()
        logger.log_config_change("a.yaml", {"x": 1})
        records = logger.query(limit=10)
        tampered = [
            PipelineAuditRecord(
                actor=r.actor,
                operation=r.operation,
                target=r.target,
                timestamp=r.timestamp,
                outcome=r.outcome,
                details={"x": 999},  # silently edited
                record_id=r.record_id,
                previous_hash=r.previous_hash,
                record_hash=r.record_hash,
            )
            for r in records
        ]
        assert logger.verify_chain(tampered) is False

    def test_rehashed_record_breaks_chain(self):
        logger = PipelineAuditLogger()
        logger.log_config_change("a.yaml", {"x": 1})
        records = logger.query(limit=10)
        stripped = [
            PipelineAuditRecord(
                actor=r.actor,
                operation=r.operation,
                target=r.target,
                timestamp=r.timestamp,
                outcome=r.outcome,
                details=r.details,
                record_id=r.record_id,
                previous_hash="deleted-entry",
                record_hash=r.record_hash,
            )
            for r in records
        ]
        assert logger.verify_chain(stripped) is False


class TestQuery:
    """Filtering by operation, target, and actor."""

    def test_filter_by_operation(self):
        logger = PipelineAuditLogger()
        logger.log_model_activation("m1", "v1", actor="alice")
        logger.log_config_change("c.yaml", {}, actor="bob")
        activations = logger.query(operation=PipelineOperation.MODEL_ACTIVATED)
        assert len(activations) == 1
        assert activations[0].target == "m1"

    def test_filter_by_actor(self):
        logger = PipelineAuditLogger()
        logger.log_model_activation("m1", "v1", actor="alice")
        logger.log_model_activation("m2", "v1", actor="bob")
        only_bob = logger.query(actor="bob")
        assert len(only_bob) == 1
        assert only_bob[0].actor == "bob"

    def test_newest_first(self):
        logger = PipelineAuditLogger()
        first = logger.log_model_activation("m1", "v1")
        second = logger.log_config_change("c.yaml", {})
        records = logger.query(limit=10)
        assert records[0].record_id == second.record_id
        assert records[-1].record_id == first.record_id


class TestStoreBackends:
    """Works with both in-memory and file-based governance stores."""

    def test_file_store_roundtrip(self, tmp_path):
        store = FileAuditStore(log_dir=tmp_path / "audit")
        logger = PipelineAuditLogger(store=store)
        logger.log_model_activation("m", "v1", actor="alice")
        logger.log_rollback("m", "v2", "v1", actor="bob")

        records = logger.query(limit=10)
        assert len(records) == 2
        operations = {r.operation for r in records}
        assert operations == {
            PipelineOperation.MODEL_ACTIVATED,
            PipelineOperation.MODEL_ROLLED_BACK,
        }
        assert logger.verify_chain(records) is True

    def test_written_events_use_governance_type(self, tmp_path):
        store = FileAuditStore(log_dir=tmp_path / "audit")
        logger = PipelineAuditLogger(store=store)
        logger.log_model_activation("m", "v1")

        events = store.query(limit=10)
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.CONFIGURATION_CHANGED


class TestRecordHashing:
    """Hash covers exactly the record content."""

    def test_hash_changes_with_any_field(self):
        base = PipelineAuditRecord(
            actor="a",
            operation=PipelineOperation.CONFIG_CHANGED,
            target="t",
            timestamp=datetime(2026, 9, 24, tzinfo=timezone.utc),
            outcome="success",
            details={"k": 1},
        )
        mutated = PipelineAuditRecord(
            actor="b",
            operation=base.operation,
            target=base.target,
            timestamp=base.timestamp,
            outcome=base.outcome,
            details=base.details,
        )
        assert base.compute_hash() != mutated.compute_hash()
        assert base.compute_hash() != ""

    def test_verify_empty_history(self):
        logger = PipelineAuditLogger()
        assert logger.verify_chain([]) is True


class TestTimezone:
    """Timestamps are UTC and strictly ordered by recording sequence."""

    def test_timestamps_are_utc_and_monotonic(self):
        logger = PipelineAuditLogger()
        before = datetime.now(timezone.utc) - timedelta(seconds=1)
        record = logger.log_model_activation("m", "v1")
        after = datetime.now(timezone.utc) + timedelta(seconds=1)
        assert before <= record.timestamp <= after
        assert record.timestamp.utcoffset() == timedelta(0)
