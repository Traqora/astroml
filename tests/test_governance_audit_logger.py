"""Tests for astroml.governance.audit_logger.

Covers AuditEvent immutability, serialization round-trips, and the
ModelAuditLogger/InMemoryAuditStore/FileAuditStore behavior.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from astroml.governance.audit_logger import (
    AuditEvent,
    AuditEventType,
    FileAuditStore,
    InMemoryAuditStore,
    ModelAuditLogger,
)


class TestAuditEventImmutability:
    def test_event_is_frozen_dataclass(self):
        assert dataclasses.fields(AuditEvent)
        event = AuditEvent(event_type=AuditEventType.MODEL_REGISTERED)
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.outcome = "failure"

    def test_cannot_reassign_any_field(self):
        event = AuditEvent(event_type=AuditEventType.DEPLOYMENT_STARTED, actor="alice")
        for field_name, value in (
            ("actor", "bob"),
            ("model_id", "other-model"),
            ("trace_id", "trace-2"),
        ):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(event, field_name, value)

    def test_default_fields_still_populated(self):
        event = AuditEvent()
        assert event.event_id
        assert event.outcome == "success"
        assert event.actor == "system"


class TestAuditEventSerialization:
    def test_to_dict_round_trip(self):
        event = AuditEvent(
            event_type=AuditEventType.DEPLOYMENT_COMPLETED,
            actor="ci-bot",
            model_id="fraud-detector:2.0.0",
            details={"duration_ms": 120},
            outcome="success",
            trace_id="trace-1",
        )
        data = event.to_dict()
        restored = AuditEvent.from_dict(data)

        assert restored.event_id == event.event_id
        assert restored.event_type == event.event_type
        assert restored.actor == event.actor
        assert restored.model_id == event.model_id
        assert restored.details == event.details
        assert restored.outcome == event.outcome
        assert restored.trace_id == event.trace_id
        assert restored.timestamp == event.timestamp

    def test_to_dict_is_json_serializable(self):
        event = AuditEvent(event_type=AuditEventType.TRAINING_FAILED, details={"epoch": 3})
        json.dumps(event.to_dict())

    def test_from_dict_defaults_missing_optional_fields(self):
        restored = AuditEvent.from_dict({"event_type": "MODEL_RETIRED"})
        assert restored.actor == "system"
        assert restored.model_id == ""
        assert restored.details == {}
        assert restored.outcome == "success"
        assert restored.trace_id is None


class TestInMemoryAuditStore:
    def test_write_and_query_by_model_id(self):
        store = InMemoryAuditStore()
        e1 = AuditEvent(model_id="model-a", event_type=AuditEventType.MODEL_REGISTERED)
        e2 = AuditEvent(model_id="model-b", event_type=AuditEventType.MODEL_REGISTERED)
        store.write(e1)
        store.write(e2)

        results = store.query(model_id="model-a")
        assert [e.event_id for e in results] == [e1.event_id]

    def test_query_most_recent_first(self):
        store = InMemoryAuditStore()
        events = [AuditEvent(model_id="m", details={"i": i}) for i in range(3)]
        for e in events:
            store.write(e)

        results = store.query(model_id="m")
        assert [e.details["i"] for e in results] == [2, 1, 0]

    def test_get_event_by_id(self):
        store = InMemoryAuditStore()
        event = AuditEvent(model_id="m")
        store.write(event)
        assert store.get_event(event.event_id) is event
        assert store.get_event("does-not-exist") is None

    def test_eviction_when_over_max_events(self):
        store = InMemoryAuditStore(max_events=2)
        first = AuditEvent(model_id="m", details={"i": 0})
        store.write(first)
        store.write(AuditEvent(model_id="m", details={"i": 1}))
        store.write(AuditEvent(model_id="m", details={"i": 2}))

        assert len(store.query(model_id="m", limit=10)) == 2
        assert store.get_event(first.event_id) is None

    def test_query_filters_by_event_type_and_actor(self):
        store = InMemoryAuditStore()
        store.write(
            AuditEvent(
                model_id="m",
                event_type=AuditEventType.DEPLOYMENT_APPROVED,
                actor="alice",
            )
        )
        store.write(
            AuditEvent(
                model_id="m",
                event_type=AuditEventType.DEPLOYMENT_REJECTED,
                actor="bob",
            )
        )

        results = store.query(
            model_id="m", event_type=AuditEventType.DEPLOYMENT_APPROVED, actor="alice"
        )
        assert len(results) == 1
        assert results[0].actor == "alice"


class TestFileAuditStore:
    def test_write_and_query_round_trip(self, tmp_path):
        store = FileAuditStore(log_dir=tmp_path)
        event = AuditEvent(
            model_id="m",
            event_type=AuditEventType.INFERENCE_COMPLETED,
            details={"latency_ms": 42},
        )
        store.write(event)

        results = store.query(model_id="m")
        assert len(results) == 1
        assert results[0].event_id == event.event_id
        assert results[0].details == {"latency_ms": 42}

    def test_get_event_by_id(self, tmp_path):
        store = FileAuditStore(log_dir=tmp_path)
        event = AuditEvent(model_id="m")
        store.write(event)
        found = store.get_event(event.event_id)
        assert found is not None
        assert found.event_id == event.event_id

    def test_rotation_on_max_file_size(self, tmp_path):
        store = FileAuditStore(log_dir=tmp_path, max_file_size_mb=0)
        store.write(AuditEvent(model_id="m", details={"i": 0}))
        first_file = store._current_file
        assert first_file.exists()

        store.write(AuditEvent(model_id="m", details={"i": 1}))

        # Any file over max_file_size triggers a rotation check on the next
        # write; both events must still be queryable regardless of whether
        # the rotated filename (second-precision timestamp) happens to
        # collide with the original.
        results = store.query(model_id="m", limit=10)
        assert len(results) == 2

    def test_corrupt_line_is_skipped_not_raised(self, tmp_path):
        store = FileAuditStore(log_dir=tmp_path)
        event = AuditEvent(model_id="m")
        store.write(event)
        current_file = store._get_current_file()
        with open(current_file, "a") as f:
            f.write("not valid json\n")

        results = store.query(model_id="m")
        assert len(results) == 1


class TestModelAuditLogger:
    def test_log_writes_to_store(self):
        store = InMemoryAuditStore()
        logger = ModelAuditLogger(store=store)
        event = logger.log(
            AuditEventType.MODEL_REGISTERED, actor="alice", model_id="m", outcome="success"
        )
        assert store.get_event(event.event_id) is event

    def test_defaults_to_in_memory_store(self):
        logger = ModelAuditLogger()
        assert isinstance(logger.store, InMemoryAuditStore)

    def test_log_event_decorator_records_success(self):
        logger = ModelAuditLogger()

        @logger.log_event(AuditEventType.TRAINING_COMPLETED)
        def train(model_id: str, actor: str = "system") -> dict:
            return {"epochs": 1}

        result = train(model_id="m", actor="ci")
        assert result == {"epochs": 1}

        trail = logger.get_model_audit_trail("m")
        assert len(trail) == 1
        assert trail[0].outcome == "success"
        assert trail[0].actor == "ci"

    def test_log_event_decorator_records_failure_and_reraises(self):
        logger = ModelAuditLogger()

        @logger.log_event(AuditEventType.TRAINING_FAILED)
        def train(model_id: str, actor: str = "system") -> dict:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            train(model_id="m", actor="ci")

        trail = logger.get_model_audit_trail("m")
        assert len(trail) == 1
        assert trail[0].outcome == "failure"
        assert trail[0].details["error"] == "boom"
        assert trail[0].details["error_type"] == "RuntimeError"

    def test_get_model_audit_trail_is_chronological(self):
        logger = ModelAuditLogger()
        for i in range(3):
            logger.log(AuditEventType.INFERENCE_REQUEST, model_id="m", details={"i": i})

        trail = logger.get_model_audit_trail("m")
        assert [e.details["i"] for e in trail] == [0, 1, 2]

    def test_export_writes_json_file(self, tmp_path):
        logger = ModelAuditLogger()
        logger.log(AuditEventType.MODEL_REGISTERED, model_id="m")
        logger.log(AuditEventType.MODEL_RETIRED, model_id="m")

        out_path = tmp_path / "audit_export.json"
        count = logger.export(out_path, model_id="m")

        assert count == 2
        exported = json.loads(out_path.read_text())
        assert len(exported) == 2
