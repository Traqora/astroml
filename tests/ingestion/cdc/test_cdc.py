"""Tests for Change Data Capture pipeline (#626)."""

from __future__ import annotations

import numpy as np
import pytest

from astroml.ingestion.cdc import (
    CDCConfig,
    CDCConnector,
    ChangeEvent,
    ChangeOperation,
    CompatibilityMode,
    DataTransformer,
    ExactlyOnceTracker,
    SchemaMigration,
    SchemaRegistry,
    SchemaVersion,
    StreamMonitor,
    StreamProcessor,
)

# ---------------------------------------------------------------------------
# CDC connector
# ---------------------------------------------------------------------------


class TestCDCConnector:
    def test_connector_lifecycle(self) -> None:
        cfg = CDCConfig(
            connector_name="test-cdc",
            database_hostname="localhost",
            database_port=5432,
        )
        connector = CDCConnector(cfg)
        assert connector.status.value == "initializing"

        connector.start()
        assert connector.status.value == "running"

        connector.pause()
        assert connector.status.value == "paused"

        connector.resume()
        assert connector.status.value == "running"

        connector.stop()
        assert connector.status.value == "stopped"

    def test_event_parsing(self) -> None:
        payload = {
            "source": {"connector": "pg", "db": "test", "table": "accounts", "lsn": 12345, "ts_ms": 1000},
            "op": "c",
            "before": None,
            "after": {"id": 1, "name": "test"},
            "transaction_id": "txn_001",
        }
        event = ChangeEvent.from_debezium(payload)
        assert event.op == ChangeOperation.CREATE
        assert event.after == {"id": 1, "name": "test"}
        assert event.lsn == 12345

    def test_simulate_events(self) -> None:
        connector = CDCConnector(CDCConfig())
        connector.start()

        events = connector.simulate_events(
            "public.accounts",
            [{"id": 1}, {"id": 2}],
            op=ChangeOperation.CREATE,
        )
        assert len(events) == 2
        assert connector.get_status()["events_processed"] == 2

    def test_subscriber_notification(self) -> None:
        connector = CDCConnector(CDCConfig())
        received: list[ChangeEvent] = []
        connector.subscribe(lambda e: received.append(e))
        connector.simulate_events("t", [{"x": 1}])
        assert len(received) == 1


# ---------------------------------------------------------------------------
# Stream processor
# ---------------------------------------------------------------------------


class TestStreamProcessor:
    def test_process_event(self) -> None:
        processor = StreamProcessor()
        event = ChangeEvent(
            source={"table": "accounts"},
            op=ChangeOperation.CREATE,
            before=None,
            after={"id": 1, "name": "alice"},
        )
        result = processor.process_event(event, offset=0)
        assert result.status == "committed"
        assert result.feature_store_key == "accounts:1"

    def test_process_batch_and_commit(self) -> None:
        processor = StreamProcessor(batch_commit_size=5)
        events = [
            ChangeEvent(
                source={"table": "t"},
                op=ChangeOperation.CREATE,
                before=None,
                after={"id": i},
            )
            for i in range(5)
        ]
        results = processor.process_batch(events)
        assert len(results) == 5
        stats = processor.get_stats()
        assert stats["total_processed"] == 5

    def test_data_transformer(self) -> None:
        def add_ts(data: dict) -> dict:
            data["processed_at"] = 999
            return data

        transformer = DataTransformer(rules=[], enrichments=[add_ts])
        result = transformer.process({"id": 1})
        assert result["processed_at"] == 999

    def test_stream_monitor(self) -> None:
        monitor = StreamMonitor(window_seconds=60)
        for _ in range(100):
            monitor.record(processing_time_ms=5.0)
        monitor.record(processing_time_ms=10.0, is_error=True)

        metrics = monitor.get_metrics()
        assert metrics["throughput_events_per_sec"] > 0
        assert metrics["total_events"] == 101
        assert metrics["total_errors"] == 1
        assert metrics["error_rate"] > 0


# ---------------------------------------------------------------------------
# Exactly-once tracker
# ---------------------------------------------------------------------------


class TestExactlyOnceTracker:
    def test_mark_and_checkpoint(self) -> None:
        tracker = ExactlyOnceTracker(checkpoint_interval=3)
        for i in range(5):
            tracker.mark_processed("topic1", 0, i)
        assert tracker.should_checkpoint()
        cp = tracker.checkpoint()
        assert "offsets" in cp

    def test_last_processed(self) -> None:
        tracker = ExactlyOnceTracker()
        tracker.mark_processed("t", 0, 42)
        assert tracker.get_last_processed("t", 0) == 42
        assert tracker.get_last_processed("t", 1) == -1


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------


class TestSchemaRegistry:
    def test_register_schema(self) -> None:
        registry = SchemaRegistry(compatibility_mode=CompatibilityMode.BACKWARD)
        v1 = registry.register_schema(
            "public.accounts",
            [{"name": "id", "type": "int"}, {"name": "name", "type": "string"}],
        )
        assert v1.version == 1
        assert len(v1.fields) == 2
        assert v1.checksum != ""

    def test_multiple_versions(self) -> None:
        registry = SchemaRegistry()
        registry.register_schema("t", [{"name": "a", "type": "int"}])
        v2 = registry.register_schema("t", [
            {"name": "a", "type": "int"},
            {"name": "b", "type": "string"},
        ])
        assert v2.version == 2
        current = registry.get_current_schema("t")
        assert current is not None
        assert current.version == 2

    def test_schema_migration(self) -> None:
        registry = SchemaRegistry(compatibility_mode=CompatibilityMode.NONE)
        registry.register_schema("t", [{"name": "id", "type": "int"}])
        registry.register_schema("t", [
            {"name": "id", "type": "int"},
            {"name": "status", "type": "string"},
        ])

        migration = SchemaMigration(
            table_name="t",
            from_version=1,
            to_version=2,
            add_fields=[{"name": "status", "type": "string", "default": "active"}],
        )
        registry.register_migration(migration)

        migrated = registry.migrate_data("t", {"id": 1}, from_version=1, to_version=2)
        assert migrated == {"id": 1, "status": "active"}

    def test_data_validation(self) -> None:
        registry = SchemaRegistry()
        registry.register_schema("t", [{"name": "x", "type": "int"}])
        errors = registry.validate_data("t", {"x": "not_int"})
        assert len(errors) == 1
        errors = registry.validate_data("t", {"x": 42})
        assert len(errors) == 0

    def test_get_all_schemas(self) -> None:
        registry = SchemaRegistry()
        registry.register_schema("t1", [{"name": "a", "type": "int"}])
        registry.register_schema("t2", [{"name": "b", "type": "string"}])
        all_schemas = registry.get_all_schemas()
        assert "t1" in all_schemas
        assert "t2" in all_schemas