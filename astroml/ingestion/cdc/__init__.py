"""Change Data Capture (CDC) pipeline (issue #626).

Provides real-time streaming data ingestion from PostgreSQL into the
AstroML feature store using Debezium and Kafka with exactly-once
processing semantics.

Components:
- CDCConnector: Debezium-based database change capture
- StreamProcessor: Kafka stream processing with transformation
- SchemaRegistry: Schema version management and migration
- StreamMonitor: Real-time throughput and latency monitoring
"""

from __future__ import annotations

from .connector import CDCConfig, CDCConnector, ChangeEvent, ChangeOperation
from .schema_registry import (
    CompatibilityMode,
    SchemaMigration,
    SchemaRegistry,
    SchemaVersion,
)
from .stream_processor import (
    DataTransformer,
    ExactlyOnceTracker,
    ProcessedEvent,
    StreamMonitor,
    StreamProcessor,
)

__all__ = [
    # Connector
    "CDCConfig",
    "CDCConnector",
    "ChangeEvent",
    "ChangeOperation",
    # Stream processor
    "DataTransformer",
    "ExactlyOnceTracker",
    "ProcessedEvent",
    "StreamMonitor",
    "StreamProcessor",
    # Schema registry
    "CompatibilityMode",
    "SchemaMigration",
    "SchemaRegistry",
    "SchemaVersion",
]