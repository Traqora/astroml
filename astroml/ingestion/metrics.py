"""Prometheus metrics for ingestion services."""
from prometheus_client import Counter, Gauge, Histogram, Summary

# Stream metrics
STREAM_RECORDS_PROCESSED = Counter(
    "astroml_ingestion_records_total",
    "Total number of Stellar records processed",
    ["stream_type", "horizon_url"]
)

STREAM_ERRORS = Counter(
    "astroml_ingestion_errors_total",
    "Total number of ingestion errors",
    ["stream_type", "horizon_url", "error_type"]
)

STREAM_CONNECTION_HEALTH = Gauge(
    "astroml_ingestion_connection_health",
    "Connection health status (1 for healthy, 0 for unhealthy)",
    ["stream_type", "horizon_url"]
)

STREAM_RATE_LIMIT_BACKOFF = Gauge(
    "astroml_ingestion_rate_limit_backoff_seconds",
    "Current rate limit backoff in seconds",
    ["stream_type", "horizon_url"]
)

STREAM_PROCESSING_LATENCY = Histogram(
    "astroml_ingestion_processing_seconds",
    "Time spent processing a batch of records",
    ["stream_type", "horizon_url"]
)

STREAM_CURSOR = Gauge(
    "astroml_ingestion_cursor",
    "Current cursor position (numeric representation if possible)",
    ["stream_type", "horizon_url"]
)

STREAM_LAG_SECONDS = Gauge(
    "astroml_ingestion_lag_seconds",
    "Current lag in seconds (time since last record's created_at)",
    ["stream_type", "horizon_url"]
)

