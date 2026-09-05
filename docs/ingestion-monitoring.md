# Ingestion monitoring

The ingestion pipeline exposes metrics and health checks so operators can
monitor backfills and detect stalls.

## Heartbeat check

`astroml.observability.ingestion.check_ingestion_heartbeat` compares the
current time to the `last_processed_at` timestamp recorded in the ingestion
state store and returns a `CheckResult`:

- `OK` — a ledger was processed within `stale_threshold_seconds`.
- `DEGRADED` — no ledger for `stale_threshold_seconds` (default 300s).
- `FAIL` — no ledger for `fail_threshold_seconds` (default 2 × stale threshold).

## Per-batch throughput metrics

`astroml.ingestion.batch_metrics.BatchMetricsRecorder` emits progress metrics
for each batch of ledgers handled by `IngestionService.ingest_stream`:

- `astroml_ingestion_batch_duration_seconds` — wall-clock time per batch.
- `astroml_ingestion_batch_ledgers_total{status="processed|skipped|error"}` —
  ledgers handled per batch.
- `astroml_ingestion_batch_throughput_ledgers_per_second` — throughput of the
  most recent batch.

Use these metrics to monitor and tune long backfills, for example:

```promql
rate(astroml_ingestion_batch_ledgers_total{status="processed"}[5m])
```

## State store timestamp

`StateStore.mark_processed` records `last_processed_at` as an ISO-8601 UTC
timestamp whenever a ledger is processed. Existing state files without the
field are treated as having no recorded ingestion time and report
`DEGRADED` until the next successful ingestion.
