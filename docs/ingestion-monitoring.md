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
- `DEGRADED` (not `FAIL`) — no heartbeat is on record at all: a state file
  written before the field existed, or none yet.

Served over HTTP as `GET /healthz/ingestion` (`api/routers/healthz.py`), and
also folded into the `GET /healthz` aggregate as the `ingestion` component.
It is intentionally excluded from the readiness gate — see
[HEALTH_CHECKS.md](HEALTH_CHECKS.md).

```bash
curl -s localhost:8000/healthz/ingestion | jq
```

Thresholds can be lowered or raised per call, or configured with
`INGESTION_STALE_THRESHOLD_SECONDS` / `INGESTION_FAIL_THRESHOLD_SECONDS`.

## Freshness metrics and alerts

`GET /metrics` re-samples the heartbeat on every scrape via
`astroml.observability.ingestion.refresh_ingestion_metrics`, so staleness keeps
climbing while ingestion is silent instead of freezing at the last value a dead
worker pushed:

- `astroml_ingestion_last_success_timestamp_seconds` — Unix timestamp of the
  most recently processed ledger.
- `astroml_ingestion_staleness_seconds` — seconds since that timestamp.

Both are `NaN` until the first heartbeat lands, and `NaN` never satisfies a
staleness comparison — so a deployment that intentionally runs no ingestion
stays quiet.

`monitoring/prometheus/alert_rules.yml` (group `astroml_ingestion_alerts`):

| Alert | Condition | Severity |
| --- | --- | --- |
| `IngestionDataStale` | `astroml_ingestion_staleness_seconds > 900` for 5m | warning |
| `IngestionHeartbeatStopped` | `time() - astroml_ingestion_last_success_timestamp_seconds > 3600` for 5m | critical |

See [runbooks/ingestion_heartbeat_stale.md](runbooks/ingestion_heartbeat_stale.md)
when one fires.

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
timestamp whenever a ledger is processed. The batched flush inside
`IngestionService.ingest_stream` stamps the same field through
`IngestionState.record_processed`, so both write paths stay in agreement.
Existing state files without the field are treated as having no recorded
ingestion time and report `DEGRADED` until the next successful ingestion.
