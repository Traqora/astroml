# Ingestion monitoring

The ingestion pipeline exposes a heartbeat check so operators can detect when
no new ledgers have been processed for a configurable period.

## Heartbeat check

`astroml.observability.ingestion.check_ingestion_heartbeat` compares the
current time to the `last_processed_at` timestamp recorded in the ingestion
state store and returns a `CheckResult`:

- `OK` — a ledger was processed within `stale_threshold_seconds`.
- `DEGRADED` — no ledger for `stale_threshold_seconds` (default 300s).
- `FAIL` — no ledger for `fail_threshold_seconds` (default 2 × stale threshold).

Example:

```python
from astroml.ingestion.state import StateStore
from astroml.observability.ingestion import check_ingestion_heartbeat

store = StateStore()
result = check_ingestion_heartbeat(store, stale_threshold_seconds=300)
print(result.status, result.remediation)
```

## Prometheus metric

`astroml.ingestion.metrics.INGESTION_LAST_PROCESSED_AT` is a Gauge that
records the Unix timestamp of the most recently processed ledger. Use it to
build alerts such as:

```promql
(time() - astroml_ingestion_last_processed_at_seconds) > 300
```

## State store timestamp

`StateStore.mark_processed` records `last_processed_at` as an ISO-8601 UTC
timestamp whenever a ledger is processed. Existing state files without the
field are treated as having no recorded ingestion time and report
`DEGRADED` until the next successful ingestion.
