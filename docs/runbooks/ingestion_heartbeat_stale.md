# Ingestion Heartbeat / Stale Data Runbook

## Alerts

| Alert | Severity | Condition |
| --- | --- | --- |
| `IngestionDataStale` | warning | `astroml_ingestion_staleness_seconds > 900` for 5m |
| `IngestionHeartbeatStopped` | critical | `time() - astroml_ingestion_last_success_timestamp_seconds > 3600` for 5m |

## What the metric actually is

Both gauges are re-sampled from the ingestion state store on every `/metrics`
scrape (`astroml.observability.ingestion.refresh_ingestion_metrics`, wired into
`GET /metrics`). They therefore keep aging while ingestion is silent, instead of
freezing at the last value a dead worker pushed.

- `astroml_ingestion_last_success_timestamp_seconds` — Unix timestamp of the
  most recently processed ledger (`last_processed_at` in
  `ingestion_state.json`).
- `astroml_ingestion_staleness_seconds` — seconds since that timestamp. Grows on
  its own between successful ingestions.

Both are `NaN` until the first heartbeat lands. `NaN` never satisfies the
comparisons above, so these alerts stay quiet in deployments that intentionally
run no ingestion (see *Silencing* below).

## Symptoms

- Dashboards show data freshness sliding; `/healthz/ingestion` returns
  `degraded` (still HTTP 200) and then `fail` (HTTP 503).
- `/healthz` aggregate reports the `ingestion` component as the worst status.
- Downstream graph snapshots and features are built on a narrowing window.

## Immediate Actions

1. **Confirm where the freshness stands**

   ```bash
   # Same numbers the alert uses, without a Prometheus round trip.
   curl -s localhost:8000/healthz/ingestion | jq
   curl -s localhost:8000/metrics | grep astroml_ingestion_staleness
   ```

   `details.last_processed_at` and `details.last_processed_ledger` tell you
   exactly which ledger the pipeline stopped on, and `details.path` names the
   state file that is being read.

2. **Check the worker is alive**

   ```bash
   kubectl get pods -n astroml -l app=astroml-ingestion
   kubectl logs -n astroml deployment/astroml-ingestion --tail=200
   ```

3. **Rule out a state-file mismatch first**

   The most common false alarm is a probe reading a *different* state file than
   the worker writes. `INGESTION_STATE_FILE` (or the default
   `<cwd>/.astroml_state/ingestion_state.json`) must resolve to the same shared
   volume in both processes. `details.path` in the probe response is
   authoritative.

   ```bash
   kubectl exec -n astroml deploy/astroml-api -- \
     stat "$INGESTION_STATE_FILE"
   ```

## Common Causes

### 1. Worker crashed or was OOM-killed

```bash
kubectl get pod -n astroml -l app=astroml-ingestion \
  -o jsonpath='{.items[*].status.containerStatuses[*].restartCount}'
kubectl describe pod -n astroml -l app=astroml-ingestion | grep -A5 "Last State"
```

**Resolution**: restart and watch it resume — state is idempotent, so a restart
re-picks from `last_processed_ledger + 1` without replaying history.

```bash
kubectl rollout restart deployment/astroml-ingestion -n astroml
kubectl logs -f -n astroml deployment/astroml-ingestion
```

### 2. Upstream (Horizon) is failing or rate limiting

Look for sustained retries or backoff in the worker logs, and cross-check the
`IngestionLagHigh` / `PersistentRateLimit` alerts. See
[ingestion_lag.md](ingestion_lag.md) for the Horizon checks.

**Resolution**: wait out the upstream incident, or add a second Horizon
endpoint. Freshness recovers on its own once fetches succeed.

### 3. Worker is alive but making no progress

A worker can hold an open connection, pass liveness, and still process nothing:
blocked on a database write, a full connection pool, or a queue that is not
draining.

```bash
kubectl top pods -n astroml -l app=astroml-ingestion
curl -s localhost:8000/healthz/db | jq .details.pool
```

**Resolution**: fix the blocking dependency (see
[db_query_latency.md](db_query_latency.md) for pool saturation), then restart the
worker if it does not unblock.

### 4. Nothing is broken — ingestion simply is not scheduled here

If this deployment is API-only, the gauges stay `NaN` and neither alert can
fire. If you are seeing them at all, a heartbeat *has* landed on this state file
at least once, so ingestion is expected here.

## Silencing

For an environment that deliberately runs no ingestion but shares a state file
with another environment:
- set `INGESTION_STATE_FILE` to a path local to the environment, or
- silence `IngestionDataStale` / `IngestionHeartbeatStopped` in Alertmanager for
  that cluster.

Do **not** raise `INGESTION_STALE_THRESHOLD_SECONDS` to hide a real stall —
that knob exists to reflect a genuinely lower-frequency pipeline, not to mute it.

## Tuning

| Variable | Default | Effect |
| --- | --- | --- |
| `INGESTION_STALE_THRESHOLD_SECONDS` | `300` | Silence before `degraded` |
| `INGESTION_FAIL_THRESHOLD_SECONDS` | `600` | Silence before `fail` |
| `INGESTION_STATE_FILE` | `<cwd>/.astroml_state/ingestion_state.json` | State file read as the heartbeat |

Pipeline that legitimately ingests less often than once every 5 minutes should
raise `INGESTION_STALE_THRESHOLD_SECONDS` and the two alert thresholds together.

## Prevention

- Alert on `IngestionDataStale` (warning) *before* `IngestionHeartbeatStopped`
  (critical) so there is time to react to a slowdown.
- Keep `restart: unless-stopped` / a Kubernetes restart policy on the worker so
  a crash self-heals into a heartbeat.
- See [../ingestion-monitoring.md](../ingestion-monitoring.md) for the full
  metric set and [../HEALTH_CHECKS.md](../HEALTH_CHECKS.md) for the probe
  contract.
