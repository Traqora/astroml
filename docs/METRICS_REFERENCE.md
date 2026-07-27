# Prometheus Metrics Reference

Metrics exported by the AstroML API and pipeline (Issues #567 and #550).
Scrape endpoint: `GET /metrics` (Prometheus text exposition format).

## Core metrics

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `http_request_duration_seconds` | Histogram | `method`, `endpoint`, `status` | Request latency. `endpoint` is the **route template** (`/api/v1/accounts/{id}`), not the raw path, so cardinality stays bounded. |
| `http_requests_total` | Counter | `method`, `endpoint`, `status` | Requests served, by response status. |
| `db_pool_size` | Gauge | `pool`, `state` | Connections per state: `configured`, `in_use`, `idle`, `overflow`, `capacity`. |
| `db_pool_utilization_ratio` | Gauge | `pool` | `in_use / capacity`, in `[0, 1]`. |
| `feature_compute_time` | Histogram | `feature` | Seconds spent computing one feature. |
| `model_inference_latency` | Histogram | `model` | Seconds spent in a model prediction call. |
| `active_jobs` | Gauge | `job_type` | Jobs currently running, e.g. `ingestion`. |

Histogram buckets: `0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30`
seconds. Each histogram also exports the standard `_bucket`, `_sum` and
`_count` series.

## Instrumenting new code

All helpers live in `astroml.observability.metrics` and work on sync *and*
async callables.

```python
from astroml.observability.metrics import (
    track_active_job,
    track_feature_compute,
    track_inference,
    track_time,
    FEATURE_COMPUTE_TIME,
)

@track_feature_compute("degree_centrality")
def compute_degree_centrality(graph):
    ...

@track_inference("gnn_fraud_v2")
async def predict(batch):
    ...

# Any histogram, any label set:
@track_time(FEATURE_COMPUTE_TIME, "temporal_decay")
def compute_temporal_decay(df):
    ...

# Gauge that stays balanced even if the job raises:
with track_active_job("ingestion"):
    run_backfill()
```

For a dynamic label value, use the Prometheus client directly:

```python
with FEATURE_COMPUTE_TIME.labels(feature_name).time():
    result = compute(feature_name)
```

### Already instrumented

| Path | Metric |
| --- | --- |
| API middleware (`api/app.py`) | `http_request_duration_seconds`, `http_requests_total` |
| `FeatureEngine.compute_feature` | `feature_compute_time` |
| `DeepSVDDFraudDetector.predict_anomaly_scores` | `model_inference_latency` |
| `IngestionService.ingest_stream` | `active_jobs{job_type="ingestion"}` |
| `/metrics`, `/healthz/db`, `/metrics/db-pool` | `db_pool_size`, `db_pool_utilization_ratio` |

## Connection pool health indicators

The pool gauges are sampled on every `/metrics` scrape and on every database
health probe. Interpretation:

| Indicator | Healthy | Investigate |
| --- | --- | --- |
| `db_pool_utilization_ratio` | < 0.8 | ≥ 0.8 sustained — leaked sessions or slow queries |
| `db_pool_size{state="overflow"}` | 0 most of the time | Persistently > 0 — steady-state size is too small |
| `db_pool_size{state="idle"}` | > 0 | Constantly 0 while `in_use` is high — pool is a bottleneck |

Pool behaviour is configured through the same environment variables the API
uses (`DB_POOL_MAX_SIZE`, `DB_POOL_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`,
`DB_POOL_RECYCLE`, `DB_POOL_PRE_PING`). Two settings matter for correctness
rather than throughput:

- **`pool_pre_ping=True`** (default) — every checkout is validated with a
  cheap round trip, so a connection killed by a database restart or an
  idle-timeout proxy surfaces as a retry rather than a request error.
- **`pool_recycle=1800`** (default) — connections older than 30 minutes are
  discarded, staying under typical proxy and `wait_timeout` limits.

Useful queries:

```promql
# Pool utilization
db_pool_utilization_ratio

# Requests that would block once the pool saturates
sum(rate(http_requests_total[5m])) by (endpoint)

# P95 latency per endpoint
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint))

# Error ratio
sum(rate(http_requests_total{status=~"5.."}[5m]))
  / sum(rate(http_requests_total[5m]))
```

## Alerts

Defined in `monitoring/prometheus/alert_rules.yml`, group `astroml_api_alerts`:

| Alert | Condition | Severity |
| --- | --- | --- |
| `DatabasePoolNearExhaustion` | `db_pool_utilization_ratio >= 0.8` for 5m | warning |
| `DatabasePoolExhausted` | `db_pool_utilization_ratio >= 1` for 1m | critical |
| `ApiHighLatency` | P95 > 1s for 5m | warning |
| `ApiHighErrorRate` | 5xx ratio > 5% for 5m | critical |
| `IngestionJobsStalled` | `active_jobs{job_type="ingestion"} == 0` for 30m | warning |
| `ModelInferenceLatencyHigh` | P95 > 2s for 10m | warning |

## Grafana dashboards

| Dashboard | File |
| --- | --- |
| API Health & Pool (these metrics) | [`monitoring/grafana/api_health_dashboard.json`](../monitoring/grafana/api_health_dashboard.json) |
| API latency | [`monitoring/grafana/api_latency_dashboard.json`](../monitoring/grafana/api_latency_dashboard.json) |
| Database performance | [`monitoring/grafana/database_performance_dashboard.json`](../monitoring/grafana/database_performance_dashboard.json) |
| Ingestion throughput | [`monitoring/grafana/ingestion_throughput_dashboard.json`](../monitoring/grafana/ingestion_throughput_dashboard.json) |

Dashboards are provisioned from `monitoring/grafana/provisioning/`.

## Related

- Health probes: [HEALTH_CHECKS.md](HEALTH_CHECKS.md)
- Implementation: `astroml/observability/metrics.py`, `astroml/db/pool_health.py`
