# Health Check Endpoints

Granular health probes for the AstroML API (Issue #569), backed by the
connection pool health checks from Issue #550.

## Endpoints

| Endpoint | Purpose | Fails (503) when |
| --- | --- | --- |
| `GET /healthz` | Aggregate status across all dependencies | Any component fails |
| `GET /healthz/live` | Liveness — the process is responsive | Never (process is up) |
| `GET /healthz/startup` | Startup probe — initialisation finished | Lifespan startup has not completed |
| `GET /healthz/ready` | Readiness gate — safe to route traffic | Startup incomplete, DB unreachable, pool exhausted, disk full |
| `GET /healthz/db` | Database connectivity + pool saturation | `SELECT 1` fails or the pool is exhausted |
| `GET /healthz/cache` | Redis connectivity | Redis unreachable **and** `HEALTHZ_CACHE_REQUIRED=true` |
| `GET /healthz/disk` | Free space on the data volume | Free space below 10% |
| `GET /metrics/db-pool` | Connection pool utilization snapshot | Pool exhausted |

The legacy `GET /health` and `GET /health/*` endpoints are unchanged; nothing
that points at them needs to move.

## Response envelope

Every probe answers with the same JSON shape:

```json
{
  "status": "ok",
  "component": "db",
  "details": { "latency_ms": 3.41, "query": "SELECT 1", "pool": { "...": "..." } },
  "remediation": "",
  "duration_ms": 3.52
}
```

Aggregate probes (`/healthz`, `/healthz/ready`) nest one entry per component:

```json
{
  "status": "degraded",
  "probe": "ready",
  "details": {
    "startup": { "status": "ok", "component": "startup", "details": { "started": true, "ready": true, "uptime_seconds": 812.4 } },
    "db":      { "status": "ok", "component": "db", "details": { "latency_ms": 2.9 } },
    "cache":   { "status": "degraded", "component": "cache", "details": { "error": "Connection refused", "required": false } },
    "disk":    { "status": "ok", "component": "disk", "details": { "free_ratio": 0.63 } }
  },
  "remediation": "Redis is unreachable. Verify REDIS_URL, that the Redis service is running, ..."
}
```

### Status vocabulary

| Status | HTTP | Meaning |
| --- | ---: | --- |
| `ok` | 200 | Fully functional |
| `degraded` | 200 | Works, but close to a limit — keep serving, page someone |
| `fail` | 503 | Unusable — Kubernetes should stop routing traffic here |

`degraded` deliberately returns 200 so a saturated-but-working pod is not
removed from the Service, which would concentrate load on the remaining pods.

### Failure examples

Database down:

```json
{
  "status": "fail",
  "component": "db",
  "details": { "error": "connection refused", "error_type": "OSError" },
  "remediation": "The database is unreachable. Verify DATABASE_URL, that the server accepts connections, and that credentials and network policy allow this pod to connect.",
  "duration_ms": 5012.4
}
```

Pool saturated (still 200, status `degraded`):

```json
{
  "status": "degraded",
  "pool": {
    "pool_size": 20, "checked_in": 2, "checked_out": 25, "overflow": 5,
    "max_overflow": 10, "capacity": 30, "utilization_percent": 83.33,
    "alert_threshold_percent": 80.0, "saturated": true, "exhausted": false
  },
  "remediation": "Pool utilization is 83%, at or above the 80% alert threshold. Check for slow queries holding connections and consider raising DB_POOL_MAX_SIZE before it exhausts."
}
```

## Readiness gating

Readiness passes only when **startup has completed** and the **database** is
reachable with a non-exhausted pool. The cache and disk checks can downgrade
readiness to `degraded` but not fail it, unless disk space is critical or
`HEALTHZ_CACHE_REQUIRED=true`.

The API lifespan calls `readiness_state.mark_started()` after the scheduler and
WebSocket poller are wired up, and `readiness_state.set_ready(False, ...)` at the
start of shutdown — so a terminating pod reports `fail` on `/healthz/ready` and
is drained from the Service before its dependencies are torn down.

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `HEALTHZ_TIMEOUT_SECONDS` | `5` | Per-dependency probe timeout |
| `HEALTHZ_DISK_PATH` | `.` | Filesystem inspected by the disk probe |
| `HEALTHZ_CACHE_REQUIRED` | `false` | When true, a Redis outage fails readiness |
| `REDIS_URL` | `redis://localhost:6379/0` | Cache endpoint pinged by `/healthz/cache` |

## Kubernetes probes

`k8s/astroml-api-deployment.yaml` wires all three probe types:

```yaml
startupProbe:
  httpGet: { path: /healthz/startup, port: 8000 }
  periodSeconds: 5
  failureThreshold: 30      # up to 150s to initialise
livenessProbe:
  httpGet: { path: /healthz/live, port: 8000 }
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 3
readinessProbe:
  httpGet: { path: /healthz/ready, port: 8000 }
  periodSeconds: 5
  timeoutSeconds: 5
  failureThreshold: 3
```

Rules of thumb:

- **Liveness must not check dependencies.** A database outage should not
  restart every API pod — that turns a dependency outage into an outage plus a
  restart storm.
- **The startup probe owns slow boots.** With a startup probe configured,
  Kubernetes suspends the liveness probe until startup succeeds, so
  `initialDelaySeconds` tuning is no longer needed on liveness.
- **Readiness is the traffic switch.** It is the only probe that should fail
  when a dependency is down.

## Related

- Metric definitions: [METRICS_REFERENCE.md](METRICS_REFERENCE.md)
- Pool internals: `astroml/db/pool_health.py`
- Probe implementation: `api/routers/healthz.py`
