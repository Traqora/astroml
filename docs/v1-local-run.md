# Running the V1 API Locally

This guide covers starting the API on your machine and making sample requests against the three V1 endpoint groups: health, model readiness, and baseline predictions. No Redis, MLflow, or other optional services are required.

---

## Prerequisites

- Python 3.11+
- Dependencies installed: `pip install -r requirements-api.txt`
- (Optional) Copy `.env.example` to `.env` and edit for your environment. If you skip this the API starts with SQLite and authentication disabled.

---

## Start the API

### Option A — Make (recommended)

```bash
make run-api
```

This runs:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Option B — Direct uvicorn

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API is ready when you see:

```
INFO:     Application startup complete.
```

It listens on **http://localhost:8000**. Interactive API docs are at **http://localhost:8000/docs**.

### Minimal environment (no .env file)

The defaults are designed to work without any configuration:

| Setting | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./astroml.db` | Created automatically on first start |
| `AUTH_ENABLED` | `true` | Set to `false` to skip auth on all requests |
| `DISABLE_SCHEDULER` | *(unset)* | Set to `true` to suppress the batch scoring scheduler |
| `DB_QUERY_TIMEOUT_SECONDS` | `5` | Increase if queries are slow in dev |
| `LLM_TIMEOUT_SECONDS` | `30` | Only relevant if `LLM_PROVIDER` is configured |

For a no-config local run:

```bash
AUTH_ENABLED=false DISABLE_SCHEDULER=true uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Sample requests

All examples below use `curl`. Replace `localhost:8000` with your host and port.

### 1. Basic health check (no auth required)

```bash
curl http://localhost:8000/health
```

Expected response (`200 OK`):

```json
{"status": "ok"}
```

### 2. Database health (feature-gated)

Requires the `health` feature to be enabled on your account, or `AUTH_ENABLED=false`.

```bash
# With AUTH_ENABLED=false
curl http://localhost:8000/health/db
```

Expected response (`200 OK`):

```json
{
  "status": "healthy",
  "latency_ms": 1.23,
  "pool_stats": {},
  "metrics": {},
  "error": null,
  "error_code": null
}
```

If the DB query exceeds `DB_QUERY_TIMEOUT_SECONDS`, the response is:

```json
{
  "type": "https://astroml.dev/problems/DB_TIMEOUT",
  "title": "Database query timed out",
  "status": 503,
  "code": "DB_TIMEOUT",
  "detail": "Database query timed out",
  "instance": "/health/db",
  "details": {}
}
```

### 3. List models — model readiness (feature-gated)

Requires the `model_readiness` feature or `AUTH_ENABLED=false`.

```bash
curl http://localhost:8000/api/v1/models
```

Expected response (`200 OK`):

```json
{
  "data": [],
  "total": 0,
  "page": 1,
  "page_size": 20
}
```

Retrieve a specific model by ID:

```bash
curl http://localhost:8000/api/v1/models/1
```

Returns `404` if no model with that ID exists, or `403 FEATURE_NOT_ENABLED` if the feature is not enabled on your account.

### 4. Baseline predictions — fraud score (feature-gated)

Requires the `baseline_predictions` feature or `AUTH_ENABLED=false`.

```bash
curl -X POST http://localhost:8000/api/v1/fraud/score \
  -H "Content-Type: application/json" \
  -d '{
    "accounts": ["GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"],
    "edges": []
  }'
```

Expected response (`200 OK`) when no model is loaded:

```json
{
  "scores": {
    "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN": 0.0
  }
}
```

Returns `0.0` for all accounts when no trained model is loaded — this is the safe default, not an error.

### 5. Fraud alerts

```bash
curl "http://localhost:8000/api/v1/fraud/alerts?page=1&page_size=10"
```

Expected response (`200 OK`):

```json
{
  "data": [],
  "page": 1,
  "page_size": 10,
  "total": 0
}
```

---

## Feature-gated endpoint errors

If authentication is enabled and your account does not have the required feature, the API returns:

```json
{
  "type": "https://astroml.dev/problems/FEATURE_NOT_ENABLED",
  "title": "The 'model_readiness' feature is not enabled for account 'GABC...'.",
  "status": 403,
  "code": "FEATURE_NOT_ENABLED",
  "detail": "The 'model_readiness' feature is not enabled for account 'GABC...'.",
  "instance": "/api/v1/models/1",
  "details": {
    "feature": "model_readiness",
    "enabled_features": []
  }
}
```

Enable features for an account using the admin endpoint (requires an `admin`-scoped token):

```bash
curl -X PUT http://localhost:8000/api/v1/accounts/GABC.../features \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"enabled_features": ["health", "model_readiness", "baseline_predictions"]}'
```

See [Account Features](../QUICKSTART_GUIDE.md#account-features-v1-api) for full details.

---

## Readiness and liveness probes

These are always available without authentication and are safe for use as health checks in Docker or Kubernetes:

```bash
curl http://localhost:8000/health/readiness   # 200 {"status":"ready"} or 503
curl http://localhost:8000/health/liveness    # 200 {"status":"alive"}
```

---

## OpenAPI / Swagger UI

The full interactive API reference is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json
