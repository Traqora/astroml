# Supported v1 Inference API

This page is the v1 inference contract. Other endpoints may exist in the
application, but only the following are supported for deployed inference
clients. They are local HTTP endpoints and do not require PostgreSQL, Redis,
GraphQL or an LLM provider.

| Method | Path | Purpose | Success | Failure |
| --- | --- | --- | --- | --- |
| `GET` | `/api/v1` | API/version handshake | `200 {"version": "...", "status": "ok"}` | `503` when the app is unhealthy |
| `GET` | `/api/v1/fraud/ready` | Fraud model readiness | `200` with `status: "ok"` | `503` with `reason: "model_unavailable"` |
| `POST` | `/api/v1/fraud/score` | Baseline fraud scores | `200` with `scores` keyed by account | `503` when the loaded scorer raises |

All three are under the v1 prefix and require the bearer-token flow described
in [API Authentication](API_AUTHENTICATION.md). `/healthz/live` and
`/healthz/ready` are Kubernetes process probes, not inference endpoints; see
[Health Checks](HEALTH_CHECKS.md).

## Readiness

Call `GET /api/v1/fraud/ready` before sending a batch or starting a client. It
checks that the configured checkpoint can be loaded into the scorer. The
configured path is `MODEL_CHECKPOINT_PATH`, or
`benchmark_results/gcn_model.pt` when unset.

```json
{
  "status": "ok",
  "service": "fraud",
  "checkpoint": "benchmark_results/gcn_model.pt",
  "ready": true
}
```

A missing or corrupt checkpoint returns `503`:

```json
{
  "detail": {
    "status": "fail",
    "service": "fraud",
    "reason": "model_unavailable",
    "checkpoint": "benchmark_results/gcn_model.pt",
    "remediation": "Set MODEL_CHECKPOINT_PATH to a valid checkpoint or activate a model."
  }
}
```

Readiness does not query the model registry or database. It uses the same
scorer loader as `/api/v1/fraud/score`, so a `200` means the next scoring call
can use the loaded model; it does not guarantee that all account IDs in a
future request are valid.

## Scoring

`POST /api/v1/fraud/score` accepts 1–50 Stellar account IDs and up to 500
edges.

```bash
curl -X POST https://<host>/api/v1/fraud/score \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "accounts": ["GABC1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"],
    "edges": []
  }'
```

The response is `{"scores": {"<account>": <float>}}`. Account validation and
the score response shape are part of the v1 contract; scores are deterministic
for a given checkpoint and input graph, but absolute values are not calibrated
across blockchains. Call the readiness endpoint first: if no scorer is loaded,
`/api/v1/fraud/score` currently returns `0.0` for every account, which is not a
validated fraud signal.

## Boundaries

Supported in v1 inference:

* one model family (the baseline graph model loaded by `api.services.scorer`)
* one scoring endpoint (`/api/v1/fraud/score`)
* one readiness endpoint (`/api/v1/fraud/ready`)
* bearer-token authentication, standard HTTP status codes and JSON bodies

Not part of the v1 inference contract:

* LLM explanations, chat, search, alerts and other DB-backed routers
* GraphQL (`/graphql`) and WebSocket endpoints
* in-process Prometheus endpoints, which are for operators rather than clients
* alternative auth schemes (API keys, session cookies, unauthenticated access)
