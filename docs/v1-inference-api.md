# V1 Inference API

The v1 inference routes are mounted under `/api/v1` in both
`astroml.api.v1_app:app` and the main `astroml.api.app:app`. Use the standalone
entry point to avoid starting the main app's database-backed batch scheduler:

```bash
uvicorn astroml.api.v1_app:app --host 127.0.0.1 --port 8000
```

The v1 routes provide a local baseline scoring path and do not query the
database, object storage, Horizon, or other external services.

## Endpoints

`GET /api/v1/health` is a liveness check and returns `{"status":"ok"}` while
the API process is serving requests.

`GET /api/v1/ready` returns `200` with `status: "ready"` when NumPy is
available and at least one predictor is registered. It returns `503` and
`status: "not_ready"` otherwise. The response includes the dependency checks
and registered predictor names. The readiness check does not probe unrelated
services.

`POST /api/v1/predict` scores a batch of feature rows. The built-in
`mean_baseline` returns the arithmetic mean of each row as a scalar score; it
is a deterministic smoke-test baseline, not a trained or calibrated model.
Call `register_predictor(name, callable)` to provide a local model. A callable
must return exactly one finite scalar score per row.

## Request and Response

```json
{
  "model_name": "mean_baseline",
  "features": [[1.0, 3.0], [2.0, 4.0]]
}
```

Successful responses use a stable schema:

```json
{
  "schema_version": "v1",
  "model_name": "mean_baseline",
  "count": 2,
  "scores": [2.0, 3.0]
}
```

`scores` are scalar model outputs, not class labels or calibrated
probabilities. The request forbids unknown fields and requires a rectangular,
finite matrix with 1–256 rows, 1–256 features per row, and at most 65,536
values total. Invalid or oversized requests return `422`; an unknown model
returns `404`; an invalid predictor output or execution failure returns `500`.

Predictor registration is process-local and is not persisted across restarts.