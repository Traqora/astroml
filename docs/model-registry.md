# Model Registry

The AstroML Model Registry provides model versioning, lifecycle management, metrics tracking, comparison, and lineage for your machine learning models trained on Stellar network data.

## Overview

The Model Registry enables you to:
- Register new model versions with optional parent lineage
- Track model metrics and performance
- Activate specific model versions for production use
- Compare multiple model versions side-by-side
- Retrieve model version lineage (parent chain)
- Store model artifacts in a configurable location

## Endpoints

### List Registered Models
Get a list of all registered model versions sorted by creation date (most recent first).

**Endpoint:** `GET /api/v1/models`

**Response:**
```json
[
  {
    "id": 1,
    "name": "fraud_detector",
    "version": "fraud_detector_v20240101_120000",
    "path": "model_store/fraud_detector/fraud_detector_v20240101_120000/model.pt",
    "metrics": {
      "auc": 0.92,
      "precision": 0.88,
      "recall": 0.85
    },
    "status": "active",
    "parent_id": null,
    "created_at": "2024-01-01T12:00:00Z"
  }
]
```

---

### Register a New Model Version
Register a new model version. If a version isn't specified, one is automatically generated using the model name and a timestamp.

**Endpoint:** `POST /api/v1/models`

**Request Body:**
```json
{
  "name": "fraud_detector",
  "version": "v1.0.0",
  "path": "/path/to/your/model.pt",
  "metrics": {
    "auc": 0.92,
    "precision": 0.88,
    "recall": 0.85
  },
  "parent_id": 1
}
```

**Fields:**
- `name` (required): The name of the model
- `version` (optional): The version string. If not provided, it will be auto-generated as `{name}_v{YYYYMMDD_HHMMSS}`
- `path` (required): Path to the model artifact file
- `metrics` (optional): Dictionary of performance metrics
- `parent_id` (optional): ID of the parent model version for lineage tracking

**Response:**
```json
{
  "id": 2,
  "name": "fraud_detector",
  "version": "v1.0.0",
  "path": "model_store/fraud_detector/v1.0.0/model.pt",
  "metrics": {
    "auc": 0.92,
    "precision": 0.88,
    "recall": 0.85
  },
  "status": "inactive",
  "parent_id": 1,
  "created_at": "2024-01-02T12:00:00Z"
}
```

**Notes:**
- If the provided `path` exists, the file will be copied to `MODEL_STORE_PATH/{name}/{version}/`
- New models are registered with `status: "inactive"` by default
- If `parent_id` is provided and doesn't exist, a 404 error is returned

---

### Activate a Model Version
Activate a specific model version and deactivate all other versions with the same model name.

**Endpoint:** `POST /api/v1/models/{model_id}/activate`

**Path Parameters:**
- `model_id`: ID of the model version to activate

**Response:**
```json
{
  "id": 2,
  "name": "fraud_detector",
  "version": "v1.0.0",
  "path": "model_store/fraud_detector/v1.0.0/model.pt",
  "metrics": {
    "auc": 0.92,
    "precision": 0.88,
    "recall": 0.85
  },
  "status": "active",
  "parent_id": 1,
  "created_at": "2024-01-02T12:00:00Z"
}
```

**Notes:**
- Activating a model version invalidates the scorer cache
- Only one version per model name can be active at a time

---

### Get Model Metrics
Retrieve stored metrics for a specific model version.

**Endpoint:** `GET /api/v1/models/{model_id}/metrics`

**Path Parameters:**
- `model_id`: ID of the model version

**Response:**
```json
{
  "id": 1,
  "name": "fraud_detector",
  "version": "fraud_detector_v20240101_120000",
  "metrics": {
    "auc": 0.92,
    "precision": 0.88,
    "recall": 0.85
  }
}
```

---

### Compare Model Versions
Compare multiple model versions, generating a report with metric deltas, and identifying the best/worst versions per metric (higher is better assumption).

**Endpoint:** `POST /api/v1/models/compare`

**Request Body:**
```json
{
  "version_ids": [1, 2, 3]
}
```

**Fields:**
- `version_ids` (required): List of at least 2 model version IDs to compare

**Response:**
```json
{
  "versions": [
    {
      "id": 1,
      "name": "fraud_detector",
      "version": "v1.0.0",
      "path": "model_store/fraud_detector/v1.0.0/model.pt",
      "metrics": {"auc": 0.90, "precision": 0.80},
      "status": "inactive",
      "parent_id": null,
      "created_at": "2024-01-01T12:00:00Z"
    },
    {
      "id": 2,
      "name": "fraud_detector",
      "version": "v1.1.0",
      "path": "model_store/fraud_detector/v1.1.0/model.pt",
      "metrics": {"auc": 0.92, "precision": 0.85},
      "status": "active",
      "parent_id": 1,
      "created_at": "2024-01-02T12:00:00Z"
    }
  ],
  "metric_deltas": [
    {
      "metric": "auc",
      "values": {"1": 0.90, "2": 0.92},
      "delta": 0.02,
      "best": 2,
      "worst": 1
    },
    {
      "metric": "precision",
      "values": {"1": 0.80, "2": 0.85},
      "delta": 0.05,
      "best": 2,
      "worst": 1
    }
  ]
}
```

**Notes:**
- Deltas are calculated from the first version in the list to the last numeric value
- Best/worst are determined by higher metric values (common for most ML metrics like AUC, precision, recall)

---

### Get Model Lineage
Retrieve the parent chain (lineage) for a model version, starting from the given version and traversing up through parents.

**Endpoint:** `GET /api/v1/models/{model_id}/lineage`

**Path Parameters:**
- `model_id`: ID of the model version

**Response:**
```json
{
  "chain": [
    {
      "id": 3,
      "name": "fraud_detector",
      "version": "v1.2.0",
      "metrics": {"auc": 0.94, "precision": 0.88},
      "created_at": "2024-01-03T12:00:00Z"
    },
    {
      "id": 2,
      "name": "fraud_detector",
      "version": "v1.1.0",
      "metrics": {"auc": 0.92, "precision": 0.85},
      "created_at": "2024-01-02T12:00:00Z"
    },
    {
      "id": 1,
      "name": "fraud_detector",
      "version": "v1.0.0",
      "metrics": {"auc": 0.90, "precision": 0.80},
      "created_at": "2024-01-01T12:00:00Z"
    }
  ]
}
```

---

The REST endpoints above are one way to manage models. AstroML also ships a
command-line interface under the `models` subcommand so you can register,
version, tag, activate and roll back models without writing HTTP calls.

## Command-Line Interface (CLI)

### Prerequisites

The `models` commands talk directly to the AstroML database through the ORM, so
they need a reachable database. Configure it either with `config/database.yaml`
or by exporting `ASTROML_DATABASE_URL`:

```bash
export ASTROML_DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/astroml"
```

Every command is invoked as a module from the repository root:

```bash
python -m astroml.cli models --help
```

> Note on the `name`/`version` pair: the CLI identifies a model version by
> `--model-name` + `--version` (a unique `(name, version)` pair), while the REST
> API above uses the numeric `model_id`. The two address the same rows in the
> `model_registry` table.

### Register a model (with tags)

Registering is also where you attach an owner and tags — there is no separate
`tag` command; tags are set on `register` (and on `version`).

```bash
python -m astroml.cli models register \
  --name fraud_detector \
  --version v1.0.0 \
  --path ./artifacts/fraud_detector_v1.pt \
  --owner alice \
  --tags fraud graph-learning stellar \
  --metrics '{"auc": 0.90, "precision": 0.80, "recall": 0.78}'
```

`--status` defaults to `inactive`, so a freshly registered model is never live
until you activate it. Omit `--metrics` to register with no metrics. Example
output:

```json
{
  "id": 1,
  "name": "fraud_detector",
  "version": "v1.0.0",
  "path": "./artifacts/fraud_detector_v1.pt",
  "owner": "alice",
  "tags": ["fraud", "graph-learning", "stellar"],
  "mlflow_run_id": null,
  "status": "inactive",
  "created_at": "2024-01-01T12:00:00"
}
```

To associate a run with MLflow, pass `--mlflow-run-id <RUN_ID>` (and later pull
its metadata with `load-metadata`, shown below).

### Add a new version

`version` adds another `(name, version)` row for an existing model, taking the
same `--owner`, `--tags`, `--mlflow-run-id`, `--metrics` and `--status` flags:

```bash
python -m astroml.cli models version \
  --model-name fraud_detector \
  --version v1.1.0 \
  --path ./artifacts/fraud_detector_v1.1.pt \
  --tags fraud graph-learning stellar \
  --metrics '{"auc": 0.92, "precision": 0.85, "recall": 0.83}'
```

If the `(name, version)` pair already exists the command prints
`Error: Model 'fraud_detector' version 'v1.1.0' already exists` and exits `1`.

### Activate a model version

There is no `activate` verb on the CLI — activation is a status transition to
`active`. Setting a version to `active` automatically flips every other version
with the same name back to `inactive`, so only one version per model is live:

```bash
python -m astroml.cli models transition \
  --model-name fraud_detector \
  --version v1.1.0 \
  --stage active
```

`--stage` accepts `inactive`, `active`, or `deprecated`. Example output:

```json
{
  "id": 2,
  "name": "fraud_detector",
  "version": "v1.1.0",
  "status": "active"
}
```

### Roll back to an earlier version

Rollback is the same mechanism: activate the older version. Because activating
deactivates the others, re-activating `v1.0.0` restores it as the single live
version:

```bash
python -m astroml.cli models transition \
  --model-name fraud_detector \
  --version v1.0.0 \
  --stage active
```

### End-to-end example: register → tag → activate → roll back

The sequence below mirrors the lifecycle described at the top of this page. Run
it from the repository root after the database is configured:

```bash
# 1. Register v1.0.0 (inactive) and tag it
python -m astroml.cli models register \
  --name fraud_detector --version v1.0.0 \
  --path ./artifacts/fraud_detector_v1.pt \
  --owner alice --tags fraud graph-learning \
  --metrics '{"auc": 0.90}'

# 2. Train/publish an improvement as v1.1.0
python -m astroml.cli models version \
  --model-name fraud_detector --version v1.1.0 \
  --path ./artifacts/fraud_detector_v1.1.pt \
  --tags fraud graph-learning \
  --metrics '{"auc": 0.94}'

# 3. Promote v1.1.0 to production (deactivates v1.0.0)
python -m astroml.cli models transition \
  --model-name fraud_detector --version v1.1.0 --stage active

# 4. v1.1.0 regressed in production — roll back by re-activating v1.0.0
python -m astroml.cli models transition \
  --model-name fraud_detector --version v1.0.0 --stage active

# 5. Confirm only v1.0.0 is active now
python -m astroml.cli models list --status active --name fraud_detector
```

### List and filter models

```bash
# All active models
python -m astroml.cli models list --status active

# Filter by owner, name, or tags (space-separated), with pagination
python -m astroml.cli models list --owner alice --tags fraud --page 1 --page-size 20
```

Results are returned most-recent-first as JSON with a `total` count and a
`data` array of model rows.

### Compare two versions

```bash
python -m astroml.cli models compare \
  --model-name fraud_detector \
  --version1 v1.0.0 \
  --version2 v1.1.0
```

The response includes both versions' metrics plus a `metrics_diff` object keyed
by metric name, e.g. `{"auc": {"version1": 0.90, "version2": 0.94}}`.

### Load MLflow run metadata

If a version was registered with `--mlflow-run-id`, pull its tracked metadata:

```bash
python -m astroml.cli models load-metadata \
  --model-name fraud_detector \
  --version v1.1.0
```

This requires MLflow to be installed; otherwise the command prints
`Error: MLflow not available: ...` and exits `1`.

---

## Configuration

### MODEL_STORE_PATH
The directory where model artifacts are stored. Can be set via environment variable:
```bash
export MODEL_STORE_PATH=/path/to/model_store
```
Default: `./model_store`

---

## Usage Examples

### Python (requests)

```python
import requests

# List models
response = requests.get("http://localhost:8000/api/v1/models")
print(response.json())

# Register a new model with parent
model_data = {
    "name": "fraud_detector",
    "version": "v1.1.0",
    "path": "./benchmark_results/gcn_model_v2.pt",
    "metrics": {"auc": 0.92, "precision": 0.88, "recall": 0.85},
    "parent_id": 1
}
response = requests.post("http://localhost:8000/api/v1/models", json=model_data)
print(response.json())

# Activate a model
model_id = response.json()["id"]
response = requests.post(f"http://localhost:8000/api/v1/models/{model_id}/activate")
print(response.json())

# Get metrics
response = requests.get(f"http://localhost:8000/api/v1/models/{model_id}/metrics")
print(response.json())

# Compare versions
response = requests.post("http://localhost:8000/api/v1/models/compare", json={
    "version_ids": [1, model_id]
})
print(response.json())

# Get lineage
response = requests.get(f"http://localhost:8000/api/v1/models/{model_id}/lineage")
print(response.json())
```

### cURL

```bash
# List models
curl http://localhost:8000/api/v1/models

# Register a new model with parent
curl -X POST http://localhost:8000/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "fraud_detector",
    "version": "v1.1.0",
    "path": "./benchmark_results/gcn_model_v2.pt",
    "metrics": {"auc": 0.92, "precision": 0.88, "recall": 0.85},
    "parent_id": 1
  }'

# Activate a model (replace 2 with your model ID)
curl -X POST http://localhost:8000/api/v1/models/2/activate

# Get metrics (replace 2 with your model ID)
curl http://localhost:8000/api/v1/models/2/metrics

# Compare versions
curl -X POST http://localhost:8000/api/v1/models/compare \
  -H "Content-Type: application/json" \
  -d '{"version_ids": [1, 2]}'

# Get lineage (replace 2 with your model ID)
curl http://localhost:8000/api/v1/models/2/lineage
```

---

## Database Schema

The `model_registry` table stores all registered model versions:

| Column          | Type       | Description                                |
|-----------------|------------|--------------------------------------------|
| `id`            | BigInt     | Primary key (auto-incrementing)            |
| `name`          | String     | Model name                                 |
| `version`       | String     | Model version                              |
| `path`          | Text       | Path to model artifact                     |
| `owner`         | String     | Owner of the model (optional)              |
| `tags`          | JSON/JSONB | List of model tags (optional)              |
| `mlflow_run_id` | String     | Associated MLflow run id (optional)        |
| `metrics`       | JSON/JSONB | Performance metrics (optional)             |
| `status`        | String     | Status: `inactive`, `active`, `deprecated` |
| `parent_id`     | BigInt     | Parent model version ID (optional)         |
| `created_at`    | DateTime   | Creation timestamp                         |

**Indexes:**
- Unique index on `(name, version)`
- Index on `status`
- Index on `parent_id`
