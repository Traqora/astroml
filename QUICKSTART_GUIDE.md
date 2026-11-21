# AstroML Quick Start Guide

## Overview

This guide explains the three improvements made to AstroML:

1. **Quick Start Command** - Single entry point for the full pipeline
2. **Benchmark Reproducibility** - Config and seed storage with results
3. **Architecture Documentation** - Detailed diagrams and module organization

---

## 1. Quick Start Command

### What It Does

The quick start command wires sample data through the complete ingestion → graph → train pipeline:

```
Generate Sample Data → Build Graph → Train Model → Save Results
```

### How to Run

#### Option A: Using Make (Recommended)

```bash
# Default: 100 ledgers, 50 accounts, 10 epochs
make quickstart

# Verbose: 200 ledgers, 100 accounts, 20 epochs
make quickstart-verbose
```

#### Option B: Using Python Module

```bash
# Default settings
python -m astroml.quick_start

# Custom parameters
python -m astroml.quick_start \
  --num-ledgers 200 \
  --num-accounts 100 \
  --epochs 20 \
  --seed 42
```

#### Option C: Using CLI

```bash
# Via CLI command
python -m astroml quickstart \
  --num-ledgers 100 \
  --num-accounts 50 \
  --epochs 10 \
  --seed 42
```

### Output Structure

```
benchmark_results/quickstart/
├── config.json          # Full configuration with random seed
├── result.json          # Training metrics and performance
└── metadata.json        # Run metadata linking config and result
```

### Example Output

```
================================================================================
AstroML Quick Start: Ingestion → Graph → Train Pipeline
================================================================================

[Step 1/5] Generating sample ledger data...
Generated 100 ledgers with 50 accounts

[Step 2/5] Building transaction graph...
Built graph with 2000 edges and 50 nodes
Graph validation: {'num_nodes': 50, 'num_edges': 2000, 'density': 0.0016}

[Step 3/5] Creating benchmark configuration...

[Step 4/5] Training baseline model...
Epoch 0: Train Loss = 0.6931, Val Loss = 0.6892
Epoch 5: Train Loss = 0.4521, Val Loss = 0.4612
Epoch 9: Train Loss = 0.3214, Val Loss = 0.3456
Training complete. Best metrics: {'auc': 0.92, 'precision': 0.88, 'recall': 0.85}

[Step 5/5] Saving benchmark results...
Saved config to benchmark_results/quickstart/config.json
Saved result to benchmark_results/quickstart/result.json
Saved metadata to benchmark_results/quickstart/metadata.json

✓ Quick start completed successfully!
Results saved to: benchmark_results/quickstart
================================================================================
```

### Configuration Parameters

```python
class QuickStartConfig:
    # Sample data parameters
    NUM_SAMPLE_LEDGERS = 100      # Number of synthetic ledgers
    NUM_ACCOUNTS = 50              # Number of accounts
    NUM_ASSETS = 5                 # Number of asset types
    TRANSACTIONS_PER_LEDGER = 20   # Transactions per ledger

    # Training parameters
    TRAIN_EPOCHS = 10              # Training epochs
    BATCH_SIZE = 16                # Batch size
    LEARNING_RATE = 0.01           # Learning rate
    RANDOM_SEED = 42               # Random seed for reproducibility

    # Output
    OUTPUT_DIR = Path("./benchmark_results/quickstart")
    STATE_DIR = Path("./.astroml_state_quickstart")
```

---

## 2. Benchmark Reproducibility

### Problem Solved

Previously, benchmark results were saved without their configuration or random seeds, making it impossible to reproduce runs.

### Solution

Each benchmark run now saves three linked files:

#### config.json

Contains the complete configuration including:

- Model name and parameters
- Data configuration (ledger range, ratios)
- Training configuration (epochs, learning rate, **random seed**)
- Device and output settings

```json
{
  "model_name": "LinkPredictor",
  "model_params": {
    "hidden_dim": 64,
    "num_layers": 2
  },
  "epochs": 10,
  "batch_size": 16,
  "learning_rate": 0.01,
  "random_seed": 42,
  "device": "cuda",
  "output_dir": "./benchmark_results/quickstart"
}
```

#### result.json

Contains all benchmark metrics:

- Model name and parameters
- Timestamp and device
- **Random seed used**
- Data statistics (nodes, edges, splits)
- Training metrics (losses, epochs, convergence)
- Performance metrics (AUC, Precision, Recall, F1)
- Resource usage (memory, GPU)

```json
{
  "model_name": "LinkPredictor",
  "model_params": {...},
  "timestamp": 1234567890.123,
  "device": "cuda",
  "random_seed": 42,
  "total_nodes": 50,
  "total_edges": 2000,
  "train_nodes": 40,
  "val_nodes": 5,
  "test_nodes": 5,
  "train_time": 12.34,
  "epochs_trained": 10,
  "best_epoch": 8,
  "train_losses": [0.693, 0.521, ...],
  "val_losses": [0.689, 0.461, ...],
  "metrics": {
    "auc": 0.92,
    "precision": 0.88,
    "recall": 0.85,
    "f1": 0.86
  },
  "peak_memory_mb": 512.5,
  "gpu_memory_mb": 2048.0
}
```

#### metadata.json

Links config and result with run metadata:

- Unique run ID
- Timestamp
- File paths
- Quick reference metrics

```json
{
  "run_id": "LinkPredictor_1234567890",
  "timestamp": "2024-05-29T10:30:45.123456",
  "model_name": "LinkPredictor",
  "random_seed": 42,
  "device": "cuda",
  "config_file": "./benchmark_results/quickstart/LinkPredictor_1234567890_config.json",
  "result_file": "./benchmark_results/quickstart/LinkPredictor_1234567890_result.json",
  "train_time_seconds": 12.34,
  "epochs_trained": 10,
  "best_metrics": {
    "auc": 0.92,
    "precision": 0.88,
    "recall": 0.85,
    "f1": 0.86
  }
}
```

### How to Reproduce a Run

1. **Find the run**: Locate the metadata.json file
2. **Load config**: Read the config.json file
3. **Set seeds**: Use the `random_seed` value
4. **Recreate**: Run with identical configuration

```python
import json
from astroml.benchmarking.config import BenchmarkConfig
from astroml.benchmarking.core import ModelBenchmark

# Load config
with open("benchmark_results/quickstart/LinkPredictor_1234567890_config.json") as f:
    config_dict = json.load(f)

config = BenchmarkConfig(**config_dict)

# Run benchmark with same config
benchmark = ModelBenchmark(config)
result = benchmark.run_benchmark()
```

### Implementation Details

The `_save_results()` method in `astroml/benchmarking/core.py` now:

1. Creates a unique run ID from model name and timestamp
2. Saves config.json with full configuration
3. Saves result.json with all metrics
4. Saves metadata.json linking the two files

```python
def _save_results(self, result: BenchmarkResult):
    """Save benchmark results and configuration to file for reproducibility."""
    output_dir = Path(self.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique run ID
    run_id = f"{result.model_name}_{int(result.timestamp)}"

    # Save config
    config_dict = asdict(self.config)
    config_path = output_dir / f"{run_id}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    # Save result
    result_dict = asdict(result)
    result_path = output_dir / f"{run_id}_result.json"
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    # Save metadata
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "config_file": str(config_path),
        "result_file": str(result_path),
        "random_seed": result.random_seed,
        ...
    }
    metadata_path = output_dir / f"{run_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
```

---

## 3. Architecture Documentation

### What Was Added

The README.md now includes:

1. **High-Level Pipeline Diagram** - Shows the 6-layer architecture
2. **Data Flow Details** - Step-by-step data transformation
3. **Module Organization** - Directory structure and responsibilities
4. **Quick Start Section** - Multiple ways to run the pipeline

### Architecture Layers

```
Layer 1: INGESTION
├─ Ledger backfill (Polars)
├─ Incremental ingestion
├─ State tracking (idempotent)
└─ PostgreSQL storage

Layer 2: NORMALIZATION
├─ Raw Stellar schema (Ledger, Transaction, Operation)
├─ Graph mirror layer (GraphAccount, GraphEdge)
└─ Composite indexes (account_id, timestamp)

Layer 3: GRAPH BUILDING
├─ Time-windowed snapshots
├─ Edge construction
├─ Node indexing
└─ Graph validation

Layer 4: FEATURE ENGINEERING
├─ Transaction frequency
├─ Asset diversity
├─ Structural importance (degree, betweenness, PageRank)
├─ Feature store & versioning
└─ Point-in-time queries

Layer 5: TRAINING
├─ Temporal train/test split
├─ Link prediction task
├─ Negative sampling
├─ PyTorch Geometric models (GCN, GraphSAGE, GAT)
└─ Early stopping

Layer 6: BENCHMARKING & EVALUATION
├─ Reproducible configs
├─ Random seed tracking
├─ Metric computation (AUC, Precision, Recall)
├─ Memory profiling
└─ Result persistence
```

### Data Flow

```
Stellar Ledger Data
    ↓
[Ingestion Service] → PostgreSQL
    ↓
[Database Schema] → Raw + Graph layers
    ↓
[Graph Snapshot] → Edge objects + node_index
    ↓
[Feature Store] → Node/edge features
    ↓
[Temporal Split] → Train/test edges
    ↓
[Link Prediction Task] → Positive/negative labels
    ↓
[Model Training] → Trained LinkPredictor
    ↓
[Benchmark Results] → config.json + result.json + metadata.json
```

### Module Organization

```
astroml/
├── ingestion/           # Ledger ingestion & state tracking
│   ├── service.py       # IngestionService (incremental, idempotent)
│   ├── state.py         # StateStore (tracks processed ledgers)
│   └── backfill.py      # Bulk ledger loading
├── db/                  # Database layer
│   ├── schema.py        # SQLAlchemy ORM models
│   └── session.py       # Database connection management
├── features/            # Feature engineering
│   ├── feature_store.py # Enterprise feature management
│   ├── graph/
│   │   └── snapshot.py  # Time-windowed graph construction
│   ├── frequency.py     # Transaction frequency features
│   ├── asset_diversity.py
│   └── gnn/             # Graph neural network layers
├── models/              # ML models
│   ├── link_predictor.py
│   ├── gcn.py
│   ├── sage.py
│   └── deep_svdd.py
├── tasks/               # Training tasks
│   └── link_prediction_task.py
├── training/            # Training utilities
│   ├── temporal_split.py # Prevent data leakage
│   └── train_link_prediction.py
├── benchmarking/        # Benchmarking framework
│   ├── core.py          # ModelBenchmark orchestrator
│   ├── config.py        # Configuration management
│   └── metrics.py       # Metric computation
├── quick_start.py       # Quick start pipeline
└── cli.py               # Command-line interface
```

---

## Files Modified/Created

### New Files

1. **astroml/quick_start.py** (350 lines)
   - `QuickStartConfig` class with default parameters
   - `set_random_seeds()` for reproducibility
   - `generate_sample_ledgers()` creates synthetic data
   - `build_sample_graph()` constructs transaction graph
   - `train_baseline_model()` trains LinkPredictor
   - `save_benchmark_config()` saves config + results
   - `run_quickstart()` orchestrates the pipeline

2. **Makefile** (30 lines)
   - `make quickstart` - Run quick start
   - `make quickstart-verbose` - Run with more data
   - `make test`, `make lint`, `make format` - Development commands
   - `make clean` - Clean build artifacts

3. **QUICKSTART_GUIDE.md** (This file)
   - Comprehensive guide to all three improvements

### Modified Files

1. **astroml/cli.py**
   - Added `quickstart` subcommand with arguments
   - Integrated `run_quickstart()` into CLI
   - Supports `--num-ledgers`, `--num-accounts`, `--epochs`, `--seed` parameters

2. **astroml/benchmarking/core.py**
   - Enhanced `_save_results()` method
   - Now saves config.json, result.json, and metadata.json
   - Generates unique run IDs
   - Stores random seed with results

3. **README.md**
   - Added detailed architecture diagrams
   - Added high-level pipeline visualization
   - Added data flow details
   - Added module organization
   - Added quick start section with 3 usage options
   - Expanded from ~100 lines to ~400 lines

---

## Usage Examples

### Example 1: Run Quick Start with Defaults

```bash
make quickstart
```

Output:

```
[Step 1/5] Generating sample ledger data...
Generated 100 ledgers with 50 accounts

[Step 2/5] Building transaction graph...
Built graph with 2000 edges and 50 nodes

[Step 3/5] Creating benchmark configuration...

[Step 4/5] Training baseline model...
Training complete. Best metrics: {'auc': 0.92, 'precision': 0.88, 'recall': 0.85}

[Step 5/5] Saving benchmark results...
✓ Quick start completed successfully!
Results saved to: benchmark_results/quickstart
```

### Example 2: Run with Custom Parameters

```bash
python -m astroml.quick_start \
  --num-ledgers 500 \
  --num-accounts 200 \
  --epochs 50 \
  --seed 123
```

### Example 3: Reproduce a Previous Run

```python
import json
from astroml.benchmarking.config import BenchmarkConfig
from astroml.benchmarking.core import ModelBenchmark

# Load previous config
with open("benchmark_results/quickstart/LinkPredictor_1234567890_config.json") as f:
    config_dict = json.load(f)

# Create config with same settings
config = BenchmarkConfig(**config_dict)

# Run benchmark - will produce identical results
benchmark = ModelBenchmark(config)
result = benchmark.run_benchmark()
```

### Example 4: Compare Multiple Runs

```bash
# Run 1: Seed 42
python -m astroml.quick_start --seed 42

# Run 2: Seed 123
python -m astroml.quick_start --seed 123

# Compare results
ls -la benchmark_results/quickstart/
# LinkPredictor_1234567890_config.json
# LinkPredictor_1234567890_result.json
# LinkPredictor_1234567890_metadata.json
# LinkPredictor_1234567891_config.json
# LinkPredictor_1234567891_result.json
# LinkPredictor_1234567891_metadata.json
```

---

## Benefits

### 1. Quick Start Command

- ✓ Single entry point for the full pipeline
- ✓ Generates sample data automatically
- ✓ Trains baseline model in seconds
- ✓ Produces reproducible results
- ✓ Great for testing and demos

### 2. Benchmark Reproducibility

- ✓ All configs stored with results
- ✓ Random seeds tracked
- ✓ Easy to reproduce runs
- ✓ Linked metadata for traceability
- ✓ Enables scientific rigor

### 3. Architecture Documentation

- ✓ Clear visual diagrams
- ✓ Data flow explanation
- ✓ Module organization
- ✓ Easier onboarding
- ✓ Better understanding of pipeline

---

## Next Steps

1. **Test the quick start**: `make quickstart`
2. **Check the output**: `ls benchmark_results/quickstart/`
3. **Review the config**: `cat benchmark_results/quickstart/config.json`
4. **Reproduce a run**: Use the config to re-run with identical settings
5. **Explore the architecture**: Read the updated README.md

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'numpy'"

**Solution**: Install dependencies

```bash
pip install -r requirements.txt
```

### Issue: "Database connection error"

**Solution**: Configure database in `config/database.yaml` or set environment variables

### Issue: "CUDA out of memory"

**Solution**: Reduce parameters

```bash
python -m astroml.quick_start --num-ledgers 50 --num-accounts 25
```

### Issue: "Results not saved"

**Solution**: Check output directory permissions

```bash
mkdir -p benchmark_results/quickstart
chmod 755 benchmark_results/quickstart
```

---

## Questions?

Refer to:

- README.md - Architecture and overview
- astroml/quick_start.py - Implementation details
- astroml/benchmarking/core.py - Benchmark framework
- astroml/cli.py - CLI integration


## Account Features (V1 API)

The V1 API enforces per-account feature gating. Each account has an `enabled_features` list; a request to a gated endpoint is rejected with `403 FEATURE_NOT_ENABLED` when the required feature is absent. This is a **deny-by-default** posture — a new account with no features set cannot call any gated endpoint.

### Feature-to-endpoint map

| Feature name | Gated endpoints |
|---|---|
| `health` | `GET /health/db`, `GET /health/db/pool`, `GET /health/db/status` |
| `model_readiness` | `GET /api/v1/models/{id}`, `POST /api/v1/models/{id}/activate`, `GET /api/v1/models/{id}/metrics` |
| `baseline_predictions` | `POST /api/v1/fraud/score` |

Basic liveness/readiness probes (`GET /health`, `/health/readiness`, `/health/liveness`) are **not** gated.

### Enabling features for an account (admin only)

```http
PUT /api/v1/accounts/{public_key}/features
Authorization: Bearer <admin-token>
Content-Type: application/json

{
  "enabled_features": ["health", "model_readiness", "baseline_predictions"]
}
```

**200 response:**

```json
{
  "public_key": "GABC...",
  "enabled_features": ["health", "model_readiness", "baseline_predictions"]
}
```

Passing an empty list removes all features. The `ApiAccount` row must already exist (created by the ingestion pipeline). Use `GET /api/v1/accounts/{public_key}` to confirm the account exists before calling this endpoint.

### Error responses

**403 — feature not enabled:**

```json
{
  "type": "https://astroml.dev/problems/FEATURE_NOT_ENABLED",
  "title": "The 'baseline_predictions' feature is not enabled for account 'GABC...'.",
  "status": 403,
  "code": "FEATURE_NOT_ENABLED",
  "detail": "The 'baseline_predictions' feature is not enabled for account 'GABC...'.",
  "instance": "/api/v1/fraud/score",
  "details": {
    "feature": "baseline_predictions",
    "enabled_features": ["health"]
  }
}
```

**404 — account not found (feature dependency):**

```json
{
  "type": "https://astroml.dev/problems/ACCOUNT_NOT_FOUND",
  "title": "No account found for user 'mykey'.",
  "status": 404,
  "code": "ACCOUNT_NOT_FOUND",
  "detail": "No account found for user 'mykey'.",
  "instance": "/api/v1/models/1",
  "details": { "subject": "mykey" }
}
```

**422 — unknown feature name in management request:**

```json
{
  "type": "https://astroml.dev/problems/VALIDATION_ERROR",
  "title": "Unknown feature name(s): ['turbo_mode'].",
  "status": 422,
  "code": "VALIDATION_ERROR",
  "detail": "Unknown feature name(s): ['turbo_mode'].",
  "instance": "/api/v1/accounts/GABC.../features",
  "details": {
    "unknown_features": ["turbo_mode"],
    "known_features": ["baseline_predictions", "health", "model_readiness"]
  }
}
```

### Local development (AUTH_ENABLED=false)

When `AUTH_ENABLED=false` (the default in local dev and tests), the feature check is bypassed and all features are treated as enabled. No `ApiAccount` row is required.


## Request Timeouts (V1 API)

All V1 endpoints that make database or external service calls enforce configurable timeouts. When a call exceeds its limit the endpoint returns a structured RFC 7807 error response.

### Environment variables

| Variable | Default | Range | Applies to |
|---|---|---|---|
| `DB_QUERY_TIMEOUT_SECONDS` | `5` | 1–300 | All gated DB queries (health DB endpoints, model readiness, fraud alerts/stats/explanation) |
| `LLM_TIMEOUT_SECONDS` | `30` | 1–300 | `GET /api/v1/fraud/{id}/explanation` LLM call |
| `EXTERNAL_HTTP_TIMEOUT_SECONDS` | `10` | 1–300 | Reserved for future outbound HTTP calls; no V1 endpoint uses this yet |

Values outside the 1–300 range cause a `ValidationError` at startup and the app will not start.

### Error responses

**503 — database query timed out:**

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

**504 — LLM service timed out:**

```json
{
  "type": "https://astroml.dev/problems/UPSTREAM_TIMEOUT",
  "title": "LLM service timed out",
  "status": 504,
  "code": "UPSTREAM_TIMEOUT",
  "detail": "LLM service timed out",
  "instance": "/api/v1/fraud/42/explanation",
  "details": {}
}
```

### Gated endpoints

| Endpoint | Timeout type |
|---|---|
| `GET /health/db`, `GET /health/db/pool`, `GET /health/db/status` | `DB_QUERY_TIMEOUT_SECONDS` |
| `GET /health/readiness` | `DB_QUERY_TIMEOUT_SECONDS` |
| `GET /api/v1/models/{id}`, `POST /api/v1/models/{id}/activate`, `GET /api/v1/models/{id}/metrics` | `DB_QUERY_TIMEOUT_SECONDS` |
| `GET /api/v1/fraud/alerts`, `GET /api/v1/fraud/stats`, `GET /api/v1/fraud/alerts/prioritized` | `DB_QUERY_TIMEOUT_SECONDS` |
| `GET /api/v1/fraud/{id}/explanation` (DB queries) | `DB_QUERY_TIMEOUT_SECONDS` |
| `GET /api/v1/fraud/{id}/explanation` (LLM call) | `LLM_TIMEOUT_SECONDS` |

The DB timeout is enforced at the PostgreSQL level via `SET LOCAL statement_timeout`. It is a no-op on SQLite (used in local dev and tests).


## Running the V1 API Locally

### Start command

```bash
make run-api
# equivalent to:
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

For a no-config local run with SQLite and no auth:

```bash
AUTH_ENABLED=false DISABLE_SCHEDULER=true uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API is ready when the console shows `Application startup complete.` Docs are at http://localhost:8000/docs.

### Sample requests

| Endpoint | Method | Feature required | No-auth default |
|---|---|---|---|
| `/health` | GET | None | `200 {"status":"ok"}` |
| `/health/readiness` | GET | None | `200 {"status":"ready"}` |
| `/health/db` | GET | `health` | `200` or `503 DB_TIMEOUT` |
| `/api/v1/models` | GET | None | `200 {"data":[],...}` |
| `/api/v1/models/{id}` | GET | `model_readiness` | `200` or `404` |
| `/api/v1/fraud/score` | POST | `baseline_predictions` | `200 {"scores":{...}}` |
| `/api/v1/fraud/alerts` | GET | None | `200 {"data":[],...}` |

See [`docs/v1-local-run.md`](docs/v1-local-run.md) for full curl examples and expected responses.
