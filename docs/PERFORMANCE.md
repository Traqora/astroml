# Performance Characteristics

This document covers measured throughput, identified bottlenecks, sizing guidelines, and regression test thresholds for AstroML.

---

## Measured Throughput

All figures are from benchmarks run on a single-node configuration (8 vCPU, 32 GB RAM, SSD-backed PostgreSQL) with the default `configs/training/default.yaml` settings unless noted.

| Stage | Metric | Measured value | Notes |
|---|---|---|---|
| Ledger ingestion | Ledgers / second | ~85 ledgers/s | Batch of 1 000 ledgers, Polars-based parsing |
| Graph building | Edges / second | ~12 000 edges/s | `GraphSnapshot` with 50 000 edges total |
| Feature computation | Features / account / second | ~2 500 features/s | All node features in `FeatureStore` |
| GCN training (CPU) | Samples / second | ~800 samples/s | 50 epochs, 10 000 edge dataset |
| GCN training (GPU) | Samples / second | ~18 000 samples/s | NVIDIA A10, same dataset |
| API response (p50) | ms | ~18 ms | `/api/v1/fraud/alerts`, warm DB |
| API response (p99) | ms | ~95 ms | Same endpoint under load |

> **Note:** These figures reflect the synthetic quick-start pipeline. Production throughput depends heavily on DB hardware, Stellar Horizon rate limits, and graph density.

---

## Scaling Bottlenecks

### 1. Database queries

The single biggest bottleneck is typically the PostgreSQL layer:

- **Composite indexes on `(account_id, timestamp)`** are essential for time-windowed graph queries. Missing or partial indexes cause full-table scans that degrade >10× on datasets > 500 k edges.
- **Feature store reads** are SQLite-backed by default. SQLite works well up to ~1 M stored feature vectors but becomes the bottleneck above that — migrate to PostgreSQL feature storage for large-scale deployments.
- **N+1 query patterns** in `GraphSnapshot.build()` can appear when fetching per-account operations. Use `selectinload` / `joinedload` if you extend the ORM query logic.

Run the built-in query profiler to identify slow queries:

```python
from astroml.db.query_profiler import QueryProfiler
profiler = QueryProfiler(session)
profiler.start()
# ... run your pipeline ...
profiler.report()
```

### 2. Memory for large graphs

`GraphSnapshot` loads the full edge set for the time window into memory as a list of `Edge` dataclass objects. Memory growth is roughly linear:

| Edges in window | Approximate RAM |
|---|---|
| 10 000 | ~20 MB |
| 100 000 | ~180 MB |
| 500 000 | ~900 MB |
| 1 000 000 | ~1.8 GB |

Mitigations:
- Reduce the time window (`--window 7d` instead of `30d`).
- Enable chunked graph building via `GraphSnapshot(chunk_size=50_000)`.
- Use the `oom_snapshot_memory_experiment.ipynb` notebook to profile a specific window before production runs.

### 3. CPU vs GPU training tradeoffs

| Scenario | Recommendation |
|---|---|
| Dataset < 50 k edges | CPU training is sufficient; GPU overhead from data transfer is not worthwhile |
| Dataset 50 k – 1 M edges | GPU provides 15–25× speedup; use `requirements-train.txt` with CUDA |
| Dataset > 1 M edges | Consider mini-batch training with `NeighborLoader` from PyTorch Geometric |
| Inference only (no training) | CPU is fine; load the saved `.pt` model file |

Set `training.device: cuda` in `configs/training/default.yaml` to enable GPU. The training loop auto-falls back to CPU if CUDA is not available.

### 4. Stellar Horizon rate limiting

The ingestion layer fetches from `https://horizon.stellar.org`. The public endpoint is rate-limited to ~100 req/s. Exceeding this returns HTTP 429. Mitigation:

- Run your own Stellar Horizon instance via the Docker setup.
- Configure `HORIZON_URL` in `.env` to point to a self-hosted node.

---

## Sizing Guidelines

### Nodes / edges per memory tier

| Available RAM | Max edges in window | Recommended `chunk_size` |
|---|---|---|
| 4 GB | ~200 000 | 25 000 |
| 8 GB | ~500 000 | 50 000 |
| 16 GB | ~1 000 000 | 100 000 |
| 32 GB | ~2 500 000 | 250 000 |
| 64 GB | ~5 000 000 | 500 000 |

> Rule of thumb: allow ~180 bytes per edge in the snapshot plus ~500 MB headroom for PyTorch tensors and PostgreSQL connection pool buffers.

### Recommended instance types

| Workload | AWS | GCP | Azure |
|---|---|---|---|
| Development / CI | `t3.large` (2 vCPU, 8 GB) | `e2-standard-2` | `Standard_B2ms` |
| Ingestion worker | `c6i.2xlarge` (8 vCPU, 16 GB) | `c2-standard-8` | `Standard_F8s_v2` |
| Graph + feature compute | `r6i.2xlarge` (8 vCPU, 64 GB) | `m2-ultramem-208` | `Standard_E8s_v4` |
| GPU training | `g4dn.xlarge` (4 vCPU, 16 GB, T4) | `n1-standard-4 + T4` | `Standard_NC4as_T4_v3` |
| Production API + DB | `m6i.2xlarge` + `db.r6g.large` RDS | `n2-standard-8` + CloudSQL | `Standard_D8s_v4` |

---

## Performance Regression Test Thresholds

These thresholds are checked as part of the CI benchmark suite (see also Issue #23 — automated performance regression detection).

| Metric | Threshold | Test file |
|---|---|---|
| Ingestion throughput | ≥ 50 ledgers/s | `tests/benchmark/test_queries.py` |
| Graph build time (10 k edges) | ≤ 2 s | `tests/test_transaction_graph.py` |
| Feature computation (1 000 accounts) | ≤ 5 s | `tests/test_frequency.py` |
| Link prediction training (10 epochs, CPU) | ≤ 60 s | `tests/test_link_prediction.py` |
| API `/health` p99 latency | ≤ 50 ms | `tests/test_healthz_endpoints.py` |
| Memory: GraphSnapshot (50 k edges) | ≤ 200 MB RSS | `tests/test_graph_memory_profile.py` |

To run the benchmark suite locally:

```bash
pytest tests/benchmark/ tests/test_transaction_graph.py tests/test_link_prediction.py -v
```

---

## Profiling Tools

AstroML ships with several profiling utilities:

```bash
# Profile feature computation end-to-end
python profile_feature_computation.py

# Memory profiling for a graph snapshot
python -c "from astroml.features.graph.memory_profile import run_profile; run_profile()"

# DB query profiler (outputs slow queries to stdout)
python -m astroml.db.query_profiler --threshold 100ms
```

See `docs/database-query-profiling.md` for deeper guidance on query optimization.
