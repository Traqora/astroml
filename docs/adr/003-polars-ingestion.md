# ADR-003: Polars for Ingestion

## Status

Accepted

## Context

AstroML processes high-volume transaction ledger streams from the Stellar network. Ingesting, parsing, filtering, and transforming hundreds of thousands of transactions into graph node and edge structures can be a CPU and memory bottleneck during bulk backfills.

Alternatives considered:
- **Pandas**: Traditional Python data processing standard, but single-threaded by default with higher memory footprint for string/object columns.
- **Pure Python lists/dicts**: Simple, but slow for multi-hundred thousand row transformations.
- **Dask / PySpark**: Scalable distributed processing, but heavy framework overhead for single-node development and quick start pipelines.

## Decision

We chose **Polars** as the primary data processing engine for ledger ingestion and normalization.

Key reasons:
- Built in Rust with multi-threaded execution out-of-the-box.
- Apache Arrow memory format reduces copy overhead and memory usage.
- Lazy evaluation API (`LazyFrame`) enables query optimization before execution.
- Delivers 5–10× throughput improvements over Pandas during bulk ledger parsing.

## Consequences

### Positive
- High throughput ledger backfills (~85 ledgers/sec).
- Significantly lower memory utilization for large batch ingestion.
- Expressive API for complex windowing and aggregations.

### Negative / Tradeoffs
- Codebase requires developers to be familiar with Polars syntax alongside Pandas.
- Conversion to PyTorch Geometric tensors requires explicit numpy/arrow array extraction steps.
