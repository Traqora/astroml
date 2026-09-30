# ADR-002: SQLite for Feature Store Metadata

## Status

Accepted

## Context

The AstroML Feature Store requires metadata tracking for feature definitions, schema versions, entity associations, and execution runs. Bulk feature vectors are stored in columnar files (Parquet), but feature catalog metadata needs fast lookup, zero-configuration local execution, and low overhead.

Alternatives considered:
- **PostgreSQL**: Robust, but adds container/service dependencies for quick-start and local CLI operations.
- **In-memory dictionary / JSON file**: Simple, but lacks concurrent safety, SQL query capability, and durability across process restarts.

## Decision

We chose **SQLite** as the default metadata engine for the AstroML Feature Store catalog.

Key reasons:
- Zero installation or network configuration required.
- Single-file database simplifies local development, testing, and CI pipelines.
- Standard SQL support for querying feature versions and lineage.
- Storing heavy feature arrays in Parquet while keeping metadata in SQLite provides optimal performance balance.

## Consequences

### Positive
- Zero setup dependency for feature store initialization.
- Fast metadata indexing and querying.
- Clean separation of catalog metadata (SQLite) and heavy numerical arrays (Parquet).

### Negative / Tradeoffs
- Concurrent writes to feature metadata are limited by SQLite file locking.
- High-concurrency enterprise deployments with >1M registered feature vectors should migrate metadata to PostgreSQL.
