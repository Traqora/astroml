# PostgreSQL Vacuum Tuning Guide

This document explains how to tune PostgreSQL's autovacuum for AstroML's
write-heavy ingestion tables, and how to verify the settings are keeping up.

## Overview

The ingestion pipeline (`astroml.ingestion`) streams Horizon ledgers into
PostgreSQL continuously via `astroml.db.session`. The tables it writes to are
almost entirely **insert-dominated** — `ledgers`, `transactions`, `operations`,
`effects`, and `normalized_transactions` (see
`astroml/db/models/__init__.py`) — with `processed_ledgers` seeing frequent
small updates to track ingestion progress.

Note that ingestion uses `session.merge()` for idempotency (see
`astroml/ingestion/stream.py`, `enhanced_stream.py`, and `batch.py`), so a
replayed or backfilled ledger produces an `UPDATE` rather than a duplicate
row. In steady-state streaming this is rare (each ledger is normally seen
once), but backfills and reprocessing runs can generate a burst of dead
tuples on top of the insert volume — the default
`autovacuum_vacuum_scale_factor` still applies and should not be disabled,
only supplemented with the insert-scale-factor settings below.

PostgreSQL's default `autovacuum` settings are tuned for a generic mixed
workload and under-react to this pattern in two ways:

- **Insert-only tables never trigger the default vacuum threshold**, which is
  based on *dead* (updated/deleted) tuples. Without `autovacuum_vacuum_insert_scale_factor`
  (PG 13+), the visibility map is never updated, so sequential scans and
  index-only scans degrade over time and a future anti-wraparound vacuum runs
  as one enormous, disruptive pass instead of many small ones.
- **High insert volume means stale planner statistics matter more than
  usual.** `autovacuum_analyze_scale_factor` defaults to reacting only after
  10% of a table changes, which on a multi-million-row `transactions` table
  is a long time to fly blind on the query planner's row estimates.

## Recommended settings

Apply these as per-table storage parameters rather than instance-wide
defaults, so smaller reference tables (`assets`, `models`, `experiments`,
etc.) keep the stock behavior:

```sql
-- Insert-heavy, append-only ingestion tables.
ALTER TABLE ledgers SET (
    autovacuum_vacuum_insert_scale_factor = 0.02,
    autovacuum_vacuum_insert_threshold = 1000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 1000
);

ALTER TABLE transactions SET (
    autovacuum_vacuum_insert_scale_factor = 0.02,
    autovacuum_vacuum_insert_threshold = 1000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 1000
);

ALTER TABLE operations SET (
    autovacuum_vacuum_insert_scale_factor = 0.02,
    autovacuum_vacuum_insert_threshold = 1000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 1000
);

ALTER TABLE effects SET (
    autovacuum_vacuum_insert_scale_factor = 0.02,
    autovacuum_vacuum_insert_threshold = 1000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 1000
);

ALTER TABLE normalized_transactions SET (
    autovacuum_vacuum_insert_scale_factor = 0.02,
    autovacuum_vacuum_insert_threshold = 1000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 1000
);

-- processed_ledgers is small but updated frequently (progress tracking);
-- it needs a *lower* dead-tuple threshold so it doesn't bloat.
ALTER TABLE processed_ledgers SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_vacuum_threshold = 50
);
```

Apply the equivalent instance-wide floor in `postgresql.conf` (or the managed
Postgres provider's parameter group) so newly created ingestion tables — for
example a future partition — inherit sane defaults before someone remembers
to set per-table overrides:

```
autovacuum_vacuum_insert_scale_factor = 0.05
autovacuum_vacuum_insert_threshold = 1000
autovacuum_analyze_scale_factor = 0.05
autovacuum_analyze_threshold = 1000
autovacuum_max_workers = 4
autovacuum_naptime = 15s
```

`autovacuum_vacuum_insert_scale_factor` requires PostgreSQL 13 or newer. On
older versions, insert-only tables must instead be vacuumed on a schedule
(see below), since there is no dead-tuple activity to trigger autovacuum.

## Verifying autovacuum is keeping up

Check when each ingestion table was last vacuumed/analyzed and how many
tuples have been inserted since:

```sql
SELECT
    relname,
    n_live_tup,
    n_dead_tup,
    n_ins_since_vacuum,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE relname IN (
    'ledgers', 'transactions', 'operations',
    'effects', 'normalized_transactions', 'processed_ledgers'
)
ORDER BY n_ins_since_vacuum DESC;
```

A healthy ingestion table should show `last_autovacuum` / `last_autoanalyze`
timestamps that advance roughly in proportion to ingestion volume, and
`n_ins_since_vacuum` should not grow unbounded between runs. If
`last_autovacuum` is `NULL` on a large, long-running table, autovacuum has
never triggered on it — a strong signal the insert-scale-factor settings
above are missing or too high.

To confirm autovacuum is not falling behind in real time, watch
`pg_stat_progress_vacuum` while a run is in flight:

```sql
SELECT * FROM pg_stat_progress_vacuum;
```

## Manual / scheduled vacuum as a fallback

For PostgreSQL versions before 13 (no insert-scale-factor), or if
`pg_stat_user_tables` shows autovacuum isn't triggering on an insert-only
table, schedule an explicit `VACUUM (ANALYZE)` during a low-traffic window
via `scripts/` cron or the existing ops scheduler:

```sql
VACUUM (ANALYZE) ledgers;
VACUUM (ANALYZE) transactions;
VACUUM (ANALYZE) operations;
VACUUM (ANALYZE) effects;
VACUUM (ANALYZE) normalized_transactions;
```

Avoid `VACUUM FULL` outside of a maintenance window: it takes an exclusive
lock on the table for the duration of the rewrite, which will block
ingestion writes and API reads.

## Related

- `docs/database-query-profiling.md` — diagnosing slow queries once
  statistics are known to be fresh.
- `astroml/db/session.py` — connection pool and engine configuration.
- `astroml/db/pool_health.py` — runtime connection-pool health checks; a
  vacuum backlog often first shows up here as growing query latency and
  pool saturation.
