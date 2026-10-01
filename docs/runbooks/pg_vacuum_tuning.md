# PostgreSQL VACUUM Tuning

## Why this matters for AstroML

The ingestion pipeline (`astroml/ingestion/normalizer.py`) writes one
`normalized_transactions` row per Horizon operation (and one per hop for
path payments), so this table grows by insert volume alone and never
gets `UPDATE`d after the fact — it is append-only, not update-heavy.
`fraud_alerts` and `feature_store`-backed tables, by contrast, are
written once and then updated repeatedly as scoring/features refresh, so
they accumulate dead tuples ("bloat") that only `VACUUM` reclaims.

The deployed Postgres (`k8s/postgres-deployment.yaml`, `postgres:15-alpine`,
256Mi-512Mi memory) runs with **no `postgresql.conf` customization** —
every setting below is the PG15 default unless otherwise noted. This doc
exists because the current defaults are undertuned for this workload,
not because anything is misconfigured relative to a prior baseline.

## Background: what autovacuum does and why defaults undertune it here

`VACUUM` reclaims space left by `UPDATE`/`DELETE` (dead tuples) and
prevents transaction ID (XID) wraparound. `autovacuum` runs it
automatically per table once dead-tuple count crosses a threshold:

```
threshold = autovacuum_vacuum_threshold + autovacuum_vacuum_scale_factor * reltuples
```

Defaults: `autovacuum_vacuum_threshold = 50`,
`autovacuum_vacuum_scale_factor = 0.2` (20% of the table). On a
100k-row table that means ~20,050 dead tuples must accumulate before
autovacuum fires — fine for a small table, but on a table with
millions of rows (which `normalized_transactions` will reach at
sustained ingestion volume), 20% dead tuples is a lot of bloat to carry
before cleanup even starts, and the resulting `VACUUM` run then has a
huge amount of work to do in one pass, competing with ingestion writes
for I/O.

## Recommended tuning

### Per-table autovacuum overrides (high-write tables)

Set directly on the table so this doesn't require a
`postgresql.conf` change/restart, and only the tables that need it are
affected:

```sql
-- normalized_transactions: append-only, high insert volume.
-- Lower the scale factor so autovacuum's ANALYZE keeps planner
-- statistics fresh without waiting for 20% growth, and lower the
-- vacuum scale factor so cleanup runs happen in smaller, more frequent
-- passes instead of one large pass that competes with ingestion writes.
ALTER TABLE normalized_transactions SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_vacuum_threshold = 1000,
  autovacuum_analyze_scale_factor = 0.02,
  autovacuum_analyze_threshold = 1000
);

-- fraud_alerts / feature_store tables: update-heavy (status/score
-- fields change after the row is created), so dead-tuple accumulation
-- happens faster per row than on an append-only table.
ALTER TABLE fraud_alerts SET (
  autovacuum_vacuum_scale_factor = 0.1,
  autovacuum_vacuum_cost_delay = 2
);
```

Apply the same pattern to any other table under sustained write load;
`pg_stat_user_tables.n_dead_tup` (see Monitoring below) is the signal
for which tables actually need an override versus which are fine on
defaults.

### Cluster-level settings

These require a `postgresql.conf` change and a restart, so they belong
in `k8s/postgres-deployment.yaml` as a mounted config, not applied
ad hoc:

```
# More vacuum workers so multiple bloated tables can be cleaned
# concurrently instead of queueing behind each other. Default is 3.
autovacuum_max_workers = 4

# Increase the I/O budget per vacuum cost-delay cycle so autovacuum
# makes faster progress under the current 256Mi-512Mi memory limits
# without starving foreground query I/O. Default is 200.
autovacuum_vacuum_cost_limit = 400

# maintenance_work_mem bounds how much of the table's dead-tuple set
# VACUUM can hold in memory per pass; too low means more passes over
# large tables. Default is 64MB; the pod's 512Mi limit has headroom
# for this at typical connection counts.
maintenance_work_mem = 128MB
```

## Monitoring bloat and vacuum activity

```sql
-- Dead tuple ratio per table — the signal that decides whether a
-- table needs a per-table override (see above), not a guess.
SELECT
  relname,
  n_live_tup,
  n_dead_tup,
  round(100.0 * n_dead_tup / GREATEST(n_live_tup + n_dead_tup, 1), 1) AS dead_pct,
  last_autovacuum,
  last_autoanalyze
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC
LIMIT 20;

-- Currently running vacuum operations and their progress (PG13+).
SELECT
  p.pid,
  p.relid::regclass AS table_name,
  p.phase,
  p.heap_blks_total,
  p.heap_blks_scanned,
  round(100.0 * p.heap_blks_scanned / GREATEST(p.heap_blks_total, 1), 1) AS pct_scanned
FROM pg_stat_progress_vacuum p;
```

Add `pg_stat_user_tables.n_dead_tup` (by table) and
`pg_stat_progress_vacuum`'s active-run count as Grafana panels
alongside the existing query-latency dashboard
(`docs/runbooks/db_query_latency.md`) rather than as a separate
dashboard — bloat and query latency share root causes often enough
that they should be read together.

## When to run a manual VACUUM instead of waiting for autovacuum

- After a large one-time backfill or bulk delete (autovacuum's
  threshold-based trigger is tuned for steady-state load, not a burst).
- When `pg_stat_user_tables.last_autovacuum` is null or far in the past
  for a table with a high `dead_pct` (autovacuum may be starved by
  other work; see cluster-level settings above).

```sql
-- ANALYZE-only refreshes planner statistics without reclaiming space;
-- cheap, safe to run anytime a table's data distribution has shifted.
ANALYZE normalized_transactions;

-- Full VACUUM ANALYZE reclaims space and updates statistics. Runs
-- concurrently with reads/writes (unlike VACUUM FULL, which locks the
-- table exclusively and should not be run against a live table here).
VACUUM ANALYZE normalized_transactions;
```

Never run `VACUUM FULL` against a table under active ingestion load —
it takes an `ACCESS EXCLUSIVE` lock for the duration, which would stall
the ingestion pipeline and any API reads against that table.
