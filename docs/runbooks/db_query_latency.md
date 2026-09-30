# Database Query Latency Runbook

## Alert
**Name**: DatabaseQueryLatencyHigh  
**Severity**: Critical  
**SLO**: P99 query latency < 500ms

## Symptoms
- Database queries are slow
- API responses degraded
- Database connection pool exhaustion

## Immediate Actions

1. **Check Database Status**
   ```bash
   # Check database pods
   kubectl get pods -n astroml -l app=postgres
   
   # Check database metrics
   kubectl exec -n astroml postgres-0 -- psql -U astroml -d astroml -c "
   SELECT state, count(*), avg(query_start) 
   FROM pg_stat_activity 
   GROUP BY state;
   "
   ```

2. **Identify Slow Queries**
   ```bash
   # Check pg_stat_statements
   kubectl exec -n astroml postgres-0 -- psql -U astroml -d astroml -c "
   SELECT query, calls, total_time, mean_time, max_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   "
   ```

3. **Check Locks**
   ```bash
   # Check for blocking locks
   kubectl exec -n astroml postgres-0 -- psql -U astroml -d astroml -c "
   SELECT pid, state, query_start, query
   FROM pg_stat_activity
   WHERE state != 'idle'
   ORDER BY query_start;
   "
   ```

## Common Causes

### 1. Missing Indexes
- Queries not using indexes
- Full table scans

**Resolution**:
```sql
-- Analyze query plan
EXPLAIN ANALYZE <slow_query>;

-- Add appropriate indexes
CREATE INDEX CONCURRENTLY idx_<column> ON <table>(<column>);
```

### 2. Lock Contention
- Long-running transactions holding locks
- Deadlocks

**Resolution**:
```bash
# Identify blocking queries
kubectl exec -n astroml postgres-0 -- psql -U astroml -d astroml -c "
SELECT pid, usename, query_start, state, query
FROM pg_stat_activity
WHERE pid IN (
    SELECT blocking_pid
    FROM pg_locks bl
    JOIN pg_stat_activity psa ON bl.pid = psa.pid
);
"

# Terminate long-running queries (carefully)
kubectl exec -n astroml postgres-0 -- psql -U astroml -d astroml -c "
SELECT pg_terminate_backend(<pid>);
"
```

### 3. Resource Exhaustion
- High CPU usage on database
- Memory pressure
- I/O bottlenecks

**Resolution**:
```bash
# Check database resource usage
kubectl top pod -n astroml postgres-0

# Check disk I/O
kubectl exec -n astroml postgres-0 -- iostat -x 1 5
```

### 4. Large Dataset Growth
- Tables grown beyond optimal size
- Statistics outdated

**Resolution**:
```sql
-- Update statistics
ANALYZE <table>;

-- Vacuum if needed
VACUUM ANALYZE <table>;
```

See [pg_vacuum_tuning.md](./pg_vacuum_tuning.md) for autovacuum threshold
tuning on high-write tables, so this becomes less likely to recur.

## Prevention

- Regular index maintenance
- Query performance monitoring
- Connection pool sizing
- Database resource limits
- Regular VACUUM/ANALYZE
