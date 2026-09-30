# Database Query Profiling Guide

This document provides guidelines for using the database query profiling features in AstroML to identify and optimize slow queries.

## Overview

The `astroml.db.query_profiler` module provides SQLAlchemy query profiling capabilities including:
- Query logging with configurable verbosity
- Slow query detection and EXPLAIN ANALYZE
- Custom logging for performance monitoring
- Integration with existing session management

## Enabling Query Profiling

### Debug Mode

Query profiling is automatically enabled when `ASTROML_DEBUG` is set:

```bash
export ASTROML_DEBUG=true
export ASTROML_SLOW_QUERY_THRESHOLD_MS=100  # Optional custom threshold
```

### Programmatic Configuration

```python
from astroml.db.query_profiler import configure_query_logging

configure_query_logging(
    log_level="DEBUG",
    enable_profiling=True,
    slow_query_threshold_ms=100
)
```

### Context Manager

For temporary profiling of specific code blocks:

```python
from astroml.db.query_profiler import profile_query_context

with profile_query_context() as profiler:
    # Run queries
    session.query(Account).all()
    
# Get statistics
stats = profiler.get_statistics()
print(f"Total queries: {stats['total_queries']}")
print(f"Slow queries: {stats['slow_queries']}")
print(f"Average duration: {stats['avg_duration_ms']:.2f}ms")
```

## Query Optimization Patterns

### 1. Use Indexes Effectively

**Problem**: Sequential scans on large tables

**Solution**: Add appropriate indexes on frequently queried columns

```python
# Before: Full table scan
session.query(Operation).filter(Operation.amount > 1000).all()

# After: Index scan (add index on amount)
# CREATE INDEX idx_operation_amount ON operations(amount);
```

### 2. Avoid N+1 Queries

**Problem**: Executing a query for each related record

**Solution**: Use eager loading with `joinedload` or `selectinload`

```python
# Before: N+1 queries
operations = session.query(Operation).all()
for op in operations:
    print(op.transaction.ledger_sequence)  # Separate query per operation

# After: Single query with eager loading
from sqlalchemy.orm import joinedload
operations = session.query(Operation).options(
    joinedload(Operation.transaction)
).all()
```

### 3. Use Specific Column Selection

**Problem**: Selecting all columns when only a few are needed

**Solution**: Select only required columns

```python
# Before: SELECT *
session.query(Account).all()

# After: SELECT id, account_id
session.query(Account.id, Account.account_id).all()
```

### 4. Optimize Joins

**Problem**: Inefficient join order or missing foreign key indexes

**Solution**: Ensure foreign keys are indexed and use appropriate join strategies

```python
# Before: Potential sequential scan
session.query(Operation).join(Transaction).filter(
    Transaction.ledger_sequence > 1000
).all()

# After: Index on foreign key and join column
# CREATE INDEX idx_operation_transaction_id ON operations(transaction_id);
# CREATE INDEX idx_transaction_ledger ON transactions(ledger_sequence);
```

### 5. Use Pagination

**Problem**: Loading entire result sets into memory

**Solution**: Use pagination with `limit` and `offset`

```python
# Before: Loads all records
session.query(Operation).all()

# After: Loads in chunks
page_size = 1000
for offset in range(0, total_count, page_size):
    session.query(Operation).limit(page_size).offset(offset).all()
```

### 6. Batch Operations

**Problem**: Individual inserts/updates for multiple records

**Solution**: Use bulk operations

```python
# Before: Individual inserts
for data in records:
    session.add(Operation(**data))
session.commit()

# After: Bulk insert
session.bulk_insert_mappings(Operation, records)
session.commit()
```

### 7. Use EXISTS Instead of IN

**Problem**: IN clause with large subqueries

**Solution**: Use EXISTS for better performance

```python
# Before: IN clause (can be slow with large lists)
session.query(Account).filter(
    Account.id.in_(large_id_list)
).all()

# After: EXISTS (better for large datasets)
from sqlalchemy import exists
session.query(Account).filter(
    exists().where(Account.id == Operation.account_id)
).all()
```

### 8. Avoid Unnecessary Sorting

**Problem**: Sorting when not needed

**Solution**: Remove ORDER BY clauses when order doesn't matter

```python
# Before: Unnecessary sort
session.query(Operation).order_by(Operation.id).all()

# After: No sort (faster)
session.query(Operation).all()
```

## Interpreting EXPLAIN ANALYZE Output

When a slow query is detected, the profiler automatically runs EXPLAIN ANALYZE and logs the output. Here's how to interpret it:

### Key Metrics

- **Execution Time**: Total time to execute the query
- **Planning Time**: Time spent by the query planner
- **Actual Time**: Time for each operation node
- **Rows**: Number of rows processed at each node
- **Loops**: Number of times the operation was repeated

### Common Performance Issues

**Sequential Scan**:
```
Seq Scan on operations (cost=0.00..1000.00 rows=10000 width=100)
```
- **Issue**: Full table scan without index
- **Fix**: Add index on filtered columns

**Nested Loop**:
```
Nested Loop (cost=0.00..5000.00 rows=1000 width=200)
```
- **Issue**: Inefficient join for large datasets
- **Fix**: Use Hash Join or Merge Join with appropriate indexes

**Hash Join**:
```
Hash Join (cost=100.00..500.00 rows=1000 width=200)
```
- **Issue**: Hash table creation overhead
- **Fix**: Ensure work_mem is sufficient or reduce dataset size

## CI/CD Integration

### Slow Query Threshold Check

Add to your CI pipeline to fail builds with slow queries:

```python
from astroml.db.query_profiler import check_slow_query_threshold_ci

# In your test suite
def test_query_performance():
    # Run your queries
    run_test_queries()
    
    # Check that no query exceeds threshold
    assert check_slow_query_threshold_ci(threshold_ms=100), \
        "Queries exceeded performance threshold"
```

### Environment Variables

Configure profiling in CI:

```yaml
# .github/workflows/tests.yml
env:
  ASTROML_DEBUG: "true"
  ASTROML_SLOW_QUERY_THRESHOLD_MS: "100"
```

## Performance Testing

### Benchmarking Queries

```python
import time
from astroml.db.query_profiler import profile_query_context

def benchmark_query(query_func, iterations=10):
    """Benchmark a query function."""
    with profile_query_context() as profiler:
        for _ in range(iterations):
            query_func()
    
    stats = profiler.get_statistics()
    avg_time = stats['avg_duration_ms']
    
    print(f"Average query time: {avg_time:.2f}ms")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Slow queries: {stats['slow_queries']}")
    
    return avg_time
```

### Comparing Query Plans

```python
from sqlalchemy import text

def compare_query_plans(session, query1, query2):
    """Compare execution plans of two queries."""
    plan1 = session.execute(text(f"EXPLAIN {query1}")).fetchall()
    plan2 = session.execute(text(f"EXPLAIN {query2}")).fetchall()
    
    print("Query 1 Plan:")
    for row in plan1:
        print(row[0])
    
    print("\nQuery 2 Plan:")
    for row in plan2:
        print(row[0])
```

## Monitoring and Alerting

### Slow Query Logger

The profiler uses a dedicated logger for slow queries:

```python
import logging

# Configure slow query logger
slow_query_logger = logging.getLogger("astroml.db.slow_queries")
slow_query_logger.setLevel(logging.WARNING)

# Add handler for alerts
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
slow_query_logger.addHandler(handler)
```

### Statistics Collection

Collect profiling statistics for monitoring:

```python
from astroml.db.query_profiler import get_query_profiler

profiler = get_query_profiler()
if profiler:
    stats = profiler.get_statistics()
    
    # Send to monitoring system
    send_to_monitoring({
        "total_queries": stats["total_queries"],
        "slow_queries": stats["slow_queries"],
        "avg_duration_ms": stats["avg_duration_ms"],
        "slow_query_rate": stats["slow_query_rate"]
    })
```

## Best Practices

1. **Profile in Development**: Enable profiling during development, not in production
2. **Set Appropriate Thresholds**: Adjust slow query threshold based on your use case
3. **Review EXPLAIN Plans**: Regularly review query plans for optimization opportunities
4. **Index Strategically**: Add indexes based on actual query patterns
5. **Monitor Over Time**: Track query performance trends to catch regressions
6. **Use Connection Pooling**: Configure appropriate pool sizes for your workload
7. **Batch Operations**: Use bulk operations for large data modifications
8. **Cache When Possible**: Use application-level caching for frequently accessed data

## Troubleshooting

### Profiler Not Enabled

**Symptom**: Queries not being profiled

**Solutions**:
- Ensure `ASTROML_DEBUG=true` is set
- Check that `configure_query_logging` is called with `enable_profiling=True`
- Verify the profiler is attached to the engine

### EXPLAIN ANALYZE Fails

**Symptom**: EXPLAIN ANALYZE errors in logs

**Solutions**:
- Check database user has EXPLAIN permissions
- Verify the query is valid SQL
- Some databases may not support EXPLAIN ANALYZE (fallback to EXPLAIN)

### High Memory Usage

**Symptom**: Profiling increases memory usage

**Solutions**:
- Disable profiling in production
- Clear profiles regularly with `profiler.clear_profiles()`
- Reduce log_all_queries setting to False

## API Reference

See `astroml.db.query_profiler` for complete API documentation.

Key classes and functions:
- `QueryProfiler`: Main profiler class
- `QueryProfile`: Dataclass for query profile information
- `configure_query_logging()`: Configure query logging and profiling
- `profile_query_context()`: Context manager for temporary profiling
- `check_slow_query_threshold_ci()`: CI/CD slow query checking
