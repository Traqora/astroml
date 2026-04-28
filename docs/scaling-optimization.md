# Scaling and Performance Optimization Guide

This guide provides best practices and strategies for scaling AstroML's ingestion and ML pipelines to handle large-scale Stellar network data efficiently.

## 🎯 Overview

As your AstroML deployment grows to handle millions of transactions and accounts, performance optimization becomes critical. This guide covers:

- **Ingestion scaling**: Processing large ledger backfills efficiently
- **Graph pipeline optimization**: Building and querying large graphs
- **ML training at scale**: Distributed training and memory management
- **Database optimization**: PostgreSQL tuning for blockchain data
- **Monitoring and profiling**: Understanding bottlenecks

## 📊 Architecture Considerations

### Typical Scaling Milestones

| Scale | Ledger Records | Accounts | Recommendations |
|-------|---|---|---|
| Development | < 100K | < 10K | Single machine, in-memory graphs |
| Production | 1M - 100M | 100K - 1M | PostgreSQL, batch processing, indexed queries |
| Enterprise | 100M+ | 1M+ | Distributed ingestion, sharded storage, feature store |

---

## 🚀 Ingestion Scaling

### 1. Batch Size Optimization

Larger batch sizes reduce database round-trips but increase memory usage.

```python
# examples/scaling_ingestion_batch.py
from astroml.ingestion.backfill import BackfillConfig

# Development: smaller batches
dev_config = BackfillConfig(
    batch_size=1000,
    start_ledger=1000000,
    end_ledger=1001000
)

# Production: larger batches with memory management
prod_config = BackfillConfig(
    batch_size=10000,  # 10x larger
    start_ledger=1000000,
    end_ledger=2000000,
    checkpoint_interval=50000,  # Save progress every 50K ledgers
    enable_memory_monitoring=True
)

# Enterprise: parallel ingestion
enterprise_config = BackfillConfig(
    batch_size=50000,
    parallel_workers=4,
    start_ledger=1000000,
    end_ledger=50000000,
    checkpoint_interval=100000
)
```

**Configuration Parameters:**

- **batch_size**: Number of transactions to process per database write (default: 1000)
  - Sweet spot: 5,000 - 50,000 depending on transaction complexity
  - Monitor memory: Each transaction ~2KB, so 50K batch ≈ 100MB baseline
  
- **checkpoint_interval**: How often to flush to disk (default: 10,000 ledgers)
  - Prevents long recovery times on failure
  - Recommended: 50K-100K ledgers for multi-hour backfills

- **parallel_workers**: Number of parallel ingestion processes (default: 1)
  - Limited by database connection pool: aim for 4-8 workers
  - Requires PostgreSQL max_connections ≥ 20 + workers

### 2. Database Connection Pooling

Configure connection pooling to avoid connection exhaustion:

```python
# config/database.yaml
database:
  host: localhost
  port: 5432
  dbname: astroml_stellar
  
  # Connection pool settings
  pool_size: 10           # Min connections to keep alive
  max_overflow: 20        # Extra connections when needed
  pool_timeout: 30        # Seconds to wait for connection
  pool_recycle: 3600      # Recycle connections after 1 hour
  
  # For production with parallel workers
  # Recommend: pool_size = 5*workers, max_overflow = 10*workers
```

### 3. Incremental Backfill Strategy

Instead of one massive backfill, use incremental windows:

```bash
#!/bin/bash
# scripts/incremental_backfill.sh

WINDOW=50000  # Ledgers per batch
START=1000000
END=50000000

for ((ledger=$START; ledger<$END; ledger+=WINDOW)); do
  next=$((ledger + WINDOW))
  echo "Backfilling ledgers $ledger to $next..."
  
  python -m astroml.ingestion.backfill \
    --start-ledger $ledger \
    --end-ledger $next \
    --batch-size 10000 \
    --checkpoint-interval $WINDOW
    
  # Allow 10 seconds for database recovery
  sleep 10
done
```

**Benefits:**
- Checkpoint recovery is faster (< 10 minutes per window)
- Database load is more predictable
- Easier to monitor and debug failures

### 4. Parallel Ingestion with Worker Processes

For ledger ranges spanning months, use multi-worker ingestion:

```python
# examples/parallel_ingestion.py
from astroml.ingestion.backfill import ParallelBackfill
from multiprocessing import cpu_count

# Configure for 4 workers
config = {
    'workers': 4,
    'batch_size': 20000,
    'checkpoint_interval': 100000,
    'start_ledger': 1000000,
    'end_ledger': 10000000,  # 9M ledgers
}

backfill = ParallelBackfill(**config)
results = backfill.run()

print(f"Ingested {results['total_transactions']} transactions")
print(f"Total time: {results['elapsed_time']/60:.1f} minutes")
print(f"Throughput: {results['throughput_tx_per_sec']:.0f} tx/sec")
```

**Worker Allocation:**
- **CPU-bound: 4 workers** (normalization, deduplication)
- **I/O-bound: 8+ workers** (database writes, disk I/O)

---

## 🕸 Graph Pipeline Optimization

### 1. Windowed Graph Construction

For large-scale graphs, construct rolling time windows instead of full snapshots:

```python
# config/configs/sampling/large_scale.yaml
graph:
  window_size: 30d
  overlap: 5d              # For temporal continuity
  
  # For 1M+ accounts
  sampling:
    strategy: degree_weighted
    sample_ratio: 0.7      # Keep 70% of edges
    min_degree: 2
  
  # Pre-filtering
  filters:
    - min_transaction_value: 0.01 XLM
    - exclude_inactive_accounts: 90d
```

### 2. Graph Caching and Materialization

Pre-compute and cache graphs for reuse:

```python
# examples/cached_graph_construction.py
from astroml.graph.cache import GraphCache
import pickle

cache = GraphCache(
    cache_dir='./cached_graphs',
    ttl_hours=24
)

# Check cache first
graph = cache.get('main_graph_30d')

if graph is None:
    # Build if not cached
    from astroml.graph.build_snapshot import build_snapshot
    
    graph = build_snapshot(
        window='30d',
        min_tx_amount=0.01,
        exclude_inactive=True
    )
    
    # Cache for reuse
    cache.set('main_graph_30d', graph)

# Now use graph for multiple downstream tasks
features = extract_features(graph)
```

### 3. Lazy Graph Loading

For production systems, load graph data on-demand:

```python
from astroml.graph.lazy import LazyGraph

# Load metadata only, defer edge loading
lazy_graph = LazyGraph.from_database(
    config_path='config/database.yaml',
    window='30d',
    lazy=True
)

# Only fetch edges when needed
neighbors = lazy_graph.neighbors(account_id)
```

---

## 🤖 ML Training at Scale

### 1. Distributed Training Setup

For multi-GPU or multi-machine training:

```python
# config/configs/training/distributed.yaml
training:
  backend: ddp              # Distributed Data Parallel
  num_gpus: 4
  num_nodes: 2              # 8 GPUs total
  
  # Batch size per GPU
  batch_size: 256
  # Effective batch size = 256 * 4 GPUs * 2 nodes = 2048
  
  # Learning rate scaling (linear scaling rule)
  lr: 0.001
  lr_scale_factor: 2        # Multiply by num_gpus
  
  # Gradient accumulation for larger effective batches
  gradient_accumulation_steps: 4
```

### 2. Memory Optimization for Large Graphs

```python
# examples/memory_efficient_training.py
import torch
from astroml.training.train_gcn import GCNTrainer

trainer = GCNTrainer(
    config_path='config/configs/training/distributed.yaml'
)

# Enable gradient checkpointing (saves memory, slower training)
trainer.model.enable_gradient_checkpointing = True

# Use mixed precision (FP16 + FP32)
trainer.use_mixed_precision = True

# Reduce model size for very large graphs
trainer.model.hidden_channels = 64  # Instead of 128
trainer.model.num_layers = 3        # Instead of 4

# Smaller batch size with more accumulation
trainer.batch_size = 128
trainer.gradient_accumulation_steps = 8
```

### 3. Feature Store Integration

Avoid recomputing features for each model:

```python
# config/configs/training/feature_store.yaml
feature_store:
  enabled: true
  backend: postgresql     # or redis for caching
  ttl_hours: 24
  
  # Cache intermediate features
  cache_embeddings: true
  cache_computed_features: true
  
  # Materialized feature views
  materialized_views:
    - user_transaction_count_30d
    - user_avg_transaction_value_30d
    - account_clustering_coefficient
```

---

## 💾 Database Optimization

### 1. PostgreSQL Configuration

For large-scale Stellar data (100M+ transactions):

```sql
-- postgresql.conf
-- Allocate 25-50% of system RAM to PostgreSQL

shared_buffers = 32GB              # 25% of 128GB RAM
effective_cache_size = 96GB        # 75% of RAM
maintenance_work_mem = 4GB
work_mem = 256MB

# Query performance
random_page_cost = 1.1             # SSD tuning
effective_io_concurrency = 200

# Connection management
max_connections = 200
max_worker_processes = 8
max_parallel_workers = 8
max_parallel_workers_per_gather = 4

# WAL configuration
wal_buffers = 16MB
checkpoint_timeout = 15min
max_wal_size = 4GB
```

### 2. Index Strategy

Create indexes strategically to avoid bloat:

```sql
-- Core transaction indexes
CREATE INDEX idx_transactions_timestamp 
  ON transactions(timestamp DESC) WHERE amount > 0.01;

CREATE INDEX idx_transactions_sender_receiver 
  ON transactions(sender_account_id, receiver_account_id, timestamp);

-- Partial index for active accounts
CREATE INDEX idx_accounts_active 
  ON accounts(account_id, last_activity)
  WHERE is_active = true;

-- For graph queries
CREATE INDEX idx_transactions_graph 
  ON transactions(sender_account_id, receiver_account_id)
  INCLUDE (amount, timestamp);
```

### 3. Query Optimization

Use materialized views for common aggregations:

```sql
-- Pre-compute frequent queries
CREATE MATERIALIZED VIEW account_stats_30d AS
SELECT 
  account_id,
  COUNT(*) as tx_count,
  SUM(amount) as total_volume,
  AVG(amount) as avg_amount,
  MAX(timestamp) as last_activity
FROM transactions
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY account_id;

-- Refresh on schedule
-- REFRESH MATERIALIZED VIEW CONCURRENTLY account_stats_30d;
```

---

## 📈 Monitoring and Profiling

### 1. Ingestion Performance Monitoring

```python
# examples/monitor_ingestion.py
from astroml.ingestion.backfill import BackfillMonitor
import logging

logging.basicConfig(level=logging.INFO)

monitor = BackfillMonitor(
    log_interval=5000,      # Log every 5K transactions
    track_memory=True,
    track_database=True
)

config = {
    'batch_size': 10000,
    'start_ledger': 1000000,
    'end_ledger': 2000000,
    'monitor': monitor
}

# Monitor will log:
# - Throughput (tx/sec)
# - Memory usage (MB)
# - Database queue depth
# - ETA to completion
```

### 2. Training Performance Profiling

```python
# examples/profile_training.py
from torch.profiler import profile, record_function
from astroml.training.train_gcn import GCNTrainer

trainer = GCNTrainer(config_path='config/configs/training/distributed.yaml')

with profile(
    activities=['cpu', 'cuda'],
    record_shapes=True
) as prof:
    trainer.train_epoch()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
```

### 3. Resource Monitoring Dashboard

Set up continuous monitoring:

```yaml
# monitoring/prometheus/astroml.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'astroml_ingestion'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/ingestion'

  - job_name: 'astroml_training'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics/training'
```

---

## 🔧 Troubleshooting Performance

### Slow Ingestion

**Symptom:** Throughput < 100 tx/sec

```bash
# Check database
VACUUM ANALYZE;               # Optimize statistics
SELECT pg_size_pretty(pg_database_size('astroml_stellar'));

# Check connection pool
SELECT count(*) FROM pg_stat_activity WHERE datname='astroml_stellar';

# Increase batch size gradually
python -m astroml.ingestion.backfill \
  --start-ledger 1000000 \
  --end-ledger 1100000 \
  --batch-size 50000         # From 10000
```

### Out of Memory During Training

```python
# Reduce model size
model.hidden_channels = 32    # From 64
model.num_layers = 2          # From 4

# Enable gradient checkpointing
model.enable_gradient_checkpointing = True

# Use smaller batches with accumulation
batch_size = 32
accumulation_steps = 8
```

### High Memory Graph Construction

```python
# sample the graph first
from astroml.graph.sampling import RandomWalkSampler

sampler = RandomWalkSampler(
    num_nodes=1000000,
    sample_size=0.5              # Keep 50% of nodes
)

subgraph = build_snapshot(
    window='30d',
    sampler=sampler
)
```

---

## 📋 Performance Checklist

### Pre-Deployment

- [ ] Database connection pool configured (pool_size ≥ 10)
- [ ] PostgreSQL parameters tuned (shared_buffers, effective_cache_size)
- [ ] Indexes created on transaction and account tables
- [ ] Incremental backfill strategy tested with target ledger range
- [ ] Batch size optimized for your data volume
- [ ] Monitoring and logging configured

### During Production

- [ ] Ingestion throughput tracked (target: 500+ tx/sec)
- [ ] Database query times monitored (p99 < 100ms)
- [ ] Memory usage tracked (should not spike > 2x baseline)
- [ ] Graph construction time profiled (target: < 10 min for 30d window)
- [ ] Training convergence validated with profiling

### Scaling Up

- [ ] Parallel workers tested (start with 2, increase to 4-8)
- [ ] Distributed training environment prepared (multi-GPU/multi-node)
- [ ] Feature store materialization automated
- [ ] Alerting configured for ingestion failures
- [ ] Capacity planning done for next 3-6 months

---

## 📚 Additional Resources

- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [PyTorch Distributed Training](https://pytorch.org/docs/stable/distributed.html)
- [Graph Sampling Techniques](https://arxiv.org/abs/1809.02779)
- [Stellar Network Documentation](https://developers.stellar.org/learn)
- [AstroML Benchmarking Suite](benchmarking.md)

---

## 💡 Best Practices Summary

1. **Start small, measure, then scale** - Profile on 1M transactions before scaling to 1B
2. **Batch processing wins** - Use incremental windows, not monolithic backfills
3. **Database is your bottleneck** - Invest in PostgreSQL tuning and indexing
4. **Monitor everything** - Throughput, memory, query times, error rates
5. **Automate recovery** - Implement checkpoint/resume for long-running pipelines
6. **Test parallel scaling linearly** - Not all workloads benefit equally from parallelization

---

**Last Updated:** 2026-04-27  
**Version:** 1.0
