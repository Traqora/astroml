# Parallel Feature Computation

The Feature Store now supports parallel computation of features to improve performance on multi-core systems.

## Overview

Feature computations can be run in parallel using `concurrent.futures.ThreadPoolExecutor`. This is particularly beneficial for:
- Large datasets with many entities
- Computationally intensive features (e.g., graph centrality measures)
- Batch feature computation for multiple features

## Configuration

### Constructor Parameters

```python
from astroml.features import FeatureStore

store = FeatureStore(
    storage_path="./feature_store",
    max_workers=4,              # Maximum number of parallel workers (default: 4)
    chunk_size=100,            # Entities per chunk (default: 100)
    enable_parallel=True,      # Enable parallel computation (default: True)
)
```

### Configuration via YAML

Create or update `config/feature_store.yaml`:

```yaml
cache:
  max_size_mb: 500
  ttl_seconds: 900
  maxsize: 128

parallel:
  max_workers: 4
  chunk_size: 100
  enable: true
```

### Convenience Function

```python
from astroml.features import create_feature_store

store = create_feature_store(
    storage_path="./feature_store",
    max_workers=8,
    chunk_size=50,
    enable_parallel=True,
)
```

## Parameters

### max_workers

- **Type**: `int`
- **Default**: `4`
- **Description**: Maximum number of parallel workers for feature computation
- **Recommendations**:
  - Set to `1` to disable parallelism
  - Set to `2-4` for I/O-bound operations
  - Set to `4-8` for CPU-bound operations
  - Do not exceed the number of CPU cores

### chunk_size

- **Type**: `int`
- **Default**: `100`
- **Description**: Number of entities to process per chunk in parallel computation
- **Trade-offs**:
  - **Smaller chunks**: More parallelism, lower memory per chunk, higher overhead
  - **Larger chunks**: Less overhead, higher memory per chunk, less parallelism
- **Recommendations**:
  - Use smaller chunks (50-100) for memory-intensive features
  - Use larger chunks (200-500) for simple aggregations
  - Adjust based on available memory

### enable_parallel

- **Type**: `bool`
- **Default**: `True`
- **Description**: Whether to enable parallel feature computation
- **Behavior**:
  - When `True` and `max_workers > 1`: Uses parallel computation for large datasets
  - When `False` or `max_workers == 1`: Always uses sequential computation
  - Parallel computation is only used when data size exceeds `chunk_size`

## Behavior

### Automatic Parallelization

Parallel computation is automatically triggered when:
1. `enable_parallel` is `True`
2. `max_workers > 1`
3. Data size exceeds `chunk_size`

For small datasets (below `chunk_size`), sequential computation is used to avoid overhead.

### Chunking Strategy

Data is split into chunks based on unique entities:
1. Extract unique entity IDs
2. Split entities into chunks of size `chunk_size`
3. Process chunks in parallel using `ThreadPoolExecutor`
4. Combine results from all chunks

### Fallback to Sequential

If parallel computation fails, the system automatically falls back to sequential computation with a warning. This ensures robustness even if:
- Thread pool initialization fails
- Chunk processing encounters unexpected errors
- Resource constraints prevent parallel execution

### Thread Safety

The implementation ensures thread safety:
- Cache operations are protected by `threading.Lock`
- Each chunk processes independent data
- No shared mutable state between workers
- Results are combined after all chunks complete

## Usage Examples

### Basic Parallel Computation

```python
from astroml.features import FeatureStore
import pandas as pd

# Create store with parallel computation enabled
store = FeatureStore(
    storage_path="./feature_store",
    max_workers=4,
    chunk_size=100,
    enable_parallel=True,
)

# Large dataset will be processed in parallel
large_data = pd.DataFrame({
    'entity_id': [...],  # 10000+ entities
    'timestamp': [...],
    'amount': [...],
})

result = store.compute_feature(
    feature_name="daily_transaction_count",
    data=large_data,
    entity_col="entity_id",
    timestamp_col="timestamp",
)
```

### Parallel Feature Fetching

```python
# Fetch multiple features in parallel
features = store.get_features_for_entities(
    feature_names=["feature1", "feature2", "feature3"],
    entity_ids=["entity1", "entity2", "entity3"],
    parallel=True,  # Enable parallel fetching
)
```

### Disable Parallelism

```python
# Disable parallel computation
store = FeatureStore(
    storage_path="./feature_store",
    max_workers=1,
    enable_parallel=False,
)

# Or disable per-call
features = store.get_features_for_entities(
    feature_names=["feature1", "feature2"],
    entity_ids=["entity1", "entity2"],
    parallel=False,
)
```

## Performance Considerations

### When to Use Parallelism

**Use parallelism when:**
- Dataset has > 1000 entities
- Features are computationally expensive (graph metrics, complex aggregations)
- Multiple features need to be computed
- System has multiple CPU cores available

**Avoid parallelism when:**
- Dataset is small (< 100 entities)
- Features are simple and fast
- Memory is constrained
- Running on single-core systems

### Expected Speedup

Speedup depends on:
- Number of workers
- Dataset size
- Feature complexity
- I/O vs CPU bound operations

Typical speedup ranges:
- **2 workers**: 1.5-1.8x
- **4 workers**: 2.5-3.5x
- **8 workers**: 3.0-5.0x

Efficiency typically decreases beyond 8 workers due to overhead.

### Memory Usage

Parallel computation increases memory usage:
- Each chunk holds a copy of the data
- Memory usage ≈ `chunk_size * row_size * max_workers`
- Monitor memory usage and adjust `chunk_size` if needed

## Benchmarking

A benchmark script is provided to measure speedup:

```bash
python benchmark_parallel_features.py
```

This script tests different:
- Data sizes (1K, 5K, 10K, 50K rows)
- Worker configurations (1, 2, 4, 8 workers)
- Feature types

## Troubleshooting

### Parallel Computation Not Triggered

If parallel computation is not being used:
1. Check `enable_parallel` is `True`
2. Check `max_workers > 1`
3. Verify data size exceeds `chunk_size`
4. Check logs for "Parallel computation failed" warnings

### Out of Memory Errors

If you encounter memory errors:
1. Reduce `chunk_size` (e.g., from 100 to 50)
2. Reduce `max_workers` (e.g., from 8 to 4)
3. Process data in smaller batches
4. Monitor memory usage during computation

### No Performance Improvement

If parallel computation doesn't improve performance:
1. Feature may be I/O-bound (limited by disk/network)
2. Overhead of chunking may exceed benefit for small datasets
3. Consider using `ProcessPoolExecutor` for CPU-bound features
4. Profile to identify bottlenecks

## Implementation Details

### ThreadPoolExecutor vs ProcessPoolExecutor

The current implementation uses `ThreadPoolExecutor` because:
- Pandas operations release the GIL for many operations
- Lower memory overhead compared to processes
- Easier to share data between workers

For CPU-bound features that don't release the GIL, consider:
- Using `ProcessPoolExecutor` (requires pickling data)
- Optimizing the feature computation itself
- Using vectorized operations

### Cache Consistency

The cache remains consistent during parallel computation:
- Cache writes are protected by locks
- Each chunk computes independently
- Cache invalidation happens after all chunks complete
- No race conditions in cache operations

## Testing

Unit tests for parallel computation are in `tests/features/test_feature_store.py`:

```bash
pytest tests/features/test_feature_store.py::TestParallelFeatureComputation -v
```

Tests cover:
- Configuration options
- Parallel vs sequential execution
- Fallback behavior
- Thread safety
- Chunking behavior
