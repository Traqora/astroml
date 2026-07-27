# Graph Batch Processing Guide

This document provides guidelines for batch processing large-scale graph operations in AstroML.

## Overview

The `astroml.features.graph.batch_processor` module provides optimized batch processing for graph building and feature computation, designed to handle graphs with 100k+ edges efficiently.

## Key Features

- **Memory-efficient iteration**: Uses `itertools.islice` to process edges without loading entire datasets into memory
- **Configurable batch sizes**: Adjust batch size based on graph size and available memory
- **Progress tracking**: Optional `tqdm` progress bars for long-running operations
- **Sparse matrix optimization**: Uses `scipy.sparse.csr_matrix` for memory-efficient adjacency matrix construction

## Batch Size Recommendations

Batch size selection depends on graph size and available memory:

| Graph Size | Edge Count | Recommended Batch Size | Memory Impact |
|------------|------------|------------------------|----------------|
| Small      | < 10k      | 5,000                  | Low           |
| Medium     | 10k - 100k | 10,000                 | Medium        |
| Large      | 100k - 1M  | 25,000                 | High          |
| X-Large    | > 1M       | 50,000                 | Very High     |

### Choosing Batch Size

**Factors to consider:**
- **Available RAM**: Larger batches require more memory per batch
- **Graph density**: Dense graphs may need smaller batches
- **Feature complexity**: Complex feature computations benefit from larger batches
- **I/O patterns**: If reading from disk, larger batches reduce I/O overhead

**General guidelines:**
- Start with the recommended size for your graph category
- Increase batch size if you have abundant memory and want faster processing
- Decrease batch size if you encounter memory errors
- Monitor memory usage with the built-in profiling tools

## Usage Examples

### Basic Batch Processing

```python
from astroml.features.graph.batch_processor import (
    batch_edges,
    BatchGraphProcessor,
    get_recommended_batch_size,
)

# Determine optimal batch size
n_edges = len(your_edges)
batch_size = get_recommended_batch_size(n_edges)

# Process edges in batches
for batch in batch_edges(your_edges, batch_size=batch_size):
    # Process each batch
    process_batch(batch)
```

### With Progress Tracking

```python
from astroml.features.graph.batch_processor import BatchGraphProcessor

processor = BatchGraphProcessor(
    batch_size=10_000,
    show_progress=True,
    progress_desc="Building graph"
)

def process_batch(batch):
    # Your processing logic
    return result

results = processor.process_edges(your_edges, process_batch)
```

### Sparse Adjacency Matrix

```python
from astroml.features.graph.batch_processor import build_adjacency_matrix_sparse

# Build sparse adjacency matrix (memory efficient for large graphs)
adj_matrix, node_index = build_adjacency_matrix_sparse(
    edges=your_edges,
    directed=True
)

# adj_matrix is a scipy.sparse.csr_matrix
print(f"Matrix shape: {adj_matrix.shape}")
print(f"Non-zero entries: {adj_matrix.nnz}")
```

### Batched Degree Centrality

```python
from astroml.features.graph.batch_processor import compute_degree_centrality_batched

# Compute degree centrality with batching
centrality = compute_degree_centrality_batched(
    edges=your_edges,
    batch_size=10_000,
    show_progress=True,
    weighted=False
)
```

## Memory Profiling

Use the built-in memory profiling to monitor resource usage:

```python
from astroml.features.graph.memory_profile import profile_graph_memory
from astroml.features.graph.batch_processor import compute_degree_centrality_batched

result, profile = profile_graph_memory(
    compute_degree_centrality_batched,
    your_edges,
    batch_size=10_000,
    show_progress=False,
    weighted=False,
    n_edges=len(your_edges)
)

print(f"Peak memory: {profile.traced_peak_mb:.2f} MB")
print(f"Duration: {profile.duration_s:.3f} s")
```

## Performance Benchmarks

Based on testing with various graph sizes:

| Edge Count | Batch Size | Processing Time | Peak Memory |
|------------|------------|-----------------|-------------|
| 10k        | 5,000      | ~0.5s           | ~50 MB      |
| 100k       | 10,000     | ~5s             | ~200 MB     |
| 500k       | 25,000     | ~30s            | ~800 MB     |
| 1M         | 50,000     | ~60s            | ~1.5 GB     |

*Note: Actual performance depends on hardware, graph structure, and feature complexity.*

## Integration with Existing Pipeline

The batch processor integrates seamlessly with existing graph operations:

```python
from astroml.features.graph.batch_processor import BatchGraphProcessor
from astroml.features.structural_importance import compute_structural_importance_metrics

processor = BatchGraphProcessor(batch_size=10_000, show_progress=True)

def compute_metrics_batch(batch):
    return compute_structural_importance_metrics(
        edges=batch,
        include_betweenness=False,  # Skip expensive metrics for large graphs
        include_closeness=False
    )

# Process in batches and combine results
batch_results = processor.process_edges(your_edges, compute_metrics_batch)
```

## Best Practices

1. **Start with recommended batch sizes** for your graph category
2. **Enable progress tracking** for long-running operations (>10k edges)
3. **Use sparse matrices** for adjacency operations on large graphs
4. **Profile memory usage** when processing new datasets
5. **Skip expensive metrics** (betweenness, closeness) for very large graphs
6. **Consider Dask** for graphs >1M edges (see `dask_builder.py`)

## Troubleshooting

### Memory Errors

**Symptom**: `MemoryError` or OOM during processing

**Solutions**:
- Reduce batch size by 50%
- Use sparse matrix operations
- Disable progress tracking (tqdm adds overhead)
- Process fewer features at once

### Slow Performance

**Symptom**: Processing takes longer than expected

**Solutions**:
- Increase batch size (if memory allows)
- Disable progress tracking
- Use sparse matrices for adjacency operations
- Consider parallel processing with Dask

### Inconsistent Results

**Symptom**: Results vary between runs

**Solutions**:
- Ensure deterministic ordering of edges
- Set random seeds for sampling operations
- Use consistent batch sizes across runs

## API Reference

See `astroml.features.graph.batch_processor` for complete API documentation.

Key functions:
- `batch_edges()`: Iterator for batching edges
- `build_adjacency_matrix_sparse()`: Sparse adjacency matrix construction
- `get_recommended_batch_size()`: Batch size recommendation
- `BatchGraphProcessor`: Class for batch processing with progress tracking
- `compute_degree_centrality_batched()`: Batched degree centrality computation
