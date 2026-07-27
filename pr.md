# Pull Request: Fix Issues #565, #549, #563, #548

## Summary
This PR addresses four medium-severity issues related to dependency management, graph building optimization, and feature store performance.

## Issues Closed

Closes #565
Closes #549
Closes #563
Closes #548

## Issues Fixed

### ✅ #565: Separate Optional Dependencies Clearly
**Component:** `pyproject.toml`, `requirements.txt`

**Changes:**
- Updated `pyproject.toml` with clearly separated optional dependencies:
  - `gpu`: PyTorch with CUDA support and torch-geometric
  - `viz`: Matplotlib and seaborn for visualization
  - `dev`: Development tools (pytest, black, mypy, ruff, etc.)
  - `notebook`: Jupyter notebook support
  - `dask`: Dask for distributed computing
- Installation examples:
  ```bash
  pip install astroml              # minimal install
  pip install astroml[gpu,dev]     # with GPU and dev tools
  pip install astroml[viz,notebook] # with visualization and notebooks
  ```

### ✅ #549: Optimize Graph Building with Dask
**Component:** `astroml/features/graph/`

**Changes:**
- Created `astroml/features/graph/dask_builder.py` with:
  - `DaskGraphBuilder` class for parallel graph construction
  - Automatic backend selection based on graph size (threshold: 1M+ edges)
  - `build_graph_auto()` function with configurable backend ("networkx", "dask", or "auto")
  - Parallel node feature computation
  - Graceful fallback to NetworkX when Dask unavailable
- Added dask as optional dependency in `pyproject.toml`
- Benchmark-ready: Designed for deterministic results
- Configuration option available via backend parameter

### ✅ #563: Implement Strict Dependency Versioning
**Component:** `requirements.txt`

**Changes:**
- Created `requirements.in` as source of truth for dependencies
- Created `requirements-lock.txt` with exact pinned versions
- Update process documented:
  1. Modify `requirements.in` with new version ranges
  2. Run `pip-compile requirements.in -o requirements-lock.txt`
  3. Test in isolation before merging
- CI should install from lock file: `pip install -r requirements-lock.txt`
- Added compatible release versioning strategy using `~=` where appropriate

### ✅ #548: Add Lazy Loading for Large Feature Store
**Component:** `astroml/features/feature_store.py`

**Changes:**
- Implemented lazy loading in `FeatureStore` class:
  - Separate metadata cache (lightweight) and value cache (on-demand)
  - LRU (Least Recently Used) eviction policy
  - Configurable memory limit (default: 500 MB)
  - TTL-based cache expiration (default: 3600 seconds)
- Added `get_cache_stats()` method for monitoring cache utilization
- Cache parameters configurable via constructor:
  ```python
  FeatureStore(
      storage_path="./features",
      max_cache_size_mb=500,
      cache_ttl_seconds=3600
  )
  ```
- Memory-efficient: Only recently accessed features kept in memory
- Benchmarked for 100+ features with automatic eviction

## Type Hints & Testing
- ✅ All new code has complete type hints
- ✅ Follows existing codebase patterns
- ✅ Graceful error handling with appropriate fallbacks

## Documentation Updates
- Optional dependencies documented in `pyproject.toml`
- Lazy loading behavior documented in docstrings
- Dask usage threshold and configuration documented
- Version update process documented

## Acceptance Criteria Met
- ✅ Code changes implemented per procedure steps
- ✅ All new code has complete type hints
- ✅ Graceful error handling for optional dependencies
- ✅ Backward compatibility maintained
- ✅ Documentation updated in relevant docstrings

## Installation & Usage

### Optional Dependencies
```bash
# Minimal install
pip install astroml

# With GPU support
pip install astroml[gpu]

# With development tools
pip install astroml[dev]

# With Dask for large graphs
pip install astroml[dask]

# Multiple extras
pip install astroml[gpu,viz,dev,dask]
```

### Lazy Feature Store
```python
from astroml.features import FeatureStore

# Initialize with custom cache settings
store = FeatureStore(
    storage_path="./features",
    max_cache_size_mb=1000,  # 1 GB cache
    cache_ttl_seconds=7200    # 2 hour TTL
)

# Check cache statistics
stats = store.get_cache_stats()
print(f"Cache utilization: {stats['cache_utilization_pct']:.2f}%")
```

### Dask Graph Building
```python
from astroml.features.graph.dask_builder import build_graph_auto

# Automatic backend selection
G = build_graph_auto(
    edges_df,
    backend="auto",          # auto-select based on size
    dask_threshold=1_000_000 # use Dask for 1M+ edges
)

# Force specific backend
G = build_graph_auto(edges_df, backend="dask")
```

## Breaking Changes
None. All changes are backward compatible.

## Notes
- Dask is an optional dependency - code gracefully falls back to NetworkX
- Feature store lazy loading is transparent to existing code
- Lock file approach enables reproducible builds
- All optional dependencies clearly separated for minimal installation footprint
