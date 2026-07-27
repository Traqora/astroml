# Issue #202: Fix - Examples with Hardcoded Paths

## Problem
Example scripts in the `examples/` directory used hardcoded or brittle paths that made them fail or require execution from specific working directories. This prevented users from running examples from anywhere in the filesystem or from different project locations.

## Solution
Updated all Python example scripts to use relative path resolution based on the script location, following the pattern already used by some examples in the repository.

## Changes Made

### Updated Files
1. **examples/feature_store_example.py**
2. **examples/deep_svdd_example.py**
3. **examples/graph_validation_demo.py**

### Implementation Pattern
All Python examples now use the following pattern at the top of the file:

```python
import sys
from pathlib import Path

# Add the parent directory to the path to import astroml
# This allows the example to run from any working directory
script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
```

### What This Achieves
- ✅ Examples can be run from any working directory
- ✅ Examples can be run from any location in the filesystem  
- ✅ Proper module imports regardless of execution context
- ✅ Follows existing patterns in `benchmark_example.py`, `calibration_example.py`, and `quick_start.py`

### Files Already Compliant
- `examples/benchmark_example.py` - already had correct setup
- `examples/calibration_example.py` - already had correct setup
- `examples/quick_start.py` - already had correct setup
- Jupyter notebooks - use relative path setup appropriate for notebooks

## Testing
- All Python files pass syntax validation
- Import tests confirm astroml can be imported from any directory
- Examples follow consistent patterns across the repository

## User Impact
Users can now run examples like:
```bash
# From any directory
python /path/to/examples/feature_store_example.py

# From project root
python examples/feature_store_example.py

# From examples directory
python feature_store_example.py

# From completely different directory
cd /tmp && python /workspaces/astroml/examples/feature_store_example.py
```

All of these will work correctly without path-related errors.
