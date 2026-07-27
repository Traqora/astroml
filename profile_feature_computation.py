"""Profile feature computation to identify bottlenecks."""

import cProfile
import pstats
import io
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from astroml.features import FeatureStore


def create_sample_data(n_rows: int = 10000) -> pd.DataFrame:
    """Create sample transaction data for profiling."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'entity_id': np.random.randint(1, 1000, n_rows),
        'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='min'),
        'amount': np.random.uniform(1, 1000, n_rows),
        'asset': np.random.choice(['BTC', 'ETH', 'USDT', 'SOL'], n_rows),
    })
    
    return data


def profile_sequential_computation():
    """Profile sequential feature computation."""
    print("Profiling sequential feature computation...")
    
    # Create feature store
    store = FeatureStore(storage_path="./profile_feature_store")
    
    # Create sample data
    data = create_sample_data(n_rows=10000)
    
    # Profile computation
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Compute multiple features sequentially
        features = ['daily_transaction_count', 'transaction_burstiness']
        for feature_name in features:
            result = store.compute_feature(
                feature_name=feature_name,
                data=data,
                entity_col='entity_id',
                timestamp_col='timestamp',
            )
            print(f"Computed {feature_name}: {len(result)} rows")
    except Exception as e:
        print(f"Error during computation: {e}")
    finally:
        profiler.disable()
    
    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    
    return profiler


if __name__ == "__main__":
    profile_sequential_computation()
