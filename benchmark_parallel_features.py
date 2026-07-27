"""Benchmark parallel vs sequential feature computation."""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from astroml.features import FeatureStore


def create_sample_data(n_rows: int = 10000, n_entities: int = 1000) -> pd.DataFrame:
    """Create sample transaction data for benchmarking."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'entity_id': np.random.randint(1, n_entities + 1, n_rows),
        'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='min'),
        'amount': np.random.uniform(1, 1000, n_rows),
        'asset': np.random.choice(['BTC', 'ETH', 'USDT', 'SOL'], n_rows),
    })
    
    return data


def benchmark_feature_computation(
    store: FeatureStore,
    feature_name: str,
    data: pd.DataFrame,
    entity_col: str = 'entity_id',
    timestamp_col: str = 'timestamp',
    n_runs: int = 3,
) -> Dict[str, float]:
    """Benchmark feature computation.
    
    Args:
        store: FeatureStore instance
        feature_name: Name of feature to compute
        data: Input data
        entity_col: Entity identifier column
        timestamp_col: Timestamp column
        n_runs: Number of benchmark runs
        
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    for run in range(n_runs):
        start_time = time.time()
        try:
            result = store.compute_feature(
                feature_name=feature_name,
                data=data,
                entity_col=entity_col,
                timestamp_col=timestamp_col,
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Run {run + 1}: {elapsed:.3f}s ({len(result)} rows)")
        except Exception as e:
            print(f"  Run {run + 1}: FAILED - {e}")
            continue
    
    if not times:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def run_benchmarks():
    """Run comprehensive benchmarks comparing parallel vs sequential execution."""
    print("=" * 70)
    print("Parallel Feature Computation Benchmark")
    print("=" * 70)
    
    # Test different data sizes
    data_sizes = [1000, 5000, 10000, 50000]
    n_entities_list = [100, 500, 1000, 5000]
    
    # Test different worker configurations
    worker_configs = [
        {"max_workers": 1, "enable_parallel": False, "name": "Sequential"},
        {"max_workers": 2, "enable_parallel": True, "name": "Parallel (2 workers)"},
        {"max_workers": 4, "enable_parallel": True, "name": "Parallel (4 workers)"},
        {"max_workers": 8, "enable_parallel": True, "name": "Parallel (8 workers)"},
    ]
    
    results = []
    
    for data_size, n_entities in zip(data_sizes, n_entities_list):
        print(f"\n{'=' * 70}")
        print(f"Data Size: {data_size} rows, {n_entities} entities")
        print(f"{'=' * 70}")
        
        data = create_sample_data(n_rows=data_size, n_entities=n_entities)
        
        config_results = {}
        
        for config in worker_configs:
            print(f"\n{config['name']}:")
            
            # Create feature store with specific configuration
            store = FeatureStore(
                storage_path=f"./benchmark_store_{config['name'].replace(' ', '_').replace('(', '').replace(')', '')}",
                max_workers=config["max_workers"],
                enable_parallel=config["enable_parallel"],
                chunk_size=100,
            )
            
            # Benchmark feature computation
            try:
                stats = benchmark_feature_computation(
                    store=store,
                    feature_name="daily_transaction_count",
                    data=data,
                    n_runs=3,
                )
                config_results[config["name"]] = stats
                print(f"  Mean: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
                config_results[config["name"]] = {"mean": 0.0, "std": 0.0}
        
        results.append({
            "data_size": data_size,
            "n_entities": n_entities,
            "results": config_results,
        })
    
    # Calculate and display speedup
    print(f"\n{'=' * 70}")
    print("Speedup Analysis")
    print(f"{'=' * 70}")
    
    for result in results:
        data_size = result["data_size"]
        n_entities = result["n_entities"]
        config_results = result["results"]
        
        sequential_time = config_results.get("Sequential", {}).get("mean", 0.0)
        
        print(f"\nData Size: {data_size} rows, {n_entities} entities")
        print(f"Sequential baseline: {sequential_time:.3f}s")
        
        for config_name in ["Parallel (2 workers)", "Parallel (4 workers)", "Parallel (8 workers)"]:
            if config_name in config_results:
                parallel_time = config_results[config_name]["mean"]
                if sequential_time > 0 and parallel_time > 0:
                    speedup = sequential_time / parallel_time
                    efficiency = speedup / int(config_name.split("(")[1].split()[0]) * 100
                    print(f"  {config_name}: {parallel_time:.3f}s ({speedup:.2f}x speedup, {efficiency:.1f}% efficiency)")
    
    # Summary statistics
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    
    for result in results:
        data_size = result["data_size"]
        config_results = result["results"]
        
        sequential_time = config_results.get("Sequential", {}).get("mean", 0.0)
        parallel_4_time = config_results.get("Parallel (4 workers)", {}).get("mean", 0.0)
        
        if sequential_time > 0 and parallel_4_time > 0:
            speedup = sequential_time / parallel_4_time
            print(f"Data size {data_size}: {speedup:.2f}x speedup with 4 workers")


if __name__ == "__main__":
    run_benchmarks()
