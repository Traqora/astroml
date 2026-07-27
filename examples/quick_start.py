#!/usr/bin/env python3
"""Quick start example for the benchmarking suite."""

import sys
from pathlib import Path

# Add the parent directory to the path to import astroml
# This allows the example to run from any working directory
script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from astroml.benchmarking import ModelBenchmark, create_config_from_template

# Use script-relative paths for outputs
OUTPUT_DIR = script_dir / "benchmark_results"


def main():
    """Quick start benchmark example."""
    print("AstroML Benchmarking Suite - Quick Start")
    print("=" * 40)
    
    # Create a simple configuration for GCN classification
    config = create_config_from_template(
        name="quick_start_gcn",
        model_name="gcn",
        task_type="classification",
        description="Quick start GCN benchmark"
    )
    
    print(f"Running {config.model.name} benchmark for {config.model.task_type}")
    print(f"Data: {config.data.num_nodes} nodes, {config.data.num_edges} edges")
    print(f"Training: {config.training.epochs} epochs")
    
    # Create and run benchmark
    benchmark = ModelBenchmark(config)
    result = benchmark.run_benchmark()
    
    # Display results
    print("\nResults:")
    print(f"  Training time: {result.train_time:.2f}s")
    print(f"  Peak memory: {result.peak_memory_mb:.1f}MB")
    
    if result.metrics:
        print("  Performance:")
        for metric, value in result.metrics.items():
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
