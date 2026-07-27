#!/usr/bin/env python3
"""Example script demonstrating the benchmarking suite."""

import sys
from pathlib import Path

# Add the parent directory to the path to import astroml
# This allows the example to run from any working directory
script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from astroml.benchmarking import (
    ModelBenchmark,
    ConfigManager,
    create_config_from_template,
    Timer,
    MemoryMonitor,
    set_random_seed,
    get_device_info
)

# Use script-relative paths for outputs
OUTPUT_DIR = script_dir / "benchmark_results"
EXAMPLE_CONFIGS_DIR = repo_root / "example_configs"


def run_basic_benchmark():
    """Run a basic benchmark using default configuration."""
    print("=== Running Basic Benchmark ===")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Display device information
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Create a simple configuration
    config = create_config_from_template(
        name="basic_gcn_test",
        model_name="gcn",
        task_type="classification",
        description="Basic GCN benchmark test",
        data_params={
            "num_nodes": 500,
            "num_features": 16,
            "num_edges": 2000,
            "num_classes": 2
        },
        training_params={
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01
        }
    )
    
    print(f"Configuration: {config.name}")
    print(f"Model: {config.model.name}")
    print(f"Task: {config.model.task_type}")
    
    # Create benchmark instance
    benchmark = ModelBenchmark(config)
    
    # Run with timing and memory monitoring
    with Timer("Total benchmark time") as timer:
        with MemoryMonitor("Memory usage"):
            results = benchmark.run_benchmark()
    
    # Print results
    print("\n=== Results ===")
    print(f"Training time: {results.train_time:.2f}s")
    print(f"Peak memory: {results.peak_memory_mb:.1f}MB")
    
    if results.metrics:
        print("\nMetrics:")
        for metric, value in results.metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nResults saved to: benchmark_results")
    print(f"Model saved to: benchmark_results")
    return results


def run_multiple_models():
    """Run benchmarks on multiple models."""
    print("\n=== Running Multiple Model Benchmarks ===")
    
    models = [
        ("gcn", "classification"),
        ("link_predictor", "link_prediction"),
        ("sage_encoder", "classification"),
        ("deep_svdd", "anomaly_detection")
    ]
    
    results = {}
    
    for model_name, task_type in models:
        print(f"\n--- Benchmarking {model_name} for {task_type} ---")
        
        config = create_config_from_template(
            name=f"{model_name}_{task_type}_test",
            model_name=model_name,
            task_type=task_type,
            description=f"{model_name} benchmark for {task_type}",
            data_params={
                "num_nodes": 300,
                "num_features": 16,
                "num_edges": 1000
            },
            training_params={
                "epochs": 30,
                "batch_size": 16
            }
        )
        
        # Adjust data for specific tasks
        if task_type == "classification":
            config.data.num_classes = 2
        elif task_type == "anomaly_detection":
            config.data.num_classes = 1
        
        benchmark = ModelBenchmark(config)
        
        try:
            with Timer(f"{model_name} benchmark"):
                result = benchmark.run_benchmark()
            
            results[model_name] = result
            print(f"✓ {model_name} completed successfully")
            
            # Print key metrics
            if result.metrics:
                key_metrics = list(result.metrics.keys())[:3]  # First 3 metrics
                print(f"  Key metrics: {', '.join(f'{k}: {v:.3f}' for k, v in result.metrics.items() if k in key_metrics)}")
            
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
    
    # Summary
    print("\n=== Summary ===")
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Training time: {result.train_time:.2f}s")
        print(f"  Peak memory: {result.peak_memory_mb:.1f}MB")
        if result.metrics:
            accuracy = result.metrics.get('accuracy', result.metrics.get('auc', 0))
            print(f"  Performance: {accuracy:.3f}")
    
    return results


def run_config_manager_example():
    """Demonstrate configuration management."""
    print("\n=== Configuration Management Example ===")
    
    # Create config manager with script-relative path
    config_manager = ConfigManager(str(EXAMPLE_CONFIGS_DIR))
    
    # Create and add default configurations
    config_manager.create_default_configs()
    
    # List available configurations
    configs = config_manager.list_configs()
    print(f"Available configurations: {configs}")
    
    # Run benchmarks using saved configurations
    for config_name in configs[:2]:  # Run first 2 configs
        print(f"\n--- Running {config_name} ---")
        
        config = config_manager.get_config(config_name)
        
        # Reduce epochs for faster demo
        config.training.epochs = 20
        config.data.num_nodes = 200
        config.data.num_edges = 500
        
        benchmark = ModelBenchmark(config)
        
        try:
            result = benchmark.run_benchmark()
            print(f"✓ {config_name} completed")
            print(f"  Training time: {result.train_time:.2f}s")
        except Exception as e:
            print(f"✗ {config_name} failed: {e}")


def run_custom_benchmark():
    """Run a benchmark with custom parameters."""
    print("\n=== Custom Benchmark Example ===")
    
    # Create custom configuration
    from astroml.benchmarking import BenchmarkConfig, ModelConfig, DataConfig, TrainingConfig
    
    config = BenchmarkConfig(
        name="custom_large_gcn",
        description="Large GCN model with custom parameters",
        model=ModelConfig(
            name="gcn",
            params={
                "in_channels": 32,
                "hidden_channels": 128,
                "out_channels": 4,
                "num_layers": 3,
                "dropout": 0.3
            },
            task_type="classification"
        ),
        data=DataConfig(
            num_nodes=1000,
            num_features=32,
            num_edges=5000,
            num_classes=4,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        ),
        training=TrainingConfig(
            epochs=40,
            batch_size=64,
            learning_rate=0.005,
            weight_decay=1e-4,
            early_stopping_patience=15
        ),
        output_dir=str(OUTPUT_DIR / "custom_results"),
        num_runs=1,
        verbose=True
    )
    
    print(f"Custom configuration: {config.name}")
    print(f"Model parameters: {config.model.params}")
    print(f"Data size: {config.data.num_nodes} nodes, {config.data.num_edges} edges")
    
    benchmark = ModelBenchmark(config)
    
    with Timer("Custom benchmark"):
        result = benchmark.run_benchmark()
    
    print("\nCustom benchmark results:")
    print(f"  Total time: {result.train_time:.2f}s")
    print(f"  Memory usage: {result.peak_memory_mb:.1f}MB")
    
    if result.metrics:
        print("  Performance metrics:")
        for metric, value in result.metrics.items():
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")
    
    return result


def main():
    """Main function to run all examples."""
    print("AstroML Benchmarking Suite Examples")
    print("=" * 50)
    
    try:
        # Run basic benchmark
        basic_results = run_basic_benchmark()
        
        # Run multiple models
        multi_results = run_multiple_models()
        
        # Run configuration management example
        run_config_manager_example()
        
        # Run custom benchmark
        custom_results = run_custom_benchmark()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
        # Final summary
        print("\nFinal Summary:")
        print(f"  Basic benchmark: {basic_results.train_time:.2f}s")
        print(f"  Multiple models: {len(multi_results)} completed")
        print(f"  Custom benchmark: {custom_results.train_time:.2f}s")
        
    except KeyboardInterrupt:
        print("\nBenchmarking interrupted by user")
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
