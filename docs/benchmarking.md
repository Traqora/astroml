# AstroML Benchmarking Suite

The AstroML benchmarking suite provides a scaffold for evaluating Graph Neural Network (GNN) models. Its built-in link-prediction workflow uses synthetic data and does not load real Stellar ledger data.

## Overview

The benchmarking suite includes:
- **Core Framework**: Main benchmark execution engine
- **Metrics Module**: Comprehensive evaluation metrics for different GNN tasks
- **Configuration Management**: Flexible configuration system for experiments
- **Utility Functions**: Helper functions for timing, memory monitoring, and more

## Quick Start

### Basic Usage

```python
from astroml.benchmarking import ModelBenchmark, create_config_from_template

# Create a simple configuration
config = create_config_from_template(
    name="my_benchmark",
    model_name="gcn",
    task_type="classification"
)

# Run benchmark
benchmark = ModelBenchmark(config)
result = benchmark.run_benchmark()

# View results
print(f"Training time: {result.training_time:.2f}s")
print(f"Metrics: {result.metrics}")
```

### Running Example Scripts

```bash
# Quick start example
python examples/quick_start.py

# Comprehensive examples
python examples/benchmark_example.py
```

## Supported Models

The benchmarking suite supports the following GNN models:

1. **GCN (Graph Convolutional Network)**
   - Task: Node classification
   - Parameters: `in_channels`, `hidden_channels`, `out_channels`, `num_layers`, `dropout`

2. **Link Predictor**
   - Task: Link prediction
   - Parameters: `in_channels`, `hidden_channels`, `num_layers`, `dropout`

3. **SAGE Encoder (GraphSAGE)**
   - Task: Inductive node classification
   - Parameters: `in_channels`, `hidden_channels`, `out_channels`, `num_layers`, `dropout`

4. **Deep SVDD**
   - Task: Anomaly detection
   - Parameters: `in_channels`, `hidden_channels`, `out_channels`, `num_layers`, `dropout`

## Supported Tasks

### Classification
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Use Case**: Node classification on Stellar network

### Link Prediction
- **Metrics**: AUC-ROC, AUC-PR, Hits@K, Mean Reciprocal Rank
- **Use Case**: Predicting transaction connections

### Anomaly Detection
- **Metrics**: AUC-ROC, Precision, Recall, F1-Score
- **Use Case**: Detecting fraudulent transactions

### Regression
- **Metrics**: MSE, MAE, R²-Score
- **Use Case**: Predicting continuous values

## Configuration

### Basic Configuration

```python
from astroml.benchmarking import BenchmarkConfig, ModelConfig, DataConfig, TrainingConfig

config = BenchmarkConfig(
    name="my_experiment",
    description="Custom benchmark experiment",
    model=ModelConfig(
        name="gcn",
        params={
            "in_channels": 16,
            "hidden_channels": 64,
            "out_channels": 2,
            "num_layers": 2,
            "dropout": 0.5
        },
        task_type="classification"
    ),
    data=DataConfig(
        num_nodes=1000,
        num_features=16,
        num_edges=5000,
        num_classes=2,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    ),
    training=TrainingConfig(
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
        early_stopping_patience=10
    )
)
```

### Configuration Management

```python
from astroml.benchmarking import ConfigManager

# Create config manager
manager = ConfigManager("./my_configs")

# Add configuration
manager.add_config(config)

# List configurations
configs = manager.list_configs()

# Load configuration
loaded_config = manager.get_config("my_experiment")

# Create default configurations
manager.create_default_configs()
```

## Data Configuration

The benchmarking suite currently uses synthetic data that mimics Stellar transaction graphs:

```python
data_config = DataConfig(
    num_nodes=1000,        # Number of nodes in the graph
    num_features=16,       # Feature dimensionality
    num_edges=5000,        # Number of edges
    num_classes=2,         # Number of classes (for classification)
    train_ratio=0.7,      # Training data ratio
    val_ratio=0.15,        # Validation data ratio
    test_ratio=0.15,      # Test data ratio
    feature_noise=0.1,     # Noise level for features
    edge_noise=0.1         # Noise level for edges
)
```

## Offline Temporal Dataset

The link-prediction workflow provides a deterministic, in-memory fixture without database, DVC, or artifact-store dependencies:

```python
from astroml.training.synthetic_dataset import generate_synthetic_ledger_dataset

dataset = generate_synthetic_ledger_dataset(seed=42)
dataset.save_metadata("artifacts/synthetic-ledger.json")
```

The generator creates directed edges without self-loops over integer timestamps in `[1, num_ledgers]`, plus seeded standard-normal node features. Metadata is written as a local JSON sidecar using `SnapshotLineage`, including node and edge counts, the configured time window, observed first/last timestamps, and seed. Empty datasets are supported and record observed timestamps as `null`. Non-empty graphs require at least two accounts. Synthetic features are not derived from edges and the fixture is not representative Stellar data.

The training CLI can persist the same metadata alongside a run:

```bash
python -m astroml.training.train_link_prediction --metadata-path artifacts/synthetic-ledger.json
```

Negative sampling uses directed non-self pairs. The requested sample count is capped by the number of eligible pairs, and random draws are bounded by `target_count * max_attempts`; a dense graph may therefore yield fewer negatives than requested, while a graph with no eligible pairs yields an empty tensor. Evaluation is defined for empty or single-class labels: AUC is the neutral sentinel `0.5`; average precision is `0.0` with no positives and `1.0` when every label is positive. These sentinel values indicate that ranking quality cannot be compared for that split.

## Training Configuration

```python
training_config = TrainingConfig(
    epochs=100,                    # Number of training epochs
    batch_size=32,                 # Batch size
    learning_rate=0.01,            # Learning rate
    weight_decay=0.0,              # Weight decay
    early_stopping_patience=10,    # Early stopping patience
    early_stopping_min_delta=1e-4, # Minimum improvement threshold
    optimizer="adam",              # Optimizer type
    loss_function="cross_entropy", # Loss function
    seed=42                        # Random seed
)
```

## Results and Outputs

### Benchmark Results

The benchmark returns a `BenchmarkResult` object containing:

```python
result = benchmark.run_benchmark()

# Access results
print(f"Training time: {result.training_time}")
print(f"Validation time: {result.validation_time}")
print(f"Test time: {result.test_time}")
print(f"Peak memory: {result.peak_memory_mb}")
print(f"Metrics: {result.metrics}")
print(f"Output directory: {result.output_dir}")
print(f"Model path: {result.model_path}")
```

### Saved Files

The benchmark automatically saves:
- **Results JSON**: Complete benchmark results and configuration
- **Model Checkpoint**: Trained model state dictionary
- **Configuration**: Benchmark configuration file

## Advanced Usage

### Custom Metrics

```python
from astroml.benchmarking import MetricCalculator

# Compute custom metrics
calculator = MetricCalculator(task_type="classification")
metrics = calculator.compute_metrics(
    y_true=[0, 1, 0, 1],
    y_pred=[0, 1, 0, 0],
    y_score=[0.2, 0.8, 0.3, 0.6]
)
```

### Memory and Timing

```python
from astroml.benchmarking import Timer, MemoryMonitor

# Time operations
with Timer("Training"):
    model.train(data)

# Monitor memory usage
with MemoryMonitor("Memory usage"):
    result = benchmark.run_benchmark()
```

### Device Management

```python
from astroml.benchmarking import get_device_info

# Get device information
device_info = get_device_info()
print(f"CUDA available: {device_info['cuda_available']}")
print(f"CPU count: {device_info['cpu_count']}")

# Set device in configuration
config.device = "cuda" if device_info['cuda_available'] else "cpu"
```

## Best Practices

### 1. Reproducibility
- Always set random seeds using `set_random_seed(42)`
- Save configurations for reproducible experiments
- Use consistent data splits

### 2. Performance Optimization
- Use appropriate batch sizes for your hardware
- Monitor memory usage to avoid OOM errors
- Use early stopping to prevent overfitting

### 3. Experiment Management
- Use descriptive names for configurations
- Save results systematically
- Document experiment parameters

### 4. Model Selection
- Start with simpler models (GCN) as baselines
- Consider task complexity when choosing models
- Validate on appropriate metrics

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller models
   - Monitor memory usage

2. **Poor Performance**
   - Check data quality and preprocessing
   - Adjust learning rate and other hyperparameters
   - Verify model architecture matches task

3. **Slow Training**
   - Use GPU acceleration
   - Optimize data loading
   - Consider model complexity

### Debug Mode

Enable verbose output for debugging:

```python
config.verbose = True
```

## Integration with Real Data

The benchmarking suite is designed to integrate with real Stellar data:

1. **Data Loading**: Replace synthetic data generation with real data ingestion
2. **Feature Engineering**: Customize feature extraction for Stellar transactions
3. **Evaluation**: Use domain-specific metrics for financial applications

## Contributing

To contribute new models or metrics:

1. Implement the model in the models directory
2. Add corresponding metrics in the metrics module
3. Update configuration templates
4. Add tests and documentation

## Examples

See the `examples/` directory for complete usage examples:
- `quick_start.py`: Basic usage example
- `benchmark_example.py`: Comprehensive examples

## API Reference

### Core Classes

- `ModelBenchmark`: Main benchmark execution class
- `BenchmarkConfig`: Complete benchmark configuration
- `BenchmarkResult`: Benchmark results container

### Configuration Classes

- `ModelConfig`: Model-specific configuration
- `DataConfig`: Data loading configuration
- `TrainingConfig`: Training parameters configuration

### Utility Classes

- `ConfigManager`: Configuration management
- `MetricCalculator`: Metrics computation
- `Timer`: Operation timing
- `MemoryMonitor`: Memory usage monitoring

For detailed API documentation, see the source code docstrings.
