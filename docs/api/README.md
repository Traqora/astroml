# AstroML API Documentation

## Overview

AstroML is a research-driven Python framework for building dynamic graph machine learning models on the Stellar Development Foundation Stellar blockchain. This comprehensive API documentation covers all modules, classes, and functions available in the AstroML framework.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Modules](#core-modules)
   - [Data Ingestion](#data-ingestion)
   - [Graph Building](#graph-building)
   - [Feature Engineering](#feature-engineering)
   - [Machine Learning Models](#machine-learning-models)
   - [Training](#training)
   - [Validation](#validation)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [API Reference](#api-reference)
   - [Error Responses](errors.md)

## Architecture Overview

AstroML follows a modular architecture designed for research-grade experimentation:

```
Stellar Ledger → Ingestion → Normalization → Graph Builder → Features → ML Models → Experiments
```

### Key Components

- **Ingestion Layer**: Handles Stellar ledger data ingestion and normalization
- **Graph Layer**: Constructs dynamic transaction graphs
- **Feature Layer**: Engineering features for blockchain accounts
- **ML Layer**: Graph neural networks and self-supervised learning
- **Training Layer**: Model training and experimentation
- **Validation Layer**: Model evaluation and validation

## Core Modules

### Data Ingestion

The ingestion module provides tools for fetching, processing, and storing Stellar blockchain data.

#### Key Classes

- **`IngestionService`**: Main service for ledger ingestion
- **`StateStore`**: Manages ingestion state and tracking
- **`IngestionResult`**: Results container for ingestion operations

#### Key Functions

- **`backfill()`**: Bulk historical data ingestion
- **`stream()`**: Real-time data streaming
- **`normalize()`**: Data normalization and cleaning

### Graph Building

The graph module constructs dynamic transaction graphs from ingested data.

#### Key Classes

- **`GraphBuilder`**: Main graph construction service
- **`TemporalGraph`**: Time-evolving graph structure
- **`NodeFeatures`**: Node feature engineering

#### Key Functions

- **`build_snapshot()`**: Create time-window graph snapshots
- **`add_temporal_edges()`**: Add time-based edges
- **`compute_features()`**: Calculate graph features

### Feature Engineering

The features module provides tools for extracting meaningful features from blockchain data.

#### Key Classes

- **`AccountFeatures`**: Account-level feature extraction
- **`TransactionFeatures`**: Transaction-level features
- **`TemporalFeatures`**: Time-based feature engineering

#### Key Functions

- **`extract_account_features()`**: Extract account characteristics
- **`compute_transaction_metrics()`**: Calculate transaction metrics
- **`temporal_aggregation()`**: Aggregate features over time windows

### Machine Learning Models

The models module provides implementations of graph neural networks and other ML models.

#### Key Classes

- **`GCN`**: Graph Convolutional Network
- **`TemporalGNN`**: Time-aware graph neural network
- **`AnomalyDetector`**: Anomaly detection models

#### Key Functions

- **`train_model()`**: Train ML models
- **`predict()`**: Make predictions
- **`evaluate()``: Model evaluation

### Training

The training module provides tools for model training and experimentation.

#### Key Classes

- **`Trainer`**: Main training orchestrator
- **`Experiment`**: Experiment management
- **`MetricsTracker`**: Training metrics tracking

#### Key Functions

- **`train_gcn()`**: Train graph convolutional networks
- **`run_experiment()`**: Execute ML experiments
- **`hyperparameter_search()`**: Hyperparameter optimization

### Validation

The validation module provides tools for model validation and evaluation.

#### Key Classes

- **`Validator`**: Model validation service
- **`MetricsCalculator`**: Metrics computation
- **`CrossValidator`**: Cross-validation utilities

#### Key Functions

- **`validate_model()`**: Validate model performance
- **`compute_metrics()`**: Calculate evaluation metrics
- **`cross_validate()``: Perform cross-validation

## Configuration

AstroML uses configuration files for database connections, model parameters, and ingestion settings.

### Database Configuration

```yaml
# config/database.yaml
database:
  host: localhost
  port: 5432
  name: astroml
  user: astroml_user
  password: secure_password
```

### Model Configuration

```yaml
# config/models.yaml
gcn:
  input_dim: 64
  hidden_dims: [128, 64]
  output_dim: 2
  dropout: 0.5
  learning_rate: 0.001
```

### Ingestion Configuration

```yaml
# config/ingestion.yaml
ingestion:
  batch_size: 1000
  max_retries: 3
  timeout: 30
  stellar_network: testnet
```

## Usage Examples

### Basic Data Ingestion

```python
from astroml.ingestion import IngestionService

# Initialize ingestion service
service = IngestionService()

# Ingest historical ledgers
result = service.ingest(
    start_ledger=1000000,
    end_ledger=1100000
)

print(f"Processed: {result.processed}")
print(f"Skipped: {result.skipped}")
```

### Graph Construction

```python
from astroml.graph import GraphBuilder

# Build graph snapshot
builder = GraphBuilder()
graph = builder.build_snapshot(
    window_days=30,
    min_transactions=10
)

print(f"Nodes: {graph.num_nodes}")
print(f"Edges: {graph.num_edges}")
```

### Model Training

```python
from astroml.models import GCN
from astroml.training import Trainer

# Initialize model
model = GCN(
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2
)

# Train model
trainer = Trainer(model)
metrics = trainer.train(
    graph=graph,
    epochs=100,
    learning_rate=0.001
)
```

### Anomaly Detection

```python
from astroml.models import AnomalyDetector

# Initialize detector
detector = AnomalyDetector()

# Detect anomalies
anomalies = detector.detect(
    graph=graph,
    threshold=0.95
)

print(f"Found {len(anomalies)} anomalies")
```

## API Reference

### Ingestion Module

#### IngestionService

```python
class IngestionService:
    def __init__(self, state_store: Optional[StateStore] = None) -> None
    def ingest(
        self,
        start_ledger: Optional[int] = None,
        end_ledger: Optional[int] = None,
        fetch_fn: Optional[Callable[[int], object]] = None,
        process_fn: Optional[Callable[[int, object], None]] = None,
    ) -> IngestionResult
```

#### StateStore

```python
class StateStore:
    def load(self) -> IngestionState
    def save(self, state: IngestionState) -> None
    def mark_processed(self, ledger_id: int) -> None
```

### Graph Module

#### GraphBuilder

```python
class GraphBuilder:
    def __init__(self, config: Optional[GraphConfig] = None) -> None
    def build_snapshot(
        self,
        window_days: int,
        min_transactions: int = 1,
        include_temporal: bool = True
    ) -> TemporalGraph
    def add_temporal_edges(self, graph: TemporalGraph) -> TemporalGraph
```

#### TemporalGraph

```python
class TemporalGraph:
    def __init__(self, nodes: List[Node], edges: List[Edge]) -> None
    def add_node(self, node: Node) -> None
    def add_edge(self, edge: Edge) -> None
    def get_subgraph(self, time_window: Tuple[datetime, datetime]) -> TemporalGraph
```

### Models Module

#### GCN

```python
class GCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.5
    ) -> None
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor
```

#### AnomalyDetector

```python
class AnomalyDetector:
    def __init__(self, model: Optional[nn.Module] = None) -> None
    def detect(
        self,
        graph: TemporalGraph,
        threshold: float = 0.95
    ) -> List[AnomalyResult]
    def train(self, graph: TemporalGraph, labels: torch.Tensor) -> None
```

### Training Module

#### Trainer

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None
    def train(
        self,
        graph: TemporalGraph,
        epochs: int,
        learning_rate: float = 0.001
    ) -> TrainingMetrics
    def evaluate(self, graph: TemporalGraph) -> EvaluationMetrics
```

### Validation Module

#### Validator

```python
class Validator:
    def __init__(self, metrics: List[str]) -> None
    def validate(
        self,
        model: nn.Module,
        test_graph: TemporalGraph,
        test_labels: torch.Tensor
    ) -> ValidationResults
    def cross_validate(
        self,
        model: nn.Module,
        graphs: List[TemporalGraph],
        k_folds: int = 5
    ) -> CrossValidationResults
```

## Error Handling

AstroML provides comprehensive error handling for common issues:

### Ingestion Errors

- **`LedgerNotFoundError`**: Raised when a requested ledger cannot be found
- **`IngestionError`**: General ingestion failures
- **`StateStoreError`**: State management issues

### Graph Errors

- **`GraphConstructionError`**: Graph building failures
- **`FeatureExtractionError`**: Feature computation issues
- **`TemporalIndexError`**: Time-based indexing problems

### Model Errors

- **`ModelTrainingError`**: Training failures
- **`PredictionError`**: Prediction issues
- **`ValidationError`**: Validation problems

## Performance Considerations

### Memory Management

- Use batch processing for large datasets
- Implement streaming for real-time ingestion
- Consider graph partitioning for large networks

### Computational Efficiency

- Leverage GPU acceleration for model training
- Use sparse matrices for large graphs
- Implement parallel processing where possible

### Storage Optimization

- Use efficient data formats (Parquet, HDF5)
- Implement data compression
- Consider distributed storage for large datasets

## Testing

AstroML includes comprehensive test coverage:

### Unit Tests

```bash
python -m pytest tests/unit/
```

### Integration Tests

```bash
python -m pytest tests/integration/
```

### End-to-End Tests

```bash
python -m pytest tests/e2e/
```

## Contributing

When contributing to AstroML:

1. Follow the existing code style
2. Add comprehensive tests
3. Update documentation
4. Ensure all tests pass
5. Submit pull requests with clear descriptions

## License

AstroML is licensed under the MIT License. See LICENSE for details.

---

This API documentation provides comprehensive coverage of the AstroML framework. For specific module details, refer to the individual module documentation files.
