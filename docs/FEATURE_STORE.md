# Feature Store Documentation

This document provides comprehensive documentation for the AstroML Feature Store, a centralized system for managing, computing, storing, and retrieving features for machine learning workflows.

## Overview

The Feature Store is designed to solve common challenges in machine learning feature engineering:

- **Feature Reuse**: Avoid recomputing the same features multiple times
- **Consistency**: Ensure features are computed consistently across training and inference
- **Versioning**: Track feature definitions and computations over time
- **Discovery**: Make features discoverable and well-documented
- **Performance**: Cache and optimize feature storage for fast access
- **Lineage**: Track feature dependencies and data provenance

## Architecture

The Feature Store consists of several key components:

### Core Components

1. **FeatureStore** - Main interface for feature management
2. **FeatureRegistry** - Registry of available feature computers
3. **FeatureStorage** - Storage backend for feature values and metadata
4. **FeatureEngine** - Computation engine for parallel feature processing
5. **FeatureTransformers** - Feature preprocessing and transformation utilities
6. **FeatureCache** - Multi-level caching system
7. **FeatureVersionManager** - Versioning and metadata management

### Data Models

- **FeatureDefinition** - Feature metadata and computation specification
- **FeatureValue** - Computed feature values with timestamps
- **FeatureSet** - Collections of related features
- **FeatureVersion** - Version information for features
- **FeatureLineage** - Dependency tracking between features

## Quick Start

### Basic Usage

```python
from astroml.features import create_feature_store
import pandas as pd

# Create feature store
store = create_feature_store("./my_feature_store")

# Load transaction data
data = pd.read_csv("transactions.csv")

# Register a custom feature
def account_balance_computer(data, entity_col, timestamp_col, **kwargs):
    """Compute account balance from transactions."""
    sent = data.groupby("src")["amount"].sum()
    received = data.groupby("dst")["amount"].sum()
    
    all_accounts = set(sent.index) | set(received.index)
    balances = {}
    
    for account in all_accounts:
        sent_amount = sent.get(account, 0)
        received_amount = received.get(account, 0)
        balances[account] = received_amount - sent_amount
    
    return pd.DataFrame(
        {"balance": list(balances.values())},
        index=list(balances.keys())
    )

# Register the feature
feature_def = store.register_feature(
    name="account_balance",
    computer=account_balance_computer,
    description="Account balance computed from transactions",
    feature_type=FeatureType.NUMERIC,
    tags=["balance", "financial"],
    owner="data_team",
)

# Compute and store the feature
computed_values = store.compute_and_store(
    feature_name="account_balance",
    data=data,
    entity_col="entity_id",
    timestamp_col="timestamp",
)

# Retrieve the feature
feature_values = store.get_feature("account_balance")
print(f"Computed balances for {len(feature_values)} accounts")
```

### Using Built-in Features

The Feature Store comes with several built-in features from the existing AstroML modules:

```python
# Compute frequency features
frequency_features = store.compute_and_store(
    feature_name="daily_transaction_count",
    data=data,
    entity_col="entity_id",
    timestamp_col="timestamp",
)

# Compute structural features
structural_features = store.compute_and_store(
    feature_name="degree_centrality",
    data=data,
    entity_col="entity_id",
    timestamp_col="timestamp",
)

# Create a feature set
feature_set = store.create_feature_set(
    name="account_features",
    feature_names=["daily_transaction_count", "degree_centrality", "account_balance"],
    description="Complete account feature set",
    entity_type="account",
)
```

## Feature Registration

### Registering Custom Features

```python
from astroml.features.feature_store import FeatureType

def custom_feature_computer(data, entity_col, timestamp_col, **kwargs):
    """Custom feature computation logic."""
    # Your feature computation code here
    result = data.groupby(entity_col).agg({
        "amount": ["sum", "mean", "count"],
        "timestamp": ["min", "max"],
    })
    
    # Flatten column names
    result.columns = ["_".join(col).strip() for col in result.columns.values]
    return result

# Register with full metadata
feature_def = store.register_feature(
    name="transaction_aggregates",
    computer=custom_feature_computer,
    description="Aggregated transaction statistics per account",
    feature_type=FeatureType.TIME_SERIES,
    tags=["aggregation", "statistics"],
    owner="ml_team",
    parameters={
        "window_size": 30,  # days
        "min_transactions": 5,
    },
)
```

### Feature Types

The Feature Store supports several feature types:

- **NUMERIC** - Numeric values (integers, floats)
- **CATEGORICAL** - Categorical values
- **BOOLEAN** - True/False values
- **TEXT** - Text values
- **VECTOR** - Multi-dimensional vectors
- **TIME_SERIES** - Time series data

### Feature Parameters

```python
# Register feature with parameters
feature_def = store.register_feature(
    name="rolling_features",
    computer=rolling_features_computer,
    description="Rolling window features",
    parameters={
        "window_size": 7,  # 7-day window
        "functions": ["mean", "std", "min", "max"],
    },
)
```

## Feature Computation

### Single Feature Computation

```python
# Compute a single feature
result = store.compute_feature(
    feature_name="account_balance",
    data=data,
    entity_col="entity_id",
    timestamp_col="timestamp",
    window_size=30,  # Custom parameter
)
```

### Batch Feature Computation

```python
# Define feature configurations
feature_configs = [
    {
        "name": "account_balance",
        "computer": "frequency_features",
        "parameters": {"window_days": 30},
    },
    {
        "name": "transaction_frequency",
        "computer": "frequency_features", 
        "parameters": {"window_days": 7},
    },
    {
        "name": "degree_centrality",
        "computer": "structural_features",
        "parameters": {},
    },
]

# Compute all features in parallel
results = store.registry.compute_features_batch(
    feature_configs=feature_configs,
    data=data,
    parallel=True,
)

for feature_name, values in results.items():
    store.store_feature(feature_name, values)
```

### Using the Computation Engine

```python
from astroml.features.feature_engine import create_computation_engine

# Create computation engine
engine = create_computation_engine(max_workers=4)

# Create computation tasks
tasks = []
for feature_name in ["feature1", "feature2", "feature3"]:
    task = engine.create_task(
        feature_name=feature_name,
        data=data,
        computer_name="frequency_features",
        entity_col="entity_id",
        timestamp_col="timestamp",
    )
    tasks.append(task)

# Submit and run tasks
engine.submit_tasks(tasks)
completed_tasks = engine.run_tasks(parallel=True)

# Process results
for task in tasks:
    if task.task_id in completed_tasks:
        completed_task = completed_tasks[task.task_id]
        if completed_task.status == ComputationStatus.COMPLETED:
            print(f"Feature {task.feature_name} computed successfully")
            store.store_feature(task.feature_name, completed_task.result)
```

## Feature Storage and Retrieval

### Basic Storage and Retrieval

```python
# Store computed feature
store.store_feature(
    feature_name="account_balance",
    values=feature_dataframe,
    metadata={
        "computed_at": datetime.utcnow().isoformat(),
        "data_source": "transactions_2023",
        "version": "1.0",
    },
)

# Retrieve feature
feature_values = store.get_feature("account_balance")

# Retrieve for specific entities
specific_values = store.get_feature(
    "account_balance",
    entity_ids=["account1", "account2", "account3"],
)

# Point-in-time retrieval (if supported)
historical_values = store.get_feature(
    "account_balance",
    entity_ids=["account1"],
    timestamp=datetime(2023, 6, 1),
)
```

### Feature Sets

```python
# Create feature set
feature_set = store.create_feature_set(
    name="risk_features",
    feature_names=[
        "account_balance",
        "transaction_frequency",
        "degree_centrality",
        "asset_diversity",
    ],
    description="Features for risk assessment",
    entity_type="account",
)

# Retrieve feature set
feature_set_data = store.get_features_for_entities(
    feature_names=["account_balance", "transaction_frequency"],
    entity_ids=["account1", "account2", "account3"],
)

# Get feature set definition
risk_features_set = store.get_feature_set("risk_features")
print(f"Feature set contains {len(risk_features_set.feature_ids)} features")
```

## Feature Transformation

### Basic Transformations

```python
from astroml.features.feature_transformers import (
    create_feature_transformer,
    TransformationType,
)

# Create transformer
transformer = create_feature_transformer()

# Add transformations
transformer.add_transformation(
    "standard_scaling",
    TransformationType.STANDARD_SCALER,
    ["account_balance", "transaction_amount"],
)

transformer.add_transformation(
    "log_transform",
    TransformationType.LOG_TRANSFORM,
    ["transaction_amount"],
    offset=1.0,
)

# Fit and transform
transformed_data = transformer.fit_transform(feature_data)

# Save transformer for later use
transformer.save("feature_transformer.pkl")
```

### Advanced Feature Engineering

```python
from astroml.features.feature_transformers import FeatureEngineering

# Create interaction features
interaction_features = FeatureEngineering.create_interaction_features(
    data=feature_data,
    columns=["balance", "frequency"],
    interaction_type="multiplication",
)

# Create polynomial features
poly_features = FeatureEngineering.create_polynomial_features(
    data=feature_data,
    columns=["balance"],
    degree=2,
)

# Create rolling features
rolling_features = FeatureEngineering.create_rolling_features(
    data=feature_data.set_index("timestamp"),
    columns=["transaction_amount"],
    window_sizes=[7, 30],
    functions=["mean", "std"],
)

# Create time features
time_features = FeatureEngineering.create_time_features(
    data=feature_data,
    timestamp_column="timestamp",
)
```

## Caching

### Memory Caching

```python
from astroml.features.feature_cache import create_feature_cache

# Create LRU cache
cache = create_feature_cache(
    strategy=CacheStrategy.LRU,
    max_size=1000,
)

# Cache will be used automatically by the feature store
store = FeatureStore(cache=cache)
```

### Disk Caching

```python
# Create disk cache for large features
cache = create_feature_cache(
    strategy=CacheStrategy.DISK,
    disk_path="./feature_cache",
    max_size=10000,
)

store = FeatureStore(cache=cache)
```

### Redis Caching

```python
# Create Redis cache for distributed environments
cache = create_feature_cache(
    strategy=CacheStrategy.REDIS,
    redis_url="redis://localhost:6379",
    ttl_seconds=3600,  # 1 hour TTL
)

store = FeatureStore(cache=cache)
```

## Feature Versioning

### Creating Versions

```python
from astroml.features.feature_versioning import create_version_manager

# Create version manager
version_manager = create_version_manager("./feature_versions")

# Create new version of a feature
version = version_manager.create_version(
    feature_name="account_balance",
    code=balance_computer_code,
    parameters={"window_days": 30},
    data_schema={"entity_id": "string", "amount": "float"},
    description="Account balance with 30-day window",
    created_by="data_team",
)

print(f"Created version {version.version} for {version.feature_name}")
```

### Managing Version Status

```python
# Update version status
version_manager.update_version_status(
    version_id=version.version_id,
    status=VersionStatus.APPROVED,
    updated_by="ml_lead",
)

# Deploy version
version_manager.update_version_status(
    version_id=version.version_id,
    status=VersionStatus.DEPLOYED,
    updated_by="ops_team",
)
```

### Version History

```python
# Get version history
history = version_manager.get_change_history(feature_name="account_balance")

for change in history:
    print(f"{change.changed_at}: {change.description}")
    print(f"  Changed by: {change.changed_by}")
    print(f"  Type: {change.change_type.value}")
```

## Performance Optimization

### Storage Optimization

```python
from astroml.features.feature_cache import create_storage_optimizer

# Create storage optimizer
optimizer = create_storage_optimizer(
    format=StorageFormat.PARQUET,
    compression="snappy",
)

# Optimize DataFrame before storage
optimized_data = optimizer.optimize_dataframe(
    data=feature_data,
    feature_name="account_balance",
)

# Save with optimal settings
optimizer.save_dataframe(optimized_data, "account_balance.parquet")
```

### Batch Processing

```python
# Use batch mode for better performance
with store.batch_mode():
    # Store multiple features
    for feature_name in feature_names:
        values = compute_feature(feature_name, data)
        store.store_feature(feature_name, values)
    
    # Cache is automatically cleared at the end
```

### Parallel Computation

```python
# Configure computation engine for parallel processing
from astroml.features.feature_engine import create_computation_engine

engine = create_computation_engine(max_workers=8)

# Process features in parallel
results = engine.compute_features_batch(
    feature_configs=feature_configs,
    data=data,
    parallel=True,
)
```

## Monitoring and Debugging

### Feature Discovery

```python
# List all available features
features = store.list_features()
for feature in features:
    print(f"{feature.name}: {feature.description}")
    print(f"  Type: {feature.feature_type.value}")
    print(f"  Tags: {', '.join(feature.tags)}")
    print(f"  Owner: {feature.owner}")
    print(f"  Status: {feature.status.value}")

# Filter features by tags
risk_features = store.list_features(tags=["risk"])
numeric_features = store.list_features(feature_type=FeatureType.NUMERIC)
```

### Cache Statistics

```python
# Get cache statistics
stats = store.cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['size']}")
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")

# Clear cache if needed
store.clear_cache()
```

### Error Handling

```python
try:
    # Compute feature
    result = store.compute_feature(
        feature_name="non_existent_feature",
        data=data,
        entity_col="entity_id",
        timestamp_col="timestamp",
    )
except ValueError as e:
    print(f"Feature computation failed: {e}")

# Check feature existence
feature_def = store.storage.get_feature_definition("feature_name_v1")
if feature_def is None:
    print("Feature not found")
```

## Best Practices

### Feature Design

1. **Descriptive Names**: Use clear, descriptive feature names
2. **Documentation**: Provide comprehensive descriptions
3. **Type Safety**: Specify correct feature types
4. **Tagging**: Use consistent tags for categorization
5. **Parameters**: Make features configurable through parameters

### Performance

1. **Caching**: Enable appropriate caching strategies
2. **Batch Operations**: Use batch mode for multiple operations
3. **Parallel Processing**: Enable parallel computation for independent features
4. **Storage Optimization**: Use optimal storage formats
5. **Indexing**: Properly index data for fast retrieval

### Version Management

1. **Semantic Versioning**: Use meaningful version numbers
2. **Change Tracking**: Document all changes thoroughly
3. **Approval Process**: Use status transitions for deployment
4. **Backward Compatibility**: Maintain compatibility when possible
5. **Deprecation**: Properly deprecate old versions

### Data Quality

1. **Validation**: Validate input data before computation
2. **Error Handling**: Handle edge cases gracefully
3. **Logging**: Log important events and errors
4. **Testing**: Test features thoroughly
5. **Monitoring**: Monitor feature quality over time

## Integration Examples

### Machine Learning Pipeline

```python
# Feature store for ML pipeline
class MLPipeline:
    def __init__(self):
        self.store = create_feature_store("./ml_feature_store")
        self.transformer = create_feature_transformer()
        
        # Setup feature transformations
        self.transformer.add_transformation(
            "scaling",
            TransformationType.STANDARD_SCALER,
            ["feature1", "feature2"],
        )
    
    def fit_features(self, training_data):
        """Compute and fit features on training data."""
        # Compute features
        feature_names = ["feature1", "feature2", "feature3"]
        
        for name in feature_names:
            self.store.compute_and_store(
                feature_name=name,
                data=training_data,
                entity_col="entity_id",
                timestamp_col="timestamp",
            )
        
        # Get features for training
        X = self.store.get_features_for_entities(
            feature_names=feature_names,
            entity_ids=training_data["entity_id"].unique(),
        )
        
        # Fit transformations
        self.transformer.fit(X)
        
        return self.transformer.transform(X)
    
    def transform_features(self, inference_data):
        """Transform features for inference."""
        feature_names = ["feature1", "feature2", "feature3"]
        
        # Get features (may use cache)
        X = self.store.get_features_for_entities(
            feature_names=feature_names,
            entity_ids=inference_data["entity_id"].unique(),
        )
        
        return self.transformer.transform(X)

# Usage
pipeline = MLPipeline()
X_train = pipeline.fit_features(training_data)
X_inference = pipeline.transform_features(inference_data)
```

### Real-time Feature Serving

```python
# Real-time feature serving
class FeatureServer:
    def __init__(self):
        self.store = create_feature_store("./realtime_store")
        
        # Configure Redis cache for real-time access
        cache = create_feature_cache(
            strategy=CacheStrategy.REDIS,
            redis_url="redis://localhost:6379",
            ttl_seconds=300,  # 5 minutes
        )
        
        self.store = FeatureStore(cache=cache)
    
    def get_features(self, entity_id, feature_names):
        """Get features for a single entity."""
        return self.store.get_features_for_entities(
            feature_names=feature_names,
            entity_ids=[entity_id],
        )
    
    def update_features(self, entity_id, new_data):
        """Update features for an entity."""
        # Recompute features
        for feature_name in self.feature_names:
            updated_values = self.store.compute_feature(
                feature_name=feature_name,
                data=new_data,
                entity_col="entity_id",
                timestamp_col="timestamp",
            )
            
            # Update cache
            self.store.store_feature(feature_name, updated_values)

# Usage
server = FeatureServer()
features = server.get_features("account123", ["balance", "frequency"])
```

## Troubleshooting

### Common Issues

1. **Feature Not Found**: Check if feature is registered and spelled correctly
2. **Memory Issues**: Reduce cache size or use disk caching
3. **Performance**: Enable parallel processing and optimize storage
4. **Version Conflicts**: Check feature version compatibility
5. **Data Issues**: Validate input data format and required columns

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("astroml.features")

# Debug feature computation
try:
    result = store.compute_feature(
        feature_name="problematic_feature",
        data=data,
        entity_col="entity_id",
        timestamp_col="timestamp",
    )
except Exception as e:
    logger.error(f"Feature computation failed: {e}")
    raise
```

### Performance Profiling

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.2f}s")

# Profile feature operations
with timer("Feature Computation"):
    result = store.compute_feature("feature_name", data, "entity_id", "timestamp")

with timer("Feature Retrieval"):
    stored_result = store.get_feature("feature_name")
```

## API Reference

### Core Classes

- **FeatureStore**: Main interface for feature management
- **FeatureDefinition**: Feature metadata and specification
- **FeatureSet**: Collection of related features
- **FeatureRegistry**: Registry of feature computers

### Configuration

- **CacheConfig**: Cache configuration options
- **StorageConfig**: Storage configuration options
- **TransformationConfig**: Transformation configuration

### Enums

- **FeatureType**: Supported feature data types
- **FeatureStatus**: Feature lifecycle status
- **CacheStrategy**: Caching strategies
- **StorageFormat**: Storage formats

For detailed API documentation, see the inline documentation in the source code.
