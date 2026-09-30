# Usage Examples and Integration Guides

## Overview

This guide provides comprehensive examples and integration patterns for using AstroML in various scenarios, from basic setup to advanced research workflows.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Ingestion Examples](#data-ingestion-examples)
3. [Graph Building Examples](#graph-building-examples)
4. [Machine Learning Examples](#machine-learning-examples)
5. [Anomaly Detection System](#anomaly-detection-system)
6. [Research Workflows](#research-workflows)
7. [Production Integration](#production-integration)
8. [Advanced Patterns](#advanced-patterns)

## Quick Start

### Basic Setup

```python
# Install AstroML
pip install astroml

# Basic imports
from astroml.ingestion import IngestionService
from astroml.graph import GraphBuilder
from astroml.models import GCN, Trainer
from astroml.features import AccountFeatures

# Initialize services
ingestion_service = IngestionService()
graph_builder = GraphBuilder()
```

### First Pipeline

```python
# Complete end-to-end pipeline
def quick_start_pipeline():
    """Quick start pipeline for AstroML."""
    
    # 1. Ingest some data
    result = ingestion_service.ingest(
        start_ledger=1000000,
        end_ledger=1000100
    )
    
    # 2. Build graph
    graph = graph_builder.build_snapshot(window_days=7)
    
    # 3. Train a simple model
    model = GCN(
        input_dim=graph.node_features.shape[1],
        hidden_dims=[64, 32],
        output_dim=2
    )
    
    trainer = Trainer(model)
    metrics = trainer.train(graph, epochs=10)
    
    print(f"Quick start complete! Accuracy: {metrics.val_accuracy[-1]:.4f}")

# Run the pipeline
quick_start_pipeline()
```

## Data Ingestion Examples

### Historical Data Backfill

```python
from astroml.ingestion import EnhancedIngestionService
from astroml.ingestion.config import IngestionConfig

def backfill_historical_data():
    """Backfill historical Stellar ledger data."""
    
    # Configure for large-scale ingestion
    config = IngestionConfig(
        batch_size=5000,
        parallel_workers=8,
        timeout=60,
        stellar_network="mainnet"
    )
    
    service = EnhancedIngestionService(config=config)
    
    # Backfill 6 months of data
    start_ledger = 1000000  # Approximate 6 months ago
    end_ledger = 1200000    # Current ledger
    
    result = service.backfill(
        start_ledger=start_ledger,
        end_ledger=end_ledger,
        batch_size=5000,
        parallel_workers=8
    )
    
    print(f"Backfill completed:")
    print(f"  Attempted: {len(result.attempted)} ledgers")
    print(f"  Processed: {len(result.processed)} ledgers")
    print(f"  Skipped: {len(result.skipped)} ledgers")
    print(f"  Success rate: {len(result.processed)/len(result.attempted)*100:.2f}%")

backfill_historical_data()
```

### Real-time Streaming

```python
import asyncio
from astroml.ingestion import HorizonStream, EnhancedStream
from astroml.ingestion.config import StreamConfig

async def setup_realtime_streaming():
    """Set up real-time data streaming from Stellar."""
    
    # Configure streaming
    config = StreamConfig(
        horizon_url="https://horizon.stellar.org",
        network="mainnet",
        reconnect_interval=5
    )
    
    stream = EnhancedStream(config, buffer_size=1000)
    
    # Define processors
    async def process_transaction(tx):
        """Process incoming transaction."""
        # Update graph with new transaction
        await update_transaction_graph(tx)
        
        # Check for anomalies in real-time
        if await is_suspicious_transaction(tx):
            await send_alert(tx)
    
    async def update_anomaly_scores(tx):
        """Update anomaly detection scores."""
        scores = await compute_realtime_scores(tx)
        await cache_scores(tx.hash, scores)
    
    async def handle_error(error, context):
        """Handle streaming errors."""
        print(f"Stream error: {error}")
        await log_error(error, context)
        await notify_admin(error)
    
    # Start streaming
    await stream.start_streaming(
        processors=[process_transaction, update_anomaly_scores],
        error_handler=handle_error
    )

# Run the streaming
asyncio.run(setup_realtime_streaming())
```

### Custom Data Processing

```python
from astroml.ingestion import IngestionService, Normalizer
from astroml.ingestion.config import NormalizationConfig

def custom_data_processing():
    """Example of custom data processing pipeline."""
    
    # Configure normalization
    norm_config = NormalizationConfig(
        standardize_timestamps=True,
        validate_addresses=True,
        filter_test_operations=True,
        min_amount_threshold=1.0  # Filter small transactions
    )
    
    normalizer = Normalizer(norm_config)
    service = IngestionService()
    
    # Custom fetch function with caching
    cache = {}
    
    def fetch_with_cache(ledger_id):
        """Fetch ledger with caching."""
        if ledger_id in cache:
            return cache[ledger_id]
        
        # Fetch from Stellar Horizon
        ledger_data = fetch_stellar_ledger(ledger_id)
        normalized = normalizer.normalize_ledger(ledger_data)
        
        cache[ledger_id] = normalized
        return normalized
    
    # Custom processing function
    def process_with_validation(ledger_id, ledger_data):
        """Process ledger with validation."""
        try:
            # Validate ledger data
            validate_ledger_integrity(ledger_data)
            
            # Extract features
            features = extract_ledger_features(ledger_data)
            
            # Store in database
            store_ledger_features(ledger_id, features)
            
            # Update analytics
            update_analytics_dashboard(ledger_data)
            
        except Exception as e:
            log_processing_error(ledger_id, e)
            raise
    
    # Run ingestion with custom processing
    result = service.ingest(
        start_ledger=1000000,
        end_ledger=1000050,
        fetch_fn=fetch_with_cache,
        process_fn=process_with_validation
    )
    
    return result

custom_data_processing()
```

## Graph Building Examples

### Temporal Graph Construction

```python
from astroml.graph import GraphBuilder, TemporalGraph
from astroml.features import TemporalFeatures
import pandas as pd

def build_temporal_graph():
    """Build a temporal graph with time-based features."""
    
    builder = GraphBuilder()
    
    # Build graph with different time windows
    time_windows = [1, 7, 30, 90]  # 1 day, 1 week, 1 month, 3 months
    
    graphs = {}
    for window in time_windows:
        graph = builder.build_snapshot(
            window_days=window,
            min_transactions=5,  # Minimum transactions per node
            include_temporal=True,
            weight_by_amount=True
        )
        
        graphs[f"{window}d"] = graph
        print(f"Built {window}day graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Create multi-scale features
    multi_scale_features = {}
    
    for node_id in graphs["1d"].nodes():
        features = {
            "degree_1d": graphs["1d"].degree(node_id),
            "degree_7d": graphs["7d"].degree(node_id),
            "degree_30d": graphs["30d"].degree(node_id),
            "degree_90d": graphs["90d"].degree(node_id),
            
            "clustering_1d": graphs["1d"].clustering_coefficient(node_id),
            "clustering_7d": graphs["7d"].clustering_coefficient(node_id),
            
            "temporal_patterns": extract_temporal_patterns(node_id, graphs)
        }
        
        multi_scale_features[node_id] = features
    
    return graphs, multi_scale_features

def extract_temporal_patterns(node_id, graphs):
    """Extract temporal patterns from multi-scale graphs."""
    patterns = {}
    
    # Degree growth rate
    degree_1d = graphs["1d"].degree(node_id)
    degree_7d = graphs["7d"].degree(node_id)
    degree_30d = graphs["30d"].degree(node_id)
    
    patterns["growth_rate_1w"] = (degree_7d - degree_1d) / max(degree_1d, 1)
    patterns["growth_rate_1m"] = (degree_30d - degree_7d) / max(degree_7d, 1)
    
    # Activity patterns
    patterns["activity_consistency"] = calculate_activity_consistency(node_id, graphs)
    
    return patterns

graphs, features = build_temporal_graph()
```

### Weighted Graph Construction

```python
def build_weighted_transaction_graph():
    """Build a graph weighted by transaction amounts and frequencies."""
    
    builder = GraphBuilder()
    
    # Configure edge weights
    edge_weights = {
        "amount_weight": 0.7,    # Weight by transaction amount
        "frequency_weight": 0.3,  # Weight by transaction frequency
        "recency_weight": 0.2,     # Weight by recency (newer transactions weigh more)
        "temporal_decay": 0.95     # Temporal decay factor
    }
    
    graph = builder.build_snapshot(
        window_days=30,
        min_transactions=1,
        include_temporal=True,
        edge_weights=edge_weights
    )
    
    # Analyze weighted graph properties
    print(f"Weighted graph statistics:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Average weighted degree: {graph.average_weighted_degree():.2f}")
    print(f"  Graph density: {graph.density():.4f}")
    
    # Find high-weight edges (important transactions)
    high_weight_edges = graph.get_top_weighted_edges(k=10)
    
    print("\nTop 10 highest weight edges:")
    for i, (source, target, weight) in enumerate(high_weight_edges):
        print(f"  {i+1}. {source} -> {target}: {weight:.4f}")
    
    return graph

weighted_graph = build_weighted_transaction_graph()
```

### Community Detection Graph

```python
import networkx as nx
from astroml.graph import GraphBuilder

def build_community_graph():
    """Build graph optimized for community detection."""
    
    builder = GraphBuilder()
    
    # Build graph with community-focused features
    graph = builder.build_snapshot(
        window_days=90,  # Longer window for stable communities
        min_transactions=10,  # Higher minimum for meaningful communities
        include_temporal=True,
        weight_by_amount=True
    )
    
    # Convert to NetworkX for community detection
    nx_graph = graph.to_networkx()
    
    # Detect communities using different algorithms
    communities = {}
    
    # Louvain method (fast, good for large graphs)
    import community as community_louvain
    louvain_communities = community_louvain.best_partition(nx_graph)
    communities["louvain"] = louvain_communities
    
    # Label propagation (very fast)
    label_communities = nx.community.label_propagation_communities(nx_graph)
    communities["label_propagation"] = {node: i for i, comm in enumerate(label_communities) for node in comm}
    
    # Analyze communities
    for method, comm_dict in communities.items():
        num_communities = len(set(comm_dict.values()))
        modularity = nx.community.modularity(nx_graph, [
            set([node for node, comm in comm_dict.items() if comm == comm_id])
            for comm_id in set(comm_dict.values())
        ])
        
        print(f"{method} communities:")
        print(f"  Number of communities: {num_communities}")
        print(f"  Modularity: {modularity:.4f}")
        
        # Find largest communities
        community_sizes = {}
        for node, comm_id in comm_dict.items():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        
        top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 communities: {[(cid, size) for cid, size in top_communities]}")
    
    return graph, communities

community_graph, communities = build_community_graph()
```

## Machine Learning Examples

### Fraud Detection Model

```python
from astroml.models import GCN, Trainer, AnomalyDetector
from astroml.features import AccountFeatures, TransactionFeatures
from astroml.training import Experiment
import torch

def build_fraud_detection_model():
    """Build a comprehensive fraud detection model."""
    
    # Build training graph
    builder = GraphBuilder()
    train_graph = builder.build_snapshot(
        window_days=60,
        min_transactions=5,
        include_temporal=True
    )
    
    # Extract rich features
    feature_extractor = AccountFeatures()
    node_features = feature_extractor.extract_all_features(train_graph)
    
    # Create labels (assuming we have some labeled fraud data)
    labels = create_fraud_labels(train_graph)
    
    # Configure model
    model = GCN(
        input_dim=node_features.shape[1],
        hidden_dims=[256, 128, 64],
        output_dim=2,  # Binary classification: fraud vs legitimate
        dropout=0.3
    )
    
    # Train with fraud detection specific settings
    trainer = Trainer(model, learning_rate=0.001)
    
    # Use class weighting for imbalanced dataset
    class_weights = calculate_class_weights(labels)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    metrics = trainer.train(
        graph=train_graph,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        criterion=criterion
    )
    
    # Evaluate on test set
    test_graph = builder.build_snapshot(
        window_days=30,
        min_transactions=5,
        include_temporal=True
    )
    
    evaluation_metrics = trainer.evaluate(test_graph)
    
    print(f"Fraud Detection Model Results:")
    print(f"  Accuracy: {evaluation_metrics.accuracy:.4f}")
    print(f"  Precision: {evaluation_metrics.precision:.4f}")
    print(f"  Recall: {evaluation_metrics.recall:.4f}")
    print(f"  F1-Score: {evaluation_metrics.f1_score:.4f}")
    print(f"  AUC-ROC: {evaluation_metrics.auc_roc:.4f}")
    
    return model, metrics, evaluation_metrics

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset."""
    n_samples = len(labels)
    n_classes = len(torch.unique(labels))
    
    class_counts = torch.bincount(labels)
    class_weights = n_samples / (n_classes * class_counts)
    
    return class_weights.float()

def create_fraud_labels(graph):
    """Create fraud labels (example implementation)."""
    # In practice, this would use known fraud cases
    # For demonstration, we'll create synthetic labels
    import random
    
    labels = []
    for node in graph.nodes():
        # Simulate fraud probability based on graph properties
        degree = graph.degree(node)
        clustering = graph.clustering_coefficient(node)
        
        # Higher degree and lower clustering might indicate fraud
        fraud_prob = (degree / 100) * (1 - clustering)
        is_fraud = 1 if random.random() < fraud_prob * 0.1 else 0  # 10% base rate
        
        labels.append(is_fraud)
    
    return torch.tensor(labels)

model, training_metrics, eval_metrics = build_fraud_detection_model()
```

### Anomaly Detection System

```python
from astroml.models import AnomalyDetector, TemporalGCN
from astroml.ingestion import HorizonStream
import asyncio

async def build_anomaly_detection_system():
    """Build a real-time anomaly detection system."""
    
    # Build temporal graph for anomaly detection
    builder = GraphBuilder()
    graph = builder.build_snapshot(
        window_days=7,
        min_transactions=3,
        include_temporal=True
    )
    
    # Create temporal GNN for anomaly detection
    anomaly_model = TemporalGCN(
        input_dim=graph.node_features.shape[1],
        hidden_dims=[128, 64],
        output_dim=64,  # Embedding dimension
        temporal_dim=32
    )
    
    # Train anomaly detector
    detector = AnomalyDetector(
        model=anomaly_model,
        threshold=0.95,
        method="autoencoder"
    )
    
    # Train on historical data
    detector.fit(graph)
    
    # Set up real-time monitoring
    stream = HorizonStream("https://horizon.stellar.org")
    
    async def monitor_transactions():
        """Monitor transactions for anomalies in real-time."""
        
        async def process_transaction(tx):
            """Process individual transaction for anomalies."""
            # Update graph with new transaction
            updated_graph = update_graph_with_transaction(graph, tx)
            
            # Check for anomalies
            anomalies = detector.detect(updated_graph)
            
            if anomalies:
                for anomaly in anomalies:
                    await handle_anomaly(anomaly, tx)
        
        async for transaction in stream.stream_transactions([process_transaction]):
            pass  # Continuous monitoring
    
    # Start monitoring
    await monitor_transactions()

async def handle_anomaly(anomaly, transaction):
    """Handle detected anomaly."""
    print(f"🚨 ANOMALY DETECTED!")
    print(f"  Node: {anomaly.node_id}")
    print(f"  Score: {anomaly.score:.4f}")
    print(f"  Reason: {anomaly.reason}")
    print(f"  Transaction: {transaction.hash}")
    
    # Send alert
    await send_security_alert(anomaly, transaction)
    
    # Log for investigation
    await log_anomaly_for_investigation(anomaly, transaction)

# Run the anomaly detection system
asyncio.run(build_anomaly_detection_system())
```

### Multi-Model Ensemble

```python
from astroml.models import GCN, GraphSAGE, GAT, Trainer
from sklearn.ensemble import VotingClassifier
import numpy as np

def build_ensemble_model():
    """Build an ensemble of different GNN models."""
    
    # Prepare data
    builder = GraphBuilder()
    graph = builder.build_snapshot(window_days=30, include_temporal=True)
    
    # Create different models
    models = {
        "GCN": GCN(
            input_dim=graph.node_features.shape[1],
            hidden_dims=[128, 64],
            output_dim=3  # Multi-class classification
        ),
        "GraphSAGE": GraphSAGE(
            input_dim=graph.node_features.shape[1],
            hidden_dims=[128, 64],
            output_dim=3
        ),
        "GAT": GAT(
            input_dim=graph.node_features.shape[1],
            hidden_dims=[128, 64],
            output_dim=3,
            num_heads=4
        )
    }
    
    # Train each model
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        trainer = Trainer(model, learning_rate=0.001)
        metrics = trainer.train(graph, epochs=50)
        
        trained_models[name] = model
        predictions[name] = trainer.predict(graph)
        
        print(f"  {name} accuracy: {metrics.val_accuracy[-1]:.4f}")
    
    # Create ensemble predictions
    ensemble_predictions = ensemble_predict(predictions)
    
    # Evaluate ensemble
    ensemble_accuracy = evaluate_ensemble(ensemble_predictions, graph)
    
    print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
    
    return trained_models, ensemble_predictions

def ensemble_predict(predictions):
    """Create ensemble predictions from multiple models."""
    # Average the probabilities
    ensemble_probs = np.mean(list(predictions.values()), axis=0)
    return np.argmax(ensemble_probs, axis=1)

def evaluate_ensemble(predictions, graph):
    """Evaluate ensemble performance."""
    # Get ground truth labels (assuming they exist)
    labels = graph.node_labels
    
    accuracy = np.mean(predictions == labels.numpy())
    return accuracy

ensemble_models, ensemble_preds = build_ensemble_model()
```

## Research Workflows

### Reproducible Experiments

```python
from astroml.training import Experiment
from astroml.models.config import GCNConfig
import json
import datetime

def fraud_detection_experiment():
    """Run a reproducible fraud detection experiment."""
    
    # Define experiment configuration
    config = {
        "experiment_name": "fraud_detection_gcn_v1",
        "dataset": "stellar_mainnet_2024_q1",
        "model_type": "GCN",
        "model_config": GCNConfig(
            input_dim=128,
            hidden_dims=[256, 128, 64],
            output_dim=2,
            dropout=0.3,
            learning_rate=0.001
        ).__dict__,
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "validation_split": 0.2,
            "early_stopping": True,
            "patience": 10
        },
        "graph_config": {
            "window_days": 60,
            "min_transactions": 5,
            "include_temporal": True,
            "weight_by_amount": True
        },
        "feature_config": {
            "account_features": True,
            "transaction_features": True,
            "temporal_features": True,
            "graph_features": True
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1", "auc_roc"],
            "cross_validation": True,
            "cv_folds": 5
        }
    }
    
    # Create experiment
    experiment = Experiment(
        name=config["experiment_name"],
        config=config,
        seed=42
    )
    
    # Prepare data
    builder = GraphBuilder()
    train_graph = builder.build_snapshot(**config["graph_config"])
    test_graph = builder.build_snapshot(window_days=30)  # Different window for test
    
    # Run experiment
    results = experiment.run(train_graph, test_graph)
    
    # Save results
    experiment.save_results(results)
    
    # Generate report
    generate_experiment_report(results, config)
    
    return results

def generate_experiment_report(results, config):
    """Generate a comprehensive experiment report."""
    
    report = {
        "experiment_name": config["experiment_name"],
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "results": {
            "training_metrics": {
                "final_train_loss": results.training_metrics.train_loss[-1],
                "final_val_loss": results.training_metrics.val_loss[-1],
                "final_train_accuracy": results.training_metrics.train_accuracy[-1],
                "final_val_accuracy": results.training_metrics.val_accuracy[-1],
                "epochs_trained": results.training_metrics.epochs_trained,
                "training_time": results.training_metrics.training_time
            },
            "evaluation_metrics": {
                "accuracy": results.evaluation_metrics.accuracy,
                "precision": results.evaluation_metrics.precision,
                "recall": results.evaluation_metrics.recall,
                "f1_score": results.evaluation_metrics.f1_score,
                "auc_roc": results.evaluation_metrics.auc_roc
            }
        }
    }
    
    # Save report
    with open(f"reports/{config['experiment_name']}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Experiment report saved to reports/{config['experiment_name']}_report.json")
    
    return report

results = fraud_detection_experiment()
```

### Hyperparameter Optimization

```python
from astroml.models import HyperparameterSearch
from astroml.models.config import GCNConfig
import optuna

def optimize_hyperparameters():
    """Optimize hyperparameters using Optuna."""
    
    def objective(trial):
        """Objective function for hyperparameter optimization."""
        
        # Define hyperparameter search space
        hidden_dims = trial.suggest_categorical("hidden_dims", [
            [64, 32],
            [128, 64],
            [256, 128],
            [512, 256],
            [128, 64, 32]
        ])
        
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Create model with suggested parameters
        model = GCN(
            input_dim=128,  # Fixed based on our features
            hidden_dims=hidden_dims,
            output_dim=2,
            dropout=dropout
        )
        
        # Train model
        trainer = Trainer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Use cross-validation
        cv_scores = []
        for fold in range(5):
            # Create train/val split for this fold
            train_graph, val_graph = create_cv_split(fold)
            
            metrics = trainer.train(
                graph=train_graph,
                epochs=50,  # Fewer epochs for faster optimization
                validation_split=0.2
            )
            
            cv_scores.append(metrics.val_accuracy[-1])
        
        # Return mean validation accuracy
        return np.mean(cv_scores)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="gcn_hyperparameter_optimization"
    )
    
    # Optimize
    study.optimize(objective, n_trials=50)
    
    # Print results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print(f"  Params: {trial.params}")
    
    # Save best parameters
    best_params = trial.params
    with open("models/best_gcn_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    return study, best_params

def create_cv_split(fold):
    """Create cross-validation split for fold."""
    # Implementation depends on your data splitting strategy
    # This is a placeholder
    train_graph = get_train_graph_for_fold(fold)
    val_graph = get_val_graph_for_fold(fold)
    return train_graph, val_graph

study, best_params = optimize_hyperparameters()
```

### Ablation Studies

```python
def ablation_study():
    """Perform ablation study to understand feature importance."""
    
    # Define different feature configurations
    feature_configs = {
        "full": {
            "account_features": True,
            "transaction_features": True,
            "temporal_features": True,
            "graph_features": True
        },
        "no_temporal": {
            "account_features": True,
            "transaction_features": True,
            "temporal_features": False,
            "graph_features": True
        },
        "no_graph": {
            "account_features": True,
            "transaction_features": True,
            "temporal_features": True,
            "graph_features": False
        },
        "account_only": {
            "account_features": True,
            "transaction_features": False,
            "temporal_features": False,
            "graph_features": False
        },
        "transaction_only": {
            "account_features": False,
            "transaction_features": True,
            "temporal_features": False,
            "graph_features": False
        }
    }
    
    results = {}
    
    for config_name, feature_config in feature_configs.items():
        print(f"Running ablation study for: {config_name}")
        
        # Build graph with specific features
        builder = GraphBuilder()
        graph = builder.build_snapshot(
            window_days=30,
            feature_config=feature_config
        )
        
        # Train model
        model = GCN(
            input_dim=graph.node_features.shape[1],
            hidden_dims=[128, 64],
            output_dim=2
        )
        
        trainer = Trainer(model)
        metrics = trainer.train(graph, epochs=50)
        
        results[config_name] = {
            "accuracy": metrics.val_accuracy[-1],
            "feature_count": graph.node_features.shape[1],
            "config": feature_config
        }
    
    # Analyze results
    print("\nAblation Study Results:")
    for config_name, result in results.items():
        print(f"{config_name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Features: {result['feature_count']}")
    
    # Find most important features
    baseline = results["full"]["accuracy"]
    
    print("\nFeature Importance (impact on accuracy):")
    for config_name, result in results.items():
        if config_name != "full":
            impact = baseline - result["accuracy"]
            print(f"  Removing {config_name}: -{impact:.4f} ({impact/baseline*100:.1f}% decrease)")
    
    return results

ablation_results = ablation_study()
```

## Production Integration

### Docker Deployment

```dockerfile
# Dockerfile for AstroML production deployment
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY astroml/ ./astroml/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 astroml
USER astroml

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import astroml; print('OK')" || exit 1

# Start application
CMD ["python", "-m", "astroml.api.server"]
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: astroml-api
  labels:
    app: astroml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: astroml-api
  template:
    metadata:
      labels:
        app: astroml-api
    spec:
      containers:
      - name: astroml-api
        image: astroml:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: astroml-secrets
              key: database-url
        - name: STELLAR_NETWORK
          value: "mainnet"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: astroml-api-service
spec:
  selector:
    app: astroml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Monitoring and Alerting

```python
from astroml.monitoring import MetricsCollector, AlertManager
import prometheus_client as prom

class ProductionMonitoring:
    """Production monitoring and alerting system."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # Prometheus metrics
        self.ingestion_rate = prom.Counter('astroml_ingestion_total', 'Total ingested ledgers')
        self.prediction_rate = prom.Counter('astroml_predictions_total', 'Total predictions made')
        self.anomaly_rate = prom.Counter('astroml_anomalies_total', 'Total anomalies detected')
        self.model_accuracy = prom.Gauge('astroml_model_accuracy', 'Current model accuracy')
        
        # Register metrics
        prom.REGISTRY.register(self.ingestion_rate)
        prom.REGISTRY.register(self.prediction_rate)
        prom.REGISTRY.register(self.anomaly_rate)
        prom.REGISTRY.register(self.model_accuracy)
    
    async def monitor_ingestion(self, ingestion_service):
        """Monitor data ingestion performance."""
        while True:
            try:
                # Get ingestion stats
                stats = ingestion_service.get_stats()
                
                # Update metrics
                self.ingestion_rate.inc(stats.processed_count)
                
                # Check for issues
                if stats.error_rate > 0.05:  # 5% error rate threshold
                    await self.alert_manager.send_alert(
                        "High ingestion error rate",
                        f"Error rate: {stats.error_rate:.2%}",
                        severity="warning"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                await self.alert_manager.send_alert(
                    "Monitoring error",
                    f"Error in ingestion monitoring: {str(e)}",
                    severity="critical"
                )
                await asyncio.sleep(60)
    
    async def monitor_model_performance(self, model, test_graph):
        """Monitor model performance and drift."""
        while True:
            try:
                # Evaluate model
                trainer = Trainer(model)
                metrics = trainer.evaluate(test_graph)
                
                # Update metrics
                self.model_accuracy.set(metrics.accuracy)
                
                # Check for performance degradation
                if metrics.accuracy < 0.8:  # 80% accuracy threshold
                    await self.alert_manager.send_alert(
                        "Model performance degradation",
                        f"Accuracy dropped to {metrics.accuracy:.4f}",
                        severity="warning"
                    )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                await self.alert_manager.send_alert(
                    "Model monitoring error",
                    f"Error in model monitoring: {str(e)}",
                    severity="critical"
                )
                await asyncio.sleep(300)

# Production monitoring setup
monitoring = ProductionMonitoring()

# Start monitoring tasks
asyncio.create_task(monitoring.monitor_ingestion(ingestion_service))
asyncio.create_task(monitoring.monitor_model_performance(model, test_graph))
```

## Advanced Patterns

### Custom Loss Functions

```python
import torch
import torch.nn as nn

class FraudDetectionLoss(nn.Module):
    """Custom loss function for fraud detection."""
    
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # Weight for normal transactions
        self.beta = beta    # Weight for fraud transactions
        self.focal_gamma = 2.0  # Focal loss parameter
    
    def forward(self, predictions, targets):
        """Calculate custom loss for fraud detection."""
        
        # Standard cross-entropy
        ce_loss = nn.CrossEntropyLoss()(predictions, targets)
        
        # Focal loss for handling class imbalance
        focal_loss = self.focal_loss(predictions, targets)
        
        # Weighted loss to emphasize fraud detection
        weights = torch.where(targets == 1, self.beta, self.alpha)
        weighted_loss = (ce_loss * weights).mean()
        
        # Combine losses
        total_loss = 0.7 * weighted_loss + 0.3 * focal_loss
        
        return total_loss
    
    def focal_loss(self, predictions, targets):
        """Calculate focal loss."""
        ce_loss = nn.CrossEntropyLoss(reduction='none')(predictions, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

# Usage in training
custom_loss = FraudDetectionLoss(alpha=0.6, beta=0.4)
trainer = Trainer(model, criterion=custom_loss)
```

### Multi-Task Learning

```python
class MultiTaskGCN(nn.Module):
    """Multi-task GCN for simultaneous fraud detection and account classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        fraud_output_dim: int = 2,
        classification_output_dim: int = 5,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Shared layers
        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(GCNConv(input_dim, hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.shared_layers.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
        
        # Task-specific heads
        self.fraud_head = GCNConv(hidden_dims[-1], fraud_output_dim)
        self.classification_head = GCNConv(hidden_dims[-1], classification_output_dim)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """Forward pass with multi-task outputs."""
        
        # Shared representation
        for conv in self.shared_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Task-specific outputs
        fraud_output = self.fraud_head(x, edge_index)
        classification_output = self.classification_head(x, edge_index)
        
        return {
            "fraud": F.log_softmax(fraud_output, dim=1),
            "classification": F.log_softmax(classification_output, dim=1)
        }

# Multi-task training
class MultiTaskTrainer:
    """Trainer for multi-task models."""
    
    def __init__(self, model, task_weights=None):
        self.model = model
        self.task_weights = task_weights or {"fraud": 0.7, "classification": 0.3}
    
    def train(self, graph, epochs=100):
        """Train multi-task model."""
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            self.model.train()
            
            # Forward pass
            outputs = self.model(graph.node_features, graph.edge_index)
            
            # Calculate losses for each task
            fraud_loss = nn.NLLLoss()(outputs["fraud"], graph.fraud_labels)
            classification_loss = nn.NLLLoss()(outputs["classification"], graph.classification_labels)
            
            # Weighted total loss
            total_loss = (
                self.task_weights["fraud"] * fraud_loss +
                self.task_weights["classification"] * classification_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Total Loss = {total_loss.item():.4f}")
```

---

This comprehensive usage guide provides practical examples for using AstroML in various scenarios, from basic setup to advanced research workflows and production deployment. Each example is designed to be copy-paste ready and includes best practices for error handling and performance optimization.
