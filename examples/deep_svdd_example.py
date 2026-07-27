"""Example usage of Deep SVDD for unsupervised fraud detection.

This example demonstrates how to use Deep SVDD for fraud detection
when labeled fraud data is scarce or unavailable.

This example can be run from any working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import classification_report, confusion_matrix

# Add the parent directory to the path to import astroml
# This allows the example to run from any working directory
script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from astroml.models.deep_svdd_trainer import FraudDetectionDeepSVDD


def create_synthetic_fraud_data(
    n_normal: int = 5000,
    n_fraud: int = 200,
    n_features: int = 12,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic transaction data for fraud detection.
    
    Args:
        n_normal: Number of normal transactions
        n_fraud: Number of fraudulent transactions
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        Tuple of (X, y) where X is features and y is labels (0=normal, 1=fraud)
    """
    np.random.seed(random_state)
    
    # Normal transactions - clustered patterns
    n_clusters = 3
    normal_data, _ = make_blobs(
        n_samples=n_normal,
        centers=n_clusters,
        n_features=n_features,
        cluster_std=1.0,
        random_state=random_state
    )
    
    # Add realistic transaction patterns
    # Feature 0: Transaction amount (log-normal distribution)
    normal_data[:, 0] = np.abs(np.random.lognormal(mean=3, sigma=1, size=n_normal))
    
    # Feature 1: Time of day (0-24 hours)
    normal_data[:, 1] = np.random.uniform(0, 24, n_normal)
    
    # Feature 2: Day of week (0-6)
    normal_data[:, 2] = np.random.randint(0, 7, n_normal)
    
    # Feature 3: Merchant category (0-9)
    normal_data[:, 3] = np.random.randint(0, 10, n_normal)
    
    # Feature 4-11: Other behavioral features
    for i in range(4, n_features):
        normal_data[:, i] = np.random.normal(0, 1, n_normal)
    
    # Fraudulent transactions - different patterns
    fraud_data = np.zeros((n_fraud, n_features))
    
    # Higher amounts for fraud
    fraud_data[:, 0] = np.abs(np.random.lognormal(mean=5, sigma=1.5, size=n_fraud))
    
    # Unusual timing patterns
    fraud_data[:, 1] = np.random.choice([2, 3, 4, 22, 23], size=n_fraud)  # Unusual hours
    
    # Random days and categories
    fraud_data[:, 2] = np.random.randint(0, 7, n_fraud)
    fraud_data[:, 3] = np.random.randint(0, 10, n_fraud)
    
    # Different behavioral patterns
    for i in range(4, n_features):
        if i % 2 == 0:
            fraud_data[:, i] = np.random.normal(2, 1.5, n_fraud)  # Higher values
        else:
            fraud_data[:, i] = np.random.normal(-2, 1.5, n_fraud)  # Lower values
    
    # Combine data
    X = np.vstack([normal_data, fraud_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    return X, y


def demonstrate_basic_usage():
    """Demonstrate basic Deep SVDD usage for fraud detection."""
    print("=" * 60)
    print("Basic Deep SVDD for Fraud Detection")
    print("=" * 60)
    
    # Create synthetic data
    X, y = create_synthetic_fraud_data(n_normal=3000, n_fraud=150)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Fraud rate: {np.mean(y):.3f} ({np.mean(y)*100:.1f}%)")
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train model
    print("\nTraining Deep SVDD model...")
    detector = FraudDetectionDeepSVDD(
        input_dim=X.shape[1],
        hidden_dims=[256, 128, 64, 32],
        nu=0.05,  # Expect 5% anomalies
        dropout=0.2
    )
    
    # Train on all data (unsupervised)
    detector.fit(
        X_train,
        epochs=50,
        validation_split=0.2,
        lr=0.001,
        loss_type='svdd',
        scheduler_type='cosine'
    )
    
    # Evaluate
    print("\nEvaluating model...")
    results = detector.evaluate_fraud_detection(X_test, y_test)
    
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
    print(f"AUC-PR: {results['auc_pr']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    
    return detector, results


def demonstrate_advanced_training():
    """Demonstrate advanced training strategies."""
    print("\n" + "=" * 60)
    print("Advanced Deep SVDD Training Strategies")
    print("=" * 60)
    
    # Create more challenging data
    X, y = create_synthetic_fraud_data(n_normal=4000, n_fraud=300)
    
    # Test different loss functions
    loss_types = ['svdd', 'soft_boundary', 'robust']
    results = {}
    
    for loss_type in loss_types:
        print(f"\nTraining with {loss_type} loss...")
        
        detector = FraudDetectionDeepSVDD(
            input_dim=X.shape[1],
            hidden_dims=[128, 64, 32],
            nu=0.07
        )
        
        detector.fit(
            X,
            epochs=30,
            validation_split=0.2,
            lr=0.001,
            loss_type=loss_type,
            scheduler_type='plateau'
        )
        
        # Evaluate
        eval_results = detector.evaluate_fraud_detection(X, y)
        results[loss_type] = eval_results
        
        print(f"  AUC-ROC: {eval_results['auc_roc']:.4f}")
        print(f"  AUC-PR: {eval_results['auc_pr']:.4f}")
        print(f"  F1: {eval_results['f1']:.4f}")
    
    # Compare results
    print("\nLoss Function Comparison:")
    print("-" * 40)
    for loss_type, metrics in results.items():
        print(f"{loss_type:15}: ROC={metrics['auc_roc']:.3f}, PR={metrics['auc_pr']:.3f}, F1={metrics['f1']:.3f}")
    
    return results


def main():
    """Run Deep SVDD examples."""
    print("Deep SVDD for Unsupervised Fraud Detection Examples")
    print("=" * 60)
    
    # Run demonstrations
    detector, basic_results = demonstrate_basic_usage()
    advanced_results = demonstrate_advanced_training()
    
    print("\n" + "=" * 60)
    print("Summary of Results")
    print("=" * 60)
    print(f"Basic Model AUC-ROC: {basic_results['auc_roc']:.4f}")
    print(f"Best Loss Function: {max(advanced_results.keys(), key=lambda k: advanced_results[k]['auc_roc'])}")
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main()
