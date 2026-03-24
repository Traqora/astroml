"""Example usage of calibration curve visualization for fraud scores.

This example demonstrates how to use the calibration analysis tools
to evaluate fraud detection models in the AstroML framework.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from astroml.validation.calibration import (
    CalibrationAnalyzer,
    create_sample_fraud_data
)


def create_realistic_fraud_models() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Create realistic fraud detection model outputs for comparison.
    
    Returns:
        Dictionary of model_name -> (y_true, y_prob)
    """
    np.random.seed(42)
    
    models = {}
    
    # Model 1: Well-calibrated baseline model
    y_true1, y_prob1 = create_sample_fraud_data(n_samples=2000, fraud_rate=0.08)
    models['Baseline Model'] = (y_true1, y_prob1)
    
    # Model 2: Overconfident model (common issue)
    y_true2, y_prob2 = create_sample_fraud_data(n_samples=2000, fraud_rate=0.08)
    # Make predictions more extreme (overconfident)
    y_prob2 = np.power(y_prob2, 0.7)  # Push probabilities toward 0 and 1
    y_prob2 = np.clip(y_prob2, 0.01, 0.99)
    models['Overconfident Model'] = (y_true2, y_prob2)
    
    # Model 3: Underconfident model
    y_true3, y_prob3 = create_sample_fraud_data(n_samples=2000, fraud_rate=0.08)
    # Make predictions more conservative (underconfident)
    y_prob3 = np.power(y_prob3, 1.5)  # Push probabilities toward 0.5
    y_prob3 = np.clip(y_prob3, 0.01, 0.99)
    models['Underconfident Model'] = (y_true3, y_prob3)
    
    # Model 4: Poorly calibrated model
    y_true4, y_prob4 = create_sample_fraud_data(n_samples=2000, fraud_rate=0.08)
    # Add systematic bias
    y_prob4 = y_prob4 * 0.7 + 0.15  # Shift predictions upward
    y_prob4 = np.clip(y_prob4, 0.01, 0.99)
    models['Poorly Calibrated Model'] = (y_true4, y_prob4)
    
    return models


def demonstrate_single_model_calibration():
    """Demonstrate calibration analysis for a single model."""
    print("=" * 60)
    print("Single Model Calibration Analysis")
    print("=" * 60)
    
    # Create sample data
    y_true, y_prob = create_sample_fraud_data(n_samples=5000, fraud_rate=0.12)
    
    # Initialize analyzer
    analyzer = CalibrationAnalyzer(n_bins=15, strategy='quantile')
    
    # Generate calibration plot
    fig = analyzer.plot_calibration_curve(
        y_true, y_prob, 
        model_name="Fraud Detection Model",
        figsize=(15, 10)
    )
    
    # Generate detailed report
    report = analyzer.generate_calibration_report(
        y_true, y_prob, 
        model_name="Fraud Detection Model"
    )
    
    print("\nCalibration Report:")
    print(report)
    
    # Save the plot
    fig.savefig('examples/single_model_calibration.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'single_model_calibration.png'")
    
    plt.show()


def demonstrate_multi_model_comparison():
    """Demonstrate calibration comparison across multiple models."""
    print("\n" + "=" * 60)
    print("Multi-Model Calibration Comparison")
    print("=" * 60)
    
    # Create multiple models
    models_data = create_realistic_fraud_models()
    
    # Initialize analyzer
    analyzer = CalibrationAnalyzer(n_bins=12, strategy='uniform')
    
    # Generate comparison plot
    fig = analyzer.plot_multiple_models(
        models_data,
        figsize=(16, 12)
    )
    
    # Generate individual reports for each model
    print("\nIndividual Model Reports:")
    print("-" * 40)
    
    for model_name, (y_true, y_prob) in models_data.items():
        print(f"\n{model_name}:")
        metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
        
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  Overconfidence: {metrics['overconfidence']:.4f}")
        print(f"  Underconfidence: {metrics['underconfidence']:.4f}")
        
        # Quick interpretation
        if metrics['overconfidence'] > 0.05:
            print("  → Model is OVERCONFIDENT")
        elif metrics['underconfidence'] > 0.05:
            print("  → Model is UNDERCONFIDENT")
        else:
            print("  → Model is reasonably calibrated")
    
    # Save the comparison plot
    fig.savefig('examples/multi_model_calibration.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'multi_model_calibration.png'")
    
    plt.show()


def demonstrate_calibration_improvement():
    """Demonstrate calibration improvement techniques."""
    print("\n" + "=" * 60)
    print("Calibration Improvement Demonstration")
    print("=" * 60)
    
    # Create poorly calibrated model
    y_true, y_prob_poor = create_sample_fraud_data(n_samples=3000, fraud_rate=0.1)
    
    # Make it poorly calibrated
    y_prob_poor = np.clip(y_prob_poor * 0.6 + 0.2, 0.01, 0.99)
    
    # Apply simple temperature scaling (calibration improvement)
    temperature = 1.5  # Temperature > 1 makes predictions less extreme
    y_prob_calibrated = 1 / (1 + np.exp((np.log(y_prob_poor / (1 - y_prob_poor)) / temperature)))
    
    # Compare before and after calibration
    models_data = {
        'Before Calibration': (y_true, y_prob_poor),
        'After Temperature Scaling': (y_true, y_prob_calibrated)
    }
    
    analyzer = CalibrationAnalyzer(n_bins=10)
    
    # Generate comparison
    fig = analyzer.plot_multiple_models(models_data, figsize=(14, 10))
    
    # Show improvement metrics
    print("\nCalibration Improvement Metrics:")
    print("-" * 40)
    
    for model_name, (y_true, y_prob) in models_data.items():
        metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
        print(f"\n{model_name}:")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
    
    # Calculate improvement
    metrics_before = analyzer.compute_calibration_metrics(y_true, y_prob_poor)
    metrics_after = analyzer.compute_calibration_metrics(y_true, y_prob_calibrated)
    
    ece_improvement = (metrics_before['ece'] - metrics_after['ece']) / metrics_before['ece'] * 100
    brier_improvement = (metrics_before['brier_score'] - metrics_after['brier_score']) / metrics_before['brier_score'] * 100
    
    print(f"\nImprovement:")
    print(f"  ECE Improvement: {ece_improvement:.1f}%")
    print(f"  Brier Score Improvement: {brier_improvement:.1f}%")
    
    # Save the plot
    fig.savefig('examples/calibration_improvement.png', dpi=300, bbox_inches='tight')
    print("\nImprovement plot saved as 'calibration_improvement.png'")
    
    plt.show()


def demonstrate_threshold_optimization():
    """Demonstrate threshold optimization based on calibration."""
    print("\n" + "=" * 60)
    print("Threshold Optimization Based on Calibration")
    print("=" * 60)
    
    # Create sample data
    y_true, y_prob = create_sample_fraud_data(n_samples=5000, fraud_rate=0.08)
    
    analyzer = CalibrationAnalyzer(n_bins=20)
    
    # Test different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    results = {
        'threshold': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'calibration_error': []
    }
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate calibration error for predictions above threshold
        mask = y_prob >= threshold
        if np.sum(mask) > 0:
            pred_mean = np.mean(y_prob[mask])
            true_rate = np.mean(y_true[mask])
            calibration_error = abs(pred_mean - true_rate)
        else:
            calibration_error = 0
        
        results['threshold'].append(threshold)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        results['calibration_error'].append(calibration_error)
    
    # Create threshold optimization plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Threshold Optimization Analysis', fontsize=16, fontweight='bold')
    
    # Precision-Recall curve
    ax1 = axes[0, 0]
    ax1.plot(results['recall'], results['precision'], 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.grid(True, alpha=0.3)
    
    # F1 Score vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(results['threshold'], results['f1'], 'g-', linewidth=2, marker='s')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Threshold')
    ax2.grid(True, alpha=0.3)
    
    # Calibration Error vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(results['threshold'], results['calibration_error'], 'r-', linewidth=2, marker='^')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Calibration Error')
    ax3.set_title('Calibration Error vs Threshold')
    ax3.grid(True, alpha=0.3)
    
    # Combined metrics
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(results['threshold'], results['f1'], 'g-', linewidth=2, label='F1 Score')
    line2 = ax4_twin.plot(results['threshold'], results['calibration_error'], 'r-', linewidth=2, label='Calibration Error')
    
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('F1 Score', color='g')
    ax4_twin.set_ylabel('Calibration Error', color='r')
    ax4.tick_params(axis='y', labelcolor='g')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    ax4.set_title('Combined Metrics')
    ax4.grid(True, alpha=0.3)
    
    # Find optimal threshold (max F1 with reasonable calibration)
    f1_array = np.array(results['f1'])
    cal_error_array = np.array(results['calibration_error'])
    
    # Filter for reasonable calibration (error < 0.1)
    reasonable_mask = cal_error_array < 0.1
    if np.any(reasonable_mask):
        optimal_idx = np.argmax(f1_array[reasonable_mask])
        optimal_threshold = results['threshold'][reasonable_mask][optimal_idx]
        optimal_f1 = results['f1'][reasonable_mask][optimal_idx]
        optimal_cal_error = results['calibration_error'][reasonable_mask][optimal_idx]
    else:
        # Fallback to max F1
        optimal_idx = np.argmax(f1_array)
        optimal_threshold = results['threshold'][optimal_idx]
        optimal_f1 = results['f1'][optimal_idx]
        optimal_cal_error = results['calibration_error'][optimal_idx]
    
    print(f"\nOptimal Threshold Analysis:")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    print(f"  F1 Score: {optimal_f1:.3f}")
    print(f"  Calibration Error: {optimal_cal_error:.3f}")
    
    plt.tight_layout()
    fig.savefig('examples/threshold_optimization.png', dpi=300, bbox_inches='tight')
    print("\nThreshold optimization plot saved as 'threshold_optimization.png'")
    
    plt.show()


def main():
    """Run all calibration analysis examples."""
    print("AstroML Calibration Analysis Examples")
    print("=====================================")
    
    # Create examples directory
    import os
    os.makedirs('examples', exist_ok=True)
    
    # Run demonstrations
    demonstrate_single_model_calibration()
    demonstrate_multi_model_comparison()
    demonstrate_calibration_improvement()
    demonstrate_threshold_optimization()
    
    print("\n" + "=" * 60)
    print("All calibration analysis examples completed!")
    print("Check the 'examples/' directory for generated plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
