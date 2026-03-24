"""Calibration curve visualization and analysis for fraud scores.

This module provides tools to assess the reliability of fraud detection
models by comparing predicted probabilities against observed outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from scipy import stats
import warnings

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CalibrationAnalyzer:
    """Comprehensive calibration analysis for fraud detection models."""
    
    def __init__(self, n_bins: int = 10, strategy: str = 'uniform'):
        """
        Initialize calibration analyzer.
        
        Args:
            n_bins: Number of bins for calibration curves
            strategy: Binning strategy ('uniform', 'quantile')
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.calibration_data = {}
        self.metrics = {}
        
    def compute_calibration_curve(self, 
                                 y_true: np.ndarray, 
                                 y_prob: np.ndarray,
                                 sample_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve for fraud scores.
        
        Args:
            y_true: True binary labels (0 = legitimate, 1 = fraudulent)
            y_prob: Predicted probabilities of fraud
            sample_weight: Optional sample weights
            
        Returns:
            Tuple of (fraction_of_positives, mean_predicted_probability)
        """
        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have the same length")
        
        if not np.all((y_prob >= 0) & (y_prob <= 1)):
            raise ValueError("y_prob must be between 0 and 1")
        
        fraction_of_positives, mean_predicted_probability = calibration_curve(
            y_true, y_prob, n_bins=self.n_bins, strategy=self.strategy,
            sample_weight=sample_weight
        )
        
        # Store calibration data
        self.calibration_data = {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_probability': mean_predicted_probability,
            'n_bins': len(fraction_of_positives)
        }
        
        return fraction_of_positives, mean_predicted_probability
    
    def compute_calibration_metrics(self,
                                  y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute comprehensive calibration metrics.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary of calibration metrics
        """
        metrics = {}
        
        # Brier Score (lower is better)
        metrics['brier_score'] = brier_score_loss(
            y_true, y_prob, sample_weight=sample_weight, pos_label=1
        )
        
        # Log Loss (lower is better)
        try:
            metrics['log_loss'] = log_loss(
                y_true, y_prob, sample_weight=sample_weight, normalize=True
            )
        except ValueError:
            # Handle edge cases where y_prob contains 0 or 1
            y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
            metrics['log_loss'] = log_loss(
                y_true, y_prob_clipped, sample_weight=sample_weight, normalize=True
            )
        
        # Expected Calibration Error (ECE)
        metrics['ece'] = self._compute_ece(y_true, y_prob, sample_weight)
        
        # Maximum Calibration Error (MCE)
        metrics['mce'] = self._compute_mce(y_true, y_prob, sample_weight)
        
        # Adaptive Calibration Error (ACE)
        metrics['ace'] = self._compute_ace(y_true, y_prob, sample_weight)
        
        # Overconfidence and Underconfidence
        metrics['overconfidence'] = self._compute_overconfidence(y_true, y_prob)
        metrics['underconfidence'] = self._compute_underconfidence(y_true, y_prob)
        
        # Sharpness (average prediction variance)
        metrics['sharpness'] = np.var(y_prob)
        
        self.metrics = metrics
        return metrics
    
    def _compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray, 
                    sample_weight: Optional[np.ndarray] = None) -> float:
        """Compute Expected Calibration Error."""
        if 'fraction_of_positives' not in self.calibration_data:
            self.compute_calibration_curve(y_true, y_prob, sample_weight)
        
        fraction_pos = self.calibration_data['fraction_of_positives']
        mean_pred = self.calibration_data['mean_predicted_probability']
        
        # Calculate bin weights
        if sample_weight is not None:
            # Weighted bin counts
            bin_counts = []
            for i in range(len(mean_pred)):
                mask = self._get_bin_mask(y_prob, i)
                bin_counts.append(np.sum(sample_weight[mask]))
        else:
            # Unweighted bin counts
            bin_counts = [len(y_prob[self._get_bin_mask(y_prob, i)]) 
                         for i in range(len(mean_pred))]
        
        bin_weights = np.array(bin_counts) / np.sum(bin_counts)
        
        # ECE = sum(|fraction_pos - mean_pred| * bin_weight)
        ece = np.sum(np.abs(fraction_pos - mean_pred) * bin_weights)
        return ece
    
    def _compute_mce(self, y_true: np.ndarray, y_prob: np.ndarray,
                    sample_weight: Optional[np.ndarray] = None) -> float:
        """Compute Maximum Calibration Error."""
        if 'fraction_of_positives' not in self.calibration_data:
            self.compute_calibration_curve(y_true, y_prob, sample_weight)
        
        fraction_pos = self.calibration_data['fraction_of_positives']
        mean_pred = self.calibration_data['mean_predicted_probability']
        
        mce = np.max(np.abs(fraction_pos - mean_pred))
        return mce
    
    def _compute_ace(self, y_true: np.ndarray, y_prob: np.ndarray,
                    sample_weight: Optional[np.ndarray] = None) -> float:
        """Compute Adaptive Calibration Error."""
        # Use quantile-based bins for ACE
        quantiles = np.quantile(y_prob, np.linspace(0, 1, self.n_bins + 1))
        
        ace = 0.0
        total_samples = len(y_true)
        
        for i in range(self.n_bins):
            mask = (y_prob >= quantiles[i]) & (y_prob < quantiles[i + 1])
            if np.sum(mask) > 0:
                bin_pred_prob = np.mean(y_prob[mask])
                bin_true_rate = np.mean(y_true[mask])
                bin_weight = np.sum(mask) / total_samples
                
                ace += np.abs(bin_pred_prob - bin_true_rate) * bin_weight
        
        return ace
    
    def _compute_overconfidence(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute overconfidence metric."""
        # Overconfidence = mean(max(p, 1-p)) - accuracy
        confidence = np.maximum(y_prob, 1 - y_prob)
        accuracy = np.mean((y_prob > 0.5) == y_true)
        overconfidence = np.mean(confidence) - accuracy
        return max(0, overconfidence)
    
    def _compute_underconfidence(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute underconfidence metric."""
        # Underconfidence = accuracy - mean(max(p, 1-p))
        confidence = np.maximum(y_prob, 1 - y_prob)
        accuracy = np.mean((y_prob > 0.5) == y_true)
        underconfidence = accuracy - np.mean(confidence)
        return max(0, underconfidence)
    
    def _get_bin_mask(self, y_prob: np.ndarray, bin_idx: int) -> np.ndarray:
        """Get mask for samples in a specific bin."""
        if self.strategy == 'uniform':
            bin_edges = np.linspace(0, 1, self.n_bins + 1)
            return (y_prob >= bin_edges[bin_idx]) & (y_prob < bin_edges[bin_idx + 1])
        else:  # quantile
            quantiles = np.quantile(y_prob, np.linspace(0, 1, self.n_bins + 1))
            return (y_prob >= quantiles[bin_idx]) & (y_prob < quantiles[bin_idx + 1])
    
    def plot_calibration_curve(self,
                             y_true: np.ndarray,
                             y_prob: np.ndarray,
                             model_name: str = "Model",
                             figsize: Tuple[int, int] = (12, 8),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive calibration curve visualization.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            model_name: Name of the model for labeling
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Compute calibration data
        fraction_pos, mean_pred = self.compute_calibration_curve(y_true, y_prob)
        metrics = self.compute_calibration_metrics(y_true, y_prob)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Calibration Analysis for {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Main calibration curve
        ax1 = axes[0, 0]
        ax1.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        ax1.plot(mean_pred, fraction_pos, 's-', label=model_name, linewidth=2, markersize=6)
        
        # Add confidence intervals
        n_samples_per_bin = []
        for i in range(len(mean_pred)):
            mask = self._get_bin_mask(y_prob, i)
            n_samples_per_bin.append(np.sum(mask))
        
        # Simple confidence intervals based on bin counts
        stderr = np.sqrt(fraction_pos * (1 - fraction_pos) / np.array(n_samples_per_bin))
        ax1.fill_between(mean_pred, fraction_pos - stderr, fraction_pos + stderr, 
                        alpha=0.2, color='blue')
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram of predictions
        ax2 = axes[0, 1]
        ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label='Legitimate', color='green')
        ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label='Fraudulent', color='red')
        ax2.set_xlabel('Predicted Fraud Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Reliability diagram with bin details
        ax3 = axes[1, 0]
        bin_counts = [np.sum(self._get_bin_mask(y_prob, i)) for i in range(len(mean_pred))]
        
        # Color bins by sample count
        colors = plt.cm.YlOrRd(np.array(bin_counts) / max(bin_counts))
        bars = ax3.bar(mean_pred, fraction_pos, width=1.0/self.n_bins, 
                      alpha=0.7, color=colors, edgecolor='black')
        
        ax3.plot([0, 1], [0, 1], 'k:', label='Perfect calibration')
        ax3.set_xlabel('Mean Predicted Probability')
        ax3.set_ylabel('Observed Fraud Rate')
        ax3.set_title('Reliability Diagram (colored by sample count)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for sample counts
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                                   norm=plt.Normalize(vmin=0, vmax=max(bin_counts)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax3, label='Sample Count per Bin')
        
        # 4. Metrics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create metrics text
        metrics_text = f"""
        Calibration Metrics:
        
        Brier Score: {metrics['brier_score']:.4f}
        Log Loss: {metrics['log_loss']:.4f}
        Expected Calibration Error (ECE): {metrics['ece']:.4f}
        Maximum Calibration Error (MCE): {metrics['mce']:.4f}
        Adaptive Calibration Error (ACE): {metrics['ace']:.4f}
        
        Confidence Metrics:
        Overconfidence: {metrics['overconfidence']:.4f}
        Underconfidence: {metrics['underconfidence']:.4f}
        Sharpness: {metrics['sharpness']:.4f}
        
        Sample Statistics:
        Total Samples: {len(y_true):,}
        Fraud Rate: {np.mean(y_true):.3f}
        Mean Prediction: {np.mean(y_prob):.3f}
        """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_models(self,
                           models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                           figsize: Tuple[int, int] = (15, 10),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare calibration curves for multiple models.
        
        Args:
            models_data: Dictionary of model_name -> (y_true, y_prob)
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Multi-Model Calibration Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
        
        # 1. Calibration curves comparison
        ax1 = axes[0, 0]
        ax1.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated', linewidth=2)
        
        for i, (model_name, (y_true, y_prob)) in enumerate(models_data.items()):
            fraction_pos, mean_pred = self.compute_calibration_curve(y_true, y_prob)
            ax1.plot(mean_pred, fraction_pos, 'o-', label=model_name, 
                    color=colors[i], linewidth=2, markersize=6)
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curves Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Metrics comparison
        ax2 = axes[0, 1]
        model_names = []
        brier_scores = []
        ece_scores = []
        
        for model_name, (y_true, y_prob) in models_data.items():
            model_names.append(model_name)
            metrics = self.compute_calibration_metrics(y_true, y_prob)
            brier_scores.append(metrics['brier_score'])
            ece_scores.append(metrics['ece'])
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax2.bar(x - width/2, brier_scores, width, label='Brier Score', alpha=0.7)
        ax2.bar(x + width/2, ece_scores, width, label='ECE', alpha=0.7)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Score')
        ax2.set_title('Calibration Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Prediction distributions comparison
        ax3 = axes[1, 0]
        
        for i, (model_name, (y_true, y_prob)) in enumerate(models_data.items()):
            ax3.hist(y_prob, bins=20, alpha=0.5, label=model_name, 
                    color=colors[i], density=True)
        
        ax3.set_xlabel('Predicted Fraud Probability')
        ax3.set_ylabel('Density')
        ax3.set_title('Prediction Distributions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Metrics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create detailed metrics table
        table_data = []
        headers = ['Model', 'Brier', 'Log Loss', 'ECE', 'MCE', 'Overconf.', 'Sharpness']
        
        for model_name, (y_true, y_prob) in models_data.items():
            metrics = self.compute_calibration_metrics(y_true, y_prob)
            row = [
                model_name,
                f"{metrics['brier_score']:.3f}",
                f"{metrics['log_loss']:.3f}",
                f"{metrics['ece']:.3f}",
                f"{metrics['mce']:.3f}",
                f"{metrics['overconfidence']:.3f}",
                f"{metrics['sharpness']:.3f}"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_calibration_report(self,
                                  y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  model_name: str = "Model",
                                  save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive calibration report.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            save_path: Optional path to save the report
            
        Returns:
            Formatted report string
        """
        metrics = self.compute_calibration_metrics(y_true, y_prob)
        
        report = f"""
# Calibration Report for {model_name}

## Summary Statistics
- Total Samples: {len(y_true):,}
- Fraud Rate: {np.mean(y_true):.3f} ({np.mean(y_true)*100:.1f}%)
- Mean Prediction: {np.mean(y_prob):.3f}
- Prediction Range: [{np.min(y_prob):.3f}, {np.max(y_prob):.3f}]

## Calibration Metrics

### Primary Metrics
- **Brier Score**: {metrics['brier_score']:.4f} {'(Lower is better)' if metrics['brier_score'] < 0.25 else '(Needs improvement)'}
- **Log Loss**: {metrics['log_loss']:.4f} {'(Lower is better)' if metrics['log_loss'] < 0.5 else '(Needs improvement)'}

### Calibration Error Metrics
- **Expected Calibration Error (ECE)**: {metrics['ece']:.4f}
- **Maximum Calibration Error (MCE)**: {metrics['mce']:.4f}
- **Adaptive Calibration Error (ACE)**: {metrics['ace']:.4f}

### Confidence Analysis
- **Overconfidence**: {metrics['overconfidence']:.4f}
- **Underconfidence**: {metrics['underconfidence']:.4f}
- **Sharpness**: {metrics['sharpness']:.4f}

## Interpretation

### Brier Score
- Excellent: < 0.1
- Good: 0.1 - 0.2
- Fair: 0.2 - 0.35
- Poor: > 0.35

### Expected Calibration Error (ECE)
- Excellent: < 0.01
- Good: 0.01 - 0.05
- Fair: 0.05 - 0.1
- Poor: > 0.1

### Recommendations
"""
        
        # Add recommendations based on metrics
        if metrics['brier_score'] > 0.25:
            report += "- Consider improving model discrimination or probability calibration\n"
        
        if metrics['ece'] > 0.05:
            report += "- Apply calibration methods (Platt scaling, isotonic regression)\n"
        
        if metrics['overconfidence'] > 0.1:
            report += "- Model is overconfident, consider temperature scaling\n"
        
        if metrics['underconfidence'] > 0.1:
            report += "- Model is underconfident, may need more training or feature engineering\n"
        
        if metrics['sharpness'] < 0.05:
            report += "- Model predictions are not very confident, consider threshold tuning\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report


def create_sample_fraud_data(n_samples: int = 10000, fraud_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample fraud detection data for testing calibration.
    
    Args:
        n_samples: Number of samples to generate
        fraud_rate: Base fraud rate
        
    Returns:
        Tuple of (y_true, y_prob)
    """
    np.random.seed(42)
    
    # Generate true labels
    y_true = np.random.choice([0, 1], size=n_samples, p=[1-fraud_rate, fraud_rate])
    
    # Generate predicted probabilities with various calibration issues
    y_prob = np.zeros(n_samples)
    
    # Well-calibrated predictions for legitimate transactions
    legit_mask = y_true == 0
    y_prob[legit_mask] = np.random.beta(2, 10, size=np.sum(legit_mask))
    
    # Slightly overconfident predictions for fraudulent transactions
    fraud_mask = y_true == 1
    y_prob[fraud_mask] = np.random.beta(8, 2, size=np.sum(fraud_mask))
    
    # Add some noise and ensure valid probability range
    y_prob = np.clip(y_prob + np.random.normal(0, 0.05, n_samples), 0.01, 0.99)
    
    return y_true, y_prob


if __name__ == "__main__":
    # Example usage
    y_true, y_prob = create_sample_fraud_data()
    
    analyzer = CalibrationAnalyzer(n_bins=10)
    
    # Generate calibration curve plot
    fig = analyzer.plot_calibration_curve(y_true, y_prob, "Sample Fraud Model")
    plt.show()
    
    # Generate report
    report = analyzer.generate_calibration_report(y_true, y_prob, "Sample Fraud Model")
    print(report)
