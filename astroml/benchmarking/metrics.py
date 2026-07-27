"""Evaluation metrics for different GNN tasks."""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, ndcg_score, mean_squared_error, mean_absolute_error
)
import torch


class ClassificationMetrics:
    """Metrics for node classification tasks."""
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC if probabilities are available
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            except Exception:
                metrics['auc'] = 0.0
        
        # Per-class metrics
        unique_labels = np.unique(y_true)
        for label in unique_labels:
            mask = (y_true == label)
            if np.sum(mask) > 0:
                label_precision = precision_score(
                    (y_true == label), (y_pred == label), zero_division=0
                )
                label_recall = recall_score(
                    (y_true == label), (y_pred == label), zero_division=0
                )
                label_f1 = f1_score(
                    (y_true == label), (y_pred == label), zero_division=0
                )
                
                metrics[f'precision_class_{label}'] = label_precision
                metrics[f'recall_class_{label}'] = label_recall
                metrics[f'f1_class_{label}'] = label_f1
        
        return metrics


class LinkPredictionMetrics:
    """Metrics for link prediction tasks."""
    
    @staticmethod
    def compute(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: Optional[np.ndarray] = None,
        k_values: list[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """Compute link prediction metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC if probabilities are available
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except Exception:
                metrics['auc'] = 0.0
        
        # Ranking metrics (for recommendation scenarios)
        if y_prob is not None:
            for k in k_values:
                try:
                    # Hits@K
                    hits_k = LinkPredictionMetrics._hits_at_k(y_true, y_prob, k)
                    metrics[f'hits_at_{k}'] = hits_k
                    
                    # Precision@K
                    precision_k = LinkPredictionMetrics._precision_at_k(y_true, y_prob, k)
                    metrics[f'precision_at_{k}'] = precision_k
                    
                    # Recall@K
                    recall_k = LinkPredictionMetrics._recall_at_k(y_true, y_prob, k)
                    metrics[f'recall_at_{k}'] = recall_k
                    
                except:
                    metrics[f'hits_at_{k}'] = 0.0
                    metrics[f'precision_at_{k}'] = 0.0
                    metrics[f'recall_at_{k}'] = 0.0
        
        return metrics
    
    @staticmethod
    def _hits_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """Compute Hits@K metric."""
        # Get top-k predictions
        top_k_indices = np.argsort(y_scores)[-k:]
        
        # Check if any true link is in top-k
        hits = 0
        for idx in top_k_indices:
            if y_true[idx] == 1:
                hits += 1
                break
        
        return hits / len(y_true)
    
    @staticmethod
    def _precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """Compute Precision@K metric."""
        top_k_indices = np.argsort(y_scores)[-k:]
        top_k_true = y_true[top_k_indices]
        
        return np.sum(top_k_true) / k
    
    @staticmethod
    def _recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """Compute Recall@K metric."""
        top_k_indices = np.argsort(y_scores)[-k:]
        top_k_true = y_true[top_k_indices]
        
        total_true = np.sum(y_true)
        if total_true == 0:
            return 0.0
        
        return np.sum(top_k_true) / total_true


class AnomalyDetectionMetrics:
    """Metrics for anomaly detection tasks."""
    
    @staticmethod
    def compute(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute anomaly detection metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC if scores are available
        if y_scores is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_scores)
            except Exception:
                metrics['auc'] = 0.0
        
        # Anomaly-specific metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        # False Positive Rate
        if fp + tn > 0:
            metrics['false_positive_rate'] = fp / (fp + tn)
        else:
            metrics['false_positive_rate'] = 0.0
        
        # False Negative Rate
        if fn + tp > 0:
            metrics['false_negative_rate'] = fn / (fn + tp)
        else:
            metrics['false_negative_rate'] = 0.0
        
        # Detection Rate (same as recall for anomalies)
        if tp + fn > 0:
            metrics['detection_rate'] = tp / (tp + fn)
        else:
            metrics['detection_rate'] = 0.0
        
        return metrics


class RegressionMetrics:
    """Metrics for regression tasks."""
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot > 0:
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            metrics['r2'] = 0.0
        
        # Mean Absolute Percentage Error
        if np.any(y_true != 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = float('inf')
        
        return metrics


class MetricCalculator:
    """Main interface for computing metrics across different tasks."""
    
    @staticmethod
    def compute_metrics(
        task_type: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute metrics based on task type."""
        task_type = task_type.lower()
        
        if task_type in ['classification', 'node_classification']:
            return ClassificationMetrics.compute(y_true, y_pred, y_prob)
        elif task_type in ['link_prediction', 'link_pred']:
            return LinkPredictionMetrics.compute(y_true, y_pred, y_prob, **kwargs)
        elif task_type in ['anomaly_detection', 'anomaly']:
            return AnomalyDetectionMetrics.compute(y_true, y_pred, y_prob)
        elif task_type in ['regression']:
            return RegressionMetrics.compute(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    @staticmethod
    def aggregate_metrics(metrics_list: list[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across multiple runs."""
        if not metrics_list:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for metrics in metrics_list:
            all_metrics.update(metrics.keys())
        
        # Compute statistics for each metric
        aggregated = {}
        for metric_name in all_metrics:
            values = [m.get(metric_name, float('nan')) for m in metrics_list]
            valid_values = [v for v in values if not np.isnan(v)]
            
            if valid_values:
                aggregated[metric_name] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values),
                    'median': np.median(valid_values),
                    'count': len(valid_values)
                }
            else:
                aggregated[metric_name] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'min': float('nan'),
                    'max': float('nan'),
                    'median': float('nan'),
                    'count': 0
                }
        
        return aggregated
