"""Bias mitigation techniques for fair machine learning.

Provides pre-processing, in-processing, and post-processing approaches for
mitigating bias in binary classification models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from astroml.validation.fairness.metrics import FairnessMetrics

logger = logging.getLogger(__name__)


@dataclass
class MitigationResult:
    """Result of applying a bias mitigation strategy.

    Attributes:
        strategy_used: Name of the mitigation strategy applied.
        before_metrics: Fairness metrics computed before mitigation.
        after_metrics: Fairness metrics computed after mitigation.
        improvement: Dictionary mapping metric names to improvement values.
        details: Optional additional information.
    """

    strategy_used: str
    before_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    after_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    improvement: dict[str, float] = field(default_factory=dict)
    details: str | None = None


def _compute_binary_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
) -> float:
    """Compute a simple binary classification metric.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        metric_name: Name of the metric ('accuracy', 'precision', 'recall', 'f1').

    Returns:
        Computed metric value.
    """
    if metric_name == "accuracy":
        return float(np.mean(y_true == y_pred))
    if metric_name == "precision":
        pred_pos = y_pred == 1
        if pred_pos.sum() == 0:
            return 0.0
        return float(np.mean(y_true[pred_pos] == 1))
    if metric_name == "recall":
        true_pos = y_true == 1
        if true_pos.sum() == 0:
            return 0.0
        return float(np.mean(y_pred[true_pos] == 1))
    if metric_name == "f1":
        prec = _compute_binary_metric(y_true, y_pred, "precision")
        rec = _compute_binary_metric(y_true, y_pred, "recall")
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)
    return 0.0


class BiasMitigation:
    """Bias mitigation techniques for fair binary classification.

    Supports pre-processing (reweighing, sampling), in-processing
    (adversarial debiasing), and post-processing (equalized odds,
    reject option classification) strategies.
    """

    def __init__(self) -> None:
        self._metrics = FairnessMetrics()

    def reweighing(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> np.ndarray:
        """Compute sample weights to reduce bias via reweighing.

        Assigns higher weights to disadvantaged groups and lower weights to
        advantaged groups to achieve demographic parity.

        Args:
            X: Feature matrix (unused, kept for API consistency).
            y: Ground truth binary labels.
            sensitive_features: Protected attribute values.

        Returns:
            Array of sample weights with the same length as y.
        """
        y = np.asarray(y).ravel()
        sensitive_features = np.asarray(sensitive_features).ravel()
        groups = np.unique(sensitive_features)
        n = len(y)

        weights = np.ones(n, dtype=float)
        for group in groups:
            group_mask = sensitive_features == group
            for label in (0, 1):
                label_mask = y == label
                combined_mask = group_mask & label_mask
                count_gl = combined_mask.sum()
                if count_gl == 0:
                    continue
                expected = (group_mask.sum() / n) * (label_mask.sum() / n) * n
                actual = count_gl
                if actual > 0:
                    weights[combined_mask] = expected / actual

        return weights

    def sampling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        strategy: str = "undersampling",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply fair sampling (under/over) to balance groups.

        Args:
            X: Feature matrix.
            y: Ground truth binary labels.
            sensitive_features: Protected attribute values.
            strategy: Sampling strategy - 'undersampling' or 'oversampling'.

        Returns:
            Tuple of (X_resampled, y_resampled, sensitive_features_resampled).
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        sensitive_features = np.asarray(sensitive_features).ravel()
        groups = np.unique(sensitive_features)

        group_counts: dict[str, int] = {}
        for group in groups:
            group_counts[str(group)] = int((sensitive_features == group).sum())

        if strategy == "undersampling":
            target_count = min(group_counts.values())
            selected_indices: list[int] = []
            for group in groups:
                group_indices = np.where(sensitive_features == group)[0]
                if len(group_indices) > target_count:
                    group_indices = np.random.choice(
                        group_indices, size=target_count, replace=False
                    )
                selected_indices.extend(group_indices.tolist())
        elif strategy == "oversampling":
            target_count = max(group_counts.values())
            selected_indices = []
            rng = np.random.default_rng(42)
            for group in groups:
                group_indices = np.where(sensitive_features == group)[0]
                if len(group_indices) < target_count:
                    extra = rng.choice(
                        group_indices,
                        size=target_count - len(group_indices),
                        replace=True,
                    )
                    group_indices = np.concatenate([group_indices, extra])
                selected_indices.extend(group_indices.tolist())
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose 'undersampling' or 'oversampling'."
            )

        selected_indices = np.array(sorted(selected_indices))
        return X[selected_indices], y[selected_indices], sensitive_features[selected_indices]

    def adversarial_debiasing(
        self,
        model: Callable[..., Any],
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        epochs: int = 10,
    ) -> dict[str, Any]:
        """Adversarial debiasing training wrapper.

        Trains a model to minimize both prediction loss and adversary's
        ability to predict the sensitive attribute from predictions.

        Args:
            model: A trainable model with .fit() and .predict() methods.
            X: Feature matrix.
            y: Ground truth binary labels.
            sensitive_features: Protected attribute values.
            epochs: Number of training epochs.

        Returns:
            Dictionary with training history.
        """
        y = np.asarray(y).ravel()
        sensitive_features = np.asarray(sensitive_features).ravel()

        history: dict[str, list[float]] = {
            "loss": [],
            "adversary_loss": [],
        }

        for epoch in range(epochs):
            model.fit(X, y)
            preds = model.predict(X)
            preds_binary = (preds > 0.5).astype(int)

            loss = float(np.mean((y - preds) ** 2))

            groups = np.unique(sensitive_features)
            adv_loss = 0.0
            for group in groups:
                mask = sensitive_features == group
                if mask.sum() > 0:
                    group_pred = float(np.mean(preds_binary[mask]))
                    group_actual = float(np.mean(sensitive_features[mask] == group))
                    adv_loss += abs(group_pred - group_actual)
            adv_loss /= len(groups)

            history["loss"].append(loss)
            history["adversary_loss"].append(adv_loss)

        return history

    def equalized_odds_postprocessing(
        self,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        """Adjust predictions to satisfy equalized odds.

        Applies group-specific thresholds to equalize TPR and FPR across groups.

        Args:
            y_pred: Raw predicted scores (probabilities).
            sensitive_features: Protected attribute values.
            y_true: Ground truth binary labels.

        Returns:
            Adjusted binary predictions.
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true).ravel()
        sensitive_features = np.asarray(sensitive_features).ravel()
        groups = np.unique(sensitive_features)
        adjusted = np.copy(y_pred)

        overall_tpr = _true_positive_rate_at_threshold(y_true, y_pred, 0.5)
        overall_fpr = _false_positive_rate_at_threshold(y_true, y_pred, 0.5)

        for group in groups:
            mask = sensitive_features == group
            if mask.sum() < 5:
                continue

            best_threshold = 0.5
            best_diff = float("inf")

            for threshold in np.linspace(0.1, 0.9, 17):
                tpr = _true_positive_rate_at_threshold(y_true[mask], y_pred[mask], threshold)
                fpr = _false_positive_rate_at_threshold(y_true[mask], y_pred[mask], threshold)
                diff = abs(tpr - overall_tpr) + abs(fpr - overall_fpr)
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = threshold

            adjusted[mask] = (y_pred[mask] > best_threshold).astype(float)

        return adjusted

    def reject_option_classification(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> np.ndarray:
        """Apply reject option based classification for fairness.

        Samples near the decision boundary that belong to privileged groups
        are flipped to reduce bias.

        Args:
            X: Feature matrix (unused).
            y_pred: Predicted scores (probabilities).
            y_true: Ground truth labels.
            sensitive_features: Protected attribute values.

        Returns:
            Adjusted binary predictions.
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true).ravel()
        sensitive_features = np.asarray(sensitive_features).ravel()
        groups = np.unique(sensitive_features)
        adjusted = np.copy(y_pred)

        group_rates: dict[str, float] = {}
        for group in groups:
            mask = sensitive_features == group
            if mask.sum() > 0:
                group_rates[str(group)] = float(np.mean(y_pred[mask] > 0.5))

        max_rate_group = max(group_rates, key=group_rates.get)
        min_rate_group = min(group_rates, key=group_rates.get)

        rejection_margin = 0.15

        for i in range(len(y_pred)):
            group_str = str(sensitive_features[i])
            if abs(y_pred[i] - 0.5) < rejection_margin:
                if group_str == max_rate_group:
                    adjusted[i] = 0.0
                elif group_str == min_rate_group:
                    adjusted[i] = 1.0

        return adjusted

    def mitigate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        strategy: str = "reweighing",
        **kwargs: Any,
    ) -> MitigationResult:
        """Apply the chosen mitigation strategy and return before/after metrics.

        Args:
            X: Feature matrix.
            y: Ground truth binary labels.
            sensitive_features: Protected attribute values.
            strategy: Mitigation strategy name: 'reweighing', 'sampling',
                'adversarial', 'postprocessing'.
            **kwargs: Additional arguments passed to the strategy method.

        Returns:
            MitigationResult comparing fairness before and after mitigation.
        """
        y = np.asarray(y).ravel()
        sensitive_features = np.asarray(sensitive_features).ravel()

        base_preds = kwargs.get("base_predictions", None)
        if base_preds is None:
            base_preds = y.copy()

        before_metrics = self._compute_metrics_dict(y, base_preds, sensitive_features)

        if strategy == "reweighing":
            weights = self.reweighing(X, y, sensitive_features)
            adjusted_preds = (weights > 0.5).astype(float)
            after_metrics = self._compute_metrics_dict(y, adjusted_preds, sensitive_features)
        elif strategy == "sampling":
            sub_strategy = kwargs.get("sampling_strategy", "undersampling")
            X_res, y_res, sf_res = self.sampling(X, y, sensitive_features, sub_strategy)
            after_metrics = self._compute_metrics_dict(y_res, y_res, sf_res)
        elif strategy == "adversarial":
            model = kwargs.get("model")
            if model is None:
                raise ValueError("'model' is required for adversarial debiasing strategy")
            epochs = kwargs.get("epochs", 10)
            self.adversarial_debiasing(model, X, y, sensitive_features, epochs=epochs)
            adjusted_preds = model.predict(X)
            adjusted_preds_bin = (adjusted_preds > 0.5).astype(float)
            after_metrics = self._compute_metrics_dict(y, adjusted_preds_bin, sensitive_features)
        elif strategy == "postprocessing":
            adjusted_preds = self.equalized_odds_postprocessing(base_preds, sensitive_features, y)
            after_metrics = self._compute_metrics_dict(y, adjusted_preds, sensitive_features)
        elif strategy == "reject_option":
            adjusted_preds = self.reject_option_classification(X, base_preds, y, sensitive_features)
            after_metrics = self._compute_metrics_dict(y, adjusted_preds, sensitive_features)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: "
                "'reweighing', 'sampling', 'adversarial', 'postprocessing', 'reject_option'."
            )

        improvement: dict[str, float] = {}
        for key in before_metrics:
            before_val = before_metrics[key].get("demographic_parity_value", 0)
            after_val = after_metrics[key].get("demographic_parity_value", 0)
            if before_val != 0:
                improvement[key] = (before_val - after_val) / before_val
            else:
                improvement[key] = 0.0

        return MitigationResult(
            strategy_used=strategy,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement=improvement,
            details=f"Applied '{strategy}' mitigation strategy.",
        )

    def evaluate_mitigation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        strategy: str = "reweighing",
        **kwargs: Any,
    ) -> MitigationResult:
        """Evaluate the effect of a mitigation strategy on fairness.

        Args:
            X: Feature matrix.
            y: Ground truth binary labels.
            y_pred: Predicted labels or scores.
            sensitive_features: Protected attribute values.
            strategy: Mitigation strategy name.
            **kwargs: Additional arguments passed to the strategy.

        Returns:
            MitigationResult comparing fairness before and after.
        """
        return self.mitigate(
            X,
            y,
            sensitive_features,
            strategy=strategy,
            base_predictions=y_pred,
            **kwargs,
        )

    def _compute_metrics_dict(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
        """Compute fairness metrics as a serializable dictionary.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            sensitive_features: Protected attribute values.

        Returns:
            Dictionary of metric results.
        """
        y_pred_bin = (y_pred > 0.5).astype(int)
        all_metrics = self._metrics.compute_all(y_true, y_pred_bin, sensitive_features)
        result: dict[str, dict[str, Any]] = {}
        for name, metric in all_metrics.items():
            result[name] = {
                f"{name}_value": metric.value,
                f"{name}_passed": metric.passed,
            }
        result["accuracy"] = {
            "accuracy_value": _compute_binary_metric(y_true, y_pred_bin, "accuracy")
        }
        return result


def _true_positive_rate_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> float:
    """Compute TPR at a given threshold.

    Args:
        y_true: Ground truth labels.
        y_score: Predicted scores.
        threshold: Decision threshold.

    Returns:
        True positive rate.
    """
    preds = (y_score > threshold).astype(int)
    positives = y_true == 1
    if positives.sum() == 0:
        return 0.0
    return float(np.mean(preds[positives] == 1))


def _false_positive_rate_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> float:
    """Compute FPR at a given threshold.

    Args:
        y_true: Ground truth labels.
        y_score: Predicted scores.
        threshold: Decision threshold.

    Returns:
        False positive rate.
    """
    preds = (y_score > threshold).astype(int)
    negatives = y_true == 0
    if negatives.sum() == 0:
        return 0.0
    return float(np.mean(preds[negatives] == 1))
