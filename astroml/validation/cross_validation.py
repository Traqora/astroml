from astroml.utils.exceptions import AstroMLError
"""Cross-validation framework and evaluation engine for AstroML.

Provides comprehensive metric computation, time-series leakage detection,
fold-by-fold execution, and reporting for standard and temporal splitters.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from astroml.validation.leakage import LeakageError
from astroml.validation.splitters import (
    BaseSplitter,
    KFoldSplitter,
    PurgedWalkForwardSplitter,
    SlidingWindowSplitter,
    TimeSeriesSplitter,
    _to_numpy_array,
    get_splitter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric Evaluator
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    scoring: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute performance metrics for a fold."""
    metrics: dict[str, float] = {}
    requested = set(scoring) if scoring else {"accuracy", "f1", "precision", "recall"}

    unique_y = np.unique(y_true)
    is_binary = len(unique_y) <= 2
    is_regression = np.issubdtype(y_true.dtype, np.floating) and len(unique_y) > 10

    if is_regression:
        if "mse" in requested or "mean_squared_error" in requested:
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        if "mae" in requested or "mean_absolute_error" in requested:
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        if "r2" in requested:
            metrics["r2"] = float(r2_score(y_true, y_pred))
        if not metrics:
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))
        return metrics

    # Classification metrics
    if "accuracy" in requested:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    average = "binary" if is_binary else "macro"

    if "precision" in requested:
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        )
    if "recall" in requested:
        metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    if "f1" in requested or "f1_score" in requested:
        metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))

    if y_prob is not None:
        if "roc_auc" in requested or "auc" in requested:
            try:
                if is_binary and y_prob.ndim > 1 and y_prob.shape[1] == 2:
                    prob_col = y_prob[:, 1]
                elif is_binary and y_prob.ndim == 1:
                    prob_col = y_prob
                else:
                    prob_col = y_prob
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, prob_col, multi_class="ovr" if not is_binary else "raise")
                )
            except AstroMLError:
                pass
        if "log_loss" in requested:
            try:
                metrics["log_loss"] = float(log_loss(y_true, y_prob))
            except AstroMLError:
                pass

    return metrics


# ---------------------------------------------------------------------------
# Cross-Validation Report
# ---------------------------------------------------------------------------


@dataclass
class CrossValidationReport:
    """Detailed summary and results of a cross-validation experiment."""

    splitter_name: str
    n_splits: int
    test_scores: list[dict[str, float]]
    train_scores: list[dict[str, float]] | None = None
    fold_sizes: list[tuple[int, int]] = field(default_factory=list)
    fit_times: list[float] = field(default_factory=list)
    score_times: list[float] = field(default_factory=list)
    leakage_verified: bool = True
    fitted_estimators: list[Any] | None = None

    @property
    def metrics_summary(self) -> dict[str, dict[str, float]]:
        """Compute aggregated statistics for all test metrics."""
        summary: dict[str, dict[str, float]] = {}
        if not self.test_scores:
            return summary

        metric_names = self.test_scores[0].keys()
        for m in metric_names:
            vals = [fold[m] for fold in self.test_scores if m in fold]
            if not vals:
                continue
            arr = np.array(vals)
            std_err = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            summary[m] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "ci_lower": float(np.mean(arr) - 1.96 * std_err),
                "ci_upper": float(np.mean(arr) + 1.96 * std_err),
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "splitter_name": self.splitter_name,
            "n_splits": self.n_splits,
            "metrics_summary": self.metrics_summary,
            "fold_scores": self.test_scores,
            "fold_sizes": self.fold_sizes,
            "fit_times": self.fit_times,
            "score_times": self.score_times,
            "leakage_verified": self.leakage_verified,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export fold-by-fold results as a pandas DataFrame."""
        rows = []
        for i, score_dict in enumerate(self.test_scores):
            row = {"fold": i + 1, **score_dict}
            if i < len(self.fold_sizes):
                row["train_size"] = self.fold_sizes[i][0]
                row["test_size"] = self.fold_sizes[i][1]
            if i < len(self.fit_times):
                row["fit_time_sec"] = self.fit_times[i]
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Generate human-readable summary text table."""
        lines = [
            f"=== Cross-Validation Report ({self.splitter_name}, {self.n_splits} folds) ===",
            f"Leakage check: {'Passed' if self.leakage_verified else 'Failed'}",
            "-" * 65,
            f"{'Metric':<18} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}",
            "-" * 65,
        ]
        for m, stats in self.metrics_summary.items():
            lines.append(
                f"{m:<18} | {stats['mean']:<10.4f} | {stats['std']:<10.4f} | "
                f"{stats['min']:<10.4f} | {stats['max']:<10.4f}"
            )
        lines.append("-" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-Validation Engine
# ---------------------------------------------------------------------------


def cross_validate(
    estimator: Any,
    X: Any,
    y: Any = None,
    *,
    groups: Any = None,
    timestamps: Any = None,
    cv: BaseSplitter | int = 5,
    scoring: str | Sequence[str] | None = None,
    return_train_score: bool = False,
    return_estimator: bool = False,
    fit_params: Mapping[str, Any] | None = None,
    check_leakage: bool = True,
) -> CrossValidationReport:
    """Run cross-validation on an estimator with comprehensive evaluation.

    Parameters
    ----------
    estimator : Any
        Estimator implementing `fit`, `predict`, and optional `predict_proba`.
    X : Any
        Feature matrix / input samples.
    y : Any, optional
        Target labels or values.
    groups : Any, optional
        Group indicators for GroupKFold.
    timestamps : Any, optional
        Timestamps for time-series validation and leakage checking.
    cv : BaseSplitter | int
        Cross-validation strategy instance or integer k.
    scoring : str | Sequence[str], optional
        Metrics to evaluate (e.g. 'accuracy', 'f1', 'precision', 'recall', 'roc_auc').
    return_train_score : bool
        Whether to calculate and return training scores.
    return_estimator : bool
        Whether to keep the fitted estimators for all folds.
    fit_params : Mapping[str, Any], optional
        Additional parameters passed to estimator.fit.
    check_leakage : bool
        Whether to enforce strict temporal leakage validation when timestamps are given.

    Returns
    -------
    CrossValidationReport
        Comprehensive cross-validation results and summary.
    """
    if isinstance(cv, int):
        splitter: BaseSplitter = KFoldSplitter(n_splits=cv)
    elif isinstance(cv, BaseSplitter):
        splitter = cv
    else:
        raise TypeError(f"cv must be int or BaseSplitter, got {type(cv)}")

    score_list = [scoring] if isinstance(scoring, str) else scoring
    fit_kwargs = dict(fit_params) if fit_params else {}

    X_arr = _to_numpy_array(X)
    y_arr = _to_numpy_array(y) if y is not None else None
    ts_arr = _to_numpy_array(timestamps) if timestamps is not None else None

    test_scores: list[dict[str, float]] = []
    train_scores: list[dict[str, float]] | None = [] if return_train_score else None
    fold_sizes: list[tuple[int, int]] = []
    fit_times: list[float] = []
    score_times: list[float] = []
    fitted_models: list[Any] | None = [] if return_estimator else None
    leakage_passed = True

    is_temporal_splitter = isinstance(
        splitter, (TimeSeriesSplitter, SlidingWindowSplitter, PurgedWalkForwardSplitter)
    )

    for train_idx, test_idx in splitter.split(X_arr, y_arr, groups=groups, timestamps=ts_arr):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        fold_sizes.append((len(train_idx), len(test_idx)))

        # Temporal leakage check
        if check_leakage and (is_temporal_splitter or ts_arr is not None) and ts_arr is not None:
            train_times = ts_arr[train_idx]
            test_times = ts_arr[test_idx]
            if len(train_times) > 0 and len(test_times) > 0:
                max_train_t = np.max(train_times)
                min_test_t = np.min(test_times)
                if max_train_t >= min_test_t:
                    leakage_passed = False
                    raise LeakageError(
                        f"Temporal leakage detected: max train timestamp ({max_train_t}) "
                        f">= min test timestamp ({min_test_t})"
                    )

        # Slice data
        X_train = X_arr[train_idx]
        X_test = X_arr[test_idx]
        y_train = y_arr[train_idx] if y_arr is not None else None
        y_test = y_arr[test_idx] if y_arr is not None else None

        # Clone estimator
        fold_estimator = copy.deepcopy(estimator)

        # Fit
        t_start_fit = time.perf_counter()
        if y_train is not None:
            fold_estimator.fit(X_train, y_train, **fit_kwargs)
        else:
            fold_estimator.fit(X_train, **fit_kwargs)
        fit_times.append(time.perf_counter() - t_start_fit)

        # Score test fold
        t_start_score = time.perf_counter()
        test_pred = fold_estimator.predict(X_test)
        test_prob = (
            fold_estimator.predict_proba(X_test)
            if hasattr(fold_estimator, "predict_proba")
            else None
        )
        score_times.append(time.perf_counter() - t_start_score)

        test_metric_dict = _compute_metrics(
            y_true=y_test if y_test is not None else np.zeros(len(X_test)),
            y_pred=test_pred,
            y_prob=test_prob,
            scoring=score_list,
        )
        test_scores.append(test_metric_dict)

        # Optional train score
        if return_train_score and train_scores is not None:
            train_pred = fold_estimator.predict(X_train)
            train_prob = (
                fold_estimator.predict_proba(X_train)
                if hasattr(fold_estimator, "predict_proba")
                else None
            )
            train_metric_dict = _compute_metrics(
                y_true=y_train if y_train is not None else np.zeros(len(X_train)),
                y_pred=train_pred,
                y_prob=train_prob,
                scoring=score_list,
            )
            train_scores.append(train_metric_dict)

        if return_estimator and fitted_models is not None:
            fitted_models.append(fold_estimator)

    return CrossValidationReport(
        splitter_name=splitter.__class__.__name__,
        n_splits=len(test_scores),
        test_scores=test_scores,
        train_scores=train_scores,
        fold_sizes=fold_sizes,
        fit_times=fit_times,
        score_times=score_times,
        leakage_verified=leakage_passed,
        fitted_estimators=fitted_models,
    )


class CrossValidator:
    """Orchestrator class for configuring and running cross-validation experiments."""

    def __init__(
        self,
        splitter: BaseSplitter | str = "kfold",
        scoring: Sequence[str] | str | None = None,
        return_train_score: bool = False,
        check_leakage: bool = True,
        **splitter_kwargs: Any,
    ) -> None:
        if isinstance(splitter, str):
            self.splitter = get_splitter(splitter, **splitter_kwargs)
        else:
            self.splitter = splitter
        self.scoring = [scoring] if isinstance(scoring, str) else scoring
        self.return_train_score = return_train_score
        self.check_leakage = check_leakage

    def evaluate(
        self,
        estimator: Any,
        X: Any,
        y: Any = None,
        groups: Any = None,
        timestamps: Any = None,
        fit_params: Mapping[str, Any] | None = None,
    ) -> CrossValidationReport:
        """Run cross-validation on given dataset."""
        return cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            groups=groups,
            timestamps=timestamps,
            cv=self.splitter,
            scoring=self.scoring,
            return_train_score=self.return_train_score,
            fit_params=fit_params,
            check_leakage=self.check_leakage,
        )
