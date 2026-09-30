"""Data property validation for ML datasets.

Implements Procedure step 5:
  5. Implement data property validation.

This module defines validation functions that assert structural and statistical
properties of raw ML datasets (feature matrices and label arrays) prior to
model training or evaluation.  Every validator returns a
:class:`DataValidationResult` that callers can inspect or log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# ── Result data class ─────────────────────────────────────────────────────────


@dataclass
class DataValidationResult:
    """Outcome of a single data validation check.

    Attributes:
        check_name: Human-readable name of the validation check.
        passed: ``True`` if the data satisfied the property.
        error_message: Description of the failure, or ``None`` on success.
    """

    check_name: str
    passed: bool
    error_message: Optional[str] = None


# ── Individual validation functions ──────────────────────────────────────────


def check_no_nan_or_inf(X: np.ndarray) -> DataValidationResult:
    """Verify that *X* contains no NaN or infinite values.

    Args:
        X: Feature matrix to inspect.

    Returns:
        A :class:`DataValidationResult` indicating pass or fail.
    """
    if not np.all(np.isfinite(X)):
        nan_count = int(np.sum(np.isnan(X)))
        inf_count = int(np.sum(np.isinf(X)))
        return DataValidationResult(
            check_name="no_nan_or_inf",
            passed=False,
            error_message=(
                f"Feature matrix contains {nan_count} NaN(s) and " f"{inf_count} infinite value(s)."
            ),
        )
    return DataValidationResult(check_name="no_nan_or_inf", passed=True)


def check_feature_matrix_shape(
    X: np.ndarray,
    expected_features: Optional[int] = None,
) -> DataValidationResult:
    """Verify that *X* is 2-D and optionally matches an expected column count.

    Args:
        X: Feature matrix to inspect.
        expected_features: If provided, the exact number of columns required.

    Returns:
        A :class:`DataValidationResult` indicating pass or fail.
    """
    if X.ndim != 2:
        return DataValidationResult(
            check_name="feature_matrix_shape",
            passed=False,
            error_message=f"Expected 2-D array, got {X.ndim}-D array.",
        )
    if expected_features is not None and X.shape[1] != expected_features:
        return DataValidationResult(
            check_name="feature_matrix_shape",
            passed=False,
            error_message=(f"Expected {expected_features} features, got {X.shape[1]}."),
        )
    return DataValidationResult(check_name="feature_matrix_shape", passed=True)


def check_label_array_shape(
    y: np.ndarray,
    X: np.ndarray,
) -> DataValidationResult:
    """Verify that label array *y* has the same length as *X*.

    Args:
        y: 1-D label array.
        X: Feature matrix used to infer the expected sample count.

    Returns:
        A :class:`DataValidationResult` indicating pass or fail.
    """
    if y.ndim != 1:
        return DataValidationResult(
            check_name="label_array_shape",
            passed=False,
            error_message=f"Label array must be 1-D, got {y.ndim}-D.",
        )
    if len(y) != X.shape[0]:
        return DataValidationResult(
            check_name="label_array_shape",
            passed=False,
            error_message=(f"Label count {len(y)} does not match sample count " f"{X.shape[0]}."),
        )
    return DataValidationResult(check_name="label_array_shape", passed=True)


def check_class_labels_in_range(
    y: np.ndarray,
    n_classes: int,
) -> DataValidationResult:
    """Verify that all labels are integers in ``[0, n_classes)``.

    Args:
        y: 1-D label array.
        n_classes: Total number of expected classes.

    Returns:
        A :class:`DataValidationResult` indicating pass or fail.
    """
    invalid = y[(y < 0) | (y >= n_classes)]
    if len(invalid) > 0:
        return DataValidationResult(
            check_name="class_labels_in_range",
            passed=False,
            error_message=(
                f"Found {len(invalid)} label(s) outside [0, {n_classes}): "
                f"{np.unique(invalid).tolist()}"
            ),
        )
    return DataValidationResult(check_name="class_labels_in_range", passed=True)


def check_minimum_samples(
    X: np.ndarray,
    min_samples: int = 1,
) -> DataValidationResult:
    """Verify that *X* contains at least *min_samples* rows.

    Args:
        X: Feature matrix to inspect.
        min_samples: Minimum number of samples required.

    Returns:
        A :class:`DataValidationResult` indicating pass or fail.
    """
    if X.shape[0] < min_samples:
        return DataValidationResult(
            check_name="minimum_samples",
            passed=False,
            error_message=(
                f"Dataset has {X.shape[0]} sample(s); at least " f"{min_samples} required."
            ),
        )
    return DataValidationResult(check_name="minimum_samples", passed=True)


def check_feature_variance(
    X: np.ndarray,
    min_variance: float = 0.0,
) -> DataValidationResult:
    """Verify that no feature column has variance below *min_variance*.

    Constant features (variance = 0) can cause numerical instability in many
    algorithms; this check surfaces such columns early.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        min_variance: Minimum acceptable per-column variance.

    Returns:
        A :class:`DataValidationResult` indicating pass or fail.
    """
    if X.shape[0] < 2:
        # Variance is undefined with a single sample.
        return DataValidationResult(
            check_name="feature_variance",
            passed=True,
        )
    variances = X.var(axis=0)
    low_var_cols: List[int] = [int(i) for i, v in enumerate(variances) if v <= min_variance]
    if low_var_cols:
        return DataValidationResult(
            check_name="feature_variance",
            passed=False,
            error_message=(f"Columns {low_var_cols} have variance <= {min_variance}."),
        )
    return DataValidationResult(check_name="feature_variance", passed=True)


# ── Composite validator ───────────────────────────────────────────────────────


@dataclass
class DataPropertyValidator:
    """Run a battery of data validation checks against a feature matrix.

    Usage::

        validator = DataPropertyValidator(expected_features=10, n_classes=2)
        results = validator.validate(X, y)

    Attributes:
        expected_features: If set, :func:`check_feature_matrix_shape` will
            enforce this column count.
        n_classes: If set together with *y*, labels are validated to be in
            ``[0, n_classes)``.
        min_samples: Minimum required sample count.
        min_variance: Minimum acceptable per-feature variance; set to ``0.0``
            to skip constant-feature detection.
    """

    expected_features: Optional[int] = None
    n_classes: Optional[int] = None
    min_samples: int = 1
    min_variance: float = 0.0

    def validate(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> List[DataValidationResult]:
        """Run all configured validation checks.

        Args:
            X: Feature matrix to validate.
            y: Optional label array; if provided, label-specific checks are
               also executed.

        Returns:
            A list of :class:`DataValidationResult` instances, one per check.
        """
        results: List[DataValidationResult] = []

        results.append(check_no_nan_or_inf(X))
        results.append(check_feature_matrix_shape(X, expected_features=self.expected_features))
        results.append(check_minimum_samples(X, min_samples=self.min_samples))
        results.append(check_feature_variance(X, min_variance=self.min_variance))

        if y is not None:
            results.append(check_label_array_shape(y, X))
            if self.n_classes is not None:
                results.append(check_class_labels_in_range(y, n_classes=self.n_classes))

        return results

    def all_passed(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> bool:
        """Return ``True`` only if every validation check passes.

        Args:
            X: Feature matrix to validate.
            y: Optional label array.

        Returns:
            ``True`` when no check fails.
        """
        return all(r.passed for r in self.validate(X, y))
