from astroml.utils.exceptions import AstroMLError
"""Data quality assertions and testing framework for pipeline data.

Issue #638 Step 1 & 6: Implements data quality assertion library,
data diff detection, and regression testing.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Data quality assertions
# ---------------------------------------------------------------------------


class AssertionSeverity(Enum):
    """Severity of a data quality assertion failure."""

    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


@dataclass
class DataTestResult:
    """Result of a single data quality test."""

    test_name: str
    passed: bool
    severity: AssertionSeverity = AssertionSeverity.ERROR
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
        }


class DataAssertion:
    """Collection of data quality assertions for pipeline testing.

    Provides common assertions for data quality: null checks, range checks,
    uniqueness, row count, distribution, and more.

    Example:
        assertions = DataAssertion()
        assertions.assert_not_null(df, "account_id")
        assertions.assert_unique(df, "account_id")
        assertions.assert_row_count(df, min_rows=100)
        assertions.assert_column_values_in_set(df, "status", {"active", "inactive"})
    """

    def __init__(self) -> None:
        self._results: list[DataTestResult] = []

    @property
    def results(self) -> list[DataTestResult]:
        return list(self._results)

    @property
    def all_passed(self) -> bool:
        return all(r.passed or r.severity == AssertionSeverity.WARNING for r in self._results)

    def _record(
        self,
        test_name: str,
        passed: bool,
        severity: AssertionSeverity = AssertionSeverity.ERROR,
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> DataTestResult:
        result = DataTestResult(
            test_name=test_name,
            passed=passed,
            severity=severity,
            message=message,
            details=details or {},
        )
        self._results.append(result)
        if not passed and severity == AssertionSeverity.FATAL:
            logger.error(f"FATAL: {test_name} - {message}")
        elif not passed:
            logger.warning(f"{severity.value.upper()}: {test_name} - {message}")
        return result

    # ── Null / missing ──────────────────────────────────────────────────

    def assert_not_null(
        self,
        data: Any,
        column: str,
        severity: AssertionSeverity = AssertionSeverity.ERROR,
    ) -> DataTestResult:
        """Assert no null values in a column."""
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            return self._record(f"assert_not_null({column})", False, severity, "Data is not a DataFrame")

        null_count = int(data[column].isnull().sum())
        passed = null_count == 0
        msg = f"Column '{column}' has {null_count} null values" if not passed else ""
        return self._record(f"assert_not_null({column})", passed, severity, msg, {"null_count": null_count})

    def assert_max_null_fraction(
        self,
        data: Any,
        column: str,
        max_fraction: float = 0.01,
        severity: AssertionSeverity = AssertionSeverity.WARNING,
    ) -> DataTestResult:
        """Assert null fraction is below threshold."""
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            return self._record(f"assert_max_null_fraction({column})", False, severity, "Data is not a DataFrame")

        null_frac = float(data[column].isnull().mean())
        passed = null_frac <= max_fraction
        msg = f"Column '{column}' null fraction {null_frac:.4f} exceeds max {max_fraction:.4f}" if not passed else ""
        return self._record(f"assert_max_null_fraction({column})", passed, severity, msg, {"null_fraction": null_frac})

    # ── Uniqueness ──────────────────────────────────────────────────────

    def assert_unique(
        self,
        data: Any,
        column: str,
        severity: AssertionSeverity = AssertionSeverity.ERROR,
    ) -> DataTestResult:
        """Assert all values in a column are unique."""
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            return self._record(f"assert_unique({column})", False, severity, "Data is not a DataFrame")

        dupes = int(data[column].duplicated().sum())
        passed = dupes == 0
        msg = f"Column '{column}' has {dupes} duplicate values" if not passed else ""
        return self._record(f"assert_unique({column})", passed, severity, msg, {"duplicate_count": dupes})

    # ── Value sets ──────────────────────────────────────────────────────

    def assert_column_values_in_set(
        self,
        data: Any,
        column: str,
        allowed_set: set[str],
        severity: AssertionSeverity = AssertionSeverity.ERROR,
    ) -> DataTestResult:
        """Assert all values in a column are within an allowed set."""
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            return self._record(f"assert_column_values_in_set({column})", False, severity, "Data is not a DataFrame")

        mask = data[column].notna()
        invalid = data.loc[mask, column].apply(lambda v: v not in allowed_set)
        invalid_count = int(invalid.sum())
        passed = invalid_count == 0
        msg = f"Column '{column}' has {invalid_count} values outside allowed set" if not passed else ""
        return self._record(
            f"assert_column_values_in_set({column})",
            passed, severity, msg,
            {"invalid_count": invalid_count, "allowed_set": sorted(allowed_set)},
        )

    # ── Row count ───────────────────────────────────────────────────────

    def assert_row_count(
        self,
        data: Any,
        min_rows: int = 0,
        max_rows: int | None = None,
        severity: AssertionSeverity = AssertionSeverity.ERROR,
    ) -> DataTestResult:
        """Assert row count is within bounds."""
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                count = len(data)
            else:
                return self._record("assert_row_count", False, severity, "Data is not a DataFrame or list")
        else:
            count = len(data)

        passed = count >= min_rows
        if max_rows is not None:
            passed = passed and count <= max_rows

        msg = f"Row count {count} outside bounds [{min_rows}, {max_rows or '∞'}]" if not passed else ""
        return self._record(
            "assert_row_count", passed, severity, msg,
            {"row_count": count, "min_rows": min_rows, "max_rows": max_rows},
        )

    # ── Statistical ─────────────────────────────────────────────────────

    def assert_column_mean_between(
        self,
        data: Any,
        column: str,
        lower: float,
        upper: float,
        severity: AssertionSeverity = AssertionSeverity.WARNING,
    ) -> DataTestResult:
        """Assert column mean is within [lower, upper]."""
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            return self._record(f"assert_column_mean_between({column})", False, severity, "Data is not a DataFrame")

        mean_val = float(data[column].mean())
        passed = lower <= mean_val <= upper
        msg = f"Column '{column}' mean {mean_val:.4f} outside [{lower:.4f}, {upper:.4f}]" if not passed else ""
        return self._record(
            f"assert_column_mean_between({column})", passed, severity, msg,
            {"mean": mean_val, "lower": lower, "upper": upper},
        )

    def assert_column_std_between(
        self,
        data: Any,
        column: str,
        lower: float,
        upper: float,
        severity: AssertionSeverity = AssertionSeverity.WARNING,
    ) -> DataTestResult:
        """Assert column standard deviation is within [lower, upper]."""
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            return self._record(f"assert_column_std_between({column})", False, severity, "Data is not a DataFrame")

        std_val = float(data[column].std())
        passed = lower <= std_val <= upper
        msg = f"Column '{column}' std {std_val:.4f} outside [{lower:.4f}, {upper:.4f}]" if not passed else ""
        return self._record(
            f"assert_column_std_between({column})", passed, severity, msg,
            {"std": std_val, "lower": lower, "upper": upper},
        )

    # ── Distributions ───────────────────────────────────────────────────

    def assert_column_distribution_change(
        self,
        data: Any,
        reference: Any,
        column: str,
        max_ks_statistic: float = 0.1,
        severity: AssertionSeverity = AssertionSeverity.WARNING,
    ) -> DataTestResult:
        """Assert Kolmogorov-Smirnov statistic between two distributions is below threshold."""
        import pandas as pd
        from scipy import stats

        if not isinstance(data, pd.DataFrame) or not isinstance(reference, pd.DataFrame):
            return self._record(f"assert_column_distribution_change({column})", False, severity, "Data is not a DataFrame")

        try:
            ks_stat, p_value = stats.ks_2samp(
                data[column].dropna().values,
                reference[column].dropna().values,
            )
            passed = float(ks_stat) <= max_ks_statistic
            msg = f"KS statistic {ks_stat:.4f} exceeds threshold {max_ks_statistic}" if not passed else ""
            return self._record(
                f"assert_column_distribution_change({column})", passed, severity, msg,
                {"ks_statistic": float(ks_stat), "p_value": float(p_value), "threshold": max_ks_statistic},
            )
        except AstroMLError as e:
            return self._record(f"assert_column_distribution_change({column})", False, severity, str(e))

    def clear(self) -> None:
        """Clear all recorded results."""
        self._results.clear()


# ---------------------------------------------------------------------------
# Data Test Suite
# ---------------------------------------------------------------------------


@dataclass
class DataTestSuite:
    """A named collection of data quality tests for a pipeline stage.

    Example:
        suite = DataTestSuite(name="Feature Engineering Tests")
        suite.add_test(lambda df: assertions.assert_not_null(df, "account_id"))
        results = suite.run(data=feature_df)
    """

    name: str
    suite_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tests: list[Callable[[Any], DataTestResult]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_test(self, test_fn: Callable[[Any], DataTestResult]) -> None:
        """Add a test to the suite."""
        self.tests.append(test_fn)

    def run(self, data: Any) -> list[DataTestResult]:
        """Run all tests in the suite against the provided data.

        Args:
            data: The data to test (usually a DataFrame).

        Returns:
            List of DataTestResult objects.
        """
        import time

        results: list[DataTestResult] = []
        logger.info(f"Running test suite: {self.name} ({len(self.tests)} tests)")

        for test_fn in self.tests:
            start = time.monotonic()
            try:
                result = test_fn(data)
            except AstroMLError as e:
                result = DataTestResult(
                    test_name=test_fn.__name__ if hasattr(test_fn, "__name__") else str(test_fn),
                    passed=False,
                    severity=AssertionSeverity.ERROR,
                    message=f"Test raised exception: {e}",
                )
            result.duration_ms = (time.monotonic() - start) * 1000
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        logger.info(f"Suite '{self.name}': {passed}/{len(results)} passed")
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "test_count": len(self.tests),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Data diff and regression detection
# ---------------------------------------------------------------------------


@dataclass
class DataDiffReport:
    """Report comparing two datasets for drift detection.

    Computes statistical comparisons between a current dataset and a
    reference (baseline) dataset.

    Example:
        diff = DataDiffReport.compare(current_df, reference_df)
        if diff.has_significant_change:
            print(diff.summary())
    """

    total_rows_current: int = 0
    total_rows_reference: int = 0
    row_diff: int = 0
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    columns_unchanged: list[str] = field(default_factory=list)
    column_diffs: dict[str, dict[str, float]] = field(default_factory=dict)
    significant_change: bool = False

    @property
    def has_significant_change(self) -> bool:
        return self.significant_change

    @classmethod
    def compare(
        cls,
        current: Any,
        reference: Any,
        key_columns: Sequence[str] | None = None,
        numeric_columns: Sequence[str] | None = None,
        max_mean_drift: float = 0.1,
    ) -> DataDiffReport:
        """Compare two DataFrames and produce a diff report.

        Args:
            current: Current dataset.
            reference: Reference (baseline) dataset.
            key_columns: Key columns to check for row-level diffs.
            numeric_columns: Numeric columns to check for drift.
            max_mean_drift: Maximum allowed mean drift for numeric columns.

        Returns:
            DataDiffReport with comparison results.
        """
        import pandas as pd

        if not isinstance(current, pd.DataFrame) or not isinstance(reference, pd.DataFrame):
            return cls()

        report = cls(
            total_rows_current=len(current),
            total_rows_reference=len(reference),
            row_diff=len(current) - len(reference),
        )

        # Column changes
        curr_cols = set(current.columns)
        ref_cols = set(reference.columns)
        report.columns_added = sorted(curr_cols - ref_cols)
        report.columns_removed = sorted(ref_cols - curr_cols)
        report.columns_unchanged = sorted(curr_cols & ref_cols)

        # Numeric drift
        numeric = list(numeric_columns) if numeric_columns else [
            c for c in report.columns_unchanged
            if pd.api.types.is_numeric_dtype(current[c]) and pd.api.types.is_numeric_dtype(reference[c])
        ]

        for col in numeric[:50]:  # Limit to 50 columns for performance
            try:
                cur_mean = float(current[col].mean())
                ref_mean = float(reference[col].mean())
                drift = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-10)

                report.column_diffs[col] = {
                    "current_mean": cur_mean,
                    "reference_mean": ref_mean,
                    "drift_ratio": drift,
                }

                if drift > max_mean_drift:
                    report.significant_change = True
            except AstroMLError:
                pass

        return report

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"DataDiffReport: {self.row_diff:+d} rows "
            f"(current={self.total_rows_current}, reference={self.total_rows_reference})",
        ]
        if self.columns_added:
            lines.append(f"  Columns added: {', '.join(self.columns_added)}")
        if self.columns_removed:
            lines.append(f"  Columns removed: {', '.join(self.columns_removed)}")
        if self.column_diffs:
            drifted = [k for k, v in self.column_diffs.items() if v.get("drift_ratio", 0) > 0.1]
            if drifted:
                lines.append(f"  Drifted columns: {', '.join(drifted)}")
        lines.append(f"  Significant change: {self.significant_change}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows_current": self.total_rows_current,
            "total_rows_reference": self.total_rows_reference,
            "row_diff": self.row_diff,
            "columns_added": self.columns_added,
            "columns_removed": self.columns_removed,
            "columns_unchanged_count": len(self.columns_unchanged),
            "column_diffs": self.column_diffs,
            "significant_change": self.significant_change,
        }


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------


@dataclass
class RegressionTest:
    """Regression test for comparing pipeline outputs against a golden baseline.

    Captures expected outputs (golden data) and compares against current runs.

    Example:
        regression = RegressionTest(name="feature_computation_v1", tolerance=1e-5)
        regression.capture_baseline(expected_output)
        # Later:
        result = regression.check(current_output)
    """

    name: str
    tolerance: float = 1e-6
    baseline: Any | None = None
    test_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def capture_baseline(self, data: Any) -> None:
        """Capture the expected baseline output."""
        self.baseline = data
        logger.info(f"Regression test '{self.name}': baseline captured")

    def check(self, current: Any) -> DataTestResult:
        """Check if current output matches the baseline within tolerance.

        Args:
            current: Current pipeline output.

        Returns:
            DataTestResult with pass/fail status.
        """
        import numpy as np
        import pandas as pd

        if self.baseline is None:
            return DataTestResult(
                test_name=f"regression:{self.name}",
                passed=False,
                message="No baseline captured",
            )

        try:
            if isinstance(self.baseline, np.ndarray) and isinstance(current, np.ndarray):
                diff = np.max(np.abs(self.baseline - current))
                passed = diff <= self.tolerance
                return DataTestResult(
                    test_name=f"regression:{self.name}",
                    passed=bool(passed),
                    message=f"Max absolute diff: {diff:.6e}" if not passed else "",
                    details={"max_abs_diff": float(diff), "tolerance": self.tolerance},
                )

            if isinstance(self.baseline, pd.DataFrame) and isinstance(current, pd.DataFrame):
                if not self.baseline.columns.equals(current.columns):
                    return DataTestResult(
                        test_name=f"regression:{self.name}",
                        passed=False,
                        message="Column mismatch",
                    )
                numeric_cols = self.baseline.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    diff = np.max(np.abs(
                        self.baseline[numeric_cols].values - current[numeric_cols].values
                    ))
                    passed = diff <= self.tolerance
                    return DataTestResult(
                        test_name=f"regression:{self.name}",
                        passed=bool(passed),
                        message=f"Max absolute diff: {diff:.6e}" if not passed else "",
                        details={"max_abs_diff": float(diff), "tolerance": self.tolerance},
                    )

            if isinstance(self.baseline, dict) and isinstance(current, dict):
                # Compare dicts recursively
                all_keys = set(self.baseline.keys()) | set(current.keys())
                for key in all_keys:
                    if key not in self.baseline or key not in current:
                        return DataTestResult(
                            test_name=f"regression:{self.name}",
                            passed=False,
                            message=f"Key '{key}' missing",
                        )
                    b_val = self.baseline[key]
                    c_val = current[key]
                    if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
                        if abs(b_val - c_val) > self.tolerance:
                            return DataTestResult(
                                test_name=f"regression:{self.name}",
                                passed=False,
                                message=f"Key '{key}' differs: {b_val} vs {c_val}",
                            )

                return DataTestResult(
                    test_name=f"regression:{self.name}",
                    passed=True,
                )

            # Default: equality comparison
            passed = self.baseline == current
            return DataTestResult(
                test_name=f"regression:{self.name}",
                passed=bool(passed),
                message="" if passed else "Output differs from baseline",
            )

        except AstroMLError as e:
            return DataTestResult(
                test_name=f"regression:{self.name}",
                passed=False,
                message=f"Comparison error: {e}",
            )


__all__ = [
    "AssertionSeverity",
    "DataTestResult",
    "DataAssertion",
    "DataTestSuite",
    "DataDiffReport",
    "RegressionTest",
]