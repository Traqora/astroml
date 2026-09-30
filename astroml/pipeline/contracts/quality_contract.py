from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityValidationResult:
    """Result of validating a DataFrame against a quality contract.

    Attributes:
        is_valid: Whether all constraints passed.
        contract_name: Name of the contract that was validated.
        violations: List of dicts describing each failed constraint.
        passed_constraints: Number of constraints that passed.
        failed_constraints: Number of constraints that failed.
        total_constraints: Total number of constraints evaluated.
    """

    is_valid: bool
    contract_name: str
    violations: list[dict[str, Any]] = field(default_factory=list)
    passed_constraints: int = 0
    failed_constraints: int = 0
    total_constraints: int = 0


SUPPORTED_CONSTRAINT_TYPES = frozenset(
    {
        "range",
        "null_ratio",
        "distinct_count",
        "unique",
        "value_set",
        "distribution_similarity",
    }
)


class QualityContract:
    """Defines data quality constraints on DataFrame columns.

    Supports constraint types: range, null_ratio, distinct_count, unique,
    value_set, distribution_similarity.

    Attributes:
        name: Human-readable name for this contract.
        version: Semantic version string.
        constraints: List of constraint dictionaries.
    """

    def __init__(self, name: str = "", version: str = "1.0.0") -> None:
        self.name = name
        self.version = version
        self.constraints: list[dict[str, Any]] = []

    def add_constraint(
        self,
        column: str,
        constraint_type: str,
        threshold: dict[str, Any] | None = None,
    ) -> None:
        """Add a quality constraint.

        Args:
            column: Target column name.
            constraint_type: One of "range", "null_ratio", "distinct_count",
                "unique", "value_set", "distribution_similarity".
            threshold: Constraint-specific parameters dict.
                For "range": {"min": ..., "max": ...}
                For "null_ratio": {"max": float}
                For "distinct_count": {"min": int, "max": int}
                For "unique": {} (no extra params)
                For "value_set": {"values": list}
                For "distribution_similarity": {"reference": list | pd.Series}

        Raises:
            ValueError: If constraint_type is not supported.
        """
        if constraint_type not in SUPPORTED_CONSTRAINT_TYPES:
            raise ValueError(
                f"Unsupported constraint type '{constraint_type}'. "
                f"Supported types: {sorted(SUPPORTED_CONSTRAINT_TYPES)}"
            )
        self.constraints.append(
            {
                "column": column,
                "type": constraint_type,
                "threshold": threshold or {},
            }
        )

    def validate(self, df: pd.DataFrame) -> QualityValidationResult:
        """Validate a DataFrame against all quality constraints.

        Args:
            df: DataFrame to validate.

        Returns:
            QualityValidationResult with pass/fail per constraint.
        """
        violations: list[dict[str, Any]] = []
        passed = 0
        failed = 0

        for constraint in self.constraints:
            col = constraint["column"]
            ctype = constraint["type"]
            threshold = constraint["threshold"]

            if col not in df.columns:
                violations.append(
                    {
                        "column": col,
                        "type": ctype,
                        "message": f"Column '{col}' not found in DataFrame",
                    }
                )
                failed += 1
                continue

            series = df[col]
            result = self._check_constraint(series, ctype, threshold)
            if result is not None:
                violations.append(
                    {
                        "column": col,
                        "type": ctype,
                        "message": result,
                        "threshold": threshold,
                    }
                )
                failed += 1
            else:
                passed += 1

        total = len(self.constraints)
        return QualityValidationResult(
            is_valid=failed == 0,
            contract_name=self.name,
            violations=violations,
            passed_constraints=passed,
            failed_constraints=failed,
            total_constraints=total,
        )

    def _check_constraint(
        self,
        series: pd.Series,
        constraint_type: str,
        threshold: dict[str, Any],
    ) -> str | None:
        """Check a single constraint against a series. Returns None if pass, error message if fail."""
        clean = series.dropna()

        if constraint_type == "range":
            return self._check_range(clean, threshold)
        elif constraint_type == "null_ratio":
            return self._check_null_ratio(series, threshold)
        elif constraint_type == "distinct_count":
            return self._check_distinct_count(clean, threshold)
        elif constraint_type == "unique":
            return self._check_unique(series)
        elif constraint_type == "value_set":
            return self._check_value_set(clean, threshold)
        elif constraint_type == "distribution_similarity":
            return self._check_distribution_similarity(clean, threshold)
        return None

    def _check_range(self, series: pd.Series, threshold: dict[str, Any]) -> str | None:
        if not pd.api.types.is_numeric_dtype(series.dtype):
            return "Column is not numeric, cannot check range"
        if "min" in threshold and (series < threshold["min"]).any():
            return f"Values below minimum {threshold['min']}"
        if "max" in threshold and (series > threshold["max"]).any():
            return f"Values above maximum {threshold['max']}"
        return None

    def _check_null_ratio(self, series: pd.Series, threshold: dict[str, Any]) -> str | None:
        max_ratio = threshold.get("max", 0.0)
        actual_ratio = float(series.isna().mean())
        if actual_ratio > max_ratio:
            return f"Null ratio {actual_ratio:.4f} exceeds max {max_ratio}"
        return None

    def _check_distinct_count(self, series: pd.Series, threshold: dict[str, Any]) -> str | None:
        distinct = int(series.nunique())
        min_val = threshold.get("min")
        max_val = threshold.get("max")
        if min_val is not None and distinct < min_val:
            return f"Distinct count {distinct} below minimum {min_val}"
        if max_val is not None and distinct > max_val:
            return f"Distinct count {distinct} exceeds maximum {max_val}"
        return None

    def _check_unique(self, series: pd.Series) -> str | None:
        non_null = series.dropna()
        if non_null.duplicated().any():
            dup_count = int(non_null.duplicated().sum())
            return f"Found {dup_count} duplicate values in unique column"
        return None

    def _check_value_set(self, series: pd.Series, threshold: dict[str, Any]) -> str | None:
        allowed = set(threshold.get("values", []))
        if not allowed:
            return None
        actual_values = set(series.unique())
        invalid = actual_values - allowed - {None}
        if invalid:
            return f"Values {sorted(invalid)} not in allowed set"
        return None

    def _check_distribution_similarity(
        self,
        series: pd.Series,
        threshold: dict[str, Any],
    ) -> str | None:
        # Simple distribution check using quantile comparison
        reference = threshold.get("reference")
        if reference is None:
            return None
        ref_series = pd.Series(reference).dropna()
        if len(ref_series) == 0 or len(series) == 0:
            return None
        ref_quantiles = ref_series.quantile([0.25, 0.5, 0.75])
        actual_quantiles = series.quantile([0.25, 0.5, 0.75])
        tolerance = threshold.get("tolerance", 0.5)
        for q in [0.25, 0.5, 0.75]:
            ref_val = ref_quantiles.get(q)
            actual_val = actual_quantiles.get(q)
            if ref_val is not None and actual_val is not None:
                diff = abs(float(actual_val) - float(ref_val))
                ref_magnitude = abs(float(ref_val)) if float(ref_val) != 0 else 1.0
                if diff / ref_magnitude > tolerance:
                    return f"Quantile {q} deviation {diff:.4f} exceeds tolerance {tolerance}"
        return None

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> QualityContract:
        """Create a QualityContract from a configuration dictionary.

        Args:
            config_dict: Dict with optional "name", "version", and "constraints" keys.
                Each constraint is a dict with "column", "type", and "threshold".

        Returns:
            A new QualityContract instance.
        """
        contract = cls(
            name=config_dict.get("name", ""),
            version=config_dict.get("version", "1.0.0"),
        )
        for constraint in config_dict.get("constraints", []):
            contract.add_constraint(
                column=constraint["column"],
                constraint_type=constraint["type"],
                threshold=constraint.get("threshold"),
            )
        return contract

    def merge(self, other: QualityContract) -> QualityContract:
        """Merge two quality contracts into a new contract.

        Args:
            other: Another QualityContract to merge with.

        Returns:
            A new QualityContract containing constraints from both.
        """
        merged = QualityContract(
            name=(
                f"{self.name}+{other.name}"
                if self.name and other.name
                else (self.name or other.name)
            ),
            version=self.version,
        )
        merged.constraints = deepcopy(self.constraints) + deepcopy(other.constraints)
        return merged
