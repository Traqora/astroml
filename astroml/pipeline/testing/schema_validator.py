from astroml.utils.exceptions import AstroMLError
"""Schema validation for data pipeline testing.

Issue #638 Step 2: Implements schema validation for pipeline stages,
supporting column types, constraints, and Great Expectations-style expectations.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------


class ColumnType(Enum):
    """Supported column types for schema validation."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"  # Accepts both int and float


@dataclass
class ColumnExpectation:
    """A single expectation for a column's values.

    Attributes:
        column: Column name.
        expectation_type: Type of check (e.g., 'not_null', 'in_set', 'range').
        kwargs: Parameters for the expectation.
    """

    column: str
    expectation_type: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "expectation_type": self.expectation_type,
            "kwargs": self.kwargs,
        }

    @classmethod
    def not_null(cls, column: str) -> ColumnExpectation:
        return cls(column=column, expectation_type="not_null")

    @classmethod
    def unique(cls, column: str) -> ColumnExpectation:
        return cls(column=column, expectation_type="unique")

    @classmethod
    def in_set(cls, column: str, values: set[str]) -> ColumnExpectation:
        return cls(column=column, expectation_type="in_set", kwargs={"values": list(values)})

    @classmethod
    def range(cls, column: str, min_value: float | None = None, max_value: float | None = None) -> ColumnExpectation:
        return cls(column=column, expectation_type="range", kwargs={"min": min_value, "max": max_value})

    @classmethod
    def regex_match(cls, column: str, pattern: str) -> ColumnExpectation:
        return cls(column=column, expectation_type="regex_match", kwargs={"pattern": pattern})

    @classmethod
    def column_type(cls, column: str, col_type: ColumnType | str) -> ColumnExpectation:
        type_val = col_type.value if isinstance(col_type, ColumnType) else col_type
        return cls(column=column, expectation_type="column_type", kwargs={"type": type_val})


@dataclass
class SchemaDefinition:
    """Definition of a table/schema with expected columns and types.

    Example:
        schema = SchemaDefinition(
            name="transaction_events",
            columns={
                "tx_id": ColumnType.STRING,
                "src_account": ColumnType.STRING,
                "dst_account": ColumnType.STRING,
                "amount": ColumnType.FLOAT,
                "timestamp": ColumnType.DATETIME,
            },
            required_columns={"tx_id", "src_account", "dst_account", "amount", "timestamp"},
            expectations=[
                ColumnExpectation.not_null("tx_id"),
                ColumnExpectation.unique("tx_id"),
                ColumnExpectation.range("amount", min_value=0),
            ],
        )
    """

    name: str
    columns: dict[str, ColumnType]
    required_columns: set[str] = field(default_factory=set)
    optional_columns: set[str] = field(default_factory=set)
    expectations: list[ColumnExpectation] = field(default_factory=list)
    version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> Any:
        if not self.required_columns:
            self.required_columns = set(self.columns.keys())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "columns": {k: v.value for k, v in self.columns.items()},
            "required_columns": sorted(self.required_columns),
            "optional_columns": sorted(self.optional_columns),
            "expectations": [e.to_dict() for e in self.expectations],
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path) -> None:
        """Save schema definition to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> SchemaDefinition:
        """Load schema definition from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        columns = {k: ColumnType(v) for k, v in data["columns"].items()}
        expectations = [ColumnExpectation(**e) for e in data.get("expectations", [])]

        return cls(
            name=data["name"],
            columns=columns,
            required_columns=set(data.get("required_columns", [])),
            optional_columns=set(data.get("optional_columns", [])),
            expectations=expectations,
            version=data.get("version", "v1"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Validation results
# ---------------------------------------------------------------------------


class ValidationStatus(Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ExpectationResult:
    """Result of a single expectation validation."""

    expectation: ColumnExpectation
    status: ValidationStatus
    message: str = ""
    observed_value: Any = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in (ValidationStatus.PASSED, ValidationStatus.WARNING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expectation": self.expectation.to_dict(),
            "status": self.status.value,
            "message": self.message,
            "observed_value": self.observed_value,
            "details": self.details,
        }


@dataclass
class SchemaValidationResult:
    """Result of a full schema validation run."""

    schema_name: str
    status: ValidationStatus = ValidationStatus.PASSED
    results: list[ExpectationResult] = field(default_factory=list)
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.status != ValidationStatus.FAILED

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.status == ValidationStatus.PASSED)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status == ValidationStatus.FAILED)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.status == ValidationStatus.WARNING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "status": self.status.value,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "warnings": self.warning_count,
            "duration_ms": round(self.duration_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Schema Validator
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Validate DataFrames against schema definitions.

    Supports column presence, type checking, and a suite of expectations
    inspired by Great Expectations.

    Example:
        validator = SchemaValidator()
        result = validator.validate(df, schema_definition)

        if not result.success:
            for r in result.results:
                if r.status == ValidationStatus.FAILED:
                    print(f"FAIL: {r.message}")
    """

    def __init__(self) -> None:
        self._schemas: dict[str, SchemaDefinition] = {}

    def register_schema(self, schema: SchemaDefinition) -> None:
        """Register a schema for validation."""
        self._schemas[schema.name] = schema

    def get_schema(self, name: str) -> SchemaDefinition | None:
        """Get a registered schema by name."""
        return self._schemas.get(name)

    def validate(
        self,
        data: Any,
        schema: SchemaDefinition | str,
        strict: bool = False,
    ) -> SchemaValidationResult:
        """Validate data against a schema definition.

        Args:
            data: DataFrame to validate.
            schema: SchemaDefinition or name of a registered schema.
            strict: If True, unknown columns cause a failure.

        Returns:
            SchemaValidationResult with detailed results.
        """
        import time

        import numpy as np
        import pandas as pd

        start = time.monotonic()

        if isinstance(schema, str):
            resolved = self._schemas.get(schema)
            if resolved is None:
                return SchemaValidationResult(
                    schema_name=schema,
                    status=ValidationStatus.FAILED,
                    results=[ExpectationResult(
                        expectation=ColumnExpectation("schema", "exists"),
                        status=ValidationStatus.FAILED,
                        message=f"Schema '{schema}' not registered",
                    )],
                )
            schema = resolved

        if not isinstance(data, pd.DataFrame):
            return SchemaValidationResult(
                schema_name=schema.name,
                status=ValidationStatus.FAILED,
                results=[ExpectationResult(
                    expectation=ColumnExpectation("data", "is_dataframe"),
                    status=ValidationStatus.FAILED,
                    message="Data must be a pandas DataFrame",
                )],
            )

        result = SchemaValidationResult(schema_name=schema.name)
        observed_columns = set(data.columns)

        # Check required columns
        for col in sorted(schema.required_columns):
            exp = ColumnExpectation(column=col, expectation_type="column_exists")
            if col not in observed_columns:
                result.results.append(ExpectationResult(
                    expectation=exp,
                    status=ValidationStatus.FAILED,
                    message=f"Required column '{col}' is missing",
                ))
            else:
                result.results.append(ExpectationResult(
                    expectation=exp,
                    status=ValidationStatus.PASSED,
                    message=f"Column '{col}' exists",
                ))

        # Check optional columns (warn if missing)
        for col in sorted(schema.optional_columns):
            exp = ColumnExpectation(column=col, expectation_type="column_exists")
            if col not in observed_columns:
                result.results.append(ExpectationResult(
                    expectation=exp,
                    status=ValidationStatus.WARNING,
                    message=f"Optional column '{col}' is missing",
                ))

        # Check unexpected columns (strict mode)
        if strict:
            expected_cols = schema.required_columns | schema.optional_columns
            extra_cols = observed_columns - expected_cols
            for col in sorted(extra_cols):
                result.results.append(ExpectationResult(
                    expectation=ColumnExpectation(column=col, expectation_type="no_extra_columns"),
                    status=ValidationStatus.FAILED,
                    message=f"Unexpected column '{col}' (strict mode)",
                ))

        # Validate column types for present columns
        common_cols = observed_columns & schema.columns.keys()
        for col in sorted(common_cols):
            expected_type = schema.columns[col]
            actual_dtype = str(data[col].dtype)
            is_valid = self._check_type(data[col], expected_type)

            exp = ColumnExpectation.column_type(col, expected_type)
            if is_valid:
                result.results.append(ExpectationResult(
                    expectation=exp,
                    status=ValidationStatus.PASSED,
                    observed_value=actual_dtype,
                ))
            else:
                result.results.append(ExpectationResult(
                    expectation=exp,
                    status=ValidationStatus.FAILED,
                    message=f"Column '{col}' expected type {expected_type.value}, got {actual_dtype}",
                    observed_value=actual_dtype,
                    details={"expected": expected_type.value, "actual": actual_dtype},
                ))

        # Run expectations
        for expectation in schema.expectations:
            if expectation.column not in observed_columns:
                result.results.append(ExpectationResult(
                    expectation=expectation,
                    status=ValidationStatus.WARNING,
                    message=f"Cannot validate expectation on missing column '{expectation.column}'",
                ))
                continue

            series = data[expectation.column]
            exp_result = self._evaluate_expectation(series, expectation)
            result.results.append(exp_result)

        # Determine overall status
        if any(r.status == ValidationStatus.FAILED for r in result.results):
            result.status = ValidationStatus.FAILED
        elif any(r.status == ValidationStatus.WARNING for r in result.results):
            result.status = ValidationStatus.WARNING

        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    @staticmethod
    def _check_type(series: Any, expected: ColumnType) -> bool:
        """Check if a pandas Series matches the expected ColumnType."""
        import numpy as np
        import pandas as pd

        try:
            if expected == ColumnType.INTEGER:
                return pd.api.types.is_integer_dtype(series)
            elif expected == ColumnType.FLOAT:
                return pd.api.types.is_float_dtype(series)
            elif expected == ColumnType.NUMERIC:
                return pd.api.types.is_numeric_dtype(series)
            elif expected == ColumnType.STRING:
                return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
            elif expected == ColumnType.BOOLEAN:
                return pd.api.types.is_bool_dtype(series)
            elif expected == ColumnType.DATETIME:
                return pd.api.types.is_datetime64_any_dtype(series)
            elif expected == ColumnType.CATEGORICAL:
                return isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series)
            return True
        except AstroMLError:
            return False

    @staticmethod
    def _evaluate_expectation(
        series: Any,
        expectation: ColumnExpectation,
    ) -> ExpectationResult:
        """Evaluate a single expectation against a column series."""
        import numpy as np

        try:
            if expectation.expectation_type == "not_null":
                null_count = int(series.isnull().sum())
                ok = null_count == 0
                msg = f"{null_count} null values found" if not ok else ""
                return ExpectationResult(
                    expectation=expectation,
                    status=ValidationStatus.FAILED if not ok else ValidationStatus.PASSED,
                    message=msg,
                    observed_value=null_count,
                )

            elif expectation.expectation_type == "unique":
                dupes = int(series.duplicated().sum())
                ok = dupes == 0
                msg = f"{dupes} duplicate values" if not ok else ""
                return ExpectationResult(
                    expectation=expectation,
                    status=ValidationStatus.FAILED if not ok else ValidationStatus.PASSED,
                    message=msg,
                    observed_value=dupes,
                )

            elif expectation.expectation_type == "in_set":
                allowed = set(expectation.kwargs["values"])
                mask = series.notna()
                invalid = series[mask].apply(lambda v: v not in allowed)
                invalid_count = int(invalid.sum())
                ok = invalid_count == 0
                msg = f"{invalid_count} values outside allowed set" if not ok else ""
                return ExpectationResult(
                    expectation=expectation,
                    status=ValidationStatus.FAILED if not ok else ValidationStatus.PASSED,
                    message=msg,
                    observed_value=invalid_count,
                )

            elif expectation.expectation_type == "range":
                min_val = expectation.kwargs.get("min")
                max_val = expectation.kwargs.get("max")
                mask = series.notna()
                ok = True
                out_of_range = 0

                if min_val is not None:
                    below = series[mask] < min_val
                    out_of_range += int(below.sum())
                    if below.any():
                        ok = False
                if max_val is not None:
                    above = series[mask] > max_val
                    out_of_range += int(above.sum())
                    if above.any():
                        ok = False

                msg = f"{out_of_range} values outside range" if not ok else ""
                return ExpectationResult(
                    expectation=expectation,
                    status=ValidationStatus.FAILED if not ok else ValidationStatus.PASSED,
                    message=msg,
                    observed_value=out_of_range,
                )

            elif expectation.expectation_type == "regex_match":
                import re
                pattern = expectation.kwargs["pattern"]
                mask = series.notna()
                non_matching = series[mask].apply(lambda v: not re.match(pattern, str(v)))
                non_match_count = int(non_matching.sum())
                ok = non_match_count == 0
                msg = f"{non_match_count} values don't match pattern" if not ok else ""
                return ExpectationResult(
                    expectation=expectation,
                    status=ValidationStatus.FAILED if not ok else ValidationStatus.PASSED,
                    message=msg,
                    observed_value=non_match_count,
                )

            elif expectation.expectation_type == "column_type":
                expected_type_str = expectation.kwargs["type"]
                try:
                    expected = ColumnType(expected_type_str)
                except ValueError:
                    return ExpectationResult(
                        expectation=expectation,
                        status=ValidationStatus.FAILED,
                        message=f"Unknown type: {expected_type_str}",
                    )
                ok = SchemaValidator._check_type(series, expected)
                return ExpectationResult(
                    expectation=expectation,
                    status=ValidationStatus.PASSED if ok else ValidationStatus.FAILED,
                    observed_value=str(series.dtype),
                )

            else:
                return ExpectationResult(
                    expectation=expectation,
                    status=ValidationStatus.WARNING,
                    message=f"Unknown expectation type: {expectation.expectation_type}",
                )

        except AstroMLError as e:
            return ExpectationResult(
                expectation=expectation,
                status=ValidationStatus.FAILED,
                message=f"Evaluation error: {e}",
            )


__all__ = [
    "ColumnType",
    "ColumnExpectation",
    "SchemaDefinition",
    "ValidationStatus",
    "ExpectationResult",
    "SchemaValidationResult",
    "SchemaValidator",
]