from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SchemaValidationResult:
    """Result of validating a DataFrame against a schema contract.

    Attributes:
        is_valid: Whether all checks passed.
        contract_name: Name of the contract that was validated.
        missing_columns: Columns expected by the contract but missing from the DataFrame.
        extra_columns: Columns present in the DataFrame but not in the contract.
        type_mismatches: List of dicts with column, expected_type, actual_type.
        null_constraint_violations: List of dicts with column, null_count for non-nullable columns.
        unique_constraint_violations: List of dicts with column, duplicate_count for unique columns.
        regex_violations: List of dicts with column, value, pattern for regex mismatches.
        errors: General error messages during validation.
    """

    is_valid: bool
    contract_name: str
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    type_mismatches: list[dict[str, Any]] = field(default_factory=list)
    null_constraint_violations: list[dict[str, Any]] = field(default_factory=list)
    unique_constraint_violations: list[dict[str, Any]] = field(default_factory=list)
    regex_violations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SchemaContract:
    """Defines expected schema for data including column names, dtypes, nullability, and uniqueness constraints.

    Attributes:
        name: Human-readable name for this contract.
        version: Semantic version string.
        columns: Dict mapping column names to their schema specifications.
    """

    def __init__(self, name: str = "", version: str = "1.0.0") -> None:
        self.name = name
        self.version = version
        self.columns: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_schema(
        cls,
        schema_dict: dict[str, Any],
        name: str = "",
        version: str = "1.0.0",
    ) -> SchemaContract:
        """Create a SchemaContract from a schema definition dictionary.

        Args:
            schema_dict: Dict with a "columns" key mapping column names to their spec.
                Each spec may contain: dtype, nullable, unique, regex.
                If schema_dict has no "columns" key, it is treated as a direct column map.
            name: Optional human-readable name.
            version: Semantic version string (default "1.0.0").

        Returns:
            A new SchemaContract instance.
        """
        contract = cls(name=name, version=version)
        columns = schema_dict if "columns" not in schema_dict else schema_dict["columns"]
        for col_name, col_spec in columns.items():
            contract.columns[col_name] = {
                "dtype": col_spec.get("dtype"),
                "nullable": col_spec.get("nullable", True),
                "unique": col_spec.get("unique", False),
                "regex": col_spec.get("regex"),
            }
        return contract

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str = "",
        version: str = "1.0.0",
    ) -> SchemaContract:
        """Infer a SchemaContract from a pandas DataFrame.

        Args:
            df: DataFrame to infer the schema from.
            name: Optional human-readable name.
            version: Semantic version string (default "1.0.0").

        Returns:
            A new SchemaContract with columns inferred from the DataFrame.
        """
        contract = cls(name=name, version=version)
        for col_name in df.columns:
            dtype_str = str(df[col_name].dtype)
            has_nulls = bool(df[col_name].isna().any())
            contract.columns[str(col_name)] = {
                "dtype": dtype_str,
                "nullable": has_nulls,
                "unique": False,
                "regex": None,
            }
        return contract

    def validate(self, df: pd.DataFrame) -> SchemaValidationResult:
        """Validate a DataFrame against this schema contract.

        Args:
            df: DataFrame to validate.

        Returns:
            SchemaValidationResult with all check results.
        """
        missing_columns: list[str] = []
        extra_columns: list[str] = []
        type_mismatches: list[dict[str, Any]] = []
        null_violations: list[dict[str, Any]] = []
        unique_violations: list[dict[str, Any]] = []
        regex_violations: list[dict[str, Any]] = []
        errors: list[str] = []

        actual_columns = set(df.columns)
        expected_columns = set(self.columns.keys())

        missing_columns = sorted(expected_columns - actual_columns)
        extra_columns = sorted(actual_columns - expected_columns)

        for col_name, col_spec in self.columns.items():
            if col_name not in df.columns:
                continue

            series = df[col_name]

            # Type check
            expected_dtype = col_spec.get("dtype")
            if expected_dtype is not None:
                actual_dtype = str(series.dtype)
                if actual_dtype != expected_dtype:
                    type_mismatches.append(
                        {
                            "column": col_name,
                            "expected_type": expected_dtype,
                            "actual_type": actual_dtype,
                        }
                    )

            # Nullability check
            nullable = col_spec.get("nullable", True)
            if not nullable:
                null_count = int(series.isna().sum())
                if null_count > 0:
                    null_violations.append(
                        {
                            "column": col_name,
                            "null_count": null_count,
                        }
                    )

            # Uniqueness check
            unique = col_spec.get("unique", False)
            if unique:
                duplicate_count = int(series.duplicated(keep=False).sum())
                if duplicate_count > 0:
                    unique_violations.append(
                        {
                            "column": col_name,
                            "duplicate_count": duplicate_count,
                        }
                    )

            # Regex check
            pattern = col_spec.get("regex")
            if pattern is not None and pattern:
                non_null = series.dropna()
                if len(non_null) > 0:
                    try:
                        compiled = re.compile(pattern)
                        mask = non_null.astype(str).str.match(compiled)
                        failing = non_null[~mask]
                        if len(failing) > 0:
                            regex_violations.append(
                                {
                                    "column": col_name,
                                    "pattern": pattern,
                                    "violating_values": failing.head(10).tolist(),
                                    "violation_count": len(failing),
                                }
                            )
                    except re.error as e:
                        errors.append(f"Regex error for column '{col_name}': {e}")

        is_valid = (
            not missing_columns
            and not type_mismatches
            and not null_violations
            and not unique_violations
            and not regex_violations
        )

        return SchemaValidationResult(
            is_valid=is_valid,
            contract_name=self.name,
            missing_columns=missing_columns,
            extra_columns=extra_columns,
            type_mismatches=type_mismatches,
            null_constraint_violations=null_violations,
            unique_constraint_violations=unique_violations,
            regex_violations=regex_violations,
            errors=errors,
        )
