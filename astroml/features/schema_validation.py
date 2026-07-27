"""Schema validation for feature store ingestion.

Provides schema checking and validation capabilities for feature data
to ensure data quality before ingestion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SchemaSeverity(Enum):
    """Severity level for schema validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class SchemaIssue:
    """A schema validation issue."""
    
    severity: SchemaSeverity
    column: str
    message: str
    expected_type: Optional[str] = None
    actual_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "severity": self.severity.value,
            "column": self.column,
            "message": self.message,
            "expected_type": self.expected_type,
            "actual_type": self.actual_type,
        }


@dataclass
class ValidationResult:
    """Result of schema validation."""
    
    is_valid: bool
    issues: List[SchemaIssue] = field(default_factory=list)
    
    @property
    def errors(self) -> List[SchemaIssue]:
        """Get error-level issues."""
        return [i for i in self.issues if i.severity == SchemaSeverity.ERROR]
    
    @property
    def warnings(self) -> List[SchemaIssue]:
        """Get warning-level issues."""
        return [i for i in self.issues if i.severity == SchemaSeverity.WARNING]
    
    def add_error(self, column: str, message: str, 
                  expected_type: Optional[str] = None, 
                  actual_type: Optional[str] = None) -> None:
        """Add an error issue."""
        self.issues.append(SchemaIssue(
            severity=SchemaSeverity.ERROR,
            column=column,
            message=message,
            expected_type=expected_type,
            actual_type=actual_type,
        ))
        self.is_valid = False
    
    def add_warning(self, column: str, message: str,
                    expected_type: Optional[str] = None,
                    actual_type: Optional[str] = None) -> None:
        """Add a warning issue."""
        self.issues.append(SchemaIssue(
            severity=SchemaSeverity.WARNING,
            column=column,
            message=message,
            expected_type=expected_type,
            actual_type=actual_type,
        ))
    
    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Validation Result: {'VALID' if self.is_valid else 'INVALID'}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        
        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  - {error.column}: {error.message}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning.column}: {warning.message}")
        
        return "\n".join(lines)


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""
    
    name: str
    dtype: str  # Expected pandas dtype
    nullable: bool = True
    unique: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[Set[Any]] = None
    regex_pattern: Optional[str] = None
    
    def validate(self, series: pd.Series, result: ValidationResult) -> None:
        """Validate a pandas Series against this schema."""
        # Check if column exists
        if series.isna().all():
            if not self.nullable:
                result.add_error(
                    self.name,
                    "Column is entirely null but nullable=False"
                )
            return
        
        # Check dtype
        actual_dtype = str(series.dtype)
        if not self._dtype_matches(actual_dtype):
            result.add_error(
                self.name,
                f"Expected dtype {self.dtype}, got {actual_dtype}",
                expected_type=self.dtype,
                actual_type=actual_dtype
            )
        
        # Check nullability
        null_count = series.isna().sum()
        if null_count > 0 and not self.nullable:
            result.add_error(
                self.name,
                f"Column has {null_count} null values but nullable=False"
            )
        
        # Check uniqueness
        if self.unique:
            duplicate_count = series.duplicated().sum()
            if duplicate_count > 0:
                result.add_error(
                    self.name,
                    f"Column has {duplicate_count} duplicate values but unique=True"
                )
        
        # Check numeric bounds
        if self.min_value is not None or self.max_value is not None:
            if pd.api.types.is_numeric_dtype(series):
                if self.min_value is not None:
                    if (series < self.min_value).any():
                        result.add_error(
                            self.name,
                            f"Values below minimum {self.min_value}"
                        )
                if self.max_value is not None:
                    if (series > self.max_value).any():
                        result.add_error(
                            self.name,
                            f"Values above maximum {self.max_value}"
                        )
        
        # Check allowed values
        if self.allowed_values is not None:
            invalid_values = set(series.dropna().unique()) - self.allowed_values
            if invalid_values:
                result.add_error(
                    self.name,
                    f"Invalid values: {invalid_values}"
                )
        
        # Check regex pattern for strings
        if self.regex_pattern is not None and pd.api.types.is_string_dtype(series):
            import re
            pattern = re.compile(self.regex_pattern)
            non_matching = series.dropna()[~series.dropna().str.match(pattern, na=False)]
            if len(non_matching) > 0:
                result.add_error(
                    self.name,
                    f"{len(non_matching)} values do not match pattern '{self.regex_pattern}'"
                )
    
    def _dtype_matches(self, actual_dtype: str) -> bool:
        """Check if actual dtype matches expected dtype."""
        # Handle dtype aliases
        dtype_map = {
            "int": ["int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"],
            "float": ["float64", "float32"],
            "str": ["object", "string"],
            "bool": ["bool"],
            "datetime": ["datetime64[ns]", "datetime64[ns, UTC]"],
        }
        
        if self.dtype in dtype_map:
            return any(actual_dtype.startswith(dt) for dt in dtype_map[self.dtype])
        
        return actual_dtype == self.dtype or actual_dtype.startswith(self.dtype)


@dataclass
class DataFrameSchema:
    """Schema definition for a DataFrame."""
    
    name: str
    columns: List[ColumnSchema]
    required_columns: Set[str] = field(default_factory=set)
    min_rows: Optional[int] = None
    max_rows: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Initialize required columns from column definitions."""
        if not self.required_columns:
            self.required_columns = {col.name for col in self.columns if not col.nullable}
    
    def validate(self, df: pd.DataFrame, result: Optional[ValidationResult] = None) -> ValidationResult:
        """Validate a DataFrame against this schema."""
        if result is None:
            result = ValidationResult(is_valid=True)
        
        # Check row count
        row_count = len(df)
        if self.min_rows is not None and row_count < self.min_rows:
            result.add_error(
                "__row_count__",
                f"DataFrame has {row_count} rows, minimum {self.min_rows} required"
            )
        
        if self.max_rows is not None and row_count > self.max_rows:
            result.add_error(
                "__row_count__",
                f"DataFrame has {row_count} rows, maximum {self.max_rows} allowed"
            )
        
        # Check required columns
        missing_columns = self.required_columns - set(df.columns)
        if missing_columns:
            result.add_error(
                "__columns__",
                f"Missing required columns: {missing_columns}"
            )
        
        # Validate each column
        for col_schema in self.columns:
            if col_schema.name in df.columns:
                col_schema.validate(df[col_schema.name], result)
            elif col_schema.name in self.required_columns:
                result.add_error(
                    col_schema.name,
                    "Required column not found in DataFrame"
                )
        
        return result


# Predefined schemas for common feature store data
FEATURE_VALUE_SCHEMA = DataFrameSchema(
    name="feature_value",
    columns=[
        ColumnSchema(name="entity_id", dtype="str", nullable=False, unique=False),
        ColumnSchema(name="value", dtype="float", nullable=True),
        ColumnSchema(name="timestamp", dtype="datetime", nullable=False),
    ],
    min_rows=1,
)

TRANSACTION_SCHEMA = DataFrameSchema(
    name="transaction",
    columns=[
        ColumnSchema(name="sender", dtype="str", nullable=False),
        ColumnSchema(name="receiver", dtype="str", nullable=True),
        ColumnSchema(name="asset", dtype="str", nullable=False),
        ColumnSchema(name="amount", dtype="float", nullable=True),
        ColumnSchema(name="timestamp", dtype="datetime", nullable=False),
    ],
    min_rows=1,
)

ACCOUNT_FEATURE_SCHEMA = DataFrameSchema(
    name="account_feature",
    columns=[
        ColumnSchema(name="account_id", dtype="str", nullable=False),
        ColumnSchema(name="feature_name", dtype="str", nullable=False),
        ColumnSchema(name="feature_value", dtype="float", nullable=True),
        ColumnSchema(name="timestamp", dtype="datetime", nullable=False),
    ],
    min_rows=1,
)


def validate_dataframe(
    df: pd.DataFrame,
    schema: Union[DataFrameSchema, str],
    strict: bool = True,
) -> ValidationResult:
    """Validate a DataFrame against a schema.
    
    Args:
        df: DataFrame to validate
        schema: Schema definition or predefined schema name
        strict: If True, errors will cause validation to fail. If False, only warnings.
        
    Returns:
        ValidationResult with issues found
    """
    # Resolve schema from name if needed
    if isinstance(schema, str):
        schema_map = {
            "feature_value": FEATURE_VALUE_SCHEMA,
            "transaction": TRANSACTION_SCHEMA,
            "account_feature": ACCOUNT_FEATURE_SCHEMA,
        }
        if schema not in schema_map:
            raise ValueError(f"Unknown schema name: {schema}")
        schema = schema_map[schema]
    
    result = schema.validate(df)
    
    # If not strict, downgrade errors to warnings
    if not strict:
        for issue in result.issues:
            if issue.severity == SchemaSeverity.ERROR:
                issue.severity = SchemaSeverity.WARNING
        result.is_valid = True
    
    return result


def dry_run_ingestion(
    df: pd.DataFrame,
    schema: Union[DataFrameSchema, str],
    log_issues: bool = True,
) -> ValidationResult:
    """Perform a dry-run validation of data before ingestion.
    
    Args:
        df: DataFrame to validate
        schema: Schema definition or predefined schema name
        log_issues: Whether to log validation issues
        
    Returns:
        ValidationResult with issues found
    """
    result = validate_dataframe(df, schema, strict=False)
    
    if log_issues:
        if result.is_valid:
            logger.info("Dry-run validation passed")
        else:
            logger.warning("Dry-run validation found issues")
        
        for issue in result.issues:
            if issue.severity == SchemaSeverity.ERROR:
                logger.error(f"Schema error: {issue.column} - {issue.message}")
            elif issue.severity == SchemaSeverity.WARNING:
                logger.warning(f"Schema warning: {issue.column} - {issue.message}")
    
    return result
