from astroml.utils.exceptions import AstroMLError
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_RESTRICTED_GLOBALS: dict[str, Any] = {
    "__builtins__": {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "round": round,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    },
}


SUPPORTED_RULE_TYPES = frozenset(
    {
        "referential_integrity",
        "functional_dependency",
        "conditional_required",
        "custom_expression",
    }
)


@dataclass
class SemanticValidationResult:
    """Result of validating a DataFrame against a semantic contract.

    Attributes:
        is_valid: Whether all rules passed.
        contract_name: Name of the contract that was validated.
        rule_results: List of dicts with rule_name, passed, message per rule.
        errors: General error messages during validation.
    """

    is_valid: bool
    contract_name: str
    rule_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SemanticContract:
    """Defines semantic relationships and business rules between DataFrame columns.

    Supports rule types: referential_integrity, functional_dependency,
    conditional_required, custom_expression.

    Attributes:
        name: Human-readable name for this contract.
        version: Semantic version string.
        rules: List of rule dictionaries.
    """

    def __init__(self, name: str = "", version: str = "1.0.0") -> None:
        self.name = name
        self.version = version
        self.rules: list[dict[str, Any]] = []

    def add_rule(
        self,
        rule_name: str,
        expression: dict[str, Any],
        description: str = "",
    ) -> None:
        """Add a semantic rule.

        Args:
            rule_name: Unique name for this rule.
            expression: Dict with "type" and type-specific parameters.
            description: Human-readable description of the rule.

        Raises:
            ValueError: If rule type is not supported.
        """
        rule_type = expression.get("type", "")
        if rule_type not in SUPPORTED_RULE_TYPES:
            raise ValueError(
                f"Unsupported rule type '{rule_type}'. "
                f"Supported types: {sorted(SUPPORTED_RULE_TYPES)}"
            )
        self.rules.append(
            {
                "rule_name": rule_name,
                "type": rule_type,
                "expression": expression,
                "description": description,
            }
        )

    def validate(self, df: pd.DataFrame) -> SemanticValidationResult:
        """Validate a DataFrame against all semantic rules.

        Args:
            df: DataFrame to validate.

        Returns:
            SemanticValidationResult with results per rule.
        """
        rule_results: list[dict[str, Any]] = []
        errors: list[str] = []

        for rule in self.rules:
            rule_name = rule["rule_name"]
            rule_type = rule["type"]
            expression = rule["expression"]

            try:
                if rule_type == "referential_integrity":
                    passed, msg = self.check_referential_integrity(
                        df,
                        foreign_key=expression.get("foreign_key", ""),
                        reference_column=expression.get("reference_column", ""),
                    )
                elif rule_type == "functional_dependency":
                    passed, msg = self.check_functional_dependency(
                        df,
                        determinant=expression.get("determinant", ""),
                        dependent=expression.get("dependent", ""),
                    )
                elif rule_type == "conditional_required":
                    passed, msg = self._check_conditional_required(
                        df,
                        condition_column=expression.get("condition_column", ""),
                        condition_value=expression.get("condition_value"),
                        required_column=expression.get("required_column", ""),
                    )
                elif rule_type == "custom_expression":
                    passed, msg = self._check_custom_expression(
                        df,
                        expr_str=expression.get("expr", ""),
                    )
                else:
                    passed, msg = False, f"Unknown rule type: {rule_type}"

                rule_results.append(
                    {
                        "rule_name": rule_name,
                        "type": rule_type,
                        "passed": passed,
                        "message": msg,
                    }
                )
            except AstroMLError as e:
                rule_results.append(
                    {
                        "rule_name": rule_name,
                        "type": rule_type,
                        "passed": False,
                        "message": str(e),
                    }
                )
                errors.append(f"Rule '{rule_name}' raised exception: {e}")

        is_valid = all(r["passed"] for r in rule_results)
        return SemanticValidationResult(
            is_valid=is_valid,
            contract_name=self.name,
            rule_results=rule_results,
            errors=errors,
        )

    def check_referential_integrity(
        self,
        df: pd.DataFrame,
        foreign_key: str,
        reference_column: str | None = None,
        reference_values: pd.Series | list | None = None,
    ) -> tuple[bool, str]:
        """Check referential integrity: all FK values exist in reference.

        Args:
            df: DataFrame to validate.
            foreign_key: Column name acting as the foreign key.
            reference_column: Column name in df that serves as the reference.
                If provided, FK values must exist in this column.
            reference_values: Explicit reference values (used if reference_column is None).

        Returns:
            Tuple of (passed, message).
        """
        if foreign_key not in df.columns:
            return False, f"Foreign key column '{foreign_key}' not found"

        if reference_column and reference_column in df.columns:
            ref_vals = set(df[reference_column].dropna().unique())
        elif reference_values is not None:
            ref_vals = set(pd.Series(reference_values).dropna().unique())
        else:
            return False, "No reference column or values provided"

        fk_vals = df[foreign_key].dropna().unique()
        missing = set(fk_vals) - ref_vals
        if missing:
            sample = sorted(missing)[:10]
            return False, (
                f"Referential integrity violation: {len(missing)} foreign key values "
                f"not found in reference. Samples: {sample}"
            )
        return True, "All foreign key values found in reference"

    def check_functional_dependency(
        self,
        df: pd.DataFrame,
        determinant: str,
        dependent: str,
    ) -> tuple[bool, str]:
        """Check functional dependency: same determinant => same dependent.

        Args:
            df: DataFrame to validate.
            determinant: Column name that determines the dependent.
            dependent: Column name that should be functionally determined.

        Returns:
            Tuple of (passed, message).
        """
        if determinant not in df.columns:
            return False, f"Determinant column '{determinant}' not found"
        if dependent not in df.columns:
            return False, f"Dependent column '{dependent}' not found"

        subset = df[[determinant, dependent]].dropna()
        if len(subset) == 0:
            return True, "No data to check functional dependency"

        grouped = subset.groupby(determinant, dropna=True)[dependent].nunique()
        violations = grouped[grouped > 1]
        if len(violations) > 0:
            sample = violations.head(10)
            return False, (
                f"Functional dependency violation: {len(violations)} determinant values "
                f"map to multiple dependent values. Samples: {sample.to_dict()}"
            )
        return True, f"Functional dependency holds: {determinant} -> {dependent}"

    def _check_conditional_required(
        self,
        df: pd.DataFrame,
        condition_column: str,
        condition_value: Any,
        required_column: str,
    ) -> tuple[bool, str]:
        """Check that when condition_column == condition_value, required_column is not null."""
        if condition_column not in df.columns:
            return False, f"Condition column '{condition_column}' not found"
        if required_column not in df.columns:
            return False, f"Required column '{required_column}' not found"

        mask = df[condition_column] == condition_value
        required_vals = df.loc[mask, required_column]
        null_count = int(required_vals.isna().sum())
        if null_count > 0:
            return False, (
                f"Conditional required violation: {null_count} rows have "
                f"'{condition_column}' = {condition_value!r} but '{required_column}' is null"
            )
        return True, f"All conditional required values present"

    def _check_custom_expression(
        self,
        df: pd.DataFrame,
        expr_str: str,
    ) -> tuple[bool, str]:
        """Evaluate a custom Python expression on the DataFrame.

        Uses a restricted set of builtins for safety. The expression can reference
        the DataFrame as 'df' and must evaluate to a boolean Series or scalar.

        Args:
            df: DataFrame to validate.
            expr_str: Python expression string. Uses 'df' as the DataFrame variable.

        Returns:
            Tuple of (passed, message).
        """
        if not expr_str:
            return False, "Empty expression"

        try:
            local_vars: dict[str, Any] = {"df": df}
            result = eval(expr_str, _RESTRICTED_GLOBALS, local_vars)
        except AstroMLError as e:
            return False, f"Expression evaluation failed: {e}"

        if isinstance(result, pd.Series):
            passed = bool(result.all())
            failure_count = int((~result).sum())
            if not passed:
                return False, f"Custom expression failed for {failure_count} rows"
            return True, "Custom expression passed for all rows"

        if isinstance(result, (bool, np.bool_)):
            if result:
                return True, "Custom expression passed"
            return False, "Custom expression returned False"

        if result:
            return True, "Custom expression passed"
        return False, f"Expression returned falsy value of type {type(result).__name__}"
