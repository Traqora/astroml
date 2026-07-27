"""Validation pipeline for comprehensive data quality checks (issue #303).

Provides a pipeline architecture for running multiple validation stages:
- Schema validation
- Data quality checks
- Business rule validation
- Statistical validation
- Custom validation rules
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum

from .api_validation import ValidationResult, ValidationError
from .data_quality import DataQualityReport, ValidationResult as DQValidationResult

logger = logging.getLogger(__name__)


class ValidationStage(Enum):
    """Validation stage identifiers."""

    SCHEMA = "schema"
    DATA_QUALITY = "data_quality"
    BUSINESS_RULES = "business_rules"
    STATISTICAL = "statistical"
    CUSTOM = "custom"


@dataclass
class PipelineResult:
    """Result of running the validation pipeline.

    Attributes:
        is_valid: Whether all validation stages passed.
        stage_results: Results per validation stage.
        total_errors: Total number of errors across all stages.
        execution_time_ms: Total pipeline execution time in milliseconds.
        timestamp: When the validation was performed.
    """

    is_valid: bool
    stage_results: Dict[str, ValidationResult] = field(default_factory=dict)
    total_errors: int = 0
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get_errors_by_stage(self, stage: ValidationStage) -> List[ValidationError]:
        """Get all errors from a specific stage."""
        result = self.stage_results.get(stage.value)
        return result.errors if result else []

    def get_all_errors(self) -> List[ValidationError]:
        """Get all errors from all stages."""
        all_errors = []
        for result in self.stage_results.values():
            all_errors.extend(result.errors)
        return all_errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for API responses."""
        return {
            "is_valid": self.is_valid,
            "total_errors": self.total_errors,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "stage_results": {
                stage: result.to_dict()
                for stage, result in self.stage_results.items()
            },
        }


class ValidationStageConfig:
    """Configuration for a single validation stage."""

    def __init__(
        self,
        stage: ValidationStage,
        enabled: bool = True,
        fail_on_error: bool = True,
        custom_validator: Optional[Callable] = None,
    ):
        self.stage = stage
        self.enabled = enabled
        self.fail_on_error = fail_on_error
        self.custom_validator = custom_validator


class ValidationPipeline:
    """Comprehensive validation pipeline for data quality checks."""

    def __init__(self):
        """Initialize the validation pipeline."""
        self.stages: Dict[ValidationStage, ValidationStageConfig] = {}
        self.metrics: Dict[str, Any] = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "average_execution_time_ms": 0.0,
        }

    def add_stage(self, config: ValidationStageConfig) -> "ValidationPipeline":
        """Add a validation stage to the pipeline."""
        self.stages[config.stage] = config
        return self

    def remove_stage(self, stage: ValidationStage) -> "ValidationPipeline":
        """Remove a validation stage from the pipeline."""
        if stage in self.stages:
            del self.stages[stage]
        return self

    def enable_stage(self, stage: ValidationStage) -> "ValidationPipeline":
        """Enable a specific validation stage."""
        if stage in self.stages:
            self.stages[stage].enabled = True
        return self

    def disable_stage(self, stage: ValidationStage) -> "ValidationPipeline":
        """Disable a specific validation stage."""
        if stage in self.stages:
            self.stages[stage].enabled = False
        return self

    def run(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Run the validation pipeline on the provided data.

        Args:
            data: Data to validate.
            context: Optional context information for validation.

        Returns:
            PipelineResult with validation status and errors.
        """
        import time

        start_time = time.time()
        stage_results: Dict[str, ValidationResult] = {}
        total_errors = 0
        is_valid = True

        context = context or {}

        for stage, config in self.stages.items():
            if not config.enabled:
                continue

            try:
                result = self._run_stage(stage, data, context)
                stage_results[stage.value] = result

                if not result.is_valid:
                    total_errors += len(result.errors)
                    if config.fail_on_error:
                        is_valid = False

            except Exception as e:
                logger.error(f"Validation stage {stage.value} failed: {e}")
                stage_results[stage.value] = ValidationResult(
                    is_valid=False,
                    errors=[ValidationError(
                        field="pipeline",
                        message=f"Stage execution failed: {str(e)}",
                        error_type="STAGE_ERROR"
                    )]
                )
                if config.fail_on_error:
                    is_valid = False

        execution_time_ms = (time.time() - start_time) * 1000

        # Update metrics
        self.metrics["total_runs"] += 1
        if is_valid:
            self.metrics["successful_runs"] += 1
        else:
            self.metrics["failed_runs"] += 1
        self.metrics["average_execution_time_ms"] = (
            (self.metrics["average_execution_time_ms"] * (self.metrics["total_runs"] - 1) + execution_time_ms)
            / self.metrics["total_runs"]
        )

        return PipelineResult(
            is_valid=is_valid,
            stage_results=stage_results,
            total_errors=total_errors,
            execution_time_ms=execution_time_ms,
        )

    def _run_stage(
        self, stage: ValidationStage, data: Dict[str, Any], context: Dict[str, Any]
    ) -> ValidationResult:
        """Run a single validation stage."""
        config = self.stages[stage]

        if config.custom_validator:
            return config.custom_validator(data, context)

        # Default stage implementations
        if stage == ValidationStage.SCHEMA:
            return self._validate_schema(data, context)
        elif stage == ValidationStage.DATA_QUALITY:
            return self._validate_data_quality(data, context)
        elif stage == ValidationStage.BUSINESS_RULES:
            return self._validate_business_rules(data, context)
        elif stage == ValidationStage.STATISTICAL:
            return self._validate_statistical(data, context)
        elif stage == ValidationStage.CUSTOM:
            return ValidationResult(is_valid=True)

        return ValidationResult(is_valid=True)

    def _validate_schema(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate data schema using Pydantic."""
        from .api_validation import validate_transaction_input, validate_account_input, validate_feature_data

        data_type = context.get("data_type", "transaction")

        if data_type == "transaction":
            return validate_transaction_input(data)
        elif data_type == "account":
            return validate_account_input(data)
        elif data_type == "feature":
            return validate_feature_data(data)
        else:
            return ValidationResult(is_valid=True)

    def _validate_data_quality(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate data quality using existing data quality module."""
        errors = []

        # Check for null values in critical fields
        critical_fields = context.get("critical_fields", [])
        for field in critical_fields:
            if field in data and data[field] is None:
                errors.append(ValidationError(
                    field=field,
                    message=f"Critical field '{field}' is null",
                    error_type="NULL_CRITICAL_FIELD"
                ))

        # Check for empty strings
        string_fields = context.get("string_fields", [])
        for field in string_fields:
            if field in data and isinstance(data[field], str) and not data[field].strip():
                errors.append(ValidationError(
                    field=field,
                    message=f"String field '{field}' is empty",
                    error_type="EMPTY_STRING_FIELD"
                ))

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_business_rules(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate business rules."""
        from .api_validation import CustomValidationRules

        errors = []

        # Transaction amount limits
        if "amount" in data:
            max_amount = context.get("max_transaction_amount", 1_000_000_000)
            result = CustomValidationRules.validate_transaction_limits(data["amount"], max_amount)
            errors.extend(result.errors)

        # Account age validation
        if "created_at" in data:
            min_age = context.get("min_account_age_days", 0)
            result = CustomValidationRules.validate_account_age(data["created_at"], min_age)
            errors.extend(result.errors)

        # Feature range validation
        if "features" in data:
            ranges = context.get("feature_ranges", {})
            if ranges:
                result = CustomValidationRules.validate_feature_range(data["features"], ranges)
                errors.extend(result.errors)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_statistical(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Validate statistical properties."""
        errors = []

        # Check for outliers in numeric fields
        numeric_fields = context.get("numeric_fields", [])
        for field in numeric_fields:
            if field in data and isinstance(data[field], (int, float)):
                value = data[field]
                # Check for extreme values
                if abs(value) > 1e15:
                    errors.append(ValidationError(
                        field=field,
                        message=f"Value {value} is extremely large",
                        error_type="EXTREME_VALUE"
                    ))

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset pipeline metrics."""
        self.metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "average_execution_time_ms": 0.0,
        }


# ─── Default Pipeline Configurations ───────────────────────────────────────

def create_transaction_pipeline() -> ValidationPipeline:
    """Create a validation pipeline for transaction data."""
    pipeline = ValidationPipeline()
    pipeline.add_stage(ValidationStageConfig(ValidationStage.SCHEMA))
    pipeline.add_stage(ValidationStageConfig(ValidationStage.DATA_QUALITY))
    pipeline.add_stage(ValidationStageConfig(ValidationStage.BUSINESS_RULES))
    return pipeline


def create_account_pipeline() -> ValidationPipeline:
    """Create a validation pipeline for account data."""
    pipeline = ValidationPipeline()
    pipeline.add_stage(ValidationStageConfig(ValidationStage.SCHEMA))
    pipeline.add_stage(ValidationStageConfig(ValidationStage.DATA_QUALITY))
    return pipeline


def create_feature_pipeline() -> ValidationPipeline:
    """Create a validation pipeline for feature data."""
    pipeline = ValidationPipeline()
    pipeline.add_stage(ValidationStageConfig(ValidationStage.SCHEMA))
    pipeline.add_stage(ValidationStageConfig(ValidationStage.STATISTICAL))
    pipeline.add_stage(ValidationStageConfig(ValidationStage.BUSINESS_RULES))
    return pipeline
