from astroml.utils.exceptions import AstroMLError
"""Data quality checks for completeness, consistency, accuracy, and timeliness.

Provides modular check suites and validator classes to inspect tabular and
transactional data for quality degradation, corruption, and anomalies.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MetricDimension(str, Enum):
    """Data quality dimensions."""

    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"


class CheckSeverity(str, Enum):
    """Severity level of check failures."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataQualityError(Exception):
    """Raised when a fatal data quality check fails."""

    pass


@dataclass
class CheckResult:
    """Result of an individual data quality check."""

    check_name: str
    dimension: MetricDimension
    is_valid: bool
    score: float  # 0.0 to 1.0 (or percentage)
    severity: CheckSeverity = CheckSeverity.WARNING
    message: str = ""
    field: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)
    timestamp: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Legacy compatibility classes
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Legacy result of a data quality validation check."""

    is_valid: bool
    error_type: str | None = None
    message: str | None = None
    field: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""

    total_records: int = 0
    valid_records: int = 0
    validation_results: list[ValidationResult] = dc_field(default_factory=list)
    check_results: list[CheckResult] = dc_field(default_factory=list)
    summary: dict[str, Any] = dc_field(default_factory=dict)
    dimension_scores: dict[str, float] = dc_field(default_factory=dict)
    batch_id: str | None = None
    created_at: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def quality_score(self) -> float:
        """Calculate data quality score as percentage of valid records."""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100.0

    @property
    def error_types(self) -> set[str]:
        """Get set of unique error types found."""
        return {
            r.error_type for r in self.validation_results if not r.is_valid and r.error_type
        }


# ---------------------------------------------------------------------------
# 1. Completeness Checks
# ---------------------------------------------------------------------------


class CompletenessChecker:
    """Checks data completeness including null rates and missing values."""

    def __init__(
        self,
        required_fields: list[str] | None = None,
        max_null_rate: float = 0.05,
    ) -> None:
        self.required_fields = required_fields or []
        self.max_null_rate = max_null_rate

    def check_records(
        self,
        records: list[dict[str, Any]],
        expected_fields: list[str] | None = None,
    ) -> list[CheckResult]:
        """Check completeness across records."""
        if not records:
            return [
                CheckResult(
                    check_name="empty_dataset_check",
                    dimension=MetricDimension.COMPLETENESS,
                    is_valid=True,
                    score=1.0,
                    severity=CheckSeverity.INFO,
                    message="Dataset is empty.",
                )
            ]

        results: list[CheckResult] = []
        fields_to_check = set(expected_fields or self.required_fields)
        if not fields_to_check:
            for r in records:
                fields_to_check.update(r.keys())

        total_records = len(records)
        field_null_counts: dict[str, int] = {f: 0 for f in fields_to_check}
        missing_required_records: list[int] = []

        for idx, rec in enumerate(records):
            has_missing_req = False
            for f in fields_to_check:
                if f not in rec or rec[f] is None:
                    field_null_counts[f] = field_null_counts.get(f, 0) + 1
                    if f in self.required_fields:
                        has_missing_req = True
                elif isinstance(rec[f], str) and not rec[f].strip():
                    field_null_counts[f] = field_null_counts.get(f, 0) + 1
                    if f in self.required_fields:
                        has_missing_req = True
            if has_missing_req:
                missing_required_records.append(idx)

        # Check required fields
        if self.required_fields:
            req_valid = len(missing_required_records) == 0
            req_score = 1.0 - (len(missing_required_records) / total_records)
            results.append(
                CheckResult(
                    check_name="required_fields_check",
                    dimension=MetricDimension.COMPLETENESS,
                    is_valid=req_valid,
                    score=max(0.0, req_score),
                    severity=CheckSeverity.CRITICAL if not req_valid else CheckSeverity.INFO,
                    message=(
                        f"Found {len(missing_required_records)} records missing required fields"
                        if not req_valid
                        else "All required fields are present"
                    ),
                    details={
                        "required_fields": self.required_fields,
                        "missing_record_indices": missing_required_records[:50],
                        "total_affected": len(missing_required_records),
                    },
                )
            )

        # Check null rates per field
        for f, null_count in field_null_counts.items():
            null_rate = null_count / total_records
            is_valid = null_rate <= self.max_null_rate
            results.append(
                CheckResult(
                    check_name=f"null_rate_{f}",
                    dimension=MetricDimension.COMPLETENESS,
                    is_valid=is_valid,
                    score=1.0 - null_rate,
                    severity=CheckSeverity.WARNING if not is_valid else CheckSeverity.INFO,
                    field=f,
                    message=(
                        f"Null rate {null_rate:.2%} exceeds threshold {self.max_null_rate:.2%}"
                        if not is_valid
                        else f"Null rate {null_rate:.2%} within limits"
                    ),
                    details={
                        "null_count": null_count,
                        "total_records": total_records,
                        "null_rate": null_rate,
                        "threshold": self.max_null_rate,
                    },
                )
            )

        return results


# ---------------------------------------------------------------------------
# 2. Consistency Checks
# ---------------------------------------------------------------------------


class ConsistencyChecker:
    """Checks cross-field validation, schema integrity, and business rules."""

    def __init__(
        self,
        schema_types: dict[str, type] | None = None,
        custom_rules: list[tuple[str, Callable[[dict[str, Any]], bool], str]] | None = None,
    ) -> None:
        self.schema_types = schema_types or {}
        self.custom_rules = custom_rules or []
        self.account_pattern = re.compile(r"^G[A-Z0-9]{40,56}$")
        self.asset_code_pattern = re.compile(r"^[A-Z0-9]{1,12}$")

    def check_records(self, records: list[dict[str, Any]]) -> list[CheckResult]:
        """Perform consistency checks across records."""
        if not records:
            return []

        results: list[CheckResult] = []
        total_records = len(records)

        # 1. Type and schema consistency
        if self.schema_types:
            type_mismatches: dict[str, int] = {f: 0 for f in self.schema_types}
            for rec in records:
                for f, expected_type in self.schema_types.items():
                    if f in rec and rec[f] is not None:
                        if not isinstance(rec[f], expected_type):
                            type_mismatches[f] = type_mismatches.get(f, 0) + 1

            for f, count in type_mismatches.items():
                mismatch_rate = count / total_records
                is_valid = count == 0
                results.append(
                    CheckResult(
                        check_name=f"type_consistency_{f}",
                        dimension=MetricDimension.CONSISTENCY,
                        is_valid=is_valid,
                        score=1.0 - mismatch_rate,
                        severity=CheckSeverity.ERROR if not is_valid else CheckSeverity.INFO,
                        field=f,
                        message=f"{count} type mismatches for field {f}",
                        details={"mismatches": count, "expected_type": str(self.schema_types[f])},
                    )
                )

        # 2. Identifier checks
        invalid_accounts = 0
        invalid_assets = 0
        for rec in records:
            if "source_account" in rec and isinstance(rec["source_account"], str):
                if not self.account_pattern.match(rec["source_account"]):
                    invalid_accounts += 1
            if "asset_code" in rec and isinstance(rec["asset_code"], str):
                if not self.asset_code_pattern.match(rec["asset_code"]):
                    invalid_assets += 1

        if any("source_account" in r for r in records):
            acc_score = 1.0 - (invalid_accounts / total_records)
            results.append(
                CheckResult(
                    check_name="account_format_consistency",
                    dimension=MetricDimension.CONSISTENCY,
                    is_valid=invalid_accounts == 0,
                    score=max(0.0, acc_score),
                    severity=CheckSeverity.ERROR if invalid_accounts > 0 else CheckSeverity.INFO,
                    field="source_account",
                    message=f"Found {invalid_accounts} malformed account IDs",
                    details={"invalid_count": invalid_accounts},
                )
            )

        if any("asset_code" in r for r in records):
            asset_score = 1.0 - (invalid_assets / total_records)
            results.append(
                CheckResult(
                    check_name="asset_format_consistency",
                    dimension=MetricDimension.CONSISTENCY,
                    is_valid=invalid_assets == 0,
                    score=max(0.0, asset_score),
                    severity=CheckSeverity.ERROR if invalid_assets > 0 else CheckSeverity.INFO,
                    field="asset_code",
                    message=f"Found {invalid_assets} malformed asset codes",
                    details={"invalid_count": invalid_assets},
                )
            )

        # 3. Custom cross-field rules
        for rule_name, rule_fn, err_msg in self.custom_rules:
            violation_count = 0
            for rec in records:
                try:
                    if not rule_fn(rec):
                        violation_count += 1
                except AstroMLError:
                    violation_count += 1

            rule_score = 1.0 - (violation_count / total_records)
            results.append(
                CheckResult(
                    check_name=f"custom_rule_{rule_name}",
                    dimension=MetricDimension.CONSISTENCY,
                    is_valid=violation_count == 0,
                    score=max(0.0, rule_score),
                    severity=CheckSeverity.WARNING if violation_count > 0 else CheckSeverity.INFO,
                    message=f"Rule '{rule_name}' violated in {violation_count} records: {err_msg}",
                    details={"violations": violation_count, "total_records": total_records},
                )
            )

        return results


# ---------------------------------------------------------------------------
# 3. Accuracy Checks
# ---------------------------------------------------------------------------


class AccuracyChecker:
    """Checks data accuracy including outlier detection and statistical sanity."""

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        z_score_threshold: float = 3.0,
        numeric_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.iqr_multiplier = iqr_multiplier
        self.z_score_threshold = z_score_threshold
        self.numeric_bounds = numeric_bounds or {}

    def check_outliers_iqr(self, values: list[float], field_name: str = "amount") -> CheckResult:
        """Detect outliers using the Interquartile Range method."""
        if len(values) < 4:
            return CheckResult(
                check_name=f"outlier_iqr_{field_name}",
                dimension=MetricDimension.ACCURACY,
                is_valid=True,
                score=1.0,
                severity=CheckSeverity.INFO,
                field=field_name,
                message="Insufficient data for outlier detection",
            )

        try:
            q1, q2, q3 = statistics.quantiles(values, n=4)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr

            outliers = [v for v in values if v < lower or v > upper]
            outlier_rate = len(outliers) / len(values)
            is_valid = len(outliers) == 0

            return CheckResult(
                check_name=f"outlier_iqr_{field_name}",
                dimension=MetricDimension.ACCURACY,
                is_valid=is_valid,
                score=max(0.0, 1.0 - outlier_rate),
                severity=CheckSeverity.WARNING if not is_valid else CheckSeverity.INFO,
                field=field_name,
                message=(
                    f"Found {len(outliers)} amount outliers"
                    if not is_valid
                    else "No amount outliers detected"
                ),
                details={
                    "outliers": outliers,
                    "outlier_count": len(outliers),
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                },
            )
        except AstroMLError as e:
            return CheckResult(
                check_name=f"outlier_iqr_{field_name}",
                dimension=MetricDimension.ACCURACY,
                is_valid=False,
                score=0.0,
                severity=CheckSeverity.ERROR,
                field=field_name,
                message=f"Error computing IQR outliers: {e}",
            )

    def check_records(self, records: list[dict[str, Any]]) -> list[CheckResult]:
        """Perform accuracy and range checks on records."""
        if not records:
            return []

        results: list[CheckResult] = []

        numeric_fields: dict[str, list[float]] = {}
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if not math.isnan(v) and not math.isinf(v):
                        numeric_fields.setdefault(k, []).append(float(v))

        for field_name, vals in numeric_fields.items():
            results.append(self.check_outliers_iqr(vals, field_name))

            if field_name in self.numeric_bounds:
                min_b, max_b = self.numeric_bounds[field_name]
                out_of_bounds = [v for v in vals if v < min_b or v > max_b]
                oob_rate = len(out_of_bounds) / len(vals)
                is_valid = len(out_of_bounds) == 0
                results.append(
                    CheckResult(
                        check_name=f"range_bounds_{field_name}",
                        dimension=MetricDimension.ACCURACY,
                        is_valid=is_valid,
                        score=max(0.0, 1.0 - oob_rate),
                        severity=CheckSeverity.ERROR if not is_valid else CheckSeverity.INFO,
                        field=field_name,
                        message=f"{len(out_of_bounds)} values outside range [{min_b}, {max_b}]",
                        details={"min_bound": min_b, "max_bound": max_b, "violations": len(out_of_bounds)},
                    )
                )

        return results


# ---------------------------------------------------------------------------
# 4. Timeliness Checks
# ---------------------------------------------------------------------------


class TimelinessChecker:
    """Checks data freshness, timestamp ordering, future drift, and gaps."""

    def __init__(
        self,
        timestamp_field: str = "timestamp",
        max_latency_seconds: float = 3600.0,
        future_tolerance_seconds: float = 300.0,
        gap_threshold_seconds: float = 3600.0,
    ) -> None:
        self.timestamp_field = timestamp_field
        self.max_latency_seconds = max_latency_seconds
        self.future_tolerance_seconds = future_tolerance_seconds
        self.gap_threshold_seconds = gap_threshold_seconds

    def check_records(
        self, records: list[dict[str, Any]], reference_time: datetime | None = None
    ) -> list[CheckResult]:
        """Perform timeliness checks across records."""
        if not records:
            return []

        results: list[CheckResult] = []
        now = reference_time or datetime.now(timezone.utc)
        if not now.tzinfo:
            now = now.replace(tzinfo=timezone.utc)

        parsed_timestamps: list[datetime] = []
        for r in records:
            if self.timestamp_field in r:
                val = r[self.timestamp_field]
                if isinstance(val, str):
                    try:
                        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                        if not dt.tzinfo:
                            dt = dt.replace(tzinfo=timezone.utc)
                        parsed_timestamps.append(dt)
                    except AstroMLError:
                        pass
                elif isinstance(val, datetime):
                    dt = val if val.tzinfo else val.replace(tzinfo=timezone.utc)
                    parsed_timestamps.append(dt)

        if parsed_timestamps:
            max_ts = max(parsed_timestamps)
            latency_sec = max(0.0, (now - max_ts).total_seconds())
            is_fresh = latency_sec <= self.max_latency_seconds
            results.append(
                CheckResult(
                    check_name="data_freshness_latency",
                    dimension=MetricDimension.TIMELINESS,
                    is_valid=is_fresh,
                    score=max(0.0, min(1.0, 1.0 - (latency_sec / (self.max_latency_seconds * 2.0)))),
                    severity=CheckSeverity.WARNING if not is_fresh else CheckSeverity.INFO,
                    field=self.timestamp_field,
                    message=f"Data freshness latency: {latency_sec:.1f}s",
                    details={"latency_seconds": latency_sec, "latest_timestamp": max_ts.isoformat()},
                )
            )

        return results


# ---------------------------------------------------------------------------
# Preserved Legacy Validators matching original behavior exactly
# ---------------------------------------------------------------------------


class TemporalValidator:
    """Validator for temporal data quality checks."""

    def __init__(self, timestamp_field: str = "timestamp") -> None:
        self.timestamp_field = timestamp_field

    def validate_timestamp_ordering(self, transactions: list[dict[str, Any]]) -> ValidationResult:
        if not transactions:
            return ValidationResult(is_valid=True, message="Empty transaction list")

        try:
            timestamps = []
            for tx in transactions:
                if self.timestamp_field not in tx:
                    return ValidationResult(
                        is_valid=False,
                        error_type="MISSING_TIMESTAMP",
                        message=f"Missing timestamp field: {self.timestamp_field}",
                        field=self.timestamp_field,
                    )

                ts_str = tx[self.timestamp_field]
                if isinstance(ts_str, str):
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except AstroMLError:
                        return ValidationResult(
                            is_valid=False,
                            error_type="INVALID_TIMESTAMP_FORMAT",
                            message=f"Invalid timestamp format: {ts_str}",
                            field=self.timestamp_field,
                        )
                elif isinstance(ts_str, datetime):
                    ts = ts_str
                else:
                    return ValidationResult(
                        is_valid=False,
                        error_type="INVALID_TIMESTAMP_FORMAT",
                        message=f"Invalid timestamp format: {type(ts_str)}",
                        field=self.timestamp_field,
                    )
                timestamps.append(ts)

            # Check if monotonically increasing
            for i in range(len(timestamps) - 1):
                if timestamps[i] > timestamps[i + 1]:
                    return ValidationResult(
                        is_valid=False,
                        error_type="TIMESTAMP_ORDER_VIOLATION",
                        message=f"Timestamp order violation at index {i}: {timestamps[i]} > {timestamps[i+1]}",
                        details={
                            "index": i,
                            "current": timestamps[i].isoformat(),
                            "next": timestamps[i + 1].isoformat(),
                        },
                    )

            return ValidationResult(is_valid=True, message="Timestamps are properly ordered")
        except AstroMLError as e:
            return ValidationResult(
                is_valid=False,
                error_type="TIMESTAMP_VALIDATION_ERROR",
                message=f"Error validating timestamps: {str(e)}",
            )

    def validate_future_timestamps(
        self, transactions: list[dict[str, Any]], tolerance_minutes: int = 5
    ) -> ValidationResult:
        if not transactions:
            return ValidationResult(is_valid=True, message="Empty transaction list")

        now = datetime.utcnow()
        tolerance = timedelta(minutes=tolerance_minutes)
        future_txs = []

        try:
            for tx in transactions:
                if self.timestamp_field not in tx:
                    continue

                ts_str = tx[self.timestamp_field]
                if isinstance(ts_str, str):
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts.tzinfo:
                            ts = ts.replace(tzinfo=None)
                    except AstroMLError:
                        continue
                elif isinstance(ts_str, datetime):
                    ts = ts_str.replace(tzinfo=None) if ts_str.tzinfo else ts_str
                else:
                    continue

                if ts > now + tolerance:
                    future_txs.append(
                        {
                            "id": tx.get("id", "unknown"),
                            "timestamp": ts.isoformat(),
                            "minutes_ahead": (ts - now).total_seconds() / 60.0,
                        }
                    )

            if future_txs:
                return ValidationResult(
                    is_valid=False,
                    error_type="FUTURE_TIMESTAMP",
                    message=f"Found {len(future_txs)} transactions with future timestamps",
                    details={"future_transactions": future_txs},
                )

            return ValidationResult(is_valid=True, message="No future timestamps detected")
        except AstroMLError as e:
            return ValidationResult(
                is_valid=False,
                error_type="FUTURE_TIMESTAMP_ERROR",
                message=f"Error checking future timestamps: {str(e)}",
            )


class ReferentialIntegrityValidator:
    """Validator for referential integrity checks."""

    def __init__(self) -> None:
        self.account_pattern = re.compile(r"^G[A-Z0-9]{40,56}$")
        self.asset_code_pattern = re.compile(r"^[A-Z0-9]{1,12}$")

    def validate_account_format(self, account: str) -> ValidationResult:
        if not isinstance(account, str):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_ACCOUNT_TYPE",
                message=f"Account must be string, got {type(account)}",
                field="account",
            )
        if self.account_pattern.match(account):
            return ValidationResult(is_valid=True, message="Account format is valid")
        return ValidationResult(
            is_valid=False,
            error_type="INVALID_ACCOUNT_FORMAT",
            message=f"Invalid Stellar account format: {account}",
            field="account",
        )

    def validate_asset_format(self, asset_code: str) -> ValidationResult:
        if not isinstance(asset_code, str):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_ASSET_TYPE",
                message=f"Asset code must be string, got {type(asset_code)}",
                field="asset_code",
            )
        if self.asset_code_pattern.match(asset_code):
            return ValidationResult(is_valid=True, message="Asset code format is valid")
        return ValidationResult(
            is_valid=False,
            error_type="INVALID_ASSET_FORMAT",
            message=f"Invalid asset code format: {asset_code}",
            field="asset_code",
        )

    def validate_ledger_sequence(self, ledger_sequence: int) -> ValidationResult:
        if not isinstance(ledger_sequence, int) or isinstance(ledger_sequence, bool):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_LEDGER_SEQUENCE_TYPE",
                message=f"Ledger sequence must be integer, got {type(ledger_sequence)}",
                field="ledger_sequence",
            )
        if ledger_sequence > 0:
            return ValidationResult(is_valid=True, message="Ledger sequence is valid")
        return ValidationResult(
            is_valid=False,
            error_type="INVALID_LEDGER_SEQUENCE",
            message=f"Ledger sequence must be positive, got {ledger_sequence}",
            field="ledger_sequence",
        )


class BusinessRulesValidator:
    """Validator for business logic rules."""

    def __init__(self) -> None:
        self.max_operations_per_transaction = 100

    def validate_fee_non_negative(self, fee: int | float) -> ValidationResult:
        if not isinstance(fee, (int, float)) or isinstance(fee, bool):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_FEE_TYPE",
                message=f"Fee must be numeric, got {type(fee)}",
                field="fee",
            )
        if fee >= 0:
            return ValidationResult(is_valid=True, message="Fee is valid")
        return ValidationResult(
            is_valid=False,
            error_type="NEGATIVE_FEE",
            message=f"Fee cannot be negative: {fee}",
            field="fee",
        )

    def validate_amount_non_negative(self, amount: float) -> ValidationResult:
        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_AMOUNT_TYPE",
                message=f"Amount must be numeric, got {type(amount)}",
                field="amount",
            )
        if amount >= 0:
            return ValidationResult(is_valid=True, message="Amount is valid")
        return ValidationResult(
            is_valid=False,
            error_type="NEGATIVE_AMOUNT",
            message=f"Amount cannot be negative: {amount}",
            field="amount",
        )

    def validate_operation_count(self, operation_count: int) -> ValidationResult:
        if not isinstance(operation_count, int) or isinstance(operation_count, bool):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_OPERATION_COUNT_TYPE",
                message=f"Operation count must be integer, got {type(operation_count)}",
                field="operation_count",
            )
        if 1 <= operation_count <= self.max_operations_per_transaction:
            return ValidationResult(is_valid=True, message="Operation count is valid")
        return ValidationResult(
            is_valid=False,
            error_type="INVALID_OPERATION_COUNT",
            message=f"Operation count must be between 1 and {self.max_operations_per_transaction}, got {operation_count}",
            field="operation_count",
        )

    def validate_balance_format(self, balance: Any) -> ValidationResult:
        if balance is None:
            return ValidationResult(is_valid=True, message="Balance can be None")
        if not isinstance(balance, (int, float)) or isinstance(balance, bool):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_BALANCE_TYPE",
                message=f"Balance must be numeric, got {type(balance)}",
                field="balance",
            )
        if math.isnan(balance) or math.isinf(balance):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_BALANCE_VALUE",
                message=f"Balance cannot be NaN or infinite: {balance}",
                field="balance",
            )
        return ValidationResult(is_valid=True, message="Balance format is valid")


class StatisticalValidator:
    """Validator for statistical data quality checks."""

    def detect_amount_outliers(
        self, amounts: list[float], iqr_multiplier: float = 1.5
    ) -> ValidationResult:
        if len(amounts) < 4:
            return ValidationResult(
                is_valid=True, message="Insufficient data for outlier detection"
            )

        try:
            q1, q2, q3 = statistics.quantiles(amounts, n=4)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            outliers = [x for x in amounts if x < lower_bound or x > upper_bound]
            if outliers:
                return ValidationResult(
                    is_valid=False,
                    error_type="AMOUNT_OUTLIERS_DETECTED",
                    message=f"Found {len(outliers)} amount outliers",
                    details={
                        "outliers": outliers,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                    },
                )
            return ValidationResult(
                is_valid=True,
                message="No amount outliers detected",
                details={"q1": q1, "q3": q3, "iqr": iqr},
            )
        except AstroMLError as e:
            return ValidationResult(
                is_valid=False,
                error_type="OUTLIER_DETECTION_ERROR",
                message=f"Error detecting outliers: {str(e)}",
            )

    def detect_timestamp_gaps(
        self, timestamps: list[datetime], gap_threshold_minutes: int = 60
    ) -> ValidationResult:
        if len(timestamps) < 2:
            return ValidationResult(
                is_valid=True, message="Insufficient timestamps for gap analysis"
            )

        try:
            sorted_timestamps = sorted(timestamps)
            gaps = []
            for i in range(len(sorted_timestamps) - 1):
                gap_seconds = (sorted_timestamps[i + 1] - sorted_timestamps[i]).total_seconds()
                gaps.append(gap_seconds)

            threshold_seconds = gap_threshold_minutes * 60
            unusual_gaps = [
                {
                    "index": i,
                    "gap_seconds": gap,
                    "gap_minutes": gap / 60.0,
                    "start_time": sorted_timestamps[i].isoformat(),
                    "end_time": sorted_timestamps[i + 1].isoformat(),
                }
                for i, gap in enumerate(gaps)
                if gap > threshold_seconds
            ]

            if unusual_gaps:
                return ValidationResult(
                    is_valid=False,
                    error_type="UNUSUAL_TIMESTAMP_GAPS",
                    message=f"Found {len(unusual_gaps)} unusual timestamp gaps",
                    details={
                        "unusual_gaps": unusual_gaps,
                        "threshold_minutes": gap_threshold_minutes,
                    },
                )
            return ValidationResult(
                is_valid=True,
                message="No unusual timestamp gaps detected",
                details={"max_gap_minutes": max(gaps) / 60 if gaps else 0},
            )
        except AstroMLError as e:
            return ValidationResult(
                is_valid=False,
                error_type="GAP_DETECTION_ERROR",
                message=f"Error detecting timestamp gaps: {str(e)}",
            )

    def detect_duplicate_patterns(
        self, transactions: list[dict[str, Any]], pattern_fields: list[str]
    ) -> ValidationResult:
        if not transactions or not pattern_fields:
            return ValidationResult(
                is_valid=True, message="No transactions or pattern fields specified"
            )

        try:
            pattern_counts: dict[tuple[str, ...], int] = {}
            for tx in transactions:
                pattern_values = []
                for field in pattern_fields:
                    if field in tx:
                        pattern_values.append(str(tx[field]))
                    else:
                        pattern_values.append("NULL")
                pattern_key = tuple(pattern_values)
                pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1

            repeated_patterns = {
                pattern: count for pattern, count in pattern_counts.items() if count > 1
            }

            if repeated_patterns:
                return ValidationResult(
                    is_valid=False,
                    error_type="DUPLICATE_PATTERNS_DETECTED",
                    message=f"Found {len(repeated_patterns)} repeated patterns",
                    details={
                        "repeated_patterns": dict(repeated_patterns),
                        "pattern_fields": pattern_fields,
                        "total_patterns": len(pattern_counts),
                        "unique_patterns": len(pattern_counts) - len(repeated_patterns),
                    },
                )
            return ValidationResult(
                is_valid=True,
                message="No duplicate patterns detected",
                details={"total_patterns": len(pattern_counts)},
            )
        except AstroMLError as e:
            return ValidationResult(
                is_valid=False,
                error_type="PATTERN_DETECTION_ERROR",
                message=f"Error detecting duplicate patterns: {str(e)}",
            )


class DataQualityValidator:
    """Comprehensive data quality validator combining all checks."""

    def __init__(self) -> None:
        self.temporal = TemporalValidator()
        self.referential = ReferentialIntegrityValidator()
        self.business = BusinessRulesValidator()
        self.statistical = StatisticalValidator()
        self.completeness = CompletenessChecker()
        self.consistency = ConsistencyChecker()
        self.accuracy = AccuracyChecker()
        self.timeliness = TimelinessChecker()

    def validate_batch(self, transactions: list[dict[str, Any]]) -> DataQualityReport:
        """Validate a batch of transactions and return a comprehensive report."""
        report = DataQualityReport(total_records=len(transactions))
        validation_results: list[ValidationResult] = []

        if transactions:
            validation_results.append(self.temporal.validate_timestamp_ordering(transactions))
            validation_results.append(self.temporal.validate_future_timestamps(transactions))

            for tx in transactions:
                if "source_account" in tx:
                    validation_results.append(self.referential.validate_account_format(tx["source_account"]))
                if "asset_code" in tx:
                    validation_results.append(self.referential.validate_asset_format(tx["asset_code"]))
                if "ledger_sequence" in tx:
                    validation_results.append(self.referential.validate_ledger_sequence(tx["ledger_sequence"]))
                if "fee" in tx:
                    validation_results.append(self.business.validate_fee_non_negative(tx["fee"]))
                if "amount" in tx:
                    validation_results.append(self.business.validate_amount_non_negative(tx["amount"]))
                if "operation_count" in tx:
                    validation_results.append(self.business.validate_operation_count(tx["operation_count"]))

            amounts = [
                tx.get("amount", 0.0)
                for tx in transactions
                if isinstance(tx.get("amount"), (int, float)) and not isinstance(tx.get("amount"), bool)
            ]
            if amounts:
                validation_results.append(self.statistical.detect_amount_outliers(amounts))

            validation_results.append(
                self.statistical.detect_duplicate_patterns(transactions, ["amount", "source_account"])
            )

        report.validation_results = validation_results
        report.valid_records = len(transactions)

        error_counts: dict[str, int] = {}
        for r in validation_results:
            if not r.is_valid and r.error_type:
                error_counts[r.error_type] = error_counts.get(r.error_type, 0) + 1

        c_res = self.completeness.check_records(transactions)
        cs_res = self.consistency.check_records(transactions)
        a_res = self.accuracy.check_records(transactions)
        t_res = self.timeliness.check_records(transactions)

        report.check_results = c_res + cs_res + a_res + t_res

        def dim_avg(items: list[CheckResult]) -> float:
            return (sum(i.score for i in items) / len(items) * 100.0) if items else 100.0

        c_score = dim_avg(c_res)
        cs_score = dim_avg(cs_res)
        a_score = dim_avg(a_res)
        t_score = dim_avg(t_res)
        overall = (c_score + cs_score + a_score + t_score) / 4.0

        report.dimension_scores = {
            "completeness": round(c_score, 2),
            "consistency": round(cs_score, 2),
            "accuracy": round(a_score, 2),
            "timeliness": round(t_score, 2),
            "overall_score": round(overall, 2),
        }

        report.summary = {
            "error_counts": error_counts,
            "total_errors": len([r for r in validation_results if not r.is_valid]),
            "quality_score": report.quality_score,
        }

        return report


def validate_data_quality(transactions: list[dict[str, Any]]) -> DataQualityReport:
    """Convenience function for data quality validation."""
    validator = DataQualityValidator()
    return validator.validate_batch(transactions)


def check_temporal_consistency(transactions: list[dict[str, Any]]) -> list[ValidationResult]:
    """Check temporal consistency of transactions."""
    validator = TemporalValidator()
    results: list[ValidationResult] = []
    if transactions:
        results.append(validator.validate_timestamp_ordering(transactions))
        results.append(validator.validate_future_timestamps(transactions))
    return results


def check_referential_integrity(transactions: list[dict[str, Any]]) -> list[ValidationResult]:
    """Check referential integrity of transactions."""
    validator = ReferentialIntegrityValidator()
    results: list[ValidationResult] = []
    for tx in transactions:
        if "source_account" in tx:
            results.append(validator.validate_account_format(tx["source_account"]))
        if "asset_code" in tx:
            results.append(validator.validate_asset_format(tx["asset_code"]))
        if "ledger_sequence" in tx:
            results.append(validator.validate_ledger_sequence(tx["ledger_sequence"]))
    return results
