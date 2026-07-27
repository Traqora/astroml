"""Extended data quality validation utilities.

This module provides additional validation functions for temporal consistency,
referential integrity, business rules, and statistical validation beyond the
basic corruption detection in the validator module.
"""
from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when a data quality check fails."""
    pass


@dataclass
class ValidationResult:
    """Result of a data quality validation check.

    Attributes:
        is_valid: Whether the data passed the validation check.
        error_type: Type of validation error that occurred.
        message: Human-readable error message.
        field: Field name where the error occurred (if applicable).
        details: Additional details about the validation result.
    """

    is_valid: bool
    error_type: Optional[str] = None
    message: Optional[str] = None
    field: Optional[str] = None
    details: Dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report for a batch of transactions.

    Attributes:
        total_records: Total number of records processed.
        valid_records: Number of records that passed all validations.
        validation_results: List of individual validation results.
        summary: Summary statistics about the data quality.
    """

    total_records: int = 0
    valid_records: int = 0
    validation_results: List[ValidationResult] = dc_field(default_factory=list)
    summary: Dict[str, Any] = dc_field(default_factory=dict)

    @property
    def quality_score(self) -> float:
        """Calculate data quality score as percentage of valid records."""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100

    @property
    def error_types(self) -> Set[str]:
        """Get set of unique error types found."""
        return {r.error_type for r in self.validation_results if not r.is_valid and r.error_type}


class TemporalValidator:
    """Validator for temporal data quality checks."""

    def __init__(self, timestamp_field: str = "timestamp"):
        """Initialize temporal validator.

        Args:
            timestamp_field: Name of the timestamp field to validate.
        """
        self.timestamp_field = timestamp_field

    def validate_timestamp_ordering(self, transactions: List[Dict[str, Any]]) -> ValidationResult:
        """Validate that timestamps are monotonically increasing within a batch.

        Args:
            transactions: List of transaction dictionaries.

        Returns:
            ValidationResult with ordering check result.
        """
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
                        field=self.timestamp_field
                    )
                
                ts_str = tx[self.timestamp_field]
                if isinstance(ts_str, str):
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                elif isinstance(ts_str, datetime):
                    ts = ts_str
                else:
                    return ValidationResult(
                        is_valid=False,
                        error_type="INVALID_TIMESTAMP_FORMAT",
                        message=f"Invalid timestamp format: {type(ts_str)}",
                        field=self.timestamp_field
                    )
                timestamps.append(ts)

            # Check if timestamps are monotonically increasing
            is_ordered = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            
            if not is_ordered:
                # Find the first out-of-order timestamp
                for i in range(len(timestamps)-1):
                    if timestamps[i] > timestamps[i+1]:
                        return ValidationResult(
                            is_valid=False,
                            error_type="TIMESTAMP_ORDER_VIOLATION",
                            message=f"Timestamp order violation at index {i}: {timestamps[i]} > {timestamps[i+1]}",
                            details={"index": i, "current": timestamps[i].isoformat(), "next": timestamps[i+1].isoformat()}
                        )

            return ValidationResult(is_valid=True, message="Timestamps are properly ordered")

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_type="TIMESTAMP_VALIDATION_ERROR",
                message=f"Error validating timestamps: {str(e)}"
            )

    def validate_future_timestamps(self, transactions: List[Dict[str, Any]], 
                                  tolerance_minutes: int = 5) -> ValidationResult:
        """Validate that no transactions have timestamps significantly in the future.

        Args:
            transactions: List of transaction dictionaries.
            tolerance_minutes: Minutes of future tolerance to account for clock skew.

        Returns:
            ValidationResult with future timestamp check result.
        """
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
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                elif isinstance(ts_str, datetime):
                    ts = ts_str
                else:
                    continue

                if ts > now + tolerance:
                    future_txs.append({
                        "id": tx.get("id", "unknown"),
                        "timestamp": ts.isoformat(),
                        "minutes_ahead": (ts - now).total_seconds() / 60
                    })

            if future_txs:
                return ValidationResult(
                    is_valid=False,
                    error_type="FUTURE_TIMESTAMP",
                    message=f"Found {len(future_txs)} transactions with future timestamps",
                    details={"future_transactions": future_txs}
                )

            return ValidationResult(is_valid=True, message="No future timestamps detected")

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_type="FUTURE_TIMESTAMP_ERROR",
                message=f"Error checking future timestamps: {str(e)}"
            )


class ReferentialIntegrityValidator:
    """Validator for referential integrity checks."""

    def __init__(self):
        """Initialize referential integrity validator."""
        self.account_pattern = re.compile(r'^G[A-Z0-9]{56}$')
        self.asset_code_pattern = re.compile(r'^[A-Z0-9]{1,12}$')

    def validate_account_format(self, account: str) -> ValidationResult:
        """Validate Stellar account address format.

        Args:
            account: Account address string to validate.

        Returns:
            ValidationResult with format check result.
        """
        if not isinstance(account, str):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_ACCOUNT_TYPE",
                message=f"Account must be string, got {type(account)}",
                field="account"
            )

        if self.account_pattern.match(account):
            return ValidationResult(is_valid=True, message="Account format is valid")
        else:
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_ACCOUNT_FORMAT",
                message=f"Invalid Stellar account format: {account}",
                field="account"
            )

    def validate_asset_format(self, asset_code: str) -> ValidationResult:
        """Validate asset code format.

        Args:
            asset_code: Asset code string to validate.

        Returns:
            ValidationResult with format check result.
        """
        if not isinstance(asset_code, str):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_ASSET_TYPE",
                message=f"Asset code must be string, got {type(asset_code)}",
                field="asset_code"
            )

        if self.asset_code_pattern.match(asset_code):
            return ValidationResult(is_valid=True, message="Asset code format is valid")
        else:
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_ASSET_FORMAT",
                message=f"Invalid asset code format: {asset_code}",
                field="asset_code"
            )

    def validate_ledger_sequence(self, ledger_sequence: int) -> ValidationResult:
        """Validate ledger sequence is positive.

        Args:
            ledger_sequence: Ledger sequence number to validate.

        Returns:
            ValidationResult with sequence check result.
        """
        if not isinstance(ledger_sequence, int):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_LEDGER_SEQUENCE_TYPE",
                message=f"Ledger sequence must be integer, got {type(ledger_sequence)}",
                field="ledger_sequence"
            )

        if ledger_sequence > 0:
            return ValidationResult(is_valid=True, message="Ledger sequence is valid")
        else:
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_LEDGER_SEQUENCE",
                message=f"Ledger sequence must be positive, got {ledger_sequence}",
                field="ledger_sequence"
            )


class BusinessRulesValidator:
    """Validator for business logic rules."""

    def __init__(self):
        """Initialize business rules validator."""
        self.max_operations_per_transaction = 100

    def validate_fee_non_negative(self, fee: int) -> ValidationResult:
        """Validate that transaction fee is non-negative.

        Args:
            fee: Transaction fee amount.

        Returns:
            ValidationResult with fee check result.
        """
        if not isinstance(fee, (int, float)):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_FEE_TYPE",
                message=f"Fee must be numeric, got {type(fee)}",
                field="fee"
            )

        if fee >= 0:
            return ValidationResult(is_valid=True, message="Fee is valid")
        else:
            return ValidationResult(
                is_valid=False,
                error_type="NEGATIVE_FEE",
                message=f"Fee cannot be negative: {fee}",
                field="fee"
            )

    def validate_amount_non_negative(self, amount: float) -> ValidationResult:
        """Validate that transaction amount is non-negative.

        Args:
            amount: Transaction amount.

        Returns:
            ValidationResult with amount check result.
        """
        if not isinstance(amount, (int, float)):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_AMOUNT_TYPE",
                message=f"Amount must be numeric, got {type(amount)}",
                field="amount"
            )

        if amount >= 0:
            return ValidationResult(is_valid=True, message="Amount is valid")
        else:
            return ValidationResult(
                is_valid=False,
                error_type="NEGATIVE_AMOUNT",
                message=f"Amount cannot be negative: {amount}",
                field="amount"
            )

    def validate_operation_count(self, operation_count: int) -> ValidationResult:
        """Validate operation count is within reasonable bounds.

        Args:
            operation_count: Number of operations in transaction.

        Returns:
            ValidationResult with operation count check result.
        """
        if not isinstance(operation_count, int):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_OPERATION_COUNT_TYPE",
                message=f"Operation count must be integer, got {type(operation_count)}",
                field="operation_count"
            )

        if 1 <= operation_count <= self.max_operations_per_transaction:
            return ValidationResult(is_valid=True, message="Operation count is valid")
        else:
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_OPERATION_COUNT",
                message=f"Operation count must be between 1 and {self.max_operations_per_transaction}, got {operation_count}",
                field="operation_count"
            )

    def validate_balance_format(self, balance: Any) -> ValidationResult:
        """Validate balance is a proper numeric value.

        Args:
            balance: Account balance to validate.

        Returns:
            ValidationResult with balance check result.
        """
        if balance is None:
            return ValidationResult(is_valid=True, message="Balance can be None")

        if not isinstance(balance, (int, float)):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_BALANCE_TYPE",
                message=f"Balance must be numeric, got {type(balance)}",
                field="balance"
            )

        # Check for NaN or infinite values
        if balance != balance or balance in [float('inf'), float('-inf')]:
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_BALANCE_VALUE",
                message=f"Balance cannot be NaN or infinite: {balance}",
                field="balance"
            )

        return ValidationResult(is_valid=True, message="Balance format is valid")


class StatisticalValidator:
    """Validator for statistical data quality checks."""

    def detect_amount_outliers(self, amounts: List[float], iqr_multiplier: float = 1.5) -> ValidationResult:
        """Detect statistical outliers in transaction amounts using IQR method.

        Args:
            amounts: List of transaction amounts.
            iqr_multiplier: Multiplier for IQR outlier detection threshold.

        Returns:
            ValidationResult with outlier detection result.
        """
        if len(amounts) < 4:  # Need at least 4 values for meaningful quartiles
            return ValidationResult(
                is_valid=True, 
                message="Insufficient data for outlier detection"
            )

        try:
            # Calculate quartiles
            q1, q2, q3 = statistics.quantiles(amounts, n=4)
            iqr = q3 - q1

            # Calculate outlier bounds
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            # Find outliers
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
                        "iqr": iqr
                    }
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    message="No amount outliers detected",
                    details={"q1": q1, "q3": q3, "iqr": iqr}
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_type="OUTLIER_DETECTION_ERROR",
                message=f"Error detecting outliers: {str(e)}"
            )

    def detect_timestamp_gaps(self, timestamps: List[datetime], 
                           gap_threshold_minutes: int = 60) -> ValidationResult:
        """Detect unusual gaps in timestamps.

        Args:
            timestamps: List of timestamp objects.
            gap_threshold_minutes: Threshold in minutes for flagging unusual gaps.

        Returns:
            ValidationResult with gap detection result.
        """
        if len(timestamps) < 2:
            return ValidationResult(
                is_valid=True,
                message="Insufficient timestamps for gap analysis"
            )

        try:
            # Sort timestamps
            sorted_timestamps = sorted(timestamps)
            
            # Calculate gaps
            gaps = []
            for i in range(len(sorted_timestamps) - 1):
                gap_seconds = (sorted_timestamps[i+1] - sorted_timestamps[i]).total_seconds()
                gaps.append(gap_seconds)

            # Find unusual gaps
            threshold_seconds = gap_threshold_minutes * 60
            unusual_gaps = [
                {
                    "index": i,
                    "gap_seconds": gap,
                    "gap_minutes": gap / 60,
                    "start_time": sorted_timestamps[i].isoformat(),
                    "end_time": sorted_timestamps[i+1].isoformat()
                }
                for i, gap in enumerate(gaps) if gap > threshold_seconds
            ]

            if unusual_gaps:
                return ValidationResult(
                    is_valid=False,
                    error_type="UNUSUAL_TIMESTAMP_GAPS",
                    message=f"Found {len(unusual_gaps)} unusual timestamp gaps",
                    details={"unusual_gaps": unusual_gaps, "threshold_minutes": gap_threshold_minutes}
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    message="No unusual timestamp gaps detected",
                    details={"max_gap_minutes": max(gaps) / 60 if gaps else 0}
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_type="GAP_DETECTION_ERROR",
                message=f"Error detecting timestamp gaps: {str(e)}"
            )

    def detect_duplicate_patterns(self, transactions: List[Dict[str, Any]], 
                                pattern_fields: List[str]) -> ValidationResult:
        """Detect patterns that might indicate data duplication issues.

        Args:
            transactions: List of transaction dictionaries.
            pattern_fields: Fields to use for pattern detection.

        Returns:
            ValidationResult with pattern detection result.
        """
        if not transactions or not pattern_fields:
            return ValidationResult(
                is_valid=True,
                message="No transactions or pattern fields specified"
            )

        try:
            # Count pattern occurrences
            pattern_counts = {}
            for tx in transactions:
                # Create pattern key from specified fields
                pattern_values = []
                for field in pattern_fields:
                    if field in tx:
                        pattern_values.append(str(tx[field]))
                    else:
                        pattern_values.append("NULL")
                
                pattern_key = tuple(pattern_values)
                pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1

            # Find repeated patterns
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
                        "unique_patterns": len(pattern_counts) - len(repeated_patterns)
                    }
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    message="No duplicate patterns detected",
                    details={"total_patterns": len(pattern_counts)}
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_type="PATTERN_DETECTION_ERROR",
                message=f"Error detecting duplicate patterns: {str(e)}"
            )


class DataQualityValidator:
    """Comprehensive data quality validator combining all validation types."""

    def __init__(self):
        """Initialize comprehensive data quality validator."""
        self.temporal = TemporalValidator()
        self.referential = ReferentialIntegrityValidator()
        self.business = BusinessRulesValidator()
        self.statistical = StatisticalValidator()

    def validate_batch(self, transactions: List[Dict[str, Any]]) -> DataQualityReport:
        """Perform comprehensive data quality validation on a batch of transactions.

        Args:
            transactions: List of transaction dictionaries to validate.

        Returns:
            DataQualityReport with comprehensive validation results.
        """
        report = DataQualityReport(total_records=len(transactions))
        validation_results = []

        # Temporal validations
        if transactions:
            temporal_order_result = self.temporal.validate_timestamp_ordering(transactions)
            validation_results.append(temporal_order_result)

            temporal_future_result = self.temporal.validate_future_timestamps(transactions)
            validation_results.append(temporal_future_result)

        # Individual transaction validations
        for tx in transactions:
            tx_results = []

            # Account format validation
            if "source_account" in tx:
                account_result = self.referential.validate_account_format(tx["source_account"])
                tx_results.append(account_result)

            # Asset format validation
            if "asset_code" in tx:
                asset_result = self.referential.validate_asset_format(tx["asset_code"])
                tx_results.append(asset_result)

            # Ledger sequence validation
            if "ledger_sequence" in tx:
                ledger_result = self.referential.validate_ledger_sequence(tx["ledger_sequence"])
                tx_results.append(ledger_result)

            # Business rule validations
            if "fee" in tx:
                fee_result = self.business.validate_fee_non_negative(tx["fee"])
                tx_results.append(fee_result)

            if "amount" in tx:
                amount_result = self.business.validate_amount_non_negative(tx["amount"])
                tx_results.append(amount_result)

            if "operation_count" in tx:
                op_count_result = self.business.validate_operation_count(tx["operation_count"])
                tx_results.append(op_count_result)

            # Add transaction results to overall results
            validation_results.extend(tx_results)

        # Statistical validations
        if transactions:
            # Amount outlier detection
            amounts = [tx.get("amount", 0) for tx in transactions if isinstance(tx.get("amount"), (int, float))]
            if amounts:
                outlier_result = self.statistical.detect_amount_outliers(amounts)
                validation_results.append(outlier_result)

            # Duplicate pattern detection
            pattern_result = self.statistical.detect_duplicate_patterns(transactions, ["amount", "source_account"])
            validation_results.append(pattern_result)

        # Compile report
        report.validation_results = validation_results
        report.valid_records = len(transactions)  # Simplified - should be based on actual validation failures
        
        # Generate summary
        error_counts = {}
        for result in validation_results:
            if not result.is_valid and result.error_type:
                error_counts[result.error_type] = error_counts.get(result.error_type, 0) + 1

        report.summary = {
            "error_counts": error_counts,
            "total_errors": len([r for r in validation_results if not r.is_valid]),
            "quality_score": report.quality_score
        }

        return report


# Convenience functions

def validate_data_quality(transactions: List[Dict[str, Any]]) -> DataQualityReport:
    """Convenience function for comprehensive data quality validation.

    Args:
        transactions: List of transaction dictionaries to validate.

    Returns:
        DataQualityReport with validation results.
    """
    validator = DataQualityValidator()
    return validator.validate_batch(transactions)


def check_temporal_consistency(transactions: List[Dict[str, Any]]) -> List[ValidationResult]:
    """Check temporal consistency of transactions.

    Args:
        transactions: List of transaction dictionaries.

    Returns:
        List of ValidationResult objects.
    """
    validator = TemporalValidator()
    results = []
    
    if transactions:
        results.append(validator.validate_timestamp_ordering(transactions))
        results.append(validator.validate_future_timestamps(transactions))
    
    return results


def check_referential_integrity(transactions: List[Dict[str, Any]]) -> List[ValidationResult]:
    """Check referential integrity of transactions.

    Args:
        transactions: List of transaction dictionaries.

    Returns:
        List of ValidationResult objects.
    """
    validator = ReferentialIntegrityValidator()
    results = []
    
    for tx in transactions:
        if "source_account" in tx:
            results.append(validator.validate_account_format(tx["source_account"]))
        if "asset_code" in tx:
            results.append(validator.validate_asset_format(tx["asset_code"]))
        if "ledger_sequence" in tx:
            results.append(validator.validate_ledger_sequence(tx["ledger_sequence"]))
    
    return results
