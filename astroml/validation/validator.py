"""Transaction validation utilities for corruption detection.

This module provides validation layers to detect corrupted transactions
before they enter the processing pipeline. A transaction is considered
corrupted if it fails schema validation, has missing required fields,
or has hash mismatches.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .hashing import compute_transaction_hash, verify_transaction_hash

logger = logging.getLogger(__name__)


class CorruptionType:
    """Constants for corruption types."""

    MISSING_FIELD = "MISSING_FIELD"
    INVALID_TYPE = "INVALID_TYPE"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    HASH_MISMATCH = "HASH_MISMATCH"
    MALFORMED_STRUCTURE = "MALFORMED_STRUCTURE"


@dataclass
class ValidationError:
    """Structured validation error information.

    Attributes:
        transaction_id: ID of the transaction that failed validation.
        error_type: Type of corruption detected.
        message: Human-readable error message.
        field: Field name where the error occurred (if applicable).
        timestamp: When the validation error was detected.
    """

    transaction_id: Optional[str]
    error_type: str
    message: str
    field: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class ValidationResult:
    """Result of transaction validation.

    Attributes:
        is_valid: Whether the transaction passed validation.
        errors: List of validation errors (empty if valid).
        transaction_id: ID of the validated transaction.
        hash: Computed hash of the transaction.
    """

    is_valid: bool
    errors: List[ValidationError]
    transaction_id: Optional[str]
    hash: str


class TransactionValidator:
    """Validator for transaction integrity and schema compliance."""

    def __init__(
        self,
        required_fields: Optional[Set[str]] = None,
        field_types: Optional[Dict[str, type]] = None,
        hash_fields: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the validator.

        Args:
            required_fields: Set of required field names. Defaults to {"id"}.
            field_types: Dict mapping field names to expected types.
            hash_fields: Set of fields to use for hash computation.
        """
        self.required_fields = required_fields or {"id"}
        self.field_types = field_types or {}
        self.hash_fields = hash_fields

    def validate(
        self,
        transaction: Dict[str, Any],
        stored_hash: Optional[str] = None,
    ) -> ValidationResult:
        """Validate a single transaction.

        Args:
            transaction: Transaction dictionary to validate.
            stored_hash: Optional pre-stored hash for verification.

        Returns:
            ValidationResult with validation status and any errors.
        """
        errors: List[ValidationError] = []
        transaction_id = transaction.get("id")

        # Check for missing required fields
        for field in self.required_fields:
            if field not in transaction or transaction[field] is None:
                errors.append(
                    ValidationError(
                        transaction_id=transaction_id,
                        error_type=CorruptionType.MISSING_FIELD,
                        message=f"Required field '{field}' is missing or null",
                        field=field,
                    )
                )

        # Check for invalid types
        for field, expected_type in self.field_types.items():
            if field in transaction and transaction[field] is not None:
                if not isinstance(transaction[field], expected_type):
                    errors.append(
                        ValidationError(
                            transaction_id=transaction_id,
                            error_type=CorruptionType.INVALID_TYPE,
                            message=f"Field '{field}' has type {type(transaction[field]).__name__}, "
                            f"expected {expected_type.__name__}",
                            field=field,
                        )
                    )

        # Check for malformed structure
        if not isinstance(transaction, dict):
            errors.append(
                ValidationError(
                    transaction_id=transaction_id,
                    error_type=CorruptionType.MALFORMED_STRUCTURE,
                    message="Transaction is not a dictionary",
                )
            )

        # Compute hash for the transaction
        tx_hash = compute_transaction_hash(
            transaction, fields=self.hash_fields, stored_hash=stored_hash
        )

        # Verify hash if stored hash is provided
        if stored_hash is not None and stored_hash != tx_hash:
            errors.append(
                ValidationError(
                    transaction_id=transaction_id,
                    error_type=CorruptionType.HASH_MISMATCH,
                    message=f"Hash mismatch: expected {stored_hash}, computed {tx_hash}",
                )
            )

        # Log validation errors
        if errors:
            for error in errors:
                logger.warning(
                    "Transaction validation failed: id=%s type=%s message=%s field=%s",
                    transaction_id,
                    error.error_type,
                    error.message,
                    error.field,
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            transaction_id=transaction_id,
            hash=tx_hash,
        )

    def validate_batch(
        self,
        transactions: List[Dict[str, Any]],
        stored_hashes: Optional[List[str]] = None,
    ) -> List[ValidationResult]:
        """Validate a batch of transactions.

        Args:
            transactions: List of transaction dictionaries to validate.
            stored_hashes: Optional list of pre-stored hashes for verification.

        Returns:
            List of ValidationResult in the same order as input transactions.
        """
        results: List[ValidationResult] = []

        for i, transaction in enumerate(transactions):
            stored_hash = None
            if stored_hashes is not None and i < len(stored_hashes):
                stored_hash = stored_hashes[i]

            result = self.validate(transaction, stored_hash=stored_hash)
            results.append(result)

        return results


def validate_transaction(
    transaction: Dict[str, Any],
    required_fields: Optional[Set[str]] = None,
    field_types: Optional[Dict[str, type]] = None,
    stored_hash: Optional[str] = None,
) -> ValidationResult:
    """Convenience function to validate a single transaction.

    Args:
        transaction: Transaction dictionary to validate.
        required_fields: Set of required field names.
        field_types: Dict mapping field names to expected types.
        stored_hash: Optional pre-stored hash for verification.

    Returns:
        ValidationResult with validation status and any errors.
    """
    validator = TransactionValidator(
        required_fields=required_fields,
        field_types=field_types,
    )
    return validator.validate(transaction, stored_hash=stored_hash)
