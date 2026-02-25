"""Integrity validation module combining deduplication and corruption detection.

This module provides a comprehensive integrity checking system that wraps
the deduplication and validation components into a single pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .dedupe import ConflictRecord, ConflictType, DeduplicationResult, Deduplicator
from .hashing import compute_transaction_hash
from .validator import (
    CorruptionType,
    ValidationError,
    ValidationResult,
    TransactionValidator,
)

logger = logging.getLogger(__name__)


@dataclass
class IntegrityResult:
    """Result of integrity validation.

    Attributes:
        valid: List of valid, unique transactions.
        duplicates: List of duplicate transactions.
        corrupted: List of corrupted transactions.
        all_hashes: Set of all hashes processed.
        validation_errors: List of all validation errors.
        conflicts: List of all conflict records.
    """

    valid: List[Dict[str, Any]] = field(default_factory=list)
    duplicates: List[Dict[str, Any]] = field(default_factory=list)
    corrupted: List[Dict[str, Any]] = field(default_factory=list)
    all_hashes: Set[str] = field(default_factory=set)
    validation_errors: List[ValidationError] = field(default_factory=list)
    conflicts: List[ConflictRecord] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if all transactions passed integrity checks."""
        return len(self.corrupted) == 0

    @property
    def has_duplicates(self) -> bool:
        """Check if any duplicates were detected."""
        return len(self.duplicates) > 0


class IntegrityValidator:
    """Combined deduplication and validation pipeline.

    This validator provides a single interface for both detecting duplicates
    and identifying corrupted transactions before processing.
    """

    def __init__(
        self,
        required_fields: Optional[Set[str]] = None,
        field_types: Optional[Dict[str, type]] = None,
        hash_fields: Optional[Set[str]] = None,
        strict: bool = False,
    ) -> None:
        """Initialize the integrity validator.

        Args:
            required_fields: Set of required field names.
            field_types: Dict mapping field names to expected types.
            hash_fields: Set of fields to use for hash computation.
            strict: If True, raise on first corruption. Defaults to False.
        """
        self._deduplicator = Deduplicator(hash_fields=hash_fields)
        self._validator = TransactionValidator(
            required_fields=required_fields,
            field_types=field_types,
            hash_fields=hash_fields,
        )
        self._strict = strict
        self._hash_fields = hash_fields

    def reset(self) -> None:
        """Reset the deduplicator state."""
        self._deduplicator.reset()

    @property
    def seen_hashes(self) -> Set[str]:
        """Return a copy of the seen hashes set."""
        return self._deduplicator.seen_hashes

    @property
    def conflicts(self) -> List[ConflictRecord]:
        """Return a copy of the conflict records."""
        return self._deduplicator.conflicts

    def validate_transaction(
        self,
        transaction: Dict[str, Any],
    ) -> ValidationResult:
        """Validate a single transaction for corruption.

        Args:
            transaction: Transaction dictionary to validate.

        Returns:
            ValidationResult with validation status.
        """
        return self._validator.validate(transaction)

    def check_duplicate(
        self,
        transaction: Dict[str, Any],
    ) -> bool:
        """Check if a transaction is a duplicate.

        Args:
            transaction: Transaction dictionary to check.

        Returns:
            True if the transaction is a duplicate.
        """
        return self._deduplicator.check(transaction)

    def add_transaction(
        self,
        transaction: Dict[str, Any],
        source: Optional[str] = None,
    ) -> bool:
        """Add a transaction to the seen set.

        Args:
            transaction: Transaction dictionary to add.
            source: Optional source identifier.

        Returns:
            True if the transaction was added (not a duplicate).
        """
        return self._deduplicator.add(transaction, source=source)

    def process(
        self,
        transactions: List[Dict[str, Any]],
        source: Optional[str] = None,
    ) -> IntegrityResult:
        """Process a batch of transactions through integrity checks.

        This method:
        1. Validates each transaction for corruption
        2. Filters out corrupted transactions
        3. Detects and logs duplicates
        4. Returns unique, valid transactions

        Args:
            transactions: List of transaction dictionaries.
            source: Optional source identifier.

        Returns:
            IntegrityResult with categorized transactions.
        """
        result = IntegrityResult()

        for transaction in transactions:
            # First, validate for corruption
            validation = self._validator.validate(transaction)

            if not validation.is_valid:
                result.corrupted.append(transaction)
                result.validation_errors.extend(validation.errors)

                if self._strict:
                    raise IntegrityError(
                        f"Corrupted transaction detected: {validation.errors[0].message}"
                    )

                # Log corruption conflict
                hash_value = compute_transaction_hash(
                    transaction, fields=self._hash_fields
                )
                conflict = ConflictRecord(
                    transaction_id=validation.transaction_id,
                    hash=hash_value,
                    conflict_type=ConflictType.CORRUPTED,
                    timestamp=validation.errors[0].timestamp if validation.errors else "",
                    source=source,
                    message=f"Corruption detected: {validation.errors[0].message}",
                )
                result.conflicts.append(conflict)
                continue

            # Check for duplicates
            hash_value = compute_transaction_hash(
                transaction, fields=self._hash_fields
            )

            if hash_value in self._deduplicator.seen_hashes:
                result.duplicates.append(transaction)
                self._deduplicator._log_conflict(
                    transaction, hash_value, ConflictType.DUPLICATE, source
                )
            else:
                self._deduplicator._seen_hashes.add(hash_value)
                result.valid.append(transaction)

            result.all_hashes.add(hash_value)

        return result

    def verify_integrity(
        self,
        transactions: List[Dict[str, Any]],
    ) -> bool:
        """Verify that a batch of transactions passes integrity checks.

        Args:
            transactions: List of transaction dictionaries to verify.

        Returns:
            True if all transactions are valid and unique.
        """
        result = self.process(transactions)
        return result.is_valid and not result.has_duplicates


class IntegrityError(Exception):
    """Raised when a strict integrity check fails."""

    pass


# Convenience functions


def check_integrity(
    transactions: List[Dict[str, Any]],
    required_fields: Optional[Set[str]] = None,
    hash_fields: Optional[Set[str]] = None,
) -> IntegrityResult:
    """Convenience function to check integrity of transactions.

    Args:
        transactions: List of transaction dictionaries.
        required_fields: Set of required field names.
        hash_fields: Set of fields to use for hash computation.

    Returns:
        IntegrityResult with categorized transactions.
    """
    validator = IntegrityValidator(
        required_fields=required_fields,
        hash_fields=hash_fields,
    )
    return validator.process(transactions)


def filter_valid_transactions(
    transactions: List[Dict[str, Any]],
    required_fields: Optional[Set[str]] = None,
    hash_fields: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Filter a list to return only valid, unique transactions.

    Args:
        transactions: List of transaction dictionaries.
        required_fields: Set of required field names.
        hash_fields: Set of fields to use for hash computation.

    Returns:
        List of valid, unique transactions.
    """
    result = check_integrity(transactions, required_fields, hash_fields)
    return result.valid
