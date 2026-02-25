"""Deduplication utilities for transaction processing.

This module provides hash-based deduplication to detect and handle
duplicate transactions. It maintains an in-memory set of seen hashes
and provides structured logging for conflicts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .hashing import compute_transaction_hash

logger = logging.getLogger(__name__)


class ConflictType:
    """Constants for conflict types."""

    DUPLICATE = "DUPLICATE"
    CORRUPTED = "CORRUPTED"


@dataclass
class ConflictRecord:
    """Structured record of a deduplication conflict.

    Attributes:
        transaction_id: ID of the conflicting transaction.
        hash: Hash of the transaction.
        conflict_type: Type of conflict (DUPLICATE or CORRUPTED).
        timestamp: When the conflict was detected.
        source: Optional source identifier for the transaction.
        message: Additional context about the conflict.
    """

    transaction_id: Optional[str]
    hash: str
    conflict_type: str
    timestamp: str
    source: Optional[str] = None
    message: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class DeduplicationResult:
    """Result of deduplication processing.

    Attributes:
        unique: List of unique transactions.
        duplicates: List of duplicate transactions.
        hashes: Set of all hashes seen.
        conflicts: List of conflict records.
    """

    unique: List[Dict[str, Any]] = field(default_factory=list)
    duplicates: List[Dict[str, Any]] = field(default_factory=list)
    hashes: Set[str] = field(default_factory=set)
    conflicts: List[ConflictRecord] = field(default_factory=list)


class Deduplicator:
    """Hash-based transaction deduplicator.

    Maintains an in-memory set of seen transaction hashes and provides
    methods to detect and filter duplicates from transaction batches.
    """

    def __init__(
        self,
        hash_fields: Optional[Set[str]] = None,
        track_conflicts: bool = True,
    ) -> None:
        """Initialize the deduplicator.

        Args:
            hash_fields: Set of fields to use for hash computation.
            track_conflicts: Whether to track conflict records.
        """
        self.hash_fields = hash_fields
        self._seen_hashes: Set[str] = set()
        self._track_conflicts = track_conflicts
        self._conflicts: List[ConflictRecord] = []

    @property
    def seen_hashes(self) -> Set[str]:
        """Return a copy of the seen hashes set."""
        return self._seen_hashes.copy()

    @property
    def conflicts(self) -> List[ConflictRecord]:
        """Return a copy of the conflict records."""
        return self._conflicts.copy()

    def reset(self) -> None:
        """Clear all seen hashes and conflict records."""
        self._seen_hashes.clear()
        self._conflicts.clear()

    def _log_conflict(
        self,
        transaction: Dict[str, Any],
        hash_value: str,
        conflict_type: str,
        source: Optional[str] = None,
    ) -> None:
        """Log a conflict with structured information.

        Args:
            transaction: The transaction that caused the conflict.
            hash_value: Hash of the transaction.
            conflict_type: Type of conflict (DUPLICATE or CORRUPTED).
            source: Optional source identifier.
        """
        transaction_id = transaction.get("id")
        message = f"Transaction {conflict_type.lower()} detected"

        conflict = ConflictRecord(
            transaction_id=transaction_id,
            hash=hash_value,
            conflict_type=conflict_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            source=source,
            message=message,
        )

        if self._track_conflicts:
            self._conflicts.append(conflict)

        logger.info(
            "Transaction conflict: id=%s hash=%s type=%s source=%s",
            transaction_id,
            hash_value,
            conflict_type,
            source,
        )

    def add(
        self,
        transaction: Dict[str, Any],
        source: Optional[str] = None,
    ) -> bool:
        """Add a transaction hash to the seen set.

        Args:
            transaction: Transaction dictionary to add.
            source: Optional source identifier.

        Returns:
            True if the transaction was added (not a duplicate).
            False if it was already seen (duplicate).
        """
        hash_value = compute_transaction_hash(transaction, fields=self.hash_fields)

        if hash_value in self._seen_hashes:
            self._log_conflict(transaction, hash_value, ConflictType.DUPLICATE, source)
            return False

        self._seen_hashes.add(hash_value)
        return True

    def check(self, transaction: Dict[str, Any]) -> bool:
        """Check if a transaction is a duplicate without adding it.

        Args:
            transaction: Transaction dictionary to check.

        Returns:
            True if the transaction is a duplicate.
        """
        hash_value = compute_transaction_hash(transaction, fields=self.hash_fields)
        return hash_value in self._seen_hashes

    def process(
        self,
        transactions: List[Dict[str, Any]],
        source: Optional[str] = None,
    ) -> DeduplicationResult:
        """Process a batch of transactions, filtering duplicates.

        Args:
            transactions: List of transaction dictionaries.
            source: Optional source identifier.

        Returns:
            DeduplicationResult with unique/duplicate splits and hashes.
        """
        result = DeduplicationResult()

        for transaction in transactions:
            hash_value = compute_transaction_hash(
                transaction, fields=self.hash_fields
            )

            if hash_value in self._seen_hashes:
                result.duplicates.append(transaction)
                if self._track_conflicts:
                    self._log_conflict(
                        transaction, hash_value, ConflictType.DUPLICATE, source
                    )
            else:
                self._seen_hashes.add(hash_value)
                result.unique.append(transaction)

            result.hashes.add(hash_value)

        return result

    def filter_duplicates(
        self,
        transactions: List[Dict[str, Any]],
        return_unique: bool = True,
    ) -> List[Dict[str, Any]]:
        """Filter duplicate transactions from a list.

        Args:
            transactions: List of transaction dictionaries.
            return_unique: If True, return unique transactions.
                          If False, return duplicate transactions.

        Returns:
            List of either unique or duplicate transactions.
        """
        result = self.process(transactions)
        return result.unique if return_unique else result.duplicates


def deduplicate(
    transactions: List[Dict[str, Any]],
    hash_fields: Optional[Set[str]] = None,
) -> DeduplicationResult:
    """Convenience function to deduplicate a list of transactions.

    Creates a new Deduplicator instance and processes the transactions.

    Args:
        transactions: List of transaction dictionaries.
        hash_fields: Set of fields to use for hash computation.

    Returns:
        DeduplicationResult with unique/duplicate splits and hashes.
    """
    dedup = Deduplicator(hash_fields=hash_fields)
    return dedup.process(transactions)
