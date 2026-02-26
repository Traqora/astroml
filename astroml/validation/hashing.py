"""Hashing utilities for transaction integrity.

This module provides deterministic hash generation for transactions
using SHA-256. The hash is computed from stable, immutable fields
to enable deduplication and integrity verification.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Default fields to include in transaction hash computation
DEFAULT_HASH_FIELDS: Set[str] = {"id", "payload", "timestamp"}


def compute_transaction_hash(
    transaction: Dict[str, Any],
    fields: Optional[Set[str]] = None,
    stored_hash: Optional[str] = None,
) -> str:
    """Compute a deterministic SHA-256 hash for a transaction.

    The hash is computed from stable, immutable fields only.
    Fields are sorted alphabetically to ensure deterministic serialization.

    Args:
        transaction: Transaction dictionary to hash.
        fields: Set of field names to include in hash computation.
                Defaults to DEFAULT_HASH_FIELDS.
        stored_hash: Optional pre-computed hash for verification.

    Returns:
        SHA-256 hex digest of the transaction.
    """
    if fields is None:
        fields = DEFAULT_HASH_FIELDS

    # Filter to only include requested fields that exist in transaction
    hash_data: Dict[str, Any] = {}
    for field in fields:
        if field in transaction:
            hash_data[field] = transaction[field]

    # Sort keys for deterministic serialization
    serialized = json.dumps(hash_data, sort_keys=True, default=str)
    hash_value = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # If there's a stored hash, verify it matches
    if stored_hash is not None and stored_hash != hash_value:
        logger.warning(
            "Hash mismatch detected: computed=%s, stored=%s",
            hash_value,
            stored_hash,
        )

    return hash_value


def verify_transaction_hash(
    transaction: Dict[str, Any],
    expected_hash: str,
    fields: Optional[Set[str]] = None,
) -> bool:
    """Verify that a transaction matches an expected hash.

    Args:
        transaction: Transaction dictionary to verify.
        expected_hash: Expected SHA-256 hash value.
        fields: Set of field names used in hash computation.

    Returns:
        True if the transaction hash matches the expected hash.
    """
    computed_hash = compute_transaction_hash(transaction, fields=fields)
    return computed_hash == expected_hash


def hash_batch(
    transactions: list[Dict[str, Any]],
    fields: Optional[Set[str]] = None,
) -> list[str]:
    """Compute hashes for a batch of transactions.

    Args:
        transactions: List of transaction dictionaries.
        fields: Set of field names to include in hash computation.

    Returns:
        List of SHA-256 hex digests in the same order as input.
    """
    return [compute_transaction_hash(tx, fields=fields) for tx in transactions]
