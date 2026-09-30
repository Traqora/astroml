"""Factory_boy factories for generating test data.

Usage:
    from tests.factories import LedgerFactory, TransactionFactory

    ledger = LedgerFactory(sequence=100)
    tx = TransactionFactory(ledger_sequence=ledger.sequence, successful=True)

Resolves #517 / #518.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import factory


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LedgerFactory(factory.Factory):
    """Generates synthetic Stellar ledger data."""

    class Meta:
        model = dict

    sequence = factory.Sequence(lambda n: n + 1)
    hash = factory.LazyFunction(lambda: factory.Faker("sha256").generate())
    prev_hash = factory.LazyFunction(lambda: factory.Faker("sha256").generate())
    closed_at = factory.LazyFunction(_utcnow)
    successful_transaction_count = factory.Faker("random_int", min=0, max=50)
    failed_transaction_count = factory.Faker("random_int", min=0, max=5)
    operation_count = factory.Faker("random_int", min=0, max=200)
    total_coins = factory.Faker("pyfloat", min_value=1e6, max_value=1e12)
    fee_pool = factory.Faker("pyfloat", min_value=100, max_value=10000)
    base_fee_in_stroops = 100
    protocol_version = 21


class TransactionFactory(factory.Factory):
    """Generates synthetic Stellar transactions."""

    class Meta:
        model = dict

    hash = factory.LazyFunction(lambda: factory.Faker("sha256").generate())
    ledger_sequence = factory.Sequence(lambda n: n + 1)
    source_account = factory.LazyFunction(
        lambda: "G" + factory.Faker("lexify", text="?" * 55).generate()
    )
    created_at = factory.LazyFunction(_utcnow)
    fee = factory.Faker("random_int", min=100, max=10000)
    operation_count = factory.Faker("random_int", min=1, max=50)
    successful = True
    memo_type = None
    memo = None


class GraphFactory(factory.Factory):
    """Builds a test graph with controlled structure."""

    class Meta:
        model = dict

    num_nodes = 50
    num_edges = 100
    seed = FIXED_SEED = 42

    @factory.lazy_attribute
    def edges(self):
        import random

        rng = random.Random(self.seed)
        return [
            (rng.randint(0, self.num_nodes - 1), rng.randint(0, self.num_nodes - 1))
            for _ in range(self.num_edges)
        ]


class FeatureDefinitionFactory(factory.Factory):
    """Creates feature metadata for testing."""

    class Meta:
        model = dict

    name = factory.Sequence(lambda n: f"feature_{n}")
    dtype = "float64"
    description = factory.Faker("sentence")
    nullable = False
    version = 1


# ─── Convenience aliases ────────────────────────────────────────────────────


def create_ledger(**kwargs: Any) -> dict:
    """Create a single ledger dict with overrides."""
    return LedgerFactory(**kwargs)


def create_transactions(count: int = 5, **kwargs: Any) -> list[dict]:
    """Create multiple transaction dicts."""
    return [TransactionFactory(**kwargs) for _ in range(count)]
