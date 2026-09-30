"""Shared fixtures for tests/validation/ test suite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture
def valid_transactions() -> list[dict[str, Any]]:
    """Three structurally complete, unique transactions."""
    return [
        {
            "id": "tx_001",
            "source_account": "GABC",
            "amount": 100.0,
            "asset_code": "XLM",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "id": "tx_002",
            "source_account": "GDEF",
            "amount": 50.5,
            "asset_code": "USDC",
            "timestamp": "2024-01-01T00:01:00Z",
        },
        {
            "id": "tx_003",
            "source_account": "GHIJ",
            "amount": 200.0,
            "asset_code": "XLM",
            "timestamp": "2024-01-01T00:02:00Z",
        },
    ]


@pytest.fixture
def transactions_with_missing_fields() -> list[dict[str, Any]]:
    """Three transactions each missing a different required field."""
    return [
        {"id": "tx_004", "amount": 100.0},  # missing source_account
        {"source_account": "GKLM", "amount": 50.0},  # missing id
        {"id": "tx_006", "source_account": None, "amount": 75.0},  # null required field
    ]


@pytest.fixture
def duplicate_transactions() -> list[dict[str, Any]]:
    """Two unique transactions plus one exact duplicate of the first."""
    base = {
        "id": "tx_007",
        "source_account": "GNOP",
        "amount": 100.0,
        "asset_code": "XLM",
        "timestamp": "2024-01-01T00:03:00Z",
    }
    return [
        base,
        {**base},  # exact duplicate
        {
            "id": "tx_008",
            "source_account": "GQRS",
            "amount": 200.0,
            "asset_code": "XLM",
            "timestamp": "2024-01-01T00:04:00Z",
        },
    ]


@pytest.fixture
def fraud_scores():
    """Well-separated fraud prediction arrays for calibration tests."""
    np.random.seed(42)
    y_true = np.array([0] * 400 + [1] * 100)
    y_prob = np.concatenate(
        [
            np.random.beta(2, 8, 400),
            np.random.beta(8, 2, 100),
        ]
    )
    y_prob = np.clip(y_prob, 0.01, 0.99)
    return y_true, y_prob
