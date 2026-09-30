"""Shared fixtures for fairness tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def fair_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a dataset with no demographic parity bias."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    sensitive = rng.integers(0, 2, size=n)
    return y_true, y_pred, sensitive


@pytest.fixture
def biased_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a dataset with clear demographic parity bias."""
    n = 200
    y_true = np.array([1, 0] * (n // 2))
    y_pred = np.array([1, 0] * (n // 2))
    sensitive = np.array([0] * (n // 2) + [1] * (n // 2))
    return y_true, y_pred, sensitive


@pytest.fixture
def biased_dataset_unequal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a dataset where group 0 gets more positive predictions."""
    n = 100
    y_true = np.array([1] * 40 + [0] * 10 + [1] * 20 + [0] * 30)
    y_pred = np.array([1] * 45 + [0] * 5 + [1] * 25 + [0] * 25)
    sensitive = np.array([0] * 50 + [1] * 50)
    return y_true, y_pred, sensitive


@pytest.fixture
def multi_attribute_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a dataset with two protected attributes."""
    n = 100
    y_true = np.random.default_rng(42).integers(0, 2, size=n)
    y_pred = y_true.copy()
    sensitive = np.column_stack(
        [
            np.random.default_rng(42).integers(0, 2, size=n),
            np.random.default_rng(99).integers(0, 3, size=n),
        ]
    )
    return y_true, y_pred, sensitive


@pytest.fixture
def sample_weights_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dataset for testing reweighing."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 3))
    y = rng.integers(0, 2, size=n)
    sensitive = rng.integers(0, 2, size=n)
    return X, y, sensitive


@pytest.fixture
def simple_classifier():
    """A simple threshold-based classifier for testing."""

    class SimpleClassifier:
        def fit(self, X, y):
            self.threshold = float(np.mean(y))

        def predict(self, X):
            return np.full(X.shape[0], 0.5)

    return SimpleClassifier()
