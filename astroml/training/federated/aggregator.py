"""Aggregation algorithms for federated model updates.

Implements FedAvg, FedProx, Trimmed Mean, Coordinate-wise Median, and Krum.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AggregationAlgorithm(str, Enum):
    """Supported federated aggregation algorithms."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    KRUM = "krum"


@dataclass
class ClientUpdate:
    """Model update submitted by a participating federated client."""

    client_id: str
    weights: dict[str, np.ndarray]
    sample_count: int
    loss: float = 0.0
    metrics: dict[str, float] = dc_field(default_factory=dict)
    round_id: int = 0


class BaseAggregator(ABC):
    """Abstract base class for federated weight aggregators."""

    @abstractmethod
    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Aggregate a collection of client updates into new global weights."""
        pass


class FedAvgAggregator(BaseAggregator):
    """Federated Averaging (FedAvg) aggregator.

    Computes weighted average of client parameters proportional to local sample counts.
    """

    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        if not updates:
            raise ValueError("No client updates provided for FedAvg aggregation.")

        total_samples = sum(u.sample_count for u in updates)
        if total_samples <= 0:
            total_samples = len(updates)

        param_names = list(updates[0].weights.keys())
        new_weights: dict[str, np.ndarray] = {}

        for p_name in param_names:
            first_shape = updates[0].weights[p_name].shape
            weighted_sum = np.zeros(first_shape, dtype=np.float64)

            for u in updates:
                weight_factor = u.sample_count / total_samples
                weighted_sum += u.weights[p_name].astype(np.float64) * weight_factor

            new_weights[p_name] = weighted_sum.astype(updates[0].weights[p_name].dtype)

        return new_weights


class FedProxAggregator(BaseAggregator):
    """FedProx aggregator with proximal regularization parameter mu.

    Handles non-IID data distribution and straggler tolerance by adjusting
    aggregation weights with proximal penalty consideration.
    """

    def __init__(self, mu: float = 0.01, learning_rate: float = 1.0) -> None:
        self.mu = mu
        self.learning_rate = learning_rate

    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        if not updates:
            raise ValueError("No client updates provided for FedProx aggregation.")

        total_samples = sum(u.sample_count for u in updates)
        if total_samples <= 0:
            total_samples = len(updates)

        param_names = list(updates[0].weights.keys())
        new_weights: dict[str, np.ndarray] = {}

        for p_name in param_names:
            first_shape = updates[0].weights[p_name].shape
            weighted_sum = np.zeros(first_shape, dtype=np.float64)

            for u in updates:
                weight_factor = u.sample_count / total_samples
                w_arr = u.weights[p_name].astype(np.float64)

                # Proximal drift adjustment if global_weights provided
                if global_weights and p_name in global_weights:
                    g_arr = global_weights[p_name].astype(np.float64)
                    # FedProx proximal pull: w_adj = w - mu * (w - g)
                    w_arr = w_arr - self.mu * (w_arr - g_arr)

                weighted_sum += w_arr * weight_factor

            if global_weights and p_name in global_weights:
                g_arr = global_weights[p_name].astype(np.float64)
                # Apply server learning rate update step
                final_w = g_arr + self.learning_rate * (weighted_sum - g_arr)
            else:
                final_w = weighted_sum

            new_weights[p_name] = final_w.astype(updates[0].weights[p_name].dtype)

        return new_weights


class TrimmedMeanAggregator(BaseAggregator):
    """Coordinate-wise Trimmed Mean aggregator.

    Trims beta proportion of smallest and largest updates coordinate-wise
    to provide Byzantine robustness.
    """

    def __init__(self, beta: float = 0.1) -> None:
        if not 0.0 <= beta < 0.5:
            raise ValueError(f"Trim factor beta must be in [0.0, 0.5), got {beta}")
        self.beta = beta

    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        if not updates:
            raise ValueError("No client updates provided for TrimmedMean aggregation.")

        n_clients = len(updates)
        k_trim = int(n_clients * self.beta)

        param_names = list(updates[0].weights.keys())
        new_weights: dict[str, np.ndarray] = {}

        for p_name in param_names:
            stacked = np.stack([u.weights[p_name] for u in updates], axis=0)  # (N, ...)

            if k_trim > 0 and n_clients > 2 * k_trim:
                sorted_stacked = np.sort(stacked, axis=0)
                trimmed = sorted_stacked[k_trim : n_clients - k_trim]
                avg = np.mean(trimmed, axis=0)
            else:
                avg = np.mean(stacked, axis=0)

            new_weights[p_name] = avg.astype(updates[0].weights[p_name].dtype)

        return new_weights


class MedianAggregator(BaseAggregator):
    """Coordinate-wise Median aggregator for robust federated learning."""

    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        if not updates:
            raise ValueError("No client updates provided for Median aggregation.")

        param_names = list(updates[0].weights.keys())
        new_weights: dict[str, np.ndarray] = {}

        for p_name in param_names:
            stacked = np.stack([u.weights[p_name] for u in updates], axis=0)
            med = np.median(stacked, axis=0)
            new_weights[p_name] = med.astype(updates[0].weights[p_name].dtype)

        return new_weights


class KrumAggregator(BaseAggregator):
    """Multi-Krum aggregator for Byzantine fault tolerance."""

    def __init__(self, num_byzantine: int = 0, m_selected: int = 1) -> None:
        self.num_byzantine = num_byzantine
        self.m_selected = m_selected

    def _flatten_weights(self, weights: dict[str, np.ndarray]) -> np.ndarray:
        flat_arrays = [w.flatten() for w in weights.values()]
        return np.concatenate(flat_arrays)

    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        if not updates:
            raise ValueError("No client updates provided for Krum aggregation.")

        n = len(updates)
        f = self.num_byzantine
        # Krum requirement: n >= 2f + 3, or fallback to closest
        num_closest = max(1, n - f - 2)

        flat_vectors = [self._flatten_weights(u.weights) for u in updates]
        dist_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.sum((flat_vectors[i] - flat_vectors[j]) ** 2))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        scores = []
        for i in range(n):
            sorted_dists = np.sort(dist_matrix[i])
            score = float(np.sum(sorted_dists[1 : num_closest + 1]))
            scores.append(score)

        m = min(self.m_selected, n)
        selected_indices = np.argsort(scores)[:m]

        selected_updates = [updates[idx] for idx in selected_indices]
        fedavg = FedAvgAggregator()
        return fedavg.aggregate(selected_updates, global_weights=global_weights)


class AggregatorFactory:
    """Factory to instantiate aggregators by algorithm name."""

    @staticmethod
    def create(algorithm: AggregationAlgorithm | str = AggregationAlgorithm.FEDAVG, **kwargs: Any) -> BaseAggregator:
        algo_str = algorithm.value if isinstance(algorithm, AggregationAlgorithm) else str(algorithm).lower()

        if algo_str == "fedavg":
            return FedAvgAggregator()
        elif algo_str == "fedprox":
            return FedProxAggregator(
                mu=kwargs.get("mu", 0.01),
                learning_rate=kwargs.get("learning_rate", 1.0),
            )
        elif algo_str == "trimmed_mean":
            return TrimmedMeanAggregator(beta=kwargs.get("beta", 0.1))
        elif algo_str == "median":
            return MedianAggregator()
        elif algo_str == "krum":
            return KrumAggregator(
                num_byzantine=kwargs.get("num_byzantine", 0),
                m_selected=kwargs.get("m_selected", 1),
            )
        else:
            raise ValueError(f"Unknown aggregation algorithm: '{algo_str}'")
