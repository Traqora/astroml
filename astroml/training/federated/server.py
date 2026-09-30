"""Federated learning server orchestrating client rounds, selection, and aggregation.

Coordinates decentralized training rounds, manages client registration,
handles secure aggregation protocols, and tracks global model convergence.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from astroml.training.federated.aggregator import (
    BaseAggregator,
    ClientUpdate,
    FedAvgAggregator,
)
from astroml.training.federated.client import FederatedClient
from astroml.training.federated.secure_aggregation import (
    MaskedUpdate,
    SecureAggregator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client Selection Strategies
# ---------------------------------------------------------------------------


class ClientSelectionStrategy(ABC):
    """Abstract base class for federated client selection policies."""

    @abstractmethod
    def select(
        self,
        registered_clients: dict[str, dict[str, Any]],
        count: int,
        round_id: int,
    ) -> list[str]:
        """Select a subset of client IDs for the next training round."""
        pass


class RandomClientSelector(ClientSelectionStrategy):
    """Uniformly samples clients at random."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def select(
        self,
        registered_clients: dict[str, dict[str, Any]],
        count: int,
        round_id: int,
    ) -> list[str]:
        client_ids = list(registered_clients.keys())
        if not client_ids:
            return []
        k = min(count, len(client_ids))
        return list(self.rng.choice(client_ids, size=k, replace=False))


class DataVolumeWeightedSelector(ClientSelectionStrategy):
    """Samples clients with probability proportional to their local sample count."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def select(
        self,
        registered_clients: dict[str, dict[str, Any]],
        count: int,
        round_id: int,
    ) -> list[str]:
        client_ids = list(registered_clients.keys())
        if not client_ids:
            return []

        counts = np.array([registered_clients[cid].get("sample_count", 1) for cid in client_ids], dtype=float)
        total = np.sum(counts)
        probs = (counts / total) if total > 0 else np.ones(len(client_ids)) / len(client_ids)

        k = min(count, len(client_ids))
        return list(self.rng.choice(client_ids, size=k, replace=False, p=probs))


class RoundRobinSelector(ClientSelectionStrategy):
    """Deterministically cycles through all registered clients round by round."""

    def select(
        self,
        registered_clients: dict[str, dict[str, Any]],
        count: int,
        round_id: int,
    ) -> list[str]:
        client_ids = sorted(registered_clients.keys())
        if not client_ids:
            return []
        n = len(client_ids)
        start_idx = (round_id * count) % n
        selected = []
        for i in range(min(count, n)):
            selected.append(client_ids[(start_idx + i) % n])
        return selected


# ---------------------------------------------------------------------------
# Federated Server
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    """Summary of a completed federated training round."""

    round_id: int
    participating_clients: list[str]
    client_count: int
    global_loss: float
    global_metrics: dict[str, float]
    aggregated_samples: int
    duration_seconds: float
    timestamp: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class FederatedServer:
    """Central server orchestrating federated learning."""

    def __init__(
        self,
        initial_weights: dict[str, np.ndarray],
        aggregator: BaseAggregator | None = None,
        selection_strategy: ClientSelectionStrategy | None = None,
        use_secure_aggregation: bool = False,
        min_clients: int = 1,
    ) -> None:
        self.global_weights: dict[str, np.ndarray] = {
            k: np.copy(v) for k, v in initial_weights.items()
        }
        self.aggregator = aggregator or FedAvgAggregator()
        self.selection_strategy = selection_strategy or RandomClientSelector()
        self.use_secure_aggregation = use_secure_aggregation
        self.min_clients = min_clients

        self.secagg = SecureAggregator() if use_secure_aggregation else None
        self._registered_clients: dict[str, dict[str, Any]] = {}
        self._pending_updates: list[ClientUpdate] = []
        self._pending_masked_updates: list[MaskedUpdate] = []
        self._round_history: list[RoundResult] = []
        self._current_round: int = 0

    @property
    def current_round(self) -> int:
        return self._current_round

    def register_client(
        self,
        client_id: str,
        sample_count: int = 100,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Register a client node with the server."""
        self._registered_clients[client_id] = {
            "client_id": client_id,
            "sample_count": sample_count,
            "metadata": metadata or {},
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Registered federated client: %s", client_id)
        return True

    def unregister_client(self, client_id: str) -> bool:
        """Unregister a client node."""
        if client_id in self._registered_clients:
            del self._registered_clients[client_id]
            logger.info("Unregistered federated client: %s", client_id)
            return True
        return False

    def list_clients(self) -> list[dict[str, Any]]:
        """List all currently registered clients."""
        return list(self._registered_clients.values())

    def select_clients(self, count: int | None = None) -> list[str]:
        """Select a subset of registered clients for the round."""
        target_count = count or len(self._registered_clients)
        return self.selection_strategy.select(
            registered_clients=self._registered_clients,
            count=target_count,
            round_id=self._current_round,
        )

    def distribute_global_weights(self) -> dict[str, np.ndarray]:
        """Return a copy of the current global model weights."""
        return {k: np.copy(v) for k, v in self.global_weights.items()}

    def submit_client_update(self, update: ClientUpdate) -> None:
        """Receive a local weight update from a client."""
        self._pending_updates.append(update)

    def submit_masked_update(self, update: MaskedUpdate) -> None:
        """Receive a masked weight update from a client for SecAgg."""
        self._pending_masked_updates.append(update)

    def aggregate_round_updates(
        self,
        participating_clients: list[str],
    ) -> dict[str, np.ndarray]:
        """Aggregate submitted updates and update the global model."""
        if self.use_secure_aggregation:
            if not self._pending_masked_updates:
                raise ValueError("No masked updates received for SecAgg.")
            new_weights = self.secagg.aggregate_masked_updates(
                self._pending_masked_updates,
                participating_clients=participating_clients,
            )
            self._pending_masked_updates.clear()
        else:
            if not self._pending_updates:
                raise ValueError("No client updates received for aggregation.")
            new_weights = self.aggregator.aggregate(
                self._pending_updates,
                global_weights=self.global_weights,
            )
            self._pending_updates.clear()

        self.global_weights = new_weights
        return self.global_weights

    def run_round(
        self,
        client_pool: dict[str, FederatedClient] | None = None,
        clients_per_round: int | None = None,
        learning_rate: float = 0.01,
        local_epochs: int = 1,
        batch_size: int = 32,
    ) -> RoundResult:
        """Orchestrate a complete synchronous federated training round."""
        start_time = datetime.now(timezone.utc)
        self._current_round += 1
        round_id = self._current_round

        # 1. Select clients
        selected_ids = self.select_clients(count=clients_per_round)
        if len(selected_ids) < self.min_clients:
            raise ValueError(
                f"Selected {len(selected_ids)} clients, but minimum required is {self.min_clients}."
            )

        losses = []
        metrics_list = []
        total_samples = 0

        # 2. Train clients
        if client_pool:
            if self.use_secure_aggregation:
                self.secagg.setup_pairwise_secrets(selected_ids)

            for cid in selected_ids:
                if cid not in client_pool:
                    continue
                client = client_pool[cid]
                client.set_weights(self.global_weights)

                # Local training
                update = client.train_epoch(
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=local_epochs,
                    global_weights=self.global_weights,
                    round_id=round_id,
                )
                losses.append(update.loss)
                metrics_list.append(update.metrics)
                total_samples += update.sample_count

                if self.use_secure_aggregation:
                    masked = self.secagg.mask_client_weights(
                        cid, update.weights, selected_ids, round_id=round_id
                    )
                    self.submit_masked_update(
                        MaskedUpdate(
                            client_id=cid,
                            masked_weights={k: v.tolist() for k, v in masked.items()},
                            sample_count=update.sample_count,
                            round_id=round_id,
                        )
                    )
                else:
                    self.submit_client_update(update)

        # 3. Aggregate
        self.aggregate_round_updates(participating_clients=selected_ids)

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Merge metrics
        merged_metrics: dict[str, float] = {"loss": round(avg_loss, 4)}
        if metrics_list:
            for k in metrics_list[0].keys():
                vals = [m[k] for m in metrics_list if k in m]
                merged_metrics[k] = round(float(np.mean(vals)), 4)

        result = RoundResult(
            round_id=round_id,
            participating_clients=selected_ids,
            client_count=len(selected_ids),
            global_loss=round(avg_loss, 4),
            global_metrics=merged_metrics,
            aggregated_samples=total_samples,
            duration_seconds=round(duration, 3),
        )
        self._round_history.append(result)
        logger.info(
            "Completed FL Round %d: %d clients, loss=%.4f",
            round_id,
            len(selected_ids),
            avg_loss,
        )
        return result

    def evaluate_global_model(
        self,
        eval_data: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, float]:
        """Evaluate global model weights on evaluation dataset."""
        X, y = eval_data
        if "weight" not in self.global_weights:
            return {"accuracy": 0.0, "loss": 0.0}

        w = self.global_weights["weight"]
        b = self.global_weights.get("bias", np.zeros(1))

        logits = np.dot(X, w) + b
        preds = (logits >= 0.0).astype(int).flatten()
        y_flat = y.flatten()

        acc = float(np.mean(preds == y_flat))
        loss = float(np.mean((preds - y_flat) ** 2))

        return {"accuracy": round(acc, 4), "loss": round(loss, 4)}

    def get_training_history(self) -> list[dict[str, Any]]:
        """Return history of all completed rounds."""
        return [
            {
                "round_id": r.round_id,
                "client_count": r.client_count,
                "participating_clients": r.participating_clients,
                "global_loss": r.global_loss,
                "global_metrics": r.global_metrics,
                "aggregated_samples": r.aggregated_samples,
                "duration_seconds": r.duration_seconds,
                "timestamp": r.timestamp,
            }
            for r in self._round_history
        ]
