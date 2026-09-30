"""Federated learning client with local model training and Differential Privacy.

Supports local gradient updates, FedProx proximal regularization, L2 norm
clipping, and Gaussian differential privacy noise injection.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from astroml.training.federated.aggregator import ClientUpdate

logger = logging.getLogger(__name__)


@dataclass
class DPConfig:
    """Differential privacy configuration for client updates."""

    enabled: bool = False
    clip_norm: float = 1.0
    noise_scale: float = 0.01  # Gaussian std dev
    target_epsilon: float = 1.0
    target_delta: float = 1e-5


class FederatedClient:
    """Decentralized federated client performing local training."""

    def __init__(
        self,
        client_id: str,
        initial_weights: dict[str, np.ndarray] | None = None,
        local_data: tuple[np.ndarray, np.ndarray] | None = None,
        dp_config: DPConfig | None = None,
        mu_prox: float = 0.0,
        random_seed: int = 42,
    ) -> None:
        self.client_id = client_id
        self.weights: dict[str, np.ndarray] = initial_weights or {}
        self.local_data = local_data
        self.dp_config = dp_config or DPConfig()
        self.mu_prox = mu_prox
        self.rng = np.random.default_rng(random_seed)

        self._privacy_spent_eps: float = 0.0
        self._privacy_spent_delta: float = 0.0
        self._total_rounds_trained: int = 0

    @property
    def sample_count(self) -> int:
        """Return number of local samples available."""
        if self.local_data is not None and len(self.local_data) > 0:
            return len(self.local_data[0])
        return 0

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Update local model weights with global weights from server."""
        self.weights = {k: np.copy(v) for k, v in weights.items()}

    def get_weights(self) -> dict[str, np.ndarray]:
        """Get copy of current local model weights."""
        return {k: np.copy(v) for k, v in self.weights.items()}

    def _apply_differential_privacy(
        self,
        weight_deltas: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Apply L2 norm clipping and Gaussian noise to parameter updates."""
        if not self.dp_config.enabled:
            return weight_deltas

        # 1. Compute total L2 norm across all parameters
        total_sq_norm = sum(float(np.sum(w**2)) for w in weight_deltas.values())
        total_norm = math.sqrt(total_sq_norm)

        # 2. Clip updates to clip_norm threshold
        clip_factor = min(1.0, self.dp_config.clip_norm / (total_norm + 1e-8))
        clipped_deltas: dict[str, np.ndarray] = {}

        for p_name, delta in weight_deltas.items():
            clipped = delta * clip_factor

            # 3. Add calibrated Gaussian noise
            noise = self.rng.normal(
                loc=0.0,
                scale=self.dp_config.noise_scale * self.dp_config.clip_norm,
                size=delta.shape,
            )
            clipped_deltas[p_name] = clipped + noise

        # 4. Track privacy budget (simple moments accounting approximation)
        self._privacy_spent_eps += self.dp_config.noise_scale * math.sqrt(2 * math.log(1.25 / self.dp_config.target_delta))
        self._privacy_spent_delta += self.dp_config.target_delta

        return clipped_deltas

    def train_epoch(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        epochs: int = 1,
        global_weights: dict[str, np.ndarray] | None = None,
        round_id: int = 0,
    ) -> ClientUpdate:
        """Perform local training using mini-batch gradient descent / linear model solver."""
        if self.local_data is None or len(self.local_data[0]) == 0:
            return ClientUpdate(
                client_id=self.client_id,
                weights=self.get_weights(),
                sample_count=0,
                loss=0.0,
                round_id=round_id,
            )

        X, y = self.local_data
        num_samples = len(X)
        ref_global = global_weights or self.weights

        # Clone starting weights
        current_w = {k: np.copy(v) for k, v in self.weights.items()}

        # Simple gradient descent for standard linear/logistic model parameters 'weight' and 'bias'
        # or generic vector parameters
        epoch_losses: list[float] = []

        for _ in range(epochs):
            indices = self.rng.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, num_samples, batch_size):
                xb = X_shuffled[start_idx : start_idx + batch_size]
                yb = y_shuffled[start_idx : start_idx + batch_size]

                if "weight" in current_w:
                    w = current_w["weight"]
                    b = current_w.get("bias", np.zeros(1))

                    # Forward pass
                    logits = np.dot(xb, w) + b
                    if yb.ndim == 1:
                        yb_col = yb.reshape(-1, 1)
                    else:
                        yb_col = yb

                    # Binary cross-entropy / MSE loss
                    preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
                    err = preds - yb_col
                    batch_loss = float(np.mean(err**2))
                    epoch_losses.append(batch_loss)

                    # Gradients
                    gw = np.dot(xb.T, err) / len(xb)
                    gb = np.mean(err, axis=0)

                    # FedProx proximal gradient adjustment
                    if self.mu_prox > 0 and "weight" in ref_global:
                        gw += self.mu_prox * (w - ref_global["weight"])
                    if self.mu_prox > 0 and "bias" in ref_global:
                        gb += self.mu_prox * (b - ref_global["bias"])

                    # Update parameters
                    current_w["weight"] -= learning_rate * gw
                    if "bias" in current_w:
                        current_w["bias"] -= learning_rate * gb
                else:
                    # Fallback for generic weights dict
                    for p_name in current_w:
                        current_w[p_name] -= learning_rate * 0.01 * current_w[p_name]
                    epoch_losses.append(0.1)

        # Compute delta from starting weights
        deltas = {k: current_w[k] - self.weights[k] for k in current_w}

        # Apply Differential Privacy
        privatized_deltas = self._apply_differential_privacy(deltas)
        final_weights = {k: self.weights[k] + privatized_deltas[k] for k in self.weights}

        self.weights = final_weights
        self._total_rounds_trained += 1
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Compute training accuracy
        metrics = {"loss": avg_loss}
        if "weight" in self.weights:
            eval_metrics = self.evaluate()
            metrics.update(eval_metrics)

        return ClientUpdate(
            client_id=self.client_id,
            weights=self.get_weights(),
            sample_count=num_samples,
            loss=avg_loss,
            metrics=metrics,
            round_id=round_id,
        )

    def evaluate(self, eval_data: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, float]:
        """Evaluate local model performance on data."""
        data = eval_data or self.local_data
        if data is None or len(data[0]) == 0 or "weight" not in self.weights:
            return {"accuracy": 0.0, "loss": 0.0}

        X, y = data
        w = self.weights["weight"]
        b = self.weights.get("bias", np.zeros(1))

        logits = np.dot(X, w) + b
        preds = (logits >= 0.0).astype(int).flatten()
        y_flat = y.flatten()

        accuracy = float(np.mean(preds == y_flat))
        loss = float(np.mean((preds - y_flat) ** 2))

        return {"accuracy": round(accuracy, 4), "loss": round(loss, 4)}

    def get_privacy_spent(self) -> tuple[float, float]:
        """Return total privacy budget spent (epsilon, delta)."""
        return self._privacy_spent_eps, self._privacy_spent_delta
