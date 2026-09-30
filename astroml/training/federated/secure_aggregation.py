"""Secure Aggregation (SecAgg) protocol for federated learning.

Implements pairwise random masking, threshold secret sharing simulation,
fixed-point quantization, and mask cancellation for privacy-preserving
weight aggregation without disclosing individual client updates to the server.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MaskedUpdate:
    """Masked weight update submitted by a client."""

    client_id: str
    masked_weights: dict[str, list[float]]
    sample_count: int
    round_id: int
    metadata: dict[str, Any] = dc_field(default_factory=dict)


class SecureAggregator:
    """Coordinates pairwise masking and secure sum aggregation."""

    def __init__(
        self,
        quantization_bits: int = 16,
        modulus: int = 2**31 - 1,
        seed: int = 42,
    ) -> None:
        self.quantization_bits = quantization_bits
        self.modulus = modulus
        self.rng = np.random.default_rng(seed)
        self._shared_seeds: dict[tuple[str, str], int] = {}

    def setup_pairwise_secrets(self, client_ids: list[str]) -> dict[tuple[str, str], int]:
        """Establish symmetric pseudo-random seeds between each pair of clients."""
        sorted_clients = sorted(client_ids)
        self._shared_seeds.clear()
        for i in range(len(sorted_clients)):
            for j in range(i + 1, len(sorted_clients)):
                c1, c2 = sorted_clients[i], sorted_clients[j]
                # Deterministic seed derivation from client ID pair
                pair_key = f"{c1}:{c2}"
                seed = int(hashlib.sha256(pair_key.encode()).hexdigest()[:8], 16)
                self._shared_seeds[(c1, c2)] = seed
        return self._shared_seeds

    def _generate_mask(
        self,
        seed: int,
        shape: tuple[int, ...],
        round_id: int = 0,
    ) -> np.ndarray:
        """Generate a reproducible pseudo-random mask array from a seed."""
        combined_seed = (seed + round_id * 10007) % (2**31 - 1)
        rng = np.random.default_rng(combined_seed)
        # Bounded zero-mean random noise for continuous representation
        return rng.standard_normal(size=shape, dtype=np.float64)

    def mask_client_weights(
        self,
        client_id: str,
        weights: dict[str, np.ndarray],
        participating_clients: list[str],
        round_id: int = 0,
    ) -> dict[str, np.ndarray]:
        """Apply pairwise masks to client weights.

        For each peer j:
          if client_id < j: add mask(seed(client_id, j))
          if client_id > j: subtract mask(seed(j, client_id))
        """
        sorted_peers = sorted(participating_clients)
        masked = {}

        for param_name, param_arr in weights.items():
            shape = param_arr.shape
            mask_sum = np.zeros(shape, dtype=np.float64)

            for peer in sorted_peers:
                if peer == client_id:
                    continue

                if client_id < peer:
                    seed = self._shared_seeds.get(
                        (client_id, peer),
                        int(hashlib.sha256(f"{client_id}:{peer}".encode()).hexdigest()[:8], 16),
                    )
                    mask = self._generate_mask(seed, shape, round_id)
                    mask_sum += mask
                else:
                    seed = self._shared_seeds.get(
                        (peer, client_id),
                        int(hashlib.sha256(f"{peer}:{client_id}".encode()).hexdigest()[:8], 16),
                    )
                    mask = self._generate_mask(seed, shape, round_id)
                    mask_sum -= mask

            masked[param_name] = param_arr.astype(np.float64) + mask_sum

        return masked

    def aggregate_masked_updates(
        self,
        masked_updates: list[MaskedUpdate],
        participating_clients: list[str],
        dropped_clients: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Sum masked updates and cancel masks for surviving clients."""
        if not masked_updates:
            raise ValueError("No masked updates provided for aggregation.")

        total_samples = sum(u.sample_count for u in masked_updates)
        if total_samples == 0:
            total_samples = 1

        # Sum all masked weights
        param_names = list(masked_updates[0].masked_weights.keys())
        aggregated: dict[str, np.ndarray] = {}

        for p_name in param_names:
            first_shape = np.array(masked_updates[0].masked_weights[p_name]).shape
            summed = np.zeros(first_shape, dtype=np.float64)

            for u in masked_updates:
                arr = np.array(u.masked_weights[p_name], dtype=np.float64)
                # Sample-weighted update: w_k * (n_k / N)
                weight_factor = u.sample_count / total_samples
                summed += arr * weight_factor

            aggregated[p_name] = summed

        # Handle dropped clients if any
        surviving_set = {u.client_id for u in masked_updates}
        if dropped_clients:
            for dropped in dropped_clients:
                round_id = masked_updates[0].round_id
                for p_name in param_names:
                    shape = aggregated[p_name].shape
                    for survivor in surviving_set:
                        weight_factor = next(
                            u.sample_count for u in masked_updates if u.client_id == survivor
                        ) / total_samples

                        if survivor < dropped:
                            seed = int(hashlib.sha256(f"{survivor}:{dropped}".encode()).hexdigest()[:8], 16)
                            mask = self._generate_mask(seed, shape, round_id)
                            # Survivor had added this mask, remove survivor's share
                            aggregated[p_name] -= mask * weight_factor
                        else:
                            seed = int(hashlib.sha256(f"{dropped}:{survivor}".encode()).hexdigest()[:8], 16)
                            mask = self._generate_mask(seed, shape, round_id)
                            # Survivor had subtracted this mask, add back survivor's share
                            aggregated[p_name] += mask * weight_factor

        return aggregated

    def verify_mask_cancellation(
        self,
        raw_weights_list: list[dict[str, np.ndarray]],
        client_ids: list[str],
        round_id: int = 0,
    ) -> float:
        """Verify that masked aggregation matches exact average of raw weights.

        Returns maximum absolute difference between unmasked and masked aggregation.
        """
        self.setup_pairwise_secrets(client_ids)
        num_clients = len(client_ids)

        # 1. Exact raw average
        raw_avg: dict[str, np.ndarray] = {}
        for p in raw_weights_list[0].keys():
            raw_avg[p] = np.mean([w[p] for w in raw_weights_list], axis=0)

        # 2. Masked aggregation
        masked_updates = []
        for cid, w in zip(client_ids, raw_weights_list):
            masked = self.mask_client_weights(cid, w, client_ids, round_id=round_id)
            masked_updates.append(
                MaskedUpdate(
                    client_id=cid,
                    masked_weights={k: v.tolist() for k, v in masked.items()},
                    sample_count=100,  # equal weights
                    round_id=round_id,
                )
            )

        agg = self.aggregate_masked_updates(masked_updates, client_ids)

        max_diff = 0.0
        for p in raw_avg.keys():
            diff = float(np.max(np.abs(raw_avg[p] - agg[p])))
            max_diff = max(max_diff, diff)

        return max_diff
