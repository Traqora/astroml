"""Traffic assignment and randomization for experiments."""

from typing import Dict, List, Tuple
import hashlib


class TrafficAssigner:
    """
    Assign traffic to variants with deterministic randomization.
    
    Uses consistent hashing to ensure users get same variant across requests.
    """

    def __init__(self, experiment_id: str):
        """Initialize assigner."""
        self.experiment_id = experiment_id

    def assign_user(
        self,
        user_id: str,
        variant_weights: Dict[str, float],
    ) -> str:
        """
        Assign user to variant using consistent hashing.

        Args:
            user_id: User identifier
            variant_weights: Dict of variant_name -> weight

        Returns:
            Assigned variant name
        """
        # Create consistent hash for user-experiment pair
        hash_input = f"{self.experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Normalize weights
        total_weight = sum(variant_weights.values())
        normalized_weights = {k: v / total_weight for k, v in variant_weights.items()}

        # Assign variant
        threshold = 0.0
        normalized_hash = (hash_value % 10000) / 10000

        for variant, weight in sorted(normalized_weights.items()):
            threshold += weight
            if normalized_hash < threshold:
                return variant

        # Fallback to first variant
        return list(variant_weights.keys())[0]

    def batch_assign(
        self,
        user_ids: List[str],
        variant_weights: Dict[str, float],
    ) -> Dict[str, List[str]]:
        """
        Assign multiple users to variants.

        Args:
            user_ids: List of user IDs
            variant_weights: Variant weights

        Returns:
            Dict mapping variant names to lists of assigned user IDs
        """
        assignments: Dict[str, List[str]] = {v: [] for v in variant_weights.keys()}

        for user_id in user_ids:
            variant = self.assign_user(user_id, variant_weights)
            assignments[variant].append(user_id)

        return assignments

    def verify_randomization(
        self,
        user_ids: List[str],
        variant_weights: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Verify that randomization matches expected weights.

        Args:
            user_ids: List of user IDs
            variant_weights: Expected variant weights

        Returns:
            Dict with actual vs expected distributions
        """
        assignments = self.batch_assign(user_ids, variant_weights)
        total = len(user_ids)

        results = {}
        for variant, users in assignments.items():
            actual_ratio = len(users) / total if total > 0 else 0
            expected_ratio = variant_weights.get(variant, 0)

            results[variant] = {
                "expected_ratio": expected_ratio,
                "actual_ratio": actual_ratio,
                "samples": len(users),
                "deviation": abs(actual_ratio - expected_ratio),
            }

        return results
