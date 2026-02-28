"""Temporal decay weighting for transaction features.

Applies exponential decay to weight recent transactions more heavily:
weight = exp(-lambda * time_since_transaction)
"""
import math
from typing import Optional, List, Dict, Any
import numpy as np


class TemporalDecayWeighter:
    """Apply exponential decay to transaction weights based on recency."""
    
    def __init__(self, lambda_param: float = 0.01):
        """Initialize temporal decay weighter.
        
        Args:
            lambda_param: Decay rate parameter. Higher values decay faster.
                         For daily data, 0.01 (~69 days half-life) to 0.1 (~7 days) typical.
        """
        if lambda_param < 0:
            raise ValueError("lambda_param must be non-negative")
        self.lambda_param = lambda_param
    
    def compute_decay_factor(self, time_delta: float) -> float:
        """Compute exponential decay factor.
        
        Args:
            time_delta: Time elapsed in same units as lambda_param
            
        Returns:
            Decay factor in (0, 1]
        """
        if time_delta < 0:
            return 1.0
        return math.exp(-self.lambda_param * time_delta)
    
    def weight_transactions(
        self,
        transactions: List[Dict[str, Any]],
        current_time: float,
        timestamp_key: str = "timestamp"
    ) -> List[float]:
        """Compute decayed weights for transactions.
        
        Args:
            transactions: List of transaction dicts with timestamp
            current_time: Reference time for decay calculation
            timestamp_key: Key in transaction dict containing timestamp
            
        Returns:
            List of decay weights corresponding to transactions
        """
        weights = []
        for txn in transactions:
            ts = txn.get(timestamp_key, current_time)
            time_delta = max(0, current_time - ts)
            weight = self.compute_decay_factor(time_delta)
            weights.append(weight)
        return weights
    
    def apply_decay_to_amount(
        self,
        amount: float,
        time_delta: float
    ) -> float:
        """Apply decay factor to transaction amount.
        
        Args:
            amount: Original transaction amount
            time_delta: Time elapsed since transaction
            
        Returns:
            Decayed amount
        """
        decay_factor = self.compute_decay_factor(time_delta)
        return amount * decay_factor
    
    def aggregate_with_decay(
        self,
        transactions: List[Dict[str, Any]],
        current_time: float,
        timestamp_key: str = "timestamp",
        amount_key: str = "amount",
        aggregation: str = "sum"
    ) -> float:
        """Aggregate transaction amounts with temporal decay applied.
        
        Args:
            transactions: List of transaction dicts
            current_time: Reference time for decay
            timestamp_key: Key for timestamp in transaction
            amount_key: Key for amount in transaction
            aggregation: Method - 'sum', 'mean', 'weighted_mean'
            
        Returns:
            Aggregated decayed amount
        """
        if not transactions:
            return 0.0
        
        decayed_amounts = []
        weights = []
        
        for txn in transactions:
            amount = float(txn.get(amount_key, 0.0))
            ts = float(txn.get(timestamp_key, current_time))
            time_delta = max(0, current_time - ts)
            decay_factor = self.compute_decay_factor(time_delta)
            decayed_amounts.append(amount * decay_factor)
            weights.append(decay_factor)
        
        if aggregation == "sum":
            return sum(decayed_amounts)
        elif aggregation == "mean":
            return sum(decayed_amounts) / len(decayed_amounts)
        elif aggregation == "weighted_mean":
            weight_sum = sum(weights)
            if weight_sum == 0:
                return 0.0
            return sum(decayed_amounts) / weight_sum
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")


def compute_decay_weights(
    transactions: List[Dict[str, Any]],
    current_time: float,
    lambda_param: float = 0.01,
    timestamp_key: str = "timestamp"
) -> np.ndarray:
    """Utility function to compute decay weights for transactions.
    
    Args:
        transactions: List of transaction dicts
        current_time: Reference time
        lambda_param: Decay rate
        timestamp_key: Key in transaction containing timestamp
        
    Returns:
        NumPy array of decay factors
    """
    weighter = TemporalDecayWeighter(lambda_param)
    weights = weighter.weight_transactions(transactions, current_time, timestamp_key)
    return np.array(weights)
