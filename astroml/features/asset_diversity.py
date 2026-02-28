"""Compute asset diversity metrics for blockchain accounts.

This module provides utilities for calculating the unique asset count and the 
Shannon entropy of the asset interaction distribution for each account.
"""
from typing import Dict, Union

import numpy as np
import pandas as pd


def compute_asset_diversity(asset_counts: pd.Series) -> Dict[str, Union[int, float]]:
    """Compute unique asset count and Shannon entropy from an asset counts series.
    
    Args:
        asset_counts: Series where index is the asset name and value is the interaction count.
        
    Returns:
        Dictionary containing 'unique_asset_count' and 'asset_entropy'.
        
    Examples:
        >>> counts = pd.Series({"XLM": 10, "USDC": 5, "BTC": 5})
        >>> metrics = compute_asset_diversity(counts)
        >>> metrics["unique_asset_count"]
        3
        >>> metrics["asset_entropy"] # log2(20/10) * 0.5 + log2(20/5) * 0.25 * 2
        1.5
    """
    total_interactions = asset_counts.sum()
    unique_assets = len(asset_counts[asset_counts > 0])
    
    if total_interactions == 0 or unique_assets == 0:
        return {
            "unique_asset_count": 0,
            "asset_entropy": 0.0
        }
        
    # Calculate probabilities
    probabilities = asset_counts / total_interactions
    
    # Filter out 0 probabilities for log2 calculation
    probabilities = probabilities[probabilities > 0]
    
    # Calculate Shannon entropy: -sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return {
        "unique_asset_count": unique_assets,
        "asset_entropy": float(entropy)
    }
