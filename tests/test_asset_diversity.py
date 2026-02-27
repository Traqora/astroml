"""Tests for astroml.features.asset_diversity."""
import numpy as np
import pandas as pd

from astroml.features.asset_diversity import compute_asset_diversity


def test_compute_asset_diversity_empty():
    counts = pd.Series(dtype=int)
    result = compute_asset_diversity(counts)
    
    assert result["unique_asset_count"] == 0
    assert result["asset_entropy"] == 0.0


def test_compute_asset_diversity_single_asset():
    counts = pd.Series({"XLM": 100})
    result = compute_asset_diversity(counts)
    
    assert result["unique_asset_count"] == 1
    assert result["asset_entropy"] == 0.0


def test_compute_asset_diversity_multiple_assets_uniform():
    # 4 assets with equal frequency -> entropy should be log2(4) = 2.0
    counts = pd.Series({"A": 10, "B": 10, "C": 10, "D": 10})
    result = compute_asset_diversity(counts)
    
    assert result["unique_asset_count"] == 4
    assert np.isclose(result["asset_entropy"], 2.0)


def test_compute_asset_diversity_multiple_assets_skewed():
    counts = pd.Series({"XLM": 50, "USDC": 50})
    result1 = compute_asset_diversity(counts)
    
    # Max entropy for 2 classes is log2(2) = 1.0
    assert result1["unique_asset_count"] == 2
    assert np.isclose(result1["asset_entropy"], 1.0)
    
    counts_skewed = pd.Series({"XLM": 99, "USDC": 1})
    result2 = compute_asset_diversity(counts_skewed)
    
    # Skewed distribution should have lower entropy
    assert result2["unique_asset_count"] == 2
    assert result2["asset_entropy"] < 1.0


def test_compute_asset_diversity_filters_zeros():
    counts = pd.Series({"XLM": 10, "USDC": 0, "BTC": 10})
    result = compute_asset_diversity(counts)
    
    assert result["unique_asset_count"] == 2
    assert np.isclose(result["asset_entropy"], 1.0)
