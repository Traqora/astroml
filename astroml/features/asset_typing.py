"""Multi-asset edge typing for the Stellar transaction graph.

Classifies asset strings (as produced by the normalizer) into three
canonical edge types used during graph construction:

- XLM        — native Stellar asset
- STABLECOIN — known fiat-pegged assets (USDC, USDT, EURC, …)
- CUSTOM     — any other issued asset
"""
from __future__ import annotations

from enum import IntEnum

# Known stablecoin asset codes on Stellar (code only, issuer-agnostic).
# Extend this set as new stablecoins are listed.
_STABLECOIN_CODES: frozenset[str] = frozenset({
    "USDC", "USDT", "EURC", "EURT", "BRLT", "NGNT", "IDRT",
    "ARST", "MXNT", "NGNC",
})


class AssetType(IntEnum):
    """Canonical edge type for a Stellar asset."""
    XLM = 0
    STABLECOIN = 1
    CUSTOM = 2


def classify_asset(asset: str) -> AssetType:
    """Return the :class:`AssetType` for a normalised asset string.

    Args:
        asset: Asset string in the form ``'XLM'``, ``'USDC:G...'``, or
               ``'CODE:ISSUER'`` as produced by the ingestion normalizer.

    Returns:
        :class:`AssetType` enum value.
    """
    if asset == "XLM":
        return AssetType.XLM

    code = asset.split(":")[0].upper()
    if code in _STABLECOIN_CODES:
        return AssetType.STABLECOIN

    return AssetType.CUSTOM
