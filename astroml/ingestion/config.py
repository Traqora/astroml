"""Configuration for Horizon streaming ingestion.

Settings are resolved from environment variables, then defaults.
All settings are exposed via the ``StreamConfig`` dataclass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

HORIZON_TESTNET_URL = "https://horizon-testnet.stellar.org"
HORIZON_MAINNET_URL = "https://horizon.stellar.org"

DEFAULT_RECONNECT_BASE_SECONDS = 1.0
DEFAULT_RECONNECT_MAX_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 0  # 0 = unlimited


@dataclass(frozen=True)
class StreamConfig:
    """Immutable configuration for a streaming session."""

    horizon_url: str = field(
        default_factory=lambda: os.environ.get(
            "ASTROML_HORIZON_URL", HORIZON_TESTNET_URL
        )
    )
    stream_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "ASTROML_STREAM_ENDPOINT", "/transactions"
        )
    )
    cursor: str | None = field(
        default_factory=lambda: os.environ.get("ASTROML_STREAM_CURSOR")
    )
    reconnect_base_seconds: float = DEFAULT_RECONNECT_BASE_SECONDS
    reconnect_max_seconds: float = DEFAULT_RECONNECT_MAX_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
