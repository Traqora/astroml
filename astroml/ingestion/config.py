"""Configuration for Horizon streaming ingestion.

Settings are resolved from environment variables, then defaults.
All settings are exposed via the ``StreamConfig`` dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from astroml.ingestion.network import NetworkProfile, StellarNetwork, profile_for

HORIZON_TESTNET_URL = "https://horizon-testnet.stellar.org"
HORIZON_MAINNET_URL = "https://horizon.stellar.org"

DEFAULT_RECONNECT_BASE_SECONDS = 1.0
DEFAULT_RECONNECT_MAX_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 0  # 0 = unlimited


@dataclass(frozen=True)
class StreamConfig:
    """Immutable configuration for a streaming session.

    ``network`` is first-class (issue #726): it selects the default Horizon
    endpoint, the network passphrase, and the per-network database schema, and
    it is what :func:`~astroml.ingestion.network.guard_write` compares against
    a store before letting a write through. Set it with
    ``ASTROML_STELLAR_NETWORK``; ``ASTROML_HORIZON_URL`` still overrides the
    endpoint alone, for a proxy or a private network.
    """

    network: StellarNetwork = field(default_factory=lambda: profile_for().network)
    horizon_url: str = field(
        default_factory=lambda: os.environ.get("ASTROML_HORIZON_URL") or profile_for().horizon_url
    )
    stream_endpoint: str = field(
        default_factory=lambda: os.environ.get("ASTROML_STREAM_ENDPOINT", "/transactions")
    )
    cursor: str | None = field(default_factory=lambda: os.environ.get("ASTROML_STREAM_CURSOR"))
    reconnect_base_seconds: float = DEFAULT_RECONNECT_BASE_SECONDS
    reconnect_max_seconds: float = DEFAULT_RECONNECT_MAX_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    persist_chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("ASTROML_PERSIST_CHUNK_SIZE", "50"))
    )

    @property
    def profile(self) -> NetworkProfile:
        """Endpoints, passphrase and schema for this config's network.

        The Horizon URL is carried across so an explicit ``horizon_url`` — a
        proxy, or a local instance — stays in effect while the passphrase and
        schema continue to come from the network itself.
        """
        return profile_for(self.network, horizon_url=self.horizon_url)
