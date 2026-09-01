"""Stellar network as a first-class ingestion concept (issue #726).

Before this, "which network" was a URL in ``StreamConfig.horizon_url`` and
nothing else. Nothing recorded which network a row came from, and nothing
stopped a process configured for testnet writing into the store holding
mainnet data. That mistake is quiet and expensive: testnet and mainnet share
address and hash *formats*, so the rows look plausible, and the damage is only
visible later as accounts that do not exist and balances that do not
reconcile — with no column to tell the two apart when cleaning up.

This module makes the network explicit:

* :class:`StellarNetwork` — pubnet, testnet, futurenet.
* Per-network Horizon and Soroban endpoints, and the network passphrase, which
  is the authoritative identifier: it is what transaction signatures commit
  to, so two deployments agreeing on it are provably on the same network.
* A separate schema (or table prefix) per network, so the stores cannot
  overlap by construction.
* :func:`guard_write` — an explicit check a writer calls before persisting.

The guard is a runtime check rather than a type, because the two things being
compared are only known at runtime: the network the process is configured for,
and the network a store was created for.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("astroml.ingestion.network")

__all__ = [
    "CrossNetworkWriteError",
    "NetworkProfile",
    "StellarNetwork",
    "guard_write",
    "profile_for",
    "resolve_network",
]


class StellarNetwork(str, Enum):
    """A Stellar network.

    ``str`` subclass so a value serialises as its name in JSON, config files
    and database columns without a custom encoder.
    """

    PUBNET = "pubnet"
    TESTNET = "testnet"
    FUTURENET = "futurenet"

    @property
    def is_production(self) -> bool:
        """Whether this network carries real value."""
        return self is StellarNetwork.PUBNET

    def __str__(self) -> str:
        return self.value


# Passphrases are fixed by the network itself and are the only identifier that
# cannot be pointed somewhere else by configuration.
_PASSPHRASES = {
    StellarNetwork.PUBNET: "Public Global Stellar Network ; September 2015",
    StellarNetwork.TESTNET: "Test SDF Network ; September 2015",
    StellarNetwork.FUTURENET: "Test SDF Future Network ; October 2022",
}

_HORIZON_URLS = {
    StellarNetwork.PUBNET: "https://horizon.stellar.org",
    StellarNetwork.TESTNET: "https://horizon-testnet.stellar.org",
    StellarNetwork.FUTURENET: "https://horizon-futurenet.stellar.org",
}

_SOROBAN_RPC_URLS = {
    StellarNetwork.PUBNET: "https://soroban.stellar.org",
    StellarNetwork.TESTNET: "https://soroban-testnet.stellar.org",
    StellarNetwork.FUTURENET: "https://rpc-futurenet.stellar.org",
}

# Aliases people actually type. "mainnet" and "public" are the names most
# operators use for pubnet, and rejecting them would be pedantry that costs a
# deployment.
_ALIASES = {
    "pubnet": StellarNetwork.PUBNET,
    "public": StellarNetwork.PUBNET,
    "mainnet": StellarNetwork.PUBNET,
    "main": StellarNetwork.PUBNET,
    "testnet": StellarNetwork.TESTNET,
    "test": StellarNetwork.TESTNET,
    "futurenet": StellarNetwork.FUTURENET,
    "future": StellarNetwork.FUTURENET,
}


class CrossNetworkWriteError(RuntimeError):
    """Raised when a write would mix data from two networks."""

    def __init__(self, writer: StellarNetwork, store: StellarNetwork, context: str = "") -> None:
        detail = f" while {context}" if context else ""
        super().__init__(
            f"refusing to write {writer} data into a {store} store{detail}. "
            "Testnet and mainnet records are indistinguishable once mixed — "
            "point the writer at the matching store, or use the per-network "
            "schema from NetworkProfile.schema."
        )
        self.writer = writer
        self.store = store


@dataclass(frozen=True)
class NetworkProfile:
    """Everything that varies between networks, resolved in one place.

    Attributes:
        network: Which network this profile describes.
        horizon_url: Horizon endpoint.
        soroban_rpc_url: Soroban RPC endpoint.
        passphrase: Network passphrase — the authoritative identifier.
        schema: Database schema (or table prefix) for this network's data.
    """

    network: StellarNetwork
    horizon_url: str
    soroban_rpc_url: str
    passphrase: str
    schema: str

    @property
    def is_production(self) -> bool:
        return self.network.is_production

    def matches(self, other: NetworkProfile | StellarNetwork | str) -> bool:
        """Whether ``other`` denotes the same network."""
        if isinstance(other, NetworkProfile):
            return other.network is self.network
        return resolve_network(other) is self.network

    def to_dict(self) -> dict[str, str]:
        """Serialisable form, for stamping onto a run or a state file."""
        return {
            "network": self.network.value,
            "horizon_url": self.horizon_url,
            "soroban_rpc_url": self.soroban_rpc_url,
            "passphrase": self.passphrase,
            "schema": self.schema,
        }


def resolve_network(value: StellarNetwork | str | None = None) -> StellarNetwork:
    """Resolve a network from a value, the environment, or the default.

    Order: the explicit argument, then ``ASTROML_STELLAR_NETWORK``, then
    testnet. Testnet is the default deliberately — an unconfigured process
    should read a network where a mistake is free, not the one holding real
    money.

    Raises:
        ValueError: If the name is not a network or a known alias.
    """
    raw = value if value is not None else os.environ.get("ASTROML_STELLAR_NETWORK")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return StellarNetwork.TESTNET

    if isinstance(raw, StellarNetwork):
        return raw

    key = str(raw).strip().lower()
    try:
        return _ALIASES[key]
    except KeyError:
        raise ValueError(
            f"unknown Stellar network {raw!r}; expected one of "
            f"{', '.join(sorted({network.value for network in StellarNetwork}))}"
        ) from None


def profile_for(
    value: StellarNetwork | str | None = None,
    *,
    horizon_url: str | None = None,
    soroban_rpc_url: str | None = None,
    schema: str | None = None,
) -> NetworkProfile:
    """Build the profile for a network, allowing endpoint overrides.

    Overrides exist for local development against a private network or a
    proxy. The passphrase is never overridable: it identifies the network, and
    a configurable identifier could not be used to detect the mix-up this
    module exists to prevent.
    """
    network = resolve_network(value)

    return NetworkProfile(
        network=network,
        horizon_url=(
            horizon_url or os.environ.get("ASTROML_HORIZON_URL") or _HORIZON_URLS[network]
        ),
        soroban_rpc_url=(
            soroban_rpc_url
            or os.environ.get("ASTROML_SOROBAN_RPC_URL")
            or _SOROBAN_RPC_URLS[network]
        ),
        passphrase=_PASSPHRASES[network],
        schema=schema or os.environ.get("ASTROML_DB_SCHEMA") or f"astroml_{network.value}",
    )


def guard_write(
    writer: NetworkProfile | StellarNetwork | str,
    store: NetworkProfile | StellarNetwork | str | None,
    *,
    context: str = "",
) -> None:
    """Refuse a write that would mix two networks.

    Call this before persisting ingested records, with the network the process
    is configured for and the network the destination store holds.

    A ``store`` of ``None`` means the destination has no recorded network —
    an empty database, or one written before this existed. That is allowed:
    refusing would make the first write to a fresh store impossible. It is
    logged, because it is also what an unstamped legacy store looks like, and
    an operator seeing this line on a store they believe is stamped has found
    a real problem.

    Raises:
        CrossNetworkWriteError: If the two networks differ.
    """
    writer_network = (
        writer.network if isinstance(writer, NetworkProfile) else resolve_network(writer)
    )

    if store is None:
        logger.info(
            "Destination has no recorded network; treating it as new and stamping it %s",
            writer_network,
        )
        return

    store_network = store.network if isinstance(store, NetworkProfile) else resolve_network(store)

    if writer_network is not store_network:
        raise CrossNetworkWriteError(writer_network, store_network, context)
