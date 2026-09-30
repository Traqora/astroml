"""Testnet vs mainnet ingestion configuration (issue #726).

"Which network" used to be a URL and nothing else: no record of which network
a row came from, and nothing stopping a testnet-configured process writing
into the mainnet store. The two are indistinguishable once mixed.
"""

from __future__ import annotations

import pytest

from astroml.ingestion.config import StreamConfig
from astroml.ingestion.network import (
    CrossNetworkWriteError,
    NetworkProfile,
    StellarNetwork,
    guard_write,
    profile_for,
    resolve_network,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Every test starts from an unconfigured environment."""
    for name in (
        "ASTROML_STELLAR_NETWORK",
        "ASTROML_HORIZON_URL",
        "ASTROML_SOROBAN_RPC_URL",
        "ASTROML_DB_SCHEMA",
    ):
        monkeypatch.delenv(name, raising=False)


class TestResolveNetwork:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("pubnet", StellarNetwork.PUBNET),
            ("mainnet", StellarNetwork.PUBNET),
            ("public", StellarNetwork.PUBNET),
            ("testnet", StellarNetwork.TESTNET),
            ("test", StellarNetwork.TESTNET),
            ("futurenet", StellarNetwork.FUTURENET),
        ],
    )
    def test_accepts_the_names_operators_actually_use(self, value, expected):
        assert resolve_network(value) is expected

    def test_is_case_and_whitespace_insensitive(self):
        assert resolve_network("  MainNet  ") is StellarNetwork.PUBNET

    def test_passes_an_enum_through(self):
        assert resolve_network(StellarNetwork.FUTURENET) is StellarNetwork.FUTURENET

    def test_reads_the_environment_when_given_nothing(self, monkeypatch):
        monkeypatch.setenv("ASTROML_STELLAR_NETWORK", "pubnet")

        assert resolve_network() is StellarNetwork.PUBNET

    def test_an_explicit_value_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("ASTROML_STELLAR_NETWORK", "pubnet")

        assert resolve_network("testnet") is StellarNetwork.TESTNET

    def test_defaults_to_testnet(self):
        """An unconfigured process should read where mistakes are free."""
        assert resolve_network() is StellarNetwork.TESTNET

    def test_an_empty_environment_value_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("ASTROML_STELLAR_NETWORK", "   ")

        assert resolve_network() is StellarNetwork.TESTNET

    def test_an_unknown_name_is_rejected_with_the_alternatives(self):
        with pytest.raises(ValueError) as excinfo:
            resolve_network("mainnett")

        assert "testnet" in str(excinfo.value)

    def test_the_enum_serialises_as_its_name(self):
        assert StellarNetwork.PUBNET == "pubnet"
        assert str(StellarNetwork.TESTNET) == "testnet"

    def test_only_pubnet_is_production(self):
        assert StellarNetwork.PUBNET.is_production
        assert not StellarNetwork.TESTNET.is_production
        assert not StellarNetwork.FUTURENET.is_production


class TestNetworkProfile:
    def test_each_network_has_its_own_endpoints(self):
        pubnet = profile_for("pubnet")
        testnet = profile_for("testnet")

        assert pubnet.horizon_url != testnet.horizon_url
        assert "horizon.stellar.org" in pubnet.horizon_url
        assert "testnet" in testnet.horizon_url

    def test_each_network_has_its_own_schema(self):
        """Separate schemas make an overlap impossible by construction."""
        schemas = {profile_for(network).schema for network in StellarNetwork}

        assert len(schemas) == len(StellarNetwork)

    def test_the_passphrase_identifies_the_network(self):
        assert profile_for("pubnet").passphrase.startswith("Public Global Stellar Network")
        assert profile_for("testnet").passphrase.startswith("Test SDF Network")
        assert profile_for("futurenet").passphrase.startswith("Test SDF Future Network")

    def test_the_passphrase_is_not_overridable(self, monkeypatch):
        """A configurable identifier could not detect a mix-up."""
        monkeypatch.setenv("ASTROML_HORIZON_URL", "http://localhost:8000")

        profile = profile_for("pubnet")

        assert profile.horizon_url == "http://localhost:8000"
        assert profile.passphrase == profile_for("pubnet").passphrase

    def test_endpoints_can_be_overridden_for_local_development(self):
        profile = profile_for("testnet", horizon_url="http://localhost:8000")

        assert profile.horizon_url == "http://localhost:8000"
        assert profile.network is StellarNetwork.TESTNET

    def test_the_environment_can_override_the_endpoint(self, monkeypatch):
        monkeypatch.setenv("ASTROML_HORIZON_URL", "https://proxy.example.com")

        assert profile_for("pubnet").horizon_url == "https://proxy.example.com"

    def test_the_schema_can_be_overridden(self, monkeypatch):
        monkeypatch.setenv("ASTROML_DB_SCHEMA", "custom_schema")

        assert profile_for("testnet").schema == "custom_schema"

    def test_matches_compares_networks_not_endpoints(self):
        default = profile_for("testnet")
        proxied = profile_for("testnet", horizon_url="http://localhost:8000")

        assert default.matches(proxied)
        assert default.matches("test")
        assert not default.matches("pubnet")

    def test_serialises_for_stamping_onto_a_run(self):
        payload = profile_for("pubnet").to_dict()

        assert payload["network"] == "pubnet"
        assert payload["schema"] == "astroml_pubnet"


class TestCrossNetworkGuard:
    """The guard is the point of the issue."""

    def test_a_matching_write_is_allowed(self):
        guard_write(profile_for("testnet"), profile_for("testnet"))

    def test_writing_testnet_data_into_a_mainnet_store_is_refused(self):
        with pytest.raises(CrossNetworkWriteError):
            guard_write(profile_for("testnet"), profile_for("pubnet"))

    def test_writing_mainnet_data_into_a_testnet_store_is_refused(self):
        """Both directions: the wrong store is wrong either way."""
        with pytest.raises(CrossNetworkWriteError):
            guard_write(profile_for("pubnet"), profile_for("testnet"))

    def test_futurenet_is_distinct_from_testnet(self):
        with pytest.raises(CrossNetworkWriteError):
            guard_write(profile_for("futurenet"), profile_for("testnet"))

    def test_the_error_names_both_networks_and_the_context(self):
        with pytest.raises(CrossNetworkWriteError) as excinfo:
            guard_write("testnet", "pubnet", context="persisting operations")

        message = str(excinfo.value)
        assert "testnet" in message
        assert "pubnet" in message
        assert "persisting operations" in message

    def test_the_error_carries_both_networks_for_a_handler(self):
        with pytest.raises(CrossNetworkWriteError) as excinfo:
            guard_write("testnet", "pubnet")

        assert excinfo.value.writer is StellarNetwork.TESTNET
        assert excinfo.value.store is StellarNetwork.PUBNET

    def test_accepts_names_as_well_as_profiles(self):
        guard_write("mainnet", "pubnet")  # aliases for the same network

    def test_an_unstamped_store_is_allowed_so_a_fresh_store_can_be_written(self):
        """Refusing would make the first write to an empty store impossible."""
        guard_write(profile_for("pubnet"), None)

    def test_an_unstamped_store_is_logged(self, caplog):
        """It is also what an unstamped legacy store looks like."""
        with caplog.at_level("INFO", logger="astroml.ingestion.network"):
            guard_write(profile_for("pubnet"), None)

        assert any("no recorded network" in record.message for record in caplog.records)


class TestStreamConfigIntegration:
    def test_defaults_to_testnet_and_its_endpoint(self):
        config = StreamConfig()

        assert config.network is StellarNetwork.TESTNET
        assert "testnet" in config.horizon_url

    def test_follows_the_environment(self, monkeypatch):
        monkeypatch.setenv("ASTROML_STELLAR_NETWORK", "pubnet")

        config = StreamConfig()

        assert config.network is StellarNetwork.PUBNET
        assert config.horizon_url == "https://horizon.stellar.org"

    def test_an_explicit_horizon_url_still_wins(self, monkeypatch):
        monkeypatch.setenv("ASTROML_STELLAR_NETWORK", "pubnet")
        monkeypatch.setenv("ASTROML_HORIZON_URL", "http://localhost:8000")

        config = StreamConfig()

        assert config.horizon_url == "http://localhost:8000"
        # ...and the network is still pubnet, so the guard and the schema are
        # unaffected by pointing at a proxy.
        assert config.network is StellarNetwork.PUBNET

    def test_the_profile_keeps_the_configured_endpoint(self):
        config = StreamConfig(network=StellarNetwork.PUBNET, horizon_url="http://localhost:8000")

        profile = config.profile

        assert profile.horizon_url == "http://localhost:8000"
        assert profile.schema == "astroml_pubnet"
        assert profile.passphrase.startswith("Public Global")

    def test_an_explicit_network_argument_is_honoured(self):
        config = StreamConfig(network=StellarNetwork.FUTURENET)

        assert config.profile.network is StellarNetwork.FUTURENET

    def test_remains_frozen(self):
        """No regression: the config is still immutable."""
        config = StreamConfig()

        with pytest.raises(Exception):
            config.network = StellarNetwork.PUBNET  # type: ignore[misc]

    def test_existing_defaults_are_unchanged(self):
        """No regression to existing ingestion flows."""
        config = StreamConfig()

        assert config.stream_endpoint == "/transactions"
        assert config.persist_chunk_size == 50
        assert config.reconnect_base_seconds == 1.0


class TestGuardInAnIngestionFlow:
    """The guard used the way a writer would use it."""

    def test_a_writer_refuses_the_wrong_store_before_persisting(self):
        persisted: list[dict] = []

        def persist(records, writer: NetworkProfile, store: NetworkProfile | None):
            guard_write(writer, store, context="persisting operations")
            persisted.extend(records)

        testnet = profile_for("testnet")
        pubnet = profile_for("pubnet")

        persist([{"id": 1}], testnet, testnet)
        assert len(persisted) == 1

        with pytest.raises(CrossNetworkWriteError):
            persist([{"id": 2}], testnet, pubnet)

        # Nothing from the refused call reached the store.
        assert len(persisted) == 1
