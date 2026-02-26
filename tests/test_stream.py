"""Tests for astroml.ingestion.stream.HorizonStreamClient.

All network and database interactions are mocked.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from astroml.ingestion.config import StreamConfig
from astroml.ingestion.stream import HorizonStreamClient


@pytest.fixture()
def config():
    return StreamConfig(
        horizon_url="https://horizon-testnet.stellar.org",
        stream_endpoint="/transactions",
        cursor="12345",
        reconnect_base_seconds=0.01,
        reconnect_max_seconds=0.05,
        max_retries=3,
    )


@pytest.fixture()
def sample_tx_event():
    """A mock SSE event with valid transaction data."""
    event = MagicMock()
    event.data = json.dumps({
        "hash": "x" * 64,
        "ledger": 100,
        "source_account": "G" + "A" * 55,
        "created_at": "2024-01-15T12:30:00Z",
        "fee_charged": "100",
        "operation_count": 1,
        "successful": True,
        "memo_type": "none",
        "paging_token": "99999",
    })
    return event


# -- Tests: URL construction --------------------------------------------------


class TestStreamURL:
    def test_build_url_with_cursor(self, config):
        client = HorizonStreamClient(config)
        url = client._build_stream_url()
        assert "cursor=12345" in url
        assert "order=asc" in url
        assert url.startswith("https://horizon-testnet.stellar.org/transactions")

    def test_build_url_defaults_to_now(self):
        cfg = StreamConfig(cursor=None)
        client = HorizonStreamClient(cfg)
        url = client._build_stream_url()
        assert "cursor=now" in url


# -- Tests: cursor tracking ---------------------------------------------------


class TestCursorTracking:
    @pytest.mark.asyncio
    async def test_cursor_updates_on_success(self, config, sample_tx_event):
        client = HorizonStreamClient(config)
        client._running = True

        with patch.object(client, "_persist_transaction", new_callable=AsyncMock):
            with patch.object(client, "_save_cursor"):
                await client._process_event(sample_tx_event)

        assert client.last_cursor == "99999"

    @pytest.mark.asyncio
    async def test_cursor_unchanged_on_failure(self, config, sample_tx_event):
        client = HorizonStreamClient(config)
        client._running = True
        client._last_cursor = "12345"

        with patch.object(
            client,
            "_persist_transaction",
            new_callable=AsyncMock,
            side_effect=Exception("DB error"),
        ):
            await client._process_event(sample_tx_event)

        assert client.last_cursor == "12345"


# -- Tests: reconnection logic ------------------------------------------------


class TestReconnection:
    @pytest.mark.asyncio
    async def test_exponential_backoff(self, config):
        client = HorizonStreamClient(config)
        client._running = True

        with patch("astroml.ingestion.stream.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._handle_reconnect(ConnectionError("test"))
            first_delay = mock_sleep.call_args[0][0]

            await client._handle_reconnect(ConnectionError("test"))
            second_delay = mock_sleep.call_args[0][0]

            assert second_delay > first_delay

    @pytest.mark.asyncio
    async def test_max_retries_stops_client(self, config):
        client = HorizonStreamClient(config)
        client._running = True
        client._retry_count = config.max_retries

        with patch("astroml.ingestion.stream.asyncio.sleep", new_callable=AsyncMock):
            await client._handle_reconnect(ConnectionError("test"))

        assert client._running is False


# -- Tests: malformed events --------------------------------------------------


class TestMalformedEvents:
    @pytest.mark.asyncio
    async def test_skips_invalid_json(self, config):
        client = HorizonStreamClient(config)
        client._running = True
        client._last_cursor = "12345"

        event = MagicMock()
        event.data = "not valid json {"

        await client._process_event(event)
        assert client.last_cursor == "12345"
