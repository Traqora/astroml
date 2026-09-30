from typing import Any, Dict, List, Optional, Union, Callable
"""Tests for enhanced streaming service with robust error handling."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from stellar_sdk.exceptions import BaseHorizonError, ConnectionError

from astroml.ingestion.enhanced_stream import (
    ConnectionHealthMonitor,
    EnhancedStellarStream,
    EnhancedStreamConfig,
    RateLimitTracker,
)


class TestRateLimitTracker:
    """Test rate limiting and adaptive throttling."""

    def test_record_request(self) -> Any:
        """Test request recording."""
        tracker = RateLimitTracker()

        # Initial state
        assert tracker.request_count == 0
        assert tracker.get_request_rate() == 0.0

        # Record requests
        tracker.record_request()
        tracker.record_request()

        assert tracker.request_count == 2
        assert tracker.get_request_rate() > 0.0

    def test_handle_rate_limit(self) -> Any:
        """Test rate limit backoff calculation."""
        tracker = RateLimitTracker(backoff_factor=2.0)

        # First rate limit
        backoff = tracker.handle_rate_limit()
        assert backoff == 2.0  # 1.0 * 2.0
        assert tracker.current_backoff == 2.0

        # Second rate limit
        backoff = tracker.handle_rate_limit()
        assert backoff == 4.0  # 2.0 * 2.0
        assert tracker.current_backoff == 4.0

        # Should cap at maximum
        tracker.current_backoff = 400.0
        backoff = tracker.handle_rate_limit()
        assert backoff == 300.0  # Capped at 300s

    def test_should_throttle(self) -> Any:
        """Test throttling logic."""
        tracker = RateLimitTracker()

        # No rate limit hit yet
        assert not tracker.should_throttle()

        # Rate limit just hit
        tracker.handle_rate_limit()
        assert tracker.should_throttle()

        # After backoff period
        with patch(
            "time.time", return_value=tracker.last_rate_limit_time + tracker.current_backoff + 1
        ):
            assert not tracker.should_throttle()


class TestConnectionHealthMonitor:
    """Test connection health monitoring."""

    def test_record_success(self) -> Any:
        """Test successful request recording."""
        monitor = ConnectionHealthMonitor()

        assert monitor.consecutive_failures == 0
        assert monitor.is_healthy

        monitor.record_success()
        assert monitor.consecutive_failures == 0
        assert monitor.is_healthy

    def test_record_failure(self) -> Any:
        """Test failure recording and health status."""
        monitor = ConnectionHealthMonitor(max_consecutive_failures=2)

        # First failure
        monitor.record_failure()
        assert monitor.consecutive_failures == 1
        assert monitor.is_healthy  # Still healthy under threshold

        # Second failure - crosses threshold
        monitor.record_failure()
        assert monitor.consecutive_failures == 2
        assert not monitor.is_healthy  # Now unhealthy

    def test_should_check_health(self) -> Any:
        """Test health check timing."""
        monitor = ConnectionHealthMonitor(check_interval=30.0)

        # Never checked before
        assert monitor.should_check_health()

        # Just checked
        monitor.last_health_check = time.time()
        assert not monitor.should_check_health()

        # Interval passed
        monitor.last_health_check = time.time() - 31.0
        assert monitor.should_check_health()

    def test_is_connection_stale(self) -> Any:
        """Test connection staleness detection."""
        monitor = ConnectionHealthMonitor(check_interval=30.0)

        # No successful requests yet
        assert not monitor.is_connection_stale()

        # Recent success
        monitor.last_successful_request = time.time() - 10.0
        assert not monitor.is_connection_stale()

        # Stale connection
        monitor.last_successful_request = time.time() - 61.0
        assert monitor.is_connection_stale()


class TestEnhancedStellarStream:
    """Test enhanced streaming functionality."""

    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return EnhancedStreamConfig(
            horizon_url="https://horizon-testnet.stellar.org",
            stream_type="effects",
            max_retries=3,
            batch_size=10,
        )

    @pytest.fixture
    def stream(self, config) -> Any:
        """Create test stream."""
        return EnhancedStellarStream(config)

    @pytest.mark.asyncio
    async def test_context_manager(self, stream):
        """Test async context manager."""
        async with stream as s:
            assert s._running is True
            assert s is stream

        assert stream._running is False

    @pytest.mark.asyncio
    async def test_check_server_health_success(self, stream):
        """Test successful server health check."""
        mock_server = MagicMock()
        mock_server.root.return_value = {"horizon_version": "3.0.0"}
        stream.server = mock_server

        result = await stream._check_server_health()

        assert result is True
        assert stream.health_monitor.is_healthy
        assert stream.health_monitor.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_check_server_health_failure(self, stream):
        """Test failed server health check."""
        mock_server = MagicMock()
        mock_server.root = MagicMock(side_effect=ConnectionError("Connection failed"))
        stream.server = mock_server

        result = await stream._check_server_health()

        assert result is False
        # The monitor record_failure was called, but threshold is 3
        assert stream.health_monitor.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_handle_rate_limit(self, stream):
        """Test rate limit handling."""
        # Create a mock BaseHorizonError with 429 status
        mock_response = MagicMock()
        mock_response.headers = MagicMock()
        mock_response.headers.get.return_value = "10"

        # We manually add .response because BaseHorizonError might not keep it
        error = BaseHorizonError(response=mock_response)
        error.response = mock_response
        error.status = 429

        stream._running = True  # Ensure running
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await stream._handle_rate_limit(error)

            # Should have called sleep with the Retry-After value
            mock_sleep.assert_called_with(10.0)
            assert stream.rate_tracker.current_backoff > 1.0

    @pytest.mark.asyncio
    async def test_handle_connection_error(self, stream):
        """Test connection error handling."""
        error = ConnectionError("Connection failed")

        with patch("asyncio.sleep") as mock_sleep:
            await stream._handle_connection_error(error)

            # Should record failure and sleep
            assert stream.health_monitor.consecutive_failures == 1
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_effects_basic(self, stream):
        """Test basic effects streaming."""
        stream._running = True
        # Mock the server effects call
        mock_effects_builder = MagicMock()
        mock_effects_builder.cursor.return_value = mock_effects_builder
        mock_effects_builder.limit.return_value = mock_effects_builder
        mock_effects_builder.call.return_value = {
            "_embedded": {
                "records": [
                    {
                        "id": "123456789",
                        "account": "GD123456789",
                        "type": "account_credited",
                        "amount": "100.0",
                        "asset_type": "native",
                        "created_at": "2023-01-01T00:00:00Z",
                        "paging_token": "123456789",
                    }
                ]
            }
        }

        stream.server = MagicMock()
        stream.server.effects.return_value = mock_effects_builder
        # Mock root for health check
        stream.server.root.return_value = {"horizon_version": "1.0.0"}

        # Mock the persistence method
        with patch.object(stream, "_persist_effect", new_callable=AsyncMock):
            # Get first batch
            effects = []
            async for effect in stream._stream_effects():
                effects.append(effect)
                break  # Just get first batch

            assert len(effects) == 1
            assert effects[0]["id"] == "123456789"

    @pytest.mark.asyncio
    async def test_stream_operations_basic(self, stream):
        """Test basic operations streaming."""
        stream._running = True
        # Change to operations stream
        stream.config.stream_type = "operations"

        # Mock the server operations call
        mock_operations_builder = MagicMock()
        mock_operations_builder.cursor.return_value = mock_operations_builder
        mock_operations_builder.limit.return_value = mock_operations_builder
        mock_operations_builder.call.return_value = {
            "_embedded": {
                "records": [
                    {
                        "id": 123456789,
                        "transaction_hash": "abcdef123456",
                        "type": "payment",
                        "source_account": "GD123456789",
                        "to": "GD987654321",
                        "amount": "100.0",
                        "asset_type": "native",
                        "created_at": "2023-01-01T00:00:00Z",
                        "paging_token": "123456789",
                    }
                ]
            }
        }

        stream.server = MagicMock()
        stream.server.operations.return_value = mock_operations_builder
        # Mock root for health check
        stream.server.root.return_value = {"horizon_version": "1.0.0"}

        # Mock the persistence method
        with patch.object(stream, "_persist_operation", new_callable=AsyncMock):
            # Get first batch
            operations = []
            async for operation in stream._stream_operations():
                operations.append(operation)
                break  # Just get first batch

            assert len(operations) == 1
            assert operations[0]["id"] == 123456789

    @pytest.mark.asyncio
    async def test_stream_with_retry_rate_limit(self, stream):
        """Test streaming with rate limit retry."""
        stream._running = True
        mock_builder = MagicMock()
        mock_builder.limit.return_value = mock_builder

        # First call raises rate limit, second succeeds
        mock_response = MagicMock()
        mock_response.headers = MagicMock()
        mock_response.headers.get.return_value = "1"
        rate_limit_error = BaseHorizonError(response=mock_response)
        rate_limit_error.status = 429

        mock_builder.call.side_effect = [rate_limit_error, {"_embedded": {"records": []}}]

        # Mock rate limit handling
        with (
            patch.object(stream, "_handle_rate_limit", new_callable=AsyncMock) as mock_handle,
            patch.object(stream, "_check_server_health", new_callable=AsyncMock, return_value=True),
        ):

            records = []
            async for record in stream._stream_with_retry(mock_builder):
                records.append(record)
                break

            # Should have handled rate limit and continued
            assert mock_handle.called

    @pytest.mark.asyncio
    async def test_get_stats(self, stream):
        """Test statistics collection."""
        # Set some internal state
        stream._cursor = "123456789"
        stream._processed_count = 100
        stream.rate_tracker.request_count = 50
        stream.health_monitor.is_healthy = True
        stream.health_monitor.consecutive_failures = 0

        stats = stream.get_stats()

        assert stats["cursor"] == "123456789"
        assert stats["processed_count"] == 100
        assert stats["request_rate"] > 0
        assert stats["is_healthy"] is True
        assert stats["consecutive_failures"] == 0
        assert "current_backoff" in stats


@pytest.mark.asyncio
async def test_integration_basic_stream():
    """Integration test for basic streaming functionality."""
    config = EnhancedStreamConfig(
        horizon_url="https://horizon-testnet.stellar.org",
        stream_type="effects",
        max_retries=1,
        batch_size=1,
    )

    stream = EnhancedStellarStream(config)

    # Mock the entire server interaction
    with patch("stellar_sdk.Server") as mock_server_class:
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_effects_builder = MagicMock()
        mock_effects_builder.cursor.return_value = mock_effects_builder
        mock_effects_builder.limit.return_value = mock_effects_builder
        mock_effects_builder.call.return_value = {"_embedded": {"records": []}}  # Empty response
        mock_server.effects.return_value = mock_effects_builder

        # Mock persistence
        with patch.object(stream, "_persist_effect", new_callable=AsyncMock):
            async with stream:
                # Should run without errors
                await asyncio.wait_for(stream.run(), timeout=1.0)


if __name__ == "__main__":
    pytest.main([__file__])
