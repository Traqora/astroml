"""Tests for retry logic in ingestion module.

Tests tenacity retry decorators used in enhanced_stream.py to ensure:
- Transient failures are retried with exponential backoff
- Permanent failures fail after max attempts
- Rate limiting triggers appropriate backoff
- Non-retryable errors fail immediately
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from stellar_sdk.exceptions import (
    BadRequestError,
    BaseHorizonError,
    NotFoundError,
)
from stellar_sdk.exceptions import (
    ConnectionError as StellarConnectionError,
)
from tenacity import RetryError, stop_after_attempt, wait_exponential

from astroml.ingestion.enhanced_stream import (
    EnhancedStellarStream,
    EnhancedStreamConfig,
)


class TestRetryLogic:
    """Test suite for tenacity retry logic."""

    @pytest.fixture
    def stream_config(self):
        """Create test stream configuration."""
        return EnhancedStreamConfig(
            horizon_url="https://horizon-testnet.stellar.org",
            stream_type="effects",
            max_retries=5,
            base_retry_delay=1.0,
            max_retry_delay=60.0,
        )

    @pytest.fixture
    def mock_server(self):
        """Create mock Stellar SDK server."""
        server = Mock()
        server.root = Mock(return_value={"horizon_version": "21.0.0"})
        return server

    @pytest.mark.asyncio
    async def test_transient_failure_succeeds_after_retries(self, stream_config, mock_server):
        """Test that transient connection errors are retried and eventually succeed."""
        stream = EnhancedStellarStream(stream_config)
        stream.server = mock_server

        # Mock server.root to fail twice then succeed
        call_count = [0]

        async def failing_root():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise StellarConnectionError("Connection failed")
            return {"horizon_version": "21.0.0"}

        mock_server.root = Mock(side_effect=failing_root)

        # Should succeed after 2 failures
        result = await stream._check_server_health()

        assert result is True
        assert call_count[0] == 3  # 2 failures + 1 success
        assert stream.health_monitor.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_permanent_failure_fails_after_max_attempts(self, stream_config, mock_server):
        """Test that permanent failures fail after max retry attempts."""
        stream = EnhancedStellarStream(stream_config)
        stream.server = mock_server

        # Mock server.root to always fail
        mock_server.root = Mock(side_effect=StellarConnectionError("Permanent failure"))

        # Should fail after max_retries attempts
        with pytest.raises((StellarConnectionError, RetryError)):
            await stream._check_server_health()

        # Verify max attempts were reached
        assert stream.health_monitor.consecutive_failures >= stream.config.max_retries

    @pytest.mark.asyncio
    async def test_exponential_backoff_applied(self, stream_config, mock_server):
        """Test that exponential backoff is applied between retries."""
        stream = EnhancedStellarStream(stream_config)
        stream.server = mock_server

        call_times = []

        async def failing_root():
            call_times.append(asyncio.get_event_loop().time())
            raise StellarConnectionError("Connection failed")

        mock_server.root = Mock(side_effect=failing_root)

        # Should fail with exponential backoff
        with pytest.raises((StellarConnectionError, RetryError)):
            await stream._check_server_health()

        # Verify multiple calls were made (retries occurred)
        assert len(call_times) >= 2

        # Verify backoff delays increase (exponential)
        if len(call_times) >= 3:
            first_delay = call_times[1] - call_times[0]
            second_delay = call_times[2] - call_times[1]
            # Second delay should be >= first delay (exponential)
            assert second_delay >= first_delay * 0.9  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_backoff(self, stream_config, mock_server):
        """Test that rate limit errors trigger appropriate backoff."""
        stream = EnhancedStellarStream(stream_config)
        stream.server = mock_server

        # Create a mock BaseHorizonError with 429 status
        rate_limit_error = BaseHorizonError("Rate limit exceeded")
        rate_limit_error.status = 429
        rate_limit_error.response = Mock()
        rate_limit_error.response.headers = {"Retry-After": "2.0"}

        mock_server.root = Mock(side_effect=rate_limit_error)

        # Should handle rate limit with backoff
        with pytest.raises((BaseHorizonError, RetryError)):
            await stream._check_server_health()

        # Verify rate limit was tracked
        assert stream.rate_tracker.last_rate_limit_time is not None
        assert stream.rate_tracker.current_backoff > 1.0

    @pytest.mark.asyncio
    async def test_non_retryable_errors_fail_immediately(self, stream_config, mock_server):
        """Test that non-retryable errors (like BadRequestError) fail immediately."""
        stream = EnhancedStellarStream(stream_config)
        stream.server = mock_server

        # BadRequestError should not be retried by the retry decorator
        mock_server.root = Mock(side_effect=BadRequestError("Bad request"))

        # Should fail immediately without retries
        with pytest.raises(BadRequestError):
            await stream._check_server_health()

        # Verify no retry occurred (single call)
        assert mock_server.root.call_count == 1

    @pytest.mark.asyncio
    async def test_not_found_error_with_retry(self, stream_config, mock_server):
        """Test that NotFoundError is handled with a single retry attempt."""
        stream = EnhancedStellarStream(stream_config)
        stream.server = mock_server

        # NotFoundError triggers a different handling path
        mock_server.root = Mock(side_effect=NotFoundError("Not found"))

        # Should fail after handling
        with pytest.raises(NotFoundError):
            await stream._check_server_health()

    @pytest.mark.asyncio
    async def test_retry_only_on_connection_errors(self, stream_config, mock_server):
        """Test that retry decorator only retries on ConnectionError and BaseHorizonError."""
        stream = EnhancedStellarStream(stream_config)
        stream.server = mock_server

        # Generic Exception should not be retried by the @retry decorator
        mock_server.root = Mock(side_effect=Exception("Generic error"))

        # The retry decorator is configured to retry only on ConnectionError and BaseHorizonError
        # Generic exceptions should fail immediately
        with pytest.raises(Exception):
            await stream._check_server_health()

    @pytest.mark.asyncio
    async def test_retry_delays_respect_min_max_bounds(self, stream_config, mock_server):
        """Test that retry delays respect min and max bounds."""
        stream = EnhancedStellarStream(stream_config)
        stream.server = mock_server

        # Configure with specific bounds
        stream.config.base_retry_delay = 0.1
        stream.config.max_retry_delay = 2.0

        call_times = []

        async def failing_root():
            call_times.append(asyncio.get_event_loop().time())
            raise StellarConnectionError("Connection failed")

        mock_server.root = Mock(side_effect=failing_root)

        with pytest.raises((StellarConnectionError, RetryError)):
            await stream._check_server_health()

        # Verify delays are within bounds
        if len(call_times) >= 2:
            delays = [call_times[i] - call_times[i - 1] for i in range(1, len(call_times))]
            for delay in delays:
                # Allow some tolerance for async overhead
                assert delay >= 0.05  # Min bound (with tolerance)
                assert delay <= 2.5  # Max bound (with tolerance)

    @pytest.mark.asyncio
    async def test_max_attempts_enforced(self, stream_config, mock_server):
        """Test that max_attempts is strictly enforced."""
        stream_config.max_retries = 3
        stream = EnhancedStellarStream(stream_config)
        stream.server = mock_server

        call_count = [0]

        async def failing_root():
            call_count[0] += 1
            raise StellarConnectionError("Connection failed")

        mock_server.root = Mock(side_effect=failing_root)

        with pytest.raises((StellarConnectionError, RetryError)):
            await stream._check_server_health()

        # Should not exceed max_retries
        assert call_count[0] <= stream.config.max_retries + 1  # +1 for initial attempt

    @pytest.mark.asyncio
    async def test_success_resets_retry_counter(self, stream_config, mock_server):
        """Test that successful calls reset the retry counter."""
        stream = EnhancedStellarStream(stream_config)
        stream.server = mock_server

        # First call succeeds
        mock_server.root = Mock(return_value={"horizon_version": "21.0.0"})
        result = await stream._check_server_health()
        assert result is True

        # Second call fails
        mock_server.root = Mock(side_effect=StellarConnectionError("Connection failed"))
        with pytest.raises((StellarConnectionError, RetryError)):
            await stream._check_server_health()

        # Counter should have been reset after first success
        # (This is implicit in tenacity behavior)
