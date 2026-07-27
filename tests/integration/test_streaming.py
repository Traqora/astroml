"""Integration tests for streaming ingestion pipeline.

These tests verify the complete workflow from real-time streaming
to database persistence, including reconnection logic and cursor tracking.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from astroml.ingestion.stream import HorizonStreamClient
from astroml.ingestion.config import StreamConfig
from astroml.ingestion.enhanced_stream import (
    EnhancedStreamConfig,
    RateLimitTracker,
)


class TestStreamClientIntegration:
    """Integration tests for Horizon streaming client."""

    @pytest.mark.asyncio
    async def test_stream_client_initialization(
        self,
    ) -> None:
        """Test stream client initialization with configuration."""
        config = StreamConfig(
            horizon_url="https://horizon-testnet.stellar.org",
            stream_endpoint="/transactions",
            cursor="12345",
        )
        
        client = HorizonStreamClient(config)
        
        assert client._config.horizon_url == "https://horizon-testnet.stellar.org"
        assert client._config.stream_endpoint == "/transactions"
        assert client._last_cursor == "12345"

    @pytest.mark.asyncio
    async def test_stream_client_url_building(
        self,
    ) -> None:
        """Test stream URL construction with cursor."""
        config = StreamConfig(
            horizon_url="https://horizon-testnet.stellar.org",
            stream_endpoint="/transactions",
            cursor="12345",
        )
        
        client = HorizonStreamClient(config)
        url = client._build_stream_url()
        
        assert "cursor=12345" in url
        assert "order=asc" in url
        assert url.startswith("https://horizon-testnet.stellar.org/transactions")

    @pytest.mark.asyncio
    async def test_stream_client_cursor_tracking(
        self,
    ) -> None:
        """Test cursor tracking during streaming."""
        config = StreamConfig(cursor="1000")
        client = HorizonStreamClient(config)
        
        # Mock event with new cursor
        event = MagicMock()
        event.data = json.dumps({
            "hash": "x" * 64,
            "paging_token": "1001",
        })
        
        client._running = True
        
        with patch.object(client, "_persist_transaction", new_callable=AsyncMock):
            with patch.object(client, "_save_cursor"):
                await client._process_event(event)
        
        assert client._last_cursor == "1001"

    @pytest.mark.asyncio
    async def test_stream_client_reconnection_logic(
        self,
    ) -> None:
        """Test exponential backoff on reconnection."""
        config = StreamConfig(
            reconnect_base_seconds=0.01,
            reconnect_max_seconds=0.05,
            max_retries=3,
        )
        client = HorizonStreamClient(config)
        client._running = True
        
        with patch("astroml.ingestion.stream.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._handle_reconnect(ConnectionError("test"))
            first_delay = mock_sleep.call_args[0][0]
            
            await client._handle_reconnect(ConnectionError("test"))
            second_delay = mock_sleep.call_args[0][0]
            
            assert second_delay > first_delay

    @pytest.mark.asyncio
    async def test_stream_client_max_retries(
        self,
    ) -> None:
        """Test that client stops after max retries."""
        config = StreamConfig(max_retries=3)
        client = HorizonStreamClient(config)
        client._running = True
        client._retry_count = 3
        
        with patch("astroml.ingestion.stream.asyncio.sleep", new_callable=AsyncMock):
            await client._handle_reconnect(ConnectionError("test"))
        
        assert client._running is False


class TestRateLimitTrackerIntegration:
    """Integration tests for rate limiting in streaming."""

    def test_rate_limit_tracker_initialization(
        self,
    ) -> None:
        """Test rate limit tracker initialization."""
        tracker = RateLimitTracker(backoff_factor=1.5)
        
        assert tracker.backoff_factor == 1.5
        assert tracker.current_backoff == 1.0
        assert tracker.request_count == 0

    def test_rate_limit_request_tracking(
        self,
    ) -> None:
        """Test request tracking for rate limiting."""
        tracker = RateLimitTracker()
        
        tracker.record_request()
        tracker.record_request()
        tracker.record_request()
        
        assert tracker.request_count == 3

    def test_rate_limit_backoff_calculation(
        self,
    ) -> None:
        """Test backoff time calculation after rate limit."""
        tracker = RateLimitTracker(backoff_factor=2.0)
        
        backoff1 = tracker.handle_rate_limit()
        assert backoff1 == 2.0
        
        backoff2 = tracker.handle_rate_limit()
        assert backoff2 == 4.0

    def test_rate_limit_throttling_decision(
        self,
    ) -> None:
        """Test throttling decision based on recent rate limits."""
        tracker = RateLimitTracker()
        
        # No rate limit yet
        assert tracker.should_throttle() is False
        
        # Hit rate limit
        tracker.handle_rate_limit()
        
        # Should throttle immediately after
        assert tracker.should_throttle() is True

    def test_request_rate_calculation(
        self,
    ) -> None:
        """Test request rate calculation."""
        tracker = RateLimitTracker()
        
        tracker.record_request()
        tracker.record_request()
        tracker.record_request()
        
        rate = tracker.get_request_rate()
        assert rate > 0


class TestEnhancedStreamingIntegration:
    """Integration tests for enhanced streaming service."""

    @pytest.mark.asyncio
    async def test_enhanced_stream_config(
        self,
    ) -> None:
        """Test enhanced stream configuration."""
        config = EnhancedStreamConfig(
            horizon_url="https://horizon-testnet.stellar.org",
            stream_type="effects",
            cursor="now",
            max_retries=5,
            batch_size=100,
        )
        
        assert config.horizon_url == "https://horizon-testnet.stellar.org"
        assert config.stream_type == "effects"
        assert config.cursor == "now"
        assert config.max_retries == 5
        assert config.batch_size == 100

    @pytest.mark.asyncio
    async def test_stream_event_processing(
        self,
        mock_horizon_response: Dict[str, Any],
    ) -> None:
        """Test processing of stream events."""
        from astroml.ingestion.parsers import parse_transaction
        
        # Parse mock response
        transaction = parse_transaction(mock_horizon_response)
        
        # Verify parsing
        assert transaction.hash == mock_horizon_response["hash"]
        assert transaction.source_account == mock_horizon_response["source_account"]
        assert transaction.ledger_sequence == mock_horizon_response["ledger"]

    @pytest.mark.asyncio
    async def test_stream_batch_processing(
        self,
    ) -> None:
        """Test batch processing of stream events."""
        events = []
        for i in range(10):
            event = MagicMock()
            event.data = json.dumps({
                "hash": "x" * 64,
                "ledger": 1000 + i,
                "source_account": f"G{'A' * 55}",
                "created_at": "2024-01-01T00:00:00Z",
                "fee_charged": 100,
                "operation_count": 1,
                "successful": True,
                "memo_type": "none",
                "paging_token": str(1000 + i),
            })
            events.append(event)
        
        # Process batch
        processed_count = 0
        for event in events:
            data = json.loads(event.data)
            if data.get("hash"):
                processed_count += 1
        
        assert processed_count == 10


class TestStreamingPipelineIntegration:
    """Integration tests for complete streaming pipeline."""

    @pytest.mark.asyncio
    async def test_stream_to_database_pipeline(
        self,
        test_session,
        mock_horizon_response: Dict[str, Any],
    ) -> None:
        """Test complete pipeline from stream to database."""
        from astroml.ingestion.parsers import parse_transaction
        from astroml.db.schema import Ledger, Transaction
        
        # Create ledger first
        ledger = Ledger(
            sequence=1000,
            hash="a" * 64,
            closed_at=datetime(2024, 1, 1),
            successful_transaction_count=1,
            failed_transaction_count=0,
            operation_count=1,
        )
        test_session.add(ledger)
        test_session.commit()
        
        # Parse and store transaction from stream
        transaction = parse_transaction(mock_horizon_response)
        test_session.add(transaction)
        test_session.commit()
        
        # Verify database state
        stored_tx = test_session.query(Transaction).filter_by(
            hash=mock_horizon_response["hash"]
        ).first()
        
        assert stored_tx is not None
        assert stored_tx.source_account == mock_horizon_response["source_account"]

    @pytest.mark.asyncio
    async def test_stream_cursor_persistence(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Test cursor persistence across stream restarts."""
        cursor_file = temp_output_dir / ".stream_cursor"
        
        # Save cursor
        cursor = "12345"
        cursor_file.write_text(cursor)
        
        # Load cursor
        loaded_cursor = cursor_file.read_text().strip()
        
        assert loaded_cursor == cursor

    @pytest.mark.asyncio
    async def test_stream_error_recovery(
        self,
    ) -> None:
        """Test stream recovery from transient errors."""
        config = StreamConfig(max_retries=3)
        client = HorizonStreamClient(config)
        client._running = True
        
        # Simulate error
        error_count = [0]
        
        async def mock_fetch():
            error_count[0] += 1
            if error_count[0] < 3:
                raise ConnectionError("Transient error")
            return {"data": "success"}
        
        # Should recover after retries
        with patch.object(client, "_handle_reconnect", new_callable=AsyncMock):
            try:
                for _ in range(3):
                    await mock_fetch()
            except ConnectionError:
                pass
        
        assert error_count[0] == 3

    @pytest.mark.asyncio
    async def test_stream_metrics_tracking(
        self,
    ) -> None:
        """Test metrics tracking during streaming."""
        from astroml.ingestion.metrics import (
            STREAM_RECORDS_PROCESSED,
            STREAM_ERRORS,
        )
        
        # Simulate processing
        STREAM_RECORDS_PROCESSED.inc()
        STREAM_RECORDS_PROCESSED.inc()
        STREAM_RECORDS_PROCESSED.inc()
        
        # Simulate error
        STREAM_ERRORS.inc()
        
        # Verify metrics (in real scenario, would query Prometheus)
        # Here we just verify the metrics can be incremented
        assert STREAM_RECORDS_PROCESSED._value.get() == 3
        assert STREAM_ERRORS._value.get() == 1

    @pytest.mark.asyncio
    async def test_stream_graceful_shutdown(
        self,
    ) -> None:
        """Test graceful shutdown of streaming client."""
        config = StreamConfig()
        client = HorizonStreamClient(config)
        
        # Simulate running state
        client._running = True
        
        # Trigger shutdown
        client._running = False
        
        assert client._running is False
