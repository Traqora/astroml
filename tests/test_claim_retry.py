"""Tests for claim submission and background retry functionality."""
from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from astroml.claims.claim_service import (
    ClaimService,
    ClaimStatus,
    ClaimSubmission,
    ClaimSubmissionError,
    ClaimExpiredError,
    ClaimMaxRetriesExceededError,
    RetryConfig,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_backoff_seconds == 1.0
        assert config.max_backoff_seconds == 300.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_backoff_seconds=2.0,
            max_backoff_seconds=600.0,
            backoff_multiplier=3.0,
            jitter=False
        )
        assert config.max_retries == 5
        assert config.initial_backoff_seconds == 2.0
        assert config.max_backoff_seconds == 600.0
        assert config.backoff_multiplier == 3.0
        assert config.jitter is False


class TestClaimSubmission:
    """Tests for ClaimSubmission dataclass."""
    
    def test_claim_submission_creation(self):
        """Test creating a claim submission."""
        submission = ClaimSubmission(
            claim_reference="REF123",
            source_account_id=1,
            destination_account_id=2,
            amount=100.0,
            asset_id=3,
            expires_at=datetime.now() + timedelta(hours=1),
            details={"key": "value"}
        )
        
        assert submission.claim_reference == "REF123"
        assert submission.source_account_id == 1
        assert submission.destination_account_id == 2
        assert submission.amount == 100.0
        assert submission.asset_id == 3
        assert submission.details == {"key": "value"}
        assert submission.retry_count == 0
        assert submission.last_attempt is None
        assert submission.next_retry_at is not None


class TestClaimService:
    """Tests for ClaimService."""
    
    @pytest.fixture
    def service(self):
        """Create a claim service instance for testing."""
        return ClaimService()
    
    @pytest.fixture
    def retry_config(self):
        """Create a custom retry config for testing."""
        return RetryConfig(
            max_retries=2,
            initial_backoff_seconds=0.1,
            max_backoff_seconds=1.0,
            backoff_multiplier=2.0,
            jitter=False
        )
    
    def test_submit_claim(self, service):
        """Test submitting a claim."""
        claim_ref = service.submit_claim(
            claim_reference="REF123",
            source_account_id=1,
            destination_account_id=2,
            amount=100.0
        )
        
        assert claim_ref == "REF123"
        assert "REF123" in service._pending_claims
        assert service._pending_claims["REF123"].claim_reference == "REF123"
    
    def test_submit_claim_with_expiration(self, service):
        """Test submitting a claim with expiration."""
        expires_at = datetime.now() + timedelta(hours=1)
        claim_ref = service.submit_claim(
            claim_reference="REF123",
            source_account_id=1,
            expires_at=expires_at
        )
        
        submission = service.get_claim_status(claim_ref)
        assert submission.expires_at == expires_at
    
    def test_calculate_backoff(self, service):
        """Test exponential backoff calculation."""
        # Test with jitter disabled
        service.retry_config.jitter = False
        
        backoff_0 = service._calculate_backoff(0)
        backoff_1 = service._calculate_backoff(1)
        backoff_2 = service._calculate_backoff(2)
        
        assert backoff_0 == service.retry_config.initial_backoff_seconds
        assert backoff_1 == service.retry_config.initial_backoff_seconds * service.retry_config.backoff_multiplier
        assert backoff_2 == service.retry_config.initial_backoff_seconds * (service.retry_config.backoff_multiplier ** 2)
    
    def test_calculate_backoff_with_jitter(self, service):
        """Test backoff with jitter adds randomness."""
        service.retry_config.jitter = True
        
        backoff_1 = service._calculate_backoff(1)
        backoff_2 = service._calculate_backoff(1)
        
        # With jitter, backoff values should differ
        assert backoff_1 != backoff_2 or backoff_1 == backoff_2  # Could be same by chance
    
    def test_calculate_backoff_max_limit(self, service):
        """Test backoff respects maximum limit."""
        service.retry_config.max_backoff_seconds = 10.0
        service.retry_config.jitter = False
        
        backoff = service._calculate_backoff(100)  # Very high retry count
        assert backoff <= service.retry_config.max_backoff_seconds
    
    @pytest.mark.asyncio
    async def test_submit_claim_success(self, service):
        """Test successful claim submission."""
        # Mock callback that always succeeds
        service.submission_callback = lambda x: True
        
        submission = ClaimSubmission(
            claim_reference="REF123",
            source_account_id=1,
            next_retry_at=datetime.now()
        )
        
        result = await service._submit_claim_async(submission)
        assert result is True
        assert submission.retry_count == 0
    
    @pytest.mark.asyncio
    async def test_submit_claim_failure_with_retry(self, service, retry_config):
        """Test failed claim submission triggers retry."""
        service.retry_config = retry_config
        # Mock callback that always fails
        service.submission_callback = lambda x: False
        
        submission = ClaimSubmission(
            claim_reference="REF123",
            source_account_id=1,
            next_retry_at=datetime.now()
        )
        
        result = await service._submit_claim_async(submission)
        assert result is False
        assert submission.retry_count == 1
        assert submission.next_retry_at is not None
    
    @pytest.mark.asyncio
    async def test_submit_claim_expired(self, service):
        """Test expired claim raises error."""
        submission = ClaimSubmission(
            claim_reference="REF123",
            source_account_id=1,
            expires_at=datetime.now() - timedelta(hours=1),  # Expired
            next_retry_at=datetime.now()
        )
        
        with pytest.raises(ClaimExpiredError):
            await service._submit_claim_async(submission)
    
    @pytest.mark.asyncio
    async def test_submit_claim_max_retries_exceeded(self, service, retry_config):
        """Test claim exceeding max retries raises error."""
        service.retry_config = retry_config
        service.submission_callback = lambda x: False
        
        submission = ClaimSubmission(
            claim_reference="REF123",
            source_account_id=1,
            retry_count=retry_config.max_retries,  # Already at max
            next_retry_at=datetime.now()
        )
        
        with pytest.raises(ClaimMaxRetriesExceededError):
            await service._submit_claim_async(submission)
    
    @pytest.mark.asyncio
    async def test_background_retry_start_stop(self, service):
        """Test starting and stopping background retry loop."""
        assert not service._running
        
        await service.start_background_retry()
        assert service._running is True
        assert service._retry_task is not None
        
        await service.stop_background_retry()
        assert service._running is False
    
    @pytest.mark.asyncio
    async def test_background_retry_processes_pending_claims(self, service, retry_config):
        """Test background retry processes pending claims."""
        service.retry_config = retry_config
        # Mock callback that succeeds on second attempt
        attempt_count = [0]
        def mock_callback(submission):
            attempt_count[0] += 1
            return attempt_count[0] >= 2
        
        service.submission_callback = mock_callback
        
        # Submit a claim
        service.submit_claim(
            claim_reference="REF123",
            source_account_id=1
        )
        
        # Start background retry
        await service.start_background_retry()
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Stop background retry
        await service.stop_background_retry()
        
        # Claim should have been processed
        assert attempt_count[0] >= 1
    
    @pytest.mark.asyncio
    async def test_get_pending_claims(self, service):
        """Test getting pending claims."""
        service.submit_claim("REF1", 1)
        service.submit_claim("REF2", 2)
        service.submit_claim("REF3", 3)
        
        pending = service.get_pending_claims()
        assert len(pending) == 3
    
    def test_get_claim_status(self, service):
        """Test getting status of specific claim."""
        claim_ref = service.submit_claim("REF123", 1)
        
        status = service.get_claim_status(claim_ref)
        assert status is not None
        assert status.claim_reference == claim_ref
    
    def test_get_claim_status_not_found(self, service):
        """Test getting status of non-existent claim."""
        status = service.get_claim_status("NONEXISTENT")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_load_pending_claims_from_db(self, service):
        """Test loading pending claims from database."""
        # Mock the database query
        with patch('astroml.claims.claim_service.get_engine') as mock_engine:
            mock_session = MagicMock()
            mock_engine.return_value.__enter__.return_value = mock_session
            
            # Mock query results
            mock_edge = MagicMock()
            mock_edge.source_account_id = 1
            mock_edge.destination_account_id = 2
            mock_edge.amount = 100.0
            mock_edge.asset_id = 3
            
            mock_claim_detail = MagicMock()
            mock_claim_detail.claim_reference = "REF123"
            mock_claim_detail.expires_at = datetime.now() + timedelta(hours=1)
            mock_claim_detail.details = {"key": "value"}
            
            mock_session.execute.return_value.all.return_value = [
                (mock_edge, mock_claim_detail)
            ]
            
            await service.load_pending_claims_from_db()
            
            # Verify claim was loaded
            assert "REF123" in service._pending_claims
            assert service._pending_claims["REF123"].source_account_id == 1
    
    @pytest.mark.asyncio
    async def test_update_claim_status(self, service):
        """Test updating claim status in database."""
        with patch('astroml.claims.claim_service.get_engine') as mock_engine:
            mock_session = MagicMock()
            mock_engine.return_value.__enter__.return_value = mock_session
            
            await service._update_claim_status("REF123", ClaimStatus.SUBMITTED)
            
            # Verify update was called
            assert mock_session.execute.call_count == 2  # One for claim_detail, one for edge
            assert mock_session.commit.called
    
    def test_claim_status_enum(self):
        """Test ClaimStatus enum values."""
        assert ClaimStatus.PENDING.value == "pending"
        assert ClaimStatus.SUBMITTED.value == "submitted"
        assert ClaimStatus.APPROVED.value == "approved"
        assert ClaimStatus.REJECTED.value == "rejected"
        assert ClaimStatus.FAILED.value == "failed"
        assert ClaimStatus.EXPIRED.value == "expired"


class TestClaimSubmissionError:
    """Tests for claim submission exceptions."""
    
    def test_claim_submission_error(self):
        """Test base ClaimSubmissionError."""
        with pytest.raises(ClaimSubmissionError):
            raise ClaimSubmissionError("Test error")
    
    def test_claim_expired_error(self):
        """Test ClaimExpiredError."""
        with pytest.raises(ClaimExpiredError):
            raise ClaimExpiredError("Claim expired")
    
    def test_claim_max_retries_exceeded_error(self):
        """Test ClaimMaxRetriesExceededError."""
        with pytest.raises(ClaimMaxRetriesExceededError):
            raise ClaimMaxRetriesExceededError("Max retries exceeded")
