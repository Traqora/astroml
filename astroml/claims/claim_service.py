"""Claim submission service with background retry mechanism.

This module provides functionality for submitting claims and automatically
retrying failed submissions in the background.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import random

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..db.schema import GraphEdge, GraphClaimDetail, GraphAccount
from ..db.session import get_engine


class ClaimStatus(str, Enum):
    """Claim status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


@dataclass
class ClaimSubmission:
    """Represents a claim submission request."""
    claim_reference: str
    source_account_id: int
    destination_account_id: Optional[int]
    amount: Optional[float]
    asset_id: Optional[int]
    expires_at: Optional[datetime]
    details: Dict = field(default_factory=dict)
    retry_count: int = 0
    last_attempt: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None


class ClaimSubmissionError(Exception):
    """Base exception for claim submission errors."""
    pass


class ClaimExpiredError(ClaimSubmissionError):
    """Raised when a claim has expired."""
    pass


class ClaimMaxRetriesExceededError(ClaimSubmissionError):
    """Raised when maximum retry attempts are exceeded."""
    pass


class ClaimService:
    """Service for managing claim submissions with background retry."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        submission_callback: Optional[Callable[[ClaimSubmission], bool]] = None
    ):
        """Initialize the claim service.
        
        Args:
            retry_config: Configuration for retry behavior
            submission_callback: Optional callback function for actual submission
        """
        self.retry_config = retry_config or RetryConfig()
        self.submission_callback = submission_callback
        self.logger = logging.getLogger(__name__)
        self._pending_claims: Dict[str, ClaimSubmission] = {}
        self._running = False
        self._retry_task: Optional[asyncio.Task] = None
    
    def submit_claim(
        self,
        claim_reference: str,
        source_account_id: int,
        destination_account_id: Optional[int] = None,
        amount: Optional[float] = None,
        asset_id: Optional[int] = None,
        expires_at: Optional[datetime] = None,
        details: Optional[Dict] = None
    ) -> str:
        """Submit a new claim.
        
        Args:
            claim_reference: Unique reference for the claim
            source_account_id: Source account ID
            destination_account_id: Destination account ID
            amount: Claim amount
            asset_id: Asset ID
            expires_at: Expiration timestamp
            details: Additional claim details
            
        Returns:
            The claim reference
        """
        submission = ClaimSubmission(
            claim_reference=claim_reference,
            source_account_id=source_account_id,
            destination_account_id=destination_account_id,
            amount=amount,
            asset_id=asset_id,
            expires_at=expires_at,
            details=details or {},
            retry_count=0,
            last_attempt=None,
            next_retry_at=datetime.now()
        )
        
        self._pending_claims[claim_reference] = submission
        self.logger.info(f"Submitted claim {claim_reference} with status pending")
        
        return claim_reference
    
    def _calculate_backoff(self, retry_count: int) -> float:
        """Calculate exponential backoff with optional jitter.
        
        Args:
            retry_count: Current retry attempt number
            
        Returns:
            Backoff time in seconds
        """
        backoff = min(
            self.retry_config.initial_backoff_seconds * 
            (self.retry_config.backoff_multiplier ** retry_count),
            self.retry_config.max_backoff_seconds
        )
        
        if self.retry_config.jitter:
            backoff = backoff * (0.5 + random.random() * 0.5)
        
        return backoff
    
    async def _submit_claim_async(self, submission: ClaimSubmission) -> bool:
        """Submit a claim asynchronously.
        
        Args:
            submission: The claim submission to process
            
        Returns:
            True if submission succeeded, False otherwise
        """
        # Check if claim has expired
        if submission.expires_at and datetime.now() > submission.expires_at:
            self.logger.warning(f"Claim {submission.claim_reference} has expired")
            await self._update_claim_status(
                submission.claim_reference,
                ClaimStatus.EXPIRED
            )
            raise ClaimExpiredError(f"Claim {submission.claim_reference} has expired")
        
        # Check if max retries exceeded
        if submission.retry_count >= self.retry_config.max_retries:
            self.logger.error(
                f"Claim {submission.claim_reference} exceeded max retries "
                f"({self.retry_config.max_retries})"
            )
            await self._update_claim_status(
                submission.claim_reference,
                ClaimStatus.FAILED
            )
            raise ClaimMaxRetriesExceededError(
                f"Claim {submission.claim_reference} exceeded max retries"
            )
        
        submission.last_attempt = datetime.now()
        
        try:
            # Use callback if provided, otherwise simulate success
            if self.submission_callback:
                success = self.submission_callback(submission)
            else:
                # Simulate submission with 80% success rate
                success = random.random() < 0.8
            
            if success:
                self.logger.info(
                    f"Claim {submission.claim_reference} submitted successfully"
                )
                await self._update_claim_status(
                    submission.claim_reference,
                    ClaimStatus.SUBMITTED
                )
                return True
            else:
                raise ClaimSubmissionError("Submission failed")
                
        except Exception as e:
            submission.retry_count += 1
            backoff = self._calculate_backoff(submission.retry_count)
            submission.next_retry_at = datetime.now() + timedelta(seconds=backoff)
            
            self.logger.warning(
                f"Claim {submission.claim_reference} submission failed "
                f"(attempt {submission.retry_count}/{self.retry_config.max_retries}), "
                f"retrying in {backoff:.2f}s. Error: {e}"
            )
            
            await self._update_claim_status(
                submission.claim_reference,
                ClaimStatus.PENDING
            )
            return False
    
    async def _update_claim_status(
        self,
        claim_reference: str,
        status: ClaimStatus
    ) -> None:
        """Update claim status in database.
        
        Args:
            claim_reference: The claim reference
            status: The new status
        """
        engine = get_engine()
        with Session(engine) as session:
            try:
                # Update claim detail status
                stmt = (
                    update(GraphClaimDetail)
                    .where(GraphClaimDetail.claim_reference == claim_reference)
                    .values(claim_status=status.value)
                )
                session.execute(stmt)
                
                # Update edge status if exists
                stmt = (
                    update(GraphEdge)
                    .where(GraphEdge.external_event_id == claim_reference)
                    .where(GraphEdge.edge_type == "claim")
                    .values(status=status.value)
                )
                session.execute(stmt)
                
                session.commit()
                self.logger.debug(f"Updated claim {claim_reference} status to {status.value}")
            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to update claim status: {e}")
    
    async def _retry_loop(self) -> None:
        """Background loop for retrying pending claims."""
        while self._running:
            now = datetime.now()
            
            # Process claims that are ready for retry
            for claim_ref, submission in list(self._pending_claims.items()):
                if submission.next_retry_at and submission.next_retry_at <= now:
                    try:
                        success = await self._submit_claim_async(submission)
                        if success:
                            # Remove from pending if successful
                            del self._pending_claims[claim_ref]
                    except (ClaimExpiredError, ClaimMaxRetriesExceededError):
                        # Remove from pending if expired or max retries exceeded
                        del self._pending_claims[claim_ref]
                    except Exception as e:
                        self.logger.error(
                            f"Unexpected error processing claim {claim_ref}: {e}"
                        )
            
            # Sleep for a short interval before next check
            await asyncio.sleep(1)
    
    async def start_background_retry(self) -> None:
        """Start the background retry loop."""
        if self._running:
            self.logger.warning("Background retry already running")
            return
        
        self._running = True
        self._retry_task = asyncio.create_task(self._retry_loop())
        self.logger.info("Background retry loop started")
    
    async def stop_background_retry(self) -> None:
        """Stop the background retry loop."""
        if not self._running:
            return
        
        self._running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Background retry loop stopped")
    
    def get_pending_claims(self) -> List[ClaimSubmission]:
        """Get all pending claims.
        
        Returns:
            List of pending claim submissions
        """
        return list(self._pending_claims.values())
    
    def get_claim_status(self, claim_reference: str) -> Optional[ClaimSubmission]:
        """Get the status of a specific claim.
        
        Args:
            claim_reference: The claim reference
            
        Returns:
            The claim submission if found, None otherwise
        """
        return self._pending_claims.get(claim_reference)
    
    async def load_pending_claims_from_db(self) -> None:
        """Load pending claims from database for retry.
        
        This is useful for recovering pending claims after a restart.
        """
        engine = get_engine()
        with Session(engine) as session:
            try:
                # Query pending claims from database
                stmt = (
                    select(GraphEdge, GraphClaimDetail)
                    .join(GraphClaimDetail, GraphEdge.id == GraphClaimDetail.edge_id)
                    .where(GraphEdge.edge_type == "claim")
                    .where(GraphClaimDetail.claim_status == ClaimStatus.PENDING.value)
                )
                
                results = session.execute(stmt).all()
                
                for edge, claim_detail in results:
                    submission = ClaimSubmission(
                        claim_reference=claim_detail.claim_reference,
                        source_account_id=edge.source_account_id,
                        destination_account_id=edge.destination_account_id,
                        amount=edge.amount,
                        asset_id=edge.asset_id,
                        expires_at=claim_detail.expires_at,
                        details=claim_detail.details or {},
                        retry_count=0,
                        last_attempt=None,
                        next_retry_at=datetime.now()
                    )
                    
                    self._pending_claims[claim_detail.claim_reference] = submission
                
                self.logger.info(f"Loaded {len(results)} pending claims from database")
                
            except Exception as e:
                self.logger.error(f"Failed to load pending claims from database: {e}")
