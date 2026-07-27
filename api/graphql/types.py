"""GraphQL type definitions for AstroML."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import strawberry
from strawberry import ID


@strawberry.type
class Account:
    """Stellar account information."""

    id: strawberry.ID
    public_key: str
    first_seen: Optional[datetime] = None
    last_active: Optional[datetime] = None
    balance: Optional[float] = None
    home_domain: Optional[str] = None
    created_at: datetime


@strawberry.type
class Transaction:
    """Blockchain transaction record."""

    hash: str
    ledger_sequence: int
    source_account: str
    destination_account: Optional[str] = None
    amount: Optional[float] = None
    asset_code: Optional[str] = None
    asset_issuer: Optional[str] = None
    fee: int
    operation_type: Optional[str] = None
    successful: bool
    memo_type: Optional[str] = None
    created_at: datetime


@strawberry.type
class FraudAlert:
    """Anomaly detection result."""

    id: strawberry.ID
    account_id: str
    pattern: Optional[str] = None
    risk_score: float
    risk_level: str
    description: Optional[str] = None
    detected_at: datetime
    resolved: bool


@strawberry.type
class LoyaltyPoints:
    """Loyalty points balance."""

    id: strawberry.ID
    account_id: str
    balance: int
    tier: str
    multiplier: float
    updated_at: datetime


@strawberry.type
class PointsTransaction:
    """Loyalty points transaction."""

    id: strawberry.ID
    account_id: str
    type: str
    points: int
    source: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime


@strawberry.type
class ModelRegistry:
    """Registered model version."""

    id: strawberry.ID
    name: str
    version: str
    path: str
    owner: Optional[str] = None
    tags: Optional[List[str]] = None
    mlflow_run_id: Optional[str] = None
    metrics: Optional[str] = None
    status: str
    parent_id: Optional[int] = None
    created_at: datetime


@strawberry.type
class User:
    """Dashboard/API user."""

    id: strawberry.ID
    username: str
    scopes: List[str]
    is_active: bool
    created_at: datetime


@strawberry.type
class ApiKey:
    """API key for machine-to-machine authentication."""

    id: strawberry.ID
    name: str
    scopes: List[str]
    expires_at: Optional[datetime] = None
    is_active: bool
    created_at: datetime


@strawberry.type
class Mentor:
    """Mentor profile."""

    id: strawberry.ID
    user_id: int
    github_username: str
    bio: Optional[str] = None
    skills: List[str]
    years_experience: int
    preferred_session_day: Optional[str] = None
    max_mentees: int
    is_available: bool
    created_at: datetime
    updated_at: datetime


@strawberry.type
class Mentee:
    """Mentee profile."""

    id: strawberry.ID
    user_id: int
    github_username: str
    bio: Optional[str] = None
    learning_interests: List[str]
    years_experience: int
    preferred_session_day: Optional[str] = None
    goals: Optional[str] = None
    created_at: datetime
    updated_at: datetime


@strawberry.type
class Mentorship:
    """Active mentorship relationship."""

    id: strawberry.ID
    mentor_id: int
    mentee_id: int
    status: str
    match_score: float
    started_at: datetime
    ended_at: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime


@strawberry.type
class Notification:
    """User notification."""

    id: strawberry.ID
    user_id: int
    event_type: str
    title: str
    content: Optional[str] = None
    link: Optional[str] = None
    actor: Optional[str] = None
    is_read: bool
    created_at: datetime


@strawberry.type
class FAQ:
    """FAQ item."""

    id: strawberry.ID
    category: str
    question: str
    answer: str
    order: int
    is_published: bool
    created_at: datetime
    updated_at: datetime


@strawberry.type
class AuditLog:
    """Audit log entry."""

    id: strawberry.ID
    timestamp: datetime
    user_id: Optional[int] = None
    username: Optional[str] = None
    auth_type: Optional[str] = None
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    status_code: Optional[int] = None
    details: Optional[str] = None


@strawberry.type
class PageInfo:
    """Pagination metadata."""

    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str] = None
    end_cursor: Optional[str] = None


@strawberry.type
class AccountConnection:
    """Paginated account connection."""

    edges: List[Account]
    page_info: PageInfo
    total_count: int


@strawberry.type
class TransactionConnection:
    """Paginated transaction connection."""

    edges: List[Transaction]
    page_info: PageInfo
    total_count: int


@strawberry.type
class FraudAlertConnection:
    """Paginated fraud alert connection."""

    edges: List[FraudAlert]
    page_info: PageInfo
    total_count: int


@strawberry.type
class MutationResult:
    """Generic mutation result."""

    success: bool
    message: Optional[str] = None
    id: Optional[str] = None


@strawberry.input
class CreateAccountInput:
    """Input for creating an account."""

    public_key: str
    home_domain: Optional[str] = None


@strawberry.input
class CreateFraudAlertInput:
    """Input for creating a fraud alert."""

    account_id: str
    pattern: Optional[str] = None
    risk_score: float
    description: Optional[str] = None


@strawberry.input
class UpdateLoyaltyPointsInput:
    """Input for updating loyalty points."""

    account_id: str
    points: int
    source: Optional[str] = None
    note: Optional[str] = None


@strawberry.type
class Subscription:
    """GraphQL subscriptions for real-time updates."""

    @strawberry.subscription
    async def transaction_created(self) -> Transaction:
        """Subscribe to new transactions."""
        from api.graphql.subscriptions import transaction_created_subscription
        async for transaction in transaction_created_subscription():
            yield transaction

    @strawberry.subscription
    async def fraud_alert_created(self) -> FraudAlert:
        """Subscribe to new fraud alerts."""
        from api.graphql.subscriptions import fraud_alert_subscription
        async for alert in fraud_alert_subscription():
            yield alert

    @strawberry.subscription
    async def loyalty_points_updated(self) -> LoyaltyPoints:
        """Subscribe to loyalty points updates."""
        from api.graphql.subscriptions import loyalty_points_subscription
        async for points in loyalty_points_subscription():
            yield points