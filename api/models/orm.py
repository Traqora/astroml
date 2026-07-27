"""SQLAlchemy ORM models for the API backend (issue #251).

Extends the existing ``astroml.db.schema.Base`` so all tables are created
by a single ``alembic upgrade head``.

Models
------
Account         — Stellar account info (public_key, first_seen, last_active, balance)
Transaction     — Blockchain transactions (hash, ledger, source, dest, amount, asset, fee)
FraudAlert      — Anomaly detection results (account_id, pattern, risk_score, detected_at)
LoyaltyPoints   — Points balance per account (account_id, balance, tier, multiplier)
PointsTransaction — Earn/redeem/adjust records
ModelRegistry   — Registered model versions (name, version, path, metrics)
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

# Reuse the project-wide declarative base so all tables live in one metadata.
from astroml.db.schema import Base

_ID = BigInteger().with_variant(Integer(), "sqlite")


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class ApiAccount(Base):
    """Stellar account info for the API layer.

    Separate from the ingestion-layer ``accounts`` table so the API can
    store richer profile data without polluting the raw schema.
    """

    __tablename__ = "api_accounts"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    public_key: Mapped[str] = mapped_column(String(56), nullable=False, unique=True)
    first_seen: Mapped[Optional[datetime]] = mapped_column()
    last_active: Mapped[Optional[datetime]] = mapped_column()
    balance: Mapped[Optional[float]] = mapped_column(Numeric)
    home_domain: Mapped[Optional[str]] = mapped_column(String(253))
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_api_accounts_public_key", "public_key"),
        Index("ix_api_accounts_last_active", "last_active"),
    )


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

class ApiTransaction(Base):
    """Blockchain transaction record for the API layer."""

    __tablename__ = "api_transactions"

    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    ledger_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    source_account: Mapped[str] = mapped_column(String(56), nullable=False)
    destination_account: Mapped[Optional[str]] = mapped_column(String(56))
    amount: Mapped[Optional[float]] = mapped_column(Numeric)
    asset_code: Mapped[Optional[str]] = mapped_column(String(12))
    asset_issuer: Mapped[Optional[str]] = mapped_column(String(56))
    fee: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    operation_type: Mapped[Optional[str]] = mapped_column(String(32))
    successful: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    memo_type: Mapped[Optional[str]] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(nullable=False)

    __table_args__ = (
        Index("ix_api_transactions_source_created_at", "source_account", "created_at"),
        Index("ix_api_transactions_dest_created_at", "destination_account", "created_at"),
        Index("ix_api_transactions_ledger", "ledger_sequence"),
    )


# ---------------------------------------------------------------------------
# FraudAlert
# ---------------------------------------------------------------------------

class FraudAlert(Base):
    """Anomaly detection result produced by the fraud scoring pipeline."""

    __tablename__ = "api_fraud_alerts"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(String(56), nullable=False)
    pattern: Mapped[Optional[str]] = mapped_column(String(64))   # e.g. sybil_cluster
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)  # low/medium/high
    description: Mapped[Optional[str]] = mapped_column(Text)
    detected_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    __table_args__ = (
        Index("ix_api_fraud_alerts_account_id", "account_id"),
        Index("ix_api_fraud_alerts_detected_at", "detected_at"),
        Index("ix_api_fraud_alerts_risk_level", "risk_level"),
        Index("ix_api_fraud_alerts_resolved", "resolved"),
    )

    @staticmethod
    def risk_level_for_score(score: float) -> str:
        if score >= 0.8:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# LoyaltyPoints
# ---------------------------------------------------------------------------

class LoyaltyPoints(Base):
    """Points balance per account."""

    __tablename__ = "loyalty_points"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(String(56), nullable=False, unique=True)
    balance: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    tier: Mapped[str] = mapped_column(String(32), nullable=False, server_default="bronze")
    multiplier: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_loyalty_points_account_id", "account_id"),
    )


# ---------------------------------------------------------------------------
# PointsTransaction
# ---------------------------------------------------------------------------

class PointsTransaction(Base):
    """Earn / redeem / adjust record for loyalty points."""

    __tablename__ = "points_transactions"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(String(56), nullable=False)
    type: Mapped[str] = mapped_column(String(16), nullable=False)   # earn|redeem|adjust
    points: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(128))
    note: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_points_transactions_account_id", "account_id"),
        Index("ix_points_transactions_created_at", "created_at"),
    )


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry(Base):
    """Registered model version for the model registry (issue #257)."""

    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    owner: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True
    )
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql")
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="inactive"
    )  # inactive | active | deprecated
    parent_id: Mapped[Optional[int]] = mapped_column(
        _ID, nullable=True
    )  # Lineage: parent model version id
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_model_registry_name_version", "name", "version", unique=True),
        Index("ix_model_registry_status", "status"),
        Index("ix_model_registry_parent_id", "parent_id"),
    )


# ---------------------------------------------------------------------------
# Auth (issue #240)
# ---------------------------------------------------------------------------

class User(Base):
    """Dashboard/API user for JWT authentication."""

    __tablename__ = "api_users"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    scopes: Mapped[Optional[list]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=False, server_default="[]"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())


class ApiKey(Base):
    """Machine-to-machine API key."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(_ID, nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    scopes: Mapped[Optional[list]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=False, server_default="[]"
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column()
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_api_keys_user_id", "user_id"),
        Index("ix_api_keys_key_hash", "key_hash"),
    )


# ---------------------------------------------------------------------------
# Mentorship (Contributors)
# ---------------------------------------------------------------------------

class Mentor(Base):
    """Mentor profile in the mentorship program."""

    __tablename__ = "mentors"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(_ID, nullable=False, unique=True)
    github_username: Mapped[str] = mapped_column(String(128), nullable=False)
    bio: Mapped[Optional[str]] = mapped_column(Text)
    skills: Mapped[Optional[list]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=False, server_default="[]"
    )  # e.g., ["ML", "Data Science", "Python"]
    years_experience: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    preferred_session_day: Mapped[Optional[str]] = mapped_column(
        String(16)
    )  # e.g., "Monday", "Flexible"
    max_mentees: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3")
    is_available: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_mentors_github_username", "github_username"),
        Index("ix_mentors_is_available", "is_available"),
    )


class Mentee(Base):
    """Mentee profile seeking mentorship."""

    __tablename__ = "mentees"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(_ID, nullable=False, unique=True)
    github_username: Mapped[str] = mapped_column(String(128), nullable=False)
    bio: Mapped[Optional[str]] = mapped_column(Text)
    learning_interests: Mapped[Optional[list]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=False, server_default="[]"
    )  # e.g., ["ML", "Python"]
    years_experience: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    preferred_session_day: Mapped[Optional[str]] = mapped_column(
        String(16)
    )  # e.g., "Wednesday", "Flexible"
    goals: Mapped[Optional[str]] = mapped_column(Text)  # mentorship goals/objectives
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_mentees_github_username", "github_username"),
    )


class Mentorship(Base):
    """Active mentorship relationship between mentor and mentee."""

    __tablename__ = "mentorships"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    mentor_id: Mapped[int] = mapped_column(_ID, nullable=False)
    mentee_id: Mapped[int] = mapped_column(_ID, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="active"
    )  # active|paused|completed
    match_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1 compatibility score
    started_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    ended_at: Mapped[Optional[datetime]] = mapped_column()
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_mentorships_mentor_id", "mentor_id"),
        Index("ix_mentorships_mentee_id", "mentee_id"),
        Index("ix_mentorships_status", "status"),
    )


class MentorshipSession(Base):
    """Record of a single mentorship session."""

    __tablename__ = "mentorship_sessions"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    mentorship_id: Mapped[int] = mapped_column(_ID, nullable=False)
    session_date: Mapped[datetime] = mapped_column(nullable=False)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    topic: Mapped[str] = mapped_column(String(256), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_mentorship_sessions_mentorship_id", "mentorship_id"),
        Index("ix_mentorship_sessions_session_date", "session_date"),
    )


class MentorshipFeedback(Base):
    """Feedback from mentor or mentee after a session."""

    __tablename__ = "mentorship_feedback"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(_ID, nullable=False)
    mentorship_id: Mapped[int] = mapped_column(_ID, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5 stars
    feedback_text: Mapped[Optional[str]] = mapped_column(Text)
    is_mentor_feedback: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_mentorship_feedback_session_id", "session_id"),
        Index("ix_mentorship_feedback_mentorship_id", "mentorship_id"),
    )


# ---------------------------------------------------------------------------
# Notifications (Contributors)
# ---------------------------------------------------------------------------

class Notification(Base):
    """User notification record."""

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(_ID, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text)
    link: Mapped[Optional[str]] = mapped_column(String(512))
    actor: Mapped[Optional[str]] = mapped_column(String(128))
    is_read: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_notifications_user_id", "user_id"),
        Index("ix_notifications_is_read", "is_read"),
        Index("ix_notifications_created_at", "created_at"),
    )


class NotificationPreference(Base):
    """User notification preferences."""

    __tablename__ = "notification_preferences"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(_ID, nullable=False, unique=True)
    email_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    slack_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    discord_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    pr_comments: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    pr_mentions: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    issue_comments: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    issue_mentions: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    review_requests: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    digest_frequency: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="weekly"
    )
    slack_webhook_url: Mapped[Optional[str]] = mapped_column(Text)
    discord_webhook_url: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_notification_preferences_user_id", "user_id"),
    )


class NotificationLog(Base):
    """Log of notification deliveries for auditing."""

    __tablename__ = "notification_logs"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    notification_id: Mapped[int] = mapped_column(_ID, nullable=False)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    delivered_at: Mapped[Optional[datetime]] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_notification_logs_notification_id", "notification_id"),
        Index("ix_notification_logs_status", "status"),
    )


# ---------------------------------------------------------------------------
# FAQ (issue #307)
# ---------------------------------------------------------------------------

class FAQ(Base):
    """FAQ item with categorization and search support."""

    __tablename__ = "faqs"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    is_published: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_faqs_category", "category"),
        Index("ix_faqs_is_published", "is_published"),
        Index("ix_faqs_order", "order"),
    )


class FAQFeedback(Base):
    """User feedback on FAQ helpfulness."""

    __tablename__ = "faq_feedback"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    faq_id: Mapped[int] = mapped_column(_ID, nullable=False)
    is_helpful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    user_comment: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_faq_feedback_faq_id", "faq_id"),
    )


class FAQSuggestion(Base):
    """User-submitted FAQ suggestions."""

    __tablename__ = "faq_suggestions"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    suggested_answer: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending"
    )  # pending|approved|rejected
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_faq_suggestions_status", "status"),
    )


# ---------------------------------------------------------------------------
# Audit Log (issue #332)
# ---------------------------------------------------------------------------

class AuditLog(Base):
    """Immutable audit log for sensitive API operations."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    user_id: Mapped[Optional[int]] = mapped_column(_ID)  # NULL for system events
    username: Mapped[Optional[str]] = mapped_column(String(64))
    auth_type: Mapped[Optional[str]] = mapped_column(String(16))  # jwt|api_key|none
    action: Mapped[str] = mapped_column(String(64), nullable=False)  # login|logout|create|update|delete
    resource_type: Mapped[Optional[str]] = mapped_column(String(64))  # user|account|transaction|model
    resource_id: Mapped[Optional[str]] = mapped_column(String(256))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))  # IPv6 support
    user_agent: Mapped[Optional[str]] = mapped_column(String(512))
    request_path: Mapped[Optional[str]] = mapped_column(String(512))
    request_method: Mapped[Optional[str]] = mapped_column(String(8))
    status_code: Mapped[Optional[int]] = mapped_column(Integer)
    details: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql")
    )

    __table_args__ = (
        Index("ix_audit_logs_timestamp", "timestamp"),
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_resource_type", "resource_type"),
        Index("ix_audit_logs_timestamp_action", "timestamp", "action"),
    )


class SupportTicket(Base):
    """Support ticket generated from a contact-form submission (issue #305)."""

    __tablename__ = "support_tickets"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    reference: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    subject: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="open")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_support_tickets_reference", "reference"),
        Index("ix_support_tickets_email", "email"),
        Index("ix_support_tickets_status", "status"),
    )


class Feedback(Base):
    """In-app user feedback: bug reports, feature requests, general comments (#308)."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(16), nullable=False)  # bug|feature|general
    message: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(254))
    # Optional screenshot stored as a data URL (data:image/png;base64,...).
    screenshot: Mapped[Optional[str]] = mapped_column(Text)
    # open|planned|in_progress|completed|declined — drives the public roadmap.
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="open")
    github_issue_url: Mapped[Optional[str]] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_feedback_category", "category"),
        Index("ix_feedback_status", "status"),
        Index("ix_feedback_created_at", "created_at"),
    )


class LLMFeedback(Base):
    """User and expert feedback for LLM outputs (#402)."""

    __tablename__ = "llm_feedback"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    feature: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    output: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text)
    user_id: Mapped[Optional[str]] = mapped_column(String(128))
    is_expert: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")
    expert_weight: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_llm_feedback_feature", "feature"),
        Index("ix_llm_feedback_created_at", "created_at"),
    )


class LLMComplianceLog(Base):
    """Compliance and audit log for all LLM interactions (issue #412)."""

    __tablename__ = "llm_compliance_logs"

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(_ID)
    username: Mapped[Optional[str]] = mapped_column(String(128))
    interaction_type: Mapped[str] = mapped_column(String(32), nullable=False)
    feature: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_redacted: Mapped[str] = mapped_column(Text, nullable=False)
    response_redacted: Mapped[str] = mapped_column(Text, nullable=False)
    model_used: Mapped[Optional[str]] = mapped_column(String(64))
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    pii_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    pii_types: Mapped[Optional[dict]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_llm_compliance_logs_user_id", "user_id"),
        Index("ix_llm_compliance_logs_created_at", "created_at"),
        Index("ix_llm_compliance_logs_interaction_type", "interaction_type"),
        Index("ix_llm_compliance_logs_pii_detected", "pii_detected"),
    )


# Backward-compatible aliases removed — use ApiAccount / ApiTransaction to avoid
# SQLAlchemy mapper name collisions with astroml.db.schema.
