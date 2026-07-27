"""SQLAlchemy models for LLM cost tracking and budgeting."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column
from astroml.db.schema import Base


class LLMCostRecord(Base):
    """Record of a single LLM API request cost and token usage."""

    __tablename__ = "llm_cost_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    team_id: Mapped[Optional[str]] = mapped_column(String(255), index=True, nullable=True)
    feature: Mapped[str] = mapped_column(String(100), index=True, nullable=False)  # e.g., 'chatbot', 'RAG', 'explanations'
    model_name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)  # e.g., 'gpt-4', 'gpt-3.5', 'claude'
    prompt_template: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True, nullable=False)


class LLMBudget(Base):
    """Budget configuration and spending tracking for a user or team."""

    __tablename__ = "llm_budgets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)  # user_id or team_id
    scope: Mapped[str] = mapped_column(String(50), default="user", nullable=False)  # 'user' or 'team'
    tier: Mapped[str] = mapped_column(String(50), default="free", nullable=False)  # 'free', 'pro', 'enterprise'
    limit_amount: Mapped[float] = mapped_column(Float, default=10.0, nullable=False)  # standard tier limits: free=10, pro=100
    current_spend: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    period: Mapped[str] = mapped_column(String(50), default="monthly", nullable=False)  # 'daily', 'monthly'
    last_alert_threshold: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # e.g. 0.5, 0.8, 1.0 to prevent duplicate alerts
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    emergency_override: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # admin override
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
