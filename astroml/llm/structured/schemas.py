"""Common Pydantic schema definitions for structured outputs."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class FraudExplanation(BaseModel):
    """Fraud alert explanation schema."""

    account_id: str = Field(..., description="Stellar account ID")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score between 0 and 1")
    reasons: List[str] = Field(..., min_length=1, description="List of fraud indicators")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the assessment")
    recommended_action: Optional[str] = Field(None, description="Suggested next steps")

    @field_validator("account_id")
    @classmethod
    def validate_account_id(cls, v: str) -> str:
        if not v or len(v) != 56:
            raise ValueError("Account ID must be 56 characters")
        return v


class ModelPrediction(BaseModel):
    """Model prediction explanation schema."""

    prediction: str = Field(..., description="Prediction label or value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    top_features: List[str] = Field(..., description="Most important features")
    feature_contributions: dict = Field(default_factory=dict, description="Feature importance scores")
    explanation: str = Field(..., description="Human-readable explanation")


class AnomalyAlert(BaseModel):
    """Anomaly detection result schema."""

    transaction_id: str = Field(..., description="Transaction hash")
    anomaly_score: float = Field(..., ge=0.0, description="Anomaly score")
    anomaly_type: str = Field(..., description="Type of anomaly detected")
    context: str = Field(..., description="Historical context")
    graph_patterns: List[str] = Field(default_factory=list, description="Relevant graph patterns")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AccountSummary(BaseModel):
    """Account activity summary schema."""

    account_id: str
    transaction_count: int = Field(..., ge=0)
    total_volume: float = Field(..., ge=0.0)
    active_days: int = Field(..., ge=0)
    risk_indicators: List[str] = Field(default_factory=list)
    summary: str = Field(..., description="Natural language summary")
