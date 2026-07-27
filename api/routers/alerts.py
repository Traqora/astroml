"""Alerts API router (issue XXX)."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.database import get_sync_db
from api.models.orm import ApiTransaction, FraudAlert
from api.schemas import (
    FraudAlertOut,
    FraudAlertsResponse,
    FraudExplanationOut,
    PrioritizedAlertOut,
    PrioritizedAlertsResponse,
    TransactionSummaryOut,
    # Predictive Alerts schemas
    BehavioralBaseline,
    BehavioralBaselineResponse,
    DeviationAlert,
    PredictiveAlertRequest,
    PredictiveAlertResponse,
    AlertGenerationRequest,
    AlertGenerationResponse,
)
from api.services.alert_prioritization import alert_prioritizer
from api.services.predictive_alerts import predictive_alert_service

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get("/prioritized", response_model=PrioritizedAlertsResponse)
def get_prioritized_alerts(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_sync_db),
):
    """Get prioritized, deduplicated fraud alerts."""
    # Fetch recent alerts
    alerts = db.scalars(
        select(FraudAlert)
        .order_by(FraudAlert.detected_at.desc())
        .limit(limit * 2)  # Fetch extra to account for deduplication
    ).all()

    total_processed = len(alerts)

    # Process alerts
    processed, reduction_pct = alert_prioritizer.process_alerts(db, alerts)

    # Convert to output model
    data = []
    for enriched in processed:
        data.append(
            PrioritizedAlertOut(
                id=enriched.alert.id,
                account_id=enriched.alert.account_id,
                pattern=enriched.alert.pattern,
                risk_score=enriched.alert.risk_score,
                risk_level=enriched.alert.risk_level,
                priority_score=enriched.priority_score,
                priority_level=enriched.priority_level,
                explanation=enriched.explanation,
                detected_at=enriched.alert.detected_at,
                recent_transactions=[
                    TransactionSummaryOut(
                        hash=tx["hash"],
                        amount=tx["amount"],
                        asset_code=tx["asset_code"],
                        destination_account=tx["destination_account"],
                        created_at=tx["created_at"],
                    ) for tx in enriched.recent_transactions
                ],
                account_activity_score=enriched.account_activity_score,
                is_duplicate=enriched.is_duplicate,
                duplicate_of=enriched.duplicate_of,
            )
        )

    return PrioritizedAlertsResponse(
        data=data[:limit],
        deduplication_reduction_pct=reduction_pct,
        total_processed=total_processed,
        total_remaining=len(data),
    )


@router.get("/predictive", response_model=PredictiveAlertResponse)
def get_predictive_alerts(
    account_id: str,
    lookback_days: int = Query(30, ge=1, le=365),
    metrics: Optional[List[str]] = Query(None),
    sensitivity: str = Query("medium", pattern="^(low|medium|high)$"),
    db: Session = Depends(get_sync_db),
):
    """
    Generate predictive alerts for account behavior changes.
    
    Analyzes historical transaction data to establish behavioral baselines
    and detects significant deviations that may indicate unusual activity.
    """
    # Validate account exists (basic check)
    if not account_id or len(account_id) < 10:  # Stellar addresses are typically longer
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid account ID format"
        )
    
    # Use the predictive alert service
    result = predictive_alert_service.generate_predictive_alerts(
        account_id=account_id,
        lookback_days=lookback_days,
        metrics=metrics,
        sensitivity=sensitivity
    )
    
    # Check if service returned an error
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )
    
    # Check for informational messages (no alerts generated)
    if "message" in result and "alerts" not in result:
        # Return empty alerts list with metadata
        return PredictiveAlertResponse(
            alerts=[],
            baselines_used=[],
            generated_at=datetime.utcnow(),
            total_analyzed=0
        )
    
    # Convert to response model
    return PredictiveAlertResponse(
        alerts=[DeviationAlert(**alert) for alert in result.get("alerts", [])],
        baselines_used=[
            BehavioralBaseline(**baseline) 
            for baseline in result.get("baselines_used", [])
        ],
        generated_at=datetime.fromisoformat(result["generated_at"]) if isinstance(result.get("generated_at"), str) else result.get("generated_at", datetime.utcnow()),
        total_analyzed=result.get("total_analyzed", 0)
    )


@router.post("/generate-explanations", response_model=AlertGenerationResponse)
def generate_alert_explanations(
    request: AlertGenerationRequest,
):
    """
    Generate natural language explanations for detected anomalies.
    """
    try:
        # Use the alert generator from the predictive service
        explanations = []
        for deviation in request.deviations:
            # Convert Pydantic model back to dict for the generator
            deviation_dict = deviation.dict()
            explanation_result = predictive_alert_service.alert_generator.generate_explanation(
                deviation.alert_id,
                deviation.account_id,
                {
                    "metric_name": deviation.metric_name,
                    "current_value": deviation.current_value,
                    "expected_value": sum(deviation.expected_range) / 2 if deviation.expected_range else 0,
                    "deviation_score": deviation.deviation_score,
                    "severity": deviation.severity
                }
            )
            explanations.append(explanation_result.get("explanation", "No explanation available"))
        
        return AlertGenerationResponse(
            alerts=request.deviations,
            explanations=explanations,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate explanations"
        )


@router.get("/predictive/status")
def get_predictive_service_status():
    """Get status of the predictive alerts service."""
    return predictive_alert_service.get_service_status()