"""Fraud Detection API — Issue #254.

Endpoints:
  POST /api/v1/fraud/score   — real-time anomaly scoring
  GET  /api/v1/fraud/alerts  — paginated fraud alerts
  GET  /api/v1/fraud/stats   — aggregated fraud statistics
  GET  /api/v1/fraud/{id}/explanation — LLM explanation for fraud alert
  GET  /api/v1/fraud/alerts/prioritized — prioritized, deduplicated alerts

Model loading
-------------
Models are loaded lazily on first request and cached in module-level state.
The active model version from the registry takes precedence over
``MODEL_CHECKPOINT_PATH`` when set.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import cast, func, select, Date
from sqlalchemy.orm import Session

from api.database import get_sync_db
from api.models.orm import FraudAlert, ApiTransaction
from api.schemas import (
    FraudAlertOut,
    FraudAlertsResponse,
    FraudStatsResponse,
    FraudExplanationOut,
    RiskPoint,
    ScoreRequest,
    ScoreResponse,
    PrioritizedAlertsResponse,
    PrioritizedAlertOut,
    TransactionSummaryOut,
)
from api.services.alert_prioritization import alert_prioritizer
from api.services.scorer import invalidate_scorer_cache, load_scorer
from api.graphql import publish_fraud_alert
from astroml.llm.explainer import FraudExplainer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/fraud", tags=["fraud"])
explainer = FraudExplainer()


def _get_scorer():
    """Load and cache the InductiveAnomalyScorer. Returns None if unavailable."""
    return load_scorer()


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/score", response_model=ScoreResponse)
async def score_accounts(body: ScoreRequest):
    """Score up to 50 accounts for anomaly/fraud risk."""
    scorer = _get_scorer()
    if scorer is None:
        scores = {acc: 0.0 for acc in body.accounts}
        return ScoreResponse(scores=scores)

    ref_time = datetime.now(timezone.utc).timestamp()
    try:
        edges = [e.model_dump() for e in body.edges]
        scores = scorer.score_new_accounts(
            edges=edges,
            account_ids=body.accounts,
            ref_time=ref_time,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Scoring failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Scoring service temporarily unavailable") from exc

    return ScoreResponse(scores=scores)


@router.get("/alerts", response_model=FraudAlertsResponse)
def get_fraud_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    risk_level: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
    db: Session = Depends(get_sync_db),
):
    """Return paginated fraud alerts, optionally filtered by risk level."""
    q = select(FraudAlert)
    if risk_level:
        q = q.where(FraudAlert.risk_level == risk_level)
    q = q.order_by(FraudAlert.detected_at.desc())

    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.offset((page - 1) * page_size).limit(page_size)).all()
    return FraudAlertsResponse(
        data=[FraudAlertOut.model_validate(r) for r in rows],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.get("/stats", response_model=FraudStatsResponse)
def get_fraud_stats(db: Session = Depends(get_sync_db)):
    """Return aggregated fraud statistics."""
    def _count(level: str) -> int:
        return db.scalar(
            select(func.count(FraudAlert.id)).where(FraudAlert.risk_level == level)
        ) or 0

    total = db.scalar(select(func.count(FraudAlert.id))) or 0
    recent = db.scalars(
        select(FraudAlert).order_by(FraudAlert.detected_at.desc()).limit(10)
    ).all()

    daily = db.execute(
        select(
            cast(FraudAlert.detected_at, Date).label("day"),
            func.avg(FraudAlert.risk_score).label("avg_score"),
        )
        .group_by("day")
        .order_by("day")
        .limit(14)
    ).all()

    return FraudStatsResponse(
        total_alerts=total,
        high_risk=_count("high"),
        medium_risk=_count("medium"),
        low_risk=_count("low"),
        recent_alerts=[FraudAlertOut.model_validate(r) for r in recent],
        risk_over_time=[
            RiskPoint(date=str(row.day), score=round(float(row.avg_score), 4))
            for row in daily
        ],
    )


@router.get("/{id}/explanation", response_model=FraudExplanationOut)
def get_fraud_explanation(id: int, db: Session = Depends(get_sync_db)):
    """Generate an explanation for a fraud alert, citing evidence."""
    alert = db.get(FraudAlert, id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    # Fetch recent transactions as evidence
    txs = db.scalars(
        select(ApiTransaction)
        .where(ApiTransaction.source_account == alert.account_id)
        .order_by(ApiTransaction.created_at.desc())
        .limit(10)
    ).all()
    
    tx_dicts = [
        {
            "hash": tx.hash,
            "amount": float(tx.amount) if tx.amount else 0.0,
            "asset_code": tx.asset_code or "XLM",
            "destination_account": tx.destination_account,
            "ledger_sequence": tx.ledger_sequence
        } for tx in txs
    ]
    
    start_time = time.time()
    
    explanation = explainer.generate_explanation(
        alert_id=alert.id,
        account_id=alert.account_id,
        pattern=alert.pattern or "unknown",
        score=alert.risk_score,
        transactions=tx_dicts
    )
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000.0
    
    return FraudExplanationOut(
        alert_id=alert.id,
        explanation=explanation,
        generated_in_ms=elapsed_ms,
        cached=elapsed_ms < 100.0  # Simple heuristic for now
    )


@router.get("/alerts/prioritized", response_model=PrioritizedAlertsResponse)
@router.get("/api/v1/alerts/prioritized", response_model=PrioritizedAlertsResponse, include_in_schema=False)
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


# ─── Fraud Alert Creation ────────────────────────────────────────────────────

async def create_fraud_alert(
    account_id: str,
    risk_score: float,
    pattern: Optional[str] = None,
    description: Optional[str] = None,
    db: Session = Depends(get_sync_db),
) -> FraudAlert:
    """
    Create a new fraud alert and publish to GraphQL subscriptions.
    
    Args:
        account_id: The account ID associated with the fraud alert
        risk_score: The risk score (0.0 to 1.0)
        pattern: Optional pattern identifier (e.g., sybil_cluster)
        description: Optional description of the alert
        db: Database session
        
    Returns:
        The created FraudAlert instance
    """
    risk_level = FraudAlert.risk_level_for_score(risk_score)
    
    alert = FraudAlert(
        account_id=account_id,
        pattern=pattern,
        risk_score=risk_score,
        risk_level=risk_level,
        description=description,
    )
    
    db.add(alert)
    db.commit()
    db.refresh(alert)
    
    # Publish to GraphQL subscription
    await publish_fraud_alert({
        "id": alert.id,
        "account_id": alert.account_id,
        "pattern": alert.pattern,
        "risk_score": alert.risk_score,
        "risk_level": alert.risk_level,
        "description": alert.description,
        "detected_at": alert.detected_at,
    })
    
    logger.info(
        "Fraud alert created: id=%d, account=%s, risk_level=%s",
        alert.id,
        alert.account_id,
        alert.risk_level,
    )
    
    return alert


# Re-export for model activation hook
__all__ = ["router", "invalidate_scorer_cache", "create_fraud_alert"]