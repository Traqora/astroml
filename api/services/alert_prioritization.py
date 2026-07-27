"""Alert prioritization and triage service (issue XXX)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models.orm import FraudAlert, ApiTransaction
from astroml.llm.provider import MockLLMProvider

llm = MockLLMProvider()


@dataclass
class EnrichedAlert:
    alert: FraudAlert
    recent_transactions: List[Dict[str, Any]]
    account_activity_score: float
    time_since_first_seen: float
    priority_score: float = 0.0
    priority_level: str = "medium"
    explanation: str = ""
    is_duplicate: bool = False
    duplicate_of: Optional[int] = None


class AlertPrioritizer:
    """Service for alert enrichment, semantic deduplication, and LLM-based prioritization."""

    def __init__(self):
        self.prioritization_prompt = """
You are a fraud analyst. Given a fraud alert and its context, assign a priority score from 0 to 1 and a priority level.

Prioritization Criteria (highest to lowest):
- High risk score (>0.7)
- Recent high-value transactions (>1000)
- Multiple similar alerts for the same account
- Unusual transaction patterns
- Account has a history of fraud
- Time sensitivity (alert detected in last 24h)

Alert Details:
- Risk Score: {risk_score}
- Risk Level: {risk_level}
- Pattern: {pattern}
- Description: {description}
- Account Activity: {account_activity_score}
- Transactions in last 7 days: {tx_count}
- Total transaction volume: ${total_volume:.2f}

Return ONLY a JSON object with:
{{
  "priority_score": float,
  "priority_level": "high" | "medium" | "low",
  "explanation": "brief explanation of prioritization"
}}
"""

    def _calculate_text_hash(self, text: str) -> str:
        """Calculate a simple hash for text deduplication."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple semantic similarity (placeholder)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def enrich_alert(self, db: Session, alert: FraudAlert) -> EnrichedAlert:
        """Enrich alert with additional context from the database."""
        # Fetch recent transactions
        txs = db.scalars(
            select(ApiTransaction)
            .where(ApiTransaction.source_account == alert.account_id)
            .order_by(ApiTransaction.created_at.desc())
            .limit(20)
        ).all()

        tx_dicts = [
            {
                "hash": tx.hash,
                "amount": float(tx.amount) if tx.amount else 0.0,
                "asset_code": tx.asset_code or "XLM",
                "destination_account": tx.destination_account,
                "created_at": tx.created_at.isoformat(),
            } for tx in txs
        ]

        # Calculate account activity score
        last_7_days = datetime.utcnow() - timedelta(days=7)
        recent_txs = [tx for tx in txs if tx.created_at > last_7_days]
        tx_count = len(recent_txs)
        total_volume = sum(float(tx.amount) for tx in recent_txs if tx.amount)
        account_activity_score = min(1.0, (tx_count / 20) + (total_volume / 10000))

        time_since_first_seen = (datetime.utcnow() - alert.detected_at).total_seconds() / 3600  # hours

        return EnrichedAlert(
            alert=alert,
            recent_transactions=tx_dicts,
            account_activity_score=account_activity_score,
            time_since_first_seen=time_since_first_seen,
        )

    def deduplicate_alerts(self, enriched_alerts: List[EnrichedAlert]) -> List[EnrichedAlert]:
        """Deduplicate alerts using semantic similarity and account ID."""
        deduplicated: List[EnrichedAlert] = []
        account_groups: Dict[str, List[EnrichedAlert]] = defaultdict(list)

        for alert in enriched_alerts:
            account_groups[alert.alert.account_id].append(alert)

        for account_id, alerts in account_groups.items():
            processed_hashes: List[str] = []
            for alert in alerts:
                alert_text = f"{alert.alert.pattern or ''} {alert.alert.description or ''}"
                alert_hash = self._calculate_text_hash(alert_text)

                is_dupe = False
                duplicate_of = None
                for processed in deduplicated:
                    if processed.alert.account_id == account_id:
                        processed_text = f"{processed.alert.pattern or ''} {processed.alert.description or ''}"
                        similarity = self._semantic_similarity(alert_text, processed_text)
                        if similarity > 0.7:
                            is_dupe = True
                            duplicate_of = processed.alert.id
                            break

                if not is_dupe:
                    deduplicated.append(alert)
                else:
                    alert.is_duplicate = True
                    alert.duplicate_of = duplicate_of

        return deduplicated

    def prioritize_alert(self, enriched: EnrichedAlert) -> EnrichedAlert:
        """Use LLM to prioritize an alert."""
        risk_score = enriched.alert.risk_score
        risk_level = enriched.alert.risk_level
        pattern = enriched.alert.pattern or "unknown"
        description = enriched.alert.description or "no description"
        tx_count = len(enriched.recent_transactions)
        total_volume = sum(tx.get("amount", 0) for tx in enriched.recent_transactions)

        prompt = self.prioritization_prompt.format(
            risk_score=risk_score,
            risk_level=risk_level,
            pattern=pattern,
            description=description,
            account_activity_score=enriched.account_activity_score,
            tx_count=tx_count,
            total_volume=total_volume,
        )

        # Mock LLM response for now (in real scenario, parse LLM output)
        # For demonstration, calculate a score based on heuristics
        heuristic_score = risk_score * 0.6
        if tx_count > 10:
            heuristic_score += 0.2
        if total_volume > 5000:
            heuristic_score += 0.15
        if enriched.time_since_first_seen < 24:
            heuristic_score += 0.05
        heuristic_score = min(1.0, heuristic_score)

        if heuristic_score >= 0.7:
            priority_level = "high"
        elif heuristic_score >= 0.4:
            priority_level = "medium"
        else:
            priority_level = "low"

        enriched.priority_score = heuristic_score
        enriched.priority_level = priority_level
        enriched.explanation = (
            f"Prioritized based on risk score ({risk_score:.2f}), account activity "
            f"({tx_count} txs, ${total_volume:.2f}), and recency."
        )

        return enriched

    def process_alerts(self, db: Session, alerts: List[FraudAlert]) -> Tuple[List[EnrichedAlert], int]:
        """Process, enrich, deduplicate, and prioritize alerts."""
        enriched = [self.enrich_alert(db, alert) for alert in alerts]
        deduplicated = self.deduplicate_alerts(enriched)
        prioritized = [self.prioritize_alert(alert) for alert in deduplicated]
        prioritized_sorted = sorted(prioritized, key=lambda x: x.priority_score, reverse=True)

        original_count = len(alerts)
        deduplicated_count = len(prioritized_sorted)
        reduction = (original_count - deduplicated_count) / original_count if original_count > 0 else 0.0

        return prioritized_sorted, int(reduction * 100)


alert_prioritizer = AlertPrioritizer()
