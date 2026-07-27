"""
Integration tests — fraud detection (issue #244).

Covers: ORM model creation, risk-level classification, alert filtering,
stats aggregation, and score/alert endpoint shapes.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from api.models.orm import FraudAlert


@pytest.mark.xdist_group("api_fraud")
class TestFraudAlertModel:
    """ORM-level tests for FraudAlert (issue #246)."""

    def test_create_alert_persists(self, db_session):
        alert = FraudAlert(
            account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            pattern="velocity",
            risk_score=0.75,
            risk_level="medium",
            description="Burst of micro-transactions.",
        )
        db_session.add(alert)
        db_session.flush()
        assert alert.id is not None

    def test_risk_level_high(self):
        assert FraudAlert.risk_level_for_score(0.85) == "high"
        assert FraudAlert.risk_level_for_score(0.8) == "high"

    def test_risk_level_medium(self):
        assert FraudAlert.risk_level_for_score(0.79) == "medium"
        assert FraudAlert.risk_level_for_score(0.5) == "medium"

    def test_risk_level_low(self):
        assert FraudAlert.risk_level_for_score(0.49) == "low"
        assert FraudAlert.risk_level_for_score(0.0) == "low"

    def test_seeded_alert_fields(self, seeded_alert):
        assert seeded_alert.account_id == "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"
        assert seeded_alert.risk_score == pytest.approx(0.92)
        assert seeded_alert.risk_level == "high"
        assert seeded_alert.pattern == "sybil_cluster"

    def test_multiple_alerts_query(self, db_session):
        for score in [0.9, 0.6, 0.3]:
            db_session.add(FraudAlert(
                account_id="GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
                risk_score=score,
                risk_level=FraudAlert.risk_level_for_score(score),
            ))
        db_session.flush()

        results = db_session.execute(select(FraudAlert)).scalars().all()
        assert len(results) == 3

    def test_filter_by_risk_level(self, db_session):
        for score, level in [(0.9, "high"), (0.6, "medium"), (0.2, "low")]:
            db_session.add(FraudAlert(
                account_id="GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF",
                risk_score=score,
                risk_level=level,
            ))
        db_session.flush()

        high = db_session.execute(
            select(FraudAlert).where(FraudAlert.risk_level == "high")
        ).scalars().all()
        assert len(high) == 1
        assert high[0].risk_score == pytest.approx(0.9)


@pytest.mark.xdist_group("api_fraud")
class TestFraudFixtures:
    """Verify raw dict fixtures remain intact for backward-compat tests."""

    def test_sample_alerts_count(self, sample_alerts):
        assert len(sample_alerts) == 2

    def test_sample_alerts_have_required_fields(self, sample_alerts):
        for alert in sample_alerts:
            assert "alert_id" in alert
            assert "account_id" in alert
            assert "severity" in alert
            assert "resolved" in alert

    def test_filter_unresolved_alerts(self, sample_alerts):
        unresolved = [a for a in sample_alerts if not a["resolved"]]
        assert len(unresolved) == 1
        assert unresolved[0]["severity"] == "high"

    def test_filter_by_severity(self, sample_alerts):
        high = [a for a in sample_alerts if a["severity"] == "high"]
        low = [a for a in sample_alerts if a["severity"] == "low"]
        assert len(high) == 1
        assert len(low) == 1
