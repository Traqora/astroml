"""Tests for ORM models — issue #231.

Verifies that all required models can be created, persisted, and queried using
the SQLite in-memory session provided by the conftest fixtures.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

# Ensure all ORM models are imported so their tables are registered on Base.metadata
import api.models.orm  # noqa: F401  (side-effect import registers tables)
from api.models.orm import (
    Account,
    FraudAlert,
    LoyaltyPoints,
    ModelRegistry,
    PointsTransaction,
    Transaction,
)


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class TestAccountModel:
    def test_create_and_read(self, db_session):
        acc = Account(
            public_key="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
            balance=100.5,
            home_domain="example.com",
        )
        db_session.add(acc)
        db_session.flush()
        db_session.refresh(acc)

        assert acc.id is not None
        assert acc.public_key == "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"
        assert float(acc.balance) == 100.5
        assert acc.home_domain == "example.com"

    def test_public_key_unique_constraint(self, db_session):
        key = "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT"
        db_session.add(Account(public_key=key))
        db_session.flush()

        db_session.add(Account(public_key=key))
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_optional_fields_default_to_none(self, db_session):
        acc = Account(
            public_key="GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF"
        )
        db_session.add(acc)
        db_session.flush()
        db_session.refresh(acc)

        assert acc.first_seen is None
        assert acc.last_active is None
        assert acc.home_domain is None


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

class TestTransactionModel:
    def test_create_and_read(self, db_session):
        txn = Transaction(
            hash="abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
            ledger_sequence=1000,
            source_account="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            destination_account="GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            amount=500.0,
            asset_code="XLM",
            fee=100,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        db_session.add(txn)
        db_session.flush()
        db_session.refresh(txn)

        assert txn.hash == "abc123def456abc123def456abc123def456abc123def456abc123def456abc1"
        assert txn.ledger_sequence == 1000
        assert txn.source_account == "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"
        assert txn.fee == 100

    def test_hash_is_primary_key(self, db_session):
        """Duplicate hash should raise IntegrityError."""
        h = "deadbeef" * 8  # 64 chars
        db_session.add(
            Transaction(
                hash=h,
                ledger_sequence=1,
                source_account="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
                fee=0,
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        )
        db_session.flush()
        db_session.add(
            Transaction(
                hash=h,
                ledger_sequence=2,
                source_account="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
                fee=0,
                created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_destination_account_optional(self, db_session):
        txn = Transaction(
            hash="0" * 64,
            ledger_sequence=99,
            source_account="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            fee=0,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        db_session.add(txn)
        db_session.flush()
        db_session.refresh(txn)
        assert txn.destination_account is None


# ---------------------------------------------------------------------------
# FraudAlert
# ---------------------------------------------------------------------------

class TestFraudAlertModel:
    def test_create_and_read(self, db_session):
        alert = FraudAlert(
            account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            pattern="sybil_cluster",
            risk_score=0.92,
            risk_level="high",
            description="Suspicious burst of transactions.",
            detected_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        db_session.add(alert)
        db_session.flush()
        db_session.refresh(alert)

        assert alert.id is not None
        assert alert.risk_score == 0.92
        assert alert.risk_level == "high"
        assert alert.description == "Suspicious burst of transactions."

    def test_resolved_defaults_to_false(self, db_session):
        alert = FraudAlert(
            account_id="GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            risk_score=0.3,
            risk_level="low",
            detected_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        db_session.add(alert)
        db_session.flush()
        db_session.refresh(alert)

        assert alert.resolved is False

    def test_resolved_can_be_set_true(self, db_session):
        alert = FraudAlert(
            account_id="GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF",
            risk_score=0.7,
            risk_level="medium",
            detected_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            resolved=True,
        )
        db_session.add(alert)
        db_session.flush()
        db_session.refresh(alert)

        assert alert.resolved is True

    def test_risk_level_helper(self):
        assert FraudAlert.risk_level_for_score(0.9) == "high"
        assert FraudAlert.risk_level_for_score(0.6) == "medium"
        assert FraudAlert.risk_level_for_score(0.2) == "low"
        # boundary: exactly 0.8 is high
        assert FraudAlert.risk_level_for_score(0.8) == "high"
        # boundary: exactly 0.5 is medium
        assert FraudAlert.risk_level_for_score(0.5) == "medium"


# ---------------------------------------------------------------------------
# LoyaltyPoints
# ---------------------------------------------------------------------------

class TestLoyaltyPointsModel:
    def test_create_and_read(self, db_session):
        lp = LoyaltyPoints(
            account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            updated_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        db_session.add(lp)
        db_session.flush()
        db_session.refresh(lp)

        assert lp.id is not None
        assert lp.account_id == "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"

    def test_balance_defaults_to_zero(self, db_session):
        lp = LoyaltyPoints(
            account_id="GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            updated_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        db_session.add(lp)
        db_session.flush()
        db_session.refresh(lp)

        assert lp.balance == 0

    def test_tier_defaults_to_bronze(self, db_session):
        lp = LoyaltyPoints(
            account_id="GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF",
            updated_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        db_session.add(lp)
        db_session.flush()
        db_session.refresh(lp)

        assert lp.tier == "bronze"

    def test_account_id_unique_constraint(self, db_session):
        key = "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"
        db_session.add(LoyaltyPoints(account_id=key, updated_at=datetime(2024, 6, 1)))
        db_session.flush()
        db_session.add(LoyaltyPoints(account_id=key, updated_at=datetime(2024, 6, 2)))
        with pytest.raises(IntegrityError):
            db_session.flush()


# ---------------------------------------------------------------------------
# PointsTransaction
# ---------------------------------------------------------------------------

class TestPointsTransactionModel:
    def test_create_earn_record(self, db_session):
        pt = PointsTransaction(
            account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            type="earn",
            points=200,
            source="onboarding_bonus",
            note="Welcome bonus",
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        db_session.add(pt)
        db_session.flush()
        db_session.refresh(pt)

        assert pt.id is not None
        assert pt.type == "earn"
        assert pt.points == 200
        assert pt.source == "onboarding_bonus"

    def test_create_redeem_record(self, db_session):
        pt = PointsTransaction(
            account_id="GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            type="redeem",
            points=100,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        db_session.add(pt)
        db_session.flush()
        db_session.refresh(pt)

        assert pt.type == "redeem"
        assert pt.source is None
        assert pt.note is None

    def test_multiple_transactions_same_account(self, db_session):
        account_id = "GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF"
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)
        for i in range(3):
            db_session.add(
                PointsTransaction(
                    account_id=account_id,
                    type="earn",
                    points=50 * (i + 1),
                    created_at=now,
                )
            )
        db_session.flush()

        from sqlalchemy import select
        rows = db_session.scalars(
            select(PointsTransaction).where(PointsTransaction.account_id == account_id)
        ).all()
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class TestModelRegistryModel:
    def test_create_and_read(self, db_session):
        model = ModelRegistry(
            name="fraud_detector",
            version="v1.0.0",
            path="/path/to/model.pt",
            metrics={"auc": 0.92, "precision": 0.88, "recall": 0.85},
            status="inactive",
        )
        db_session.add(model)
        db_session.flush()
        db_session.refresh(model)

        assert model.id is not None
        assert model.name == "fraud_detector"
        assert model.version == "v1.0.0"
        assert model.path == "/path/to/model.pt"
        assert model.metrics == {"auc": 0.92, "precision": 0.88, "recall": 0.85}
        assert model.status == "inactive"
        assert model.parent_id is None

    def test_create_with_parent(self, db_session):
        parent = ModelRegistry(
            name="fraud_detector",
            version="v1.0.0",
            path="/path/to/parent.pt",
        )
        db_session.add(parent)
        db_session.flush()
        
        child = ModelRegistry(
            name="fraud_detector",
            version="v1.1.0",
            path="/path/to/child.pt",
            parent_id=parent.id,
        )
        db_session.add(child)
        db_session.flush()
        db_session.refresh(child)
        
        assert child.parent_id == parent.id

    def test_status_defaults_to_inactive(self, db_session):
        model = ModelRegistry(
            name="fraud_detector",
            version="v1.0.0",
            path="/path/to/model.pt",
        )
        db_session.add(model)
        db_session.flush()
        db_session.refresh(model)
        assert model.status == "inactive"

    def test_metrics_optional(self, db_session):
        model = ModelRegistry(
            name="fraud_detector",
            version="v1.0.0",
            path="/path/to/model.pt",
        )
        db_session.add(model)
        db_session.flush()
        db_session.refresh(model)
        assert model.metrics is None

    def test_name_version_unique_constraint(self, db_session):
        # Add first model
        db_session.add(ModelRegistry(
            name="fraud_detector",
            version="v1.0.0",
            path="/path/to/model1.pt"
        ))
        db_session.flush()

        # Try to add another with same name and version
        db_session.add(ModelRegistry(
            name="fraud_detector",
            version="v1.0.0",
            path="/path/to/model2.pt"
        ))
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_same_name_different_version_allowed(self, db_session):
        db_session.add(ModelRegistry(
            name="fraud_detector",
            version="v1.0.0",
            path="/path/to/model1.pt"
        ))
        db_session.add(ModelRegistry(
            name="fraud_detector",
            version="v1.1.0",
            path="/path/to/model2.pt"
        ))
        db_session.flush()

        from sqlalchemy import select
        rows = db_session.scalars(
            select(ModelRegistry).where(ModelRegistry.name == "fraud_detector")
        ).all()
        assert len(rows) == 2

    def test_different_name_same_version_allowed(self, db_session):
        db_session.add(ModelRegistry(
            name="fraud_detector",
            version="v1.0.0",
            path="/path/to/model1.pt"
        ))
        db_session.add(ModelRegistry(
            name="anomaly_detector",
            version="v1.0.0",
            path="/path/to/model2.pt"
        ))
        db_session.flush()

        from sqlalchemy import select
        rows = db_session.scalars(select(ModelRegistry)).all()
        assert len(rows) == 2

    def test_stage_transitions(self, db_session):
        model = ModelRegistry(
            name="fraud_detector",
            version="v1.0.0",
            path="/path/to/model.pt"
        )
        db_session.add(model)
        db_session.flush()

        # Test transition: inactive → active
        model.status = "active"
        db_session.flush()
        db_session.refresh(model)
        assert model.status == "active"

        # Test transition: active → deprecated
        model.status = "deprecated"
        db_session.flush()
        db_session.refresh(model)
        assert model.status == "deprecated"

        # Test transition: deprecated → inactive
        model.status = "inactive"
        db_session.flush()
        db_session.refresh(model)
        assert model.status == "inactive"
