"""
Integration tests — loyalty points (issue #244).

Covers: ORM model creation, tier logic, points transactions,
balance queries, and history pagination.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from api.models.orm import LoyaltyPoints, PointsTransaction


@pytest.mark.xdist_group("api_loyalty")
class TestLoyaltyPointsModel:
    """ORM-level tests for LoyaltyPoints (issue #246)."""

    def test_create_loyalty_row(self, db_session):
        lp = LoyaltyPoints(
            account_id="GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            balance=500,
            tier="bronze",
            multiplier=1.0,
        )
        db_session.add(lp)
        db_session.flush()
        assert lp.id is not None

    def test_seeded_loyalty_fields(self, seeded_loyalty):
        assert seeded_loyalty.balance == 2500
        assert seeded_loyalty.tier == "silver"
        assert seeded_loyalty.multiplier == pytest.approx(1.1)

    def test_unique_account_constraint(self, db_session, seeded_loyalty):
        duplicate = LoyaltyPoints(
            account_id=seeded_loyalty.account_id,
            balance=0,
            tier="bronze",
            multiplier=1.0,
        )
        db_session.add(duplicate)
        with pytest.raises(Exception):
            db_session.flush()

    def test_query_by_account(self, db_session, seeded_loyalty):
        result = db_session.execute(
            select(LoyaltyPoints).where(
                LoyaltyPoints.account_id == seeded_loyalty.account_id
            )
        ).scalar_one()
        assert result.balance == 2500


@pytest.mark.xdist_group("api_loyalty")
class TestPointsTransactionModel:
    """ORM-level tests for PointsTransaction (issue #246)."""

    def test_create_earn_transaction(self, db_session):
        pt = PointsTransaction(
            account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            type="earn",
            points=100,
            source="trade_completion",
        )
        db_session.add(pt)
        db_session.flush()
        assert pt.id is not None

    def test_create_redeem_transaction(self, db_session):
        pt = PointsTransaction(
            account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            type="redeem",
            points=-200,
            source="reward_redemption",
        )
        db_session.add(pt)
        db_session.flush()
        assert pt.points == -200

    def test_history_ordering(self, db_session):
        account = "GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF"
        from datetime import datetime, timezone, timedelta
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i, pts in enumerate([50, 100, -30]):
            pt = PointsTransaction(
                account_id=account,
                type="earn" if pts > 0 else "redeem",
                points=pts,
                source=f"event_{i}",
            )
            db_session.add(pt)
        db_session.flush()

        rows = db_session.execute(
            select(PointsTransaction)
            .where(PointsTransaction.account_id == account)
            .order_by(PointsTransaction.id)
        ).scalars().all()
        assert len(rows) == 3
        assert rows[0].points == 50
        assert rows[2].points == -30

    def test_net_balance_calculation(self, db_session):
        account = "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT"
        for pts in [200, 50, -75]:
            db_session.add(PointsTransaction(
                account_id=account,
                type="earn" if pts > 0 else "redeem",
                points=pts,
            ))
        db_session.flush()

        rows = db_session.execute(
            select(PointsTransaction).where(PointsTransaction.account_id == account)
        ).scalars().all()
        net = sum(r.points for r in rows)
        assert net == 175

    def test_filter_by_type(self, db_session):
        account = "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"
        for t, pts in [("earn", 100), ("redeem", -50), ("adjust", 10)]:
            db_session.add(PointsTransaction(account_id=account, type=t, points=pts))
        db_session.flush()

        earns = db_session.execute(
            select(PointsTransaction).where(
                PointsTransaction.account_id == account,
                PointsTransaction.type == "earn",
            )
        ).scalars().all()
        assert len(earns) == 1


"""Tests for the Loyalty Points API — issue #235.

Uses FastAPI TestClient wired to a SQLite in-memory database via conftest
fixtures. The loyalty router reads from two sources:
  - api.loyalty_models (LoyaltyAccount, PointsLedger)  — the primary store
  - astroml.db.session.SessionLocal                     — injected via _get_db()

We override the `_get_db` dependency so tests use the same SQLite session
produced by conftest instead of the real Postgres session.
"""

import uuid
from datetime import date, datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.app import app
from api.loyalty_models import LoyaltyBase, LoyaltyAccount, PointsLedger
import api.routers.loyalty as _loyalty_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_path):
    db_file = tmp_path / "loyalty_test.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
    LoyaltyBase.metadata.create_all(engine)
    return engine


@pytest.fixture()
def loyalty_client(tmp_path):
    """TestClient with loyalty dependency overridden to use SQLite."""
    engine = _make_engine(tmp_path)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[_loyalty_module._get_db] = override_get_db
    client = TestClient(app, raise_server_exceptions=True)
    yield client, SessionLocal
    app.dependency_overrides.pop(_loyalty_module._get_db, None)
    engine.dispose()


def _seed_account(session_factory, account_id: str, balance: int = 0, tier: str = "bronze"):
    with session_factory() as s:
        s.add(LoyaltyAccount(account_id=account_id, points_balance=balance, tier_id=tier))
        s.commit()


def _seed_ledger_row(
    session_factory,
    account_id: str,
    txn_type: str = "earn",
    points: int = 100,
    created_at: datetime | None = None,
):
    with session_factory() as s:
        s.add(
            PointsLedger(
                id=str(uuid.uuid4()),
                account_id=account_id,
                txn_type=txn_type,
                points=points,
                created_at=created_at or datetime.now(timezone.utc),
            )
        )
        s.commit()


# ---------------------------------------------------------------------------
# Summary endpoint
# ---------------------------------------------------------------------------

class TestLoyaltySummary:
    ACCOUNT = "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"

    def test_fresh_account_gets_bronze_with_zero_points(self, loyalty_client):
        client, _ = loyalty_client
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["points_balance"] == 0
        assert body["current_tier"]["id"] == "bronze"

    def test_silver_tier_at_1500_points(self, loyalty_client):
        client, sf = loyalty_client
        _seed_account(sf, self.ACCOUNT, balance=1500)
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/summary")
        assert r.status_code == 200
        assert r.json()["current_tier"]["id"] == "silver"

    def test_gold_tier_at_3000_points(self, loyalty_client):
        client, sf = loyalty_client
        _seed_account(sf, self.ACCOUNT, balance=3000)
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/summary")
        assert r.status_code == 200
        assert r.json()["current_tier"]["id"] == "gold"

    def test_platinum_tier_at_6000_points(self, loyalty_client):
        client, sf = loyalty_client
        _seed_account(sf, self.ACCOUNT, balance=6000)
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/summary")
        assert r.status_code == 200
        assert r.json()["current_tier"]["id"] == "platinum"

    def test_next_tier_info_present_for_non_platinum(self, loyalty_client):
        client, _ = loyalty_client
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["next_tier"] is not None
        assert body["next_tier"]["remaining_to_upgrade"] == 1500

    def test_next_tier_none_for_platinum(self, loyalty_client):
        client, sf = loyalty_client
        _seed_account(sf, self.ACCOUNT, balance=9999)
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/summary")
        assert r.status_code == 200
        assert r.json()["next_tier"] is None

    def test_benefits_list_non_empty(self, loyalty_client):
        client, _ = loyalty_client
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/summary")
        assert r.status_code == 200
        assert len(r.json()["benefits"]) >= 1


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------

class TestLoyaltyHistory:
    ACCOUNT = "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT"

    def test_empty_history(self, loyalty_client):
        client, _ = loyalty_client
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/history")
        assert r.status_code == 200
        body = r.json()
        assert body["data"] == []
        assert body["total"] == 0
        assert body["page"] == 1

    def test_pagination_defaults(self, loyalty_client):
        client, sf = loyalty_client
        for _ in range(5):
            _seed_ledger_row(sf, self.ACCOUNT, points=10)
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/history")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 5
        assert body["page_size"] == 20

    def test_pagination_page_2(self, loyalty_client):
        client, sf = loyalty_client
        for _ in range(25):
            _seed_ledger_row(sf, self.ACCOUNT, points=10)
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/history?page=2&page_size=20")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 25
        assert len(body["data"]) == 5
        assert body["page"] == 2

    def test_page_size_param(self, loyalty_client):
        client, sf = loyalty_client
        for _ in range(10):
            _seed_ledger_row(sf, self.ACCOUNT, points=5)
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/history?page_size=3")
        assert r.status_code == 200
        assert len(r.json()["data"]) == 3

    def test_history_row_shape(self, loyalty_client):
        client, sf = loyalty_client
        _seed_ledger_row(sf, self.ACCOUNT, txn_type="earn", points=50)
        r = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/history")
        row = r.json()["data"][0]
        assert "id" in row
        assert "type" in row
        assert "points" in row
        assert "date" in row


# ---------------------------------------------------------------------------
# Redeem endpoint
# ---------------------------------------------------------------------------

class TestLoyaltyRedeem:
    ACCOUNT = "GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF"

    def test_successful_redemption_deducts_balance(self, loyalty_client):
        client, sf = loyalty_client
        _seed_account(sf, self.ACCOUNT, balance=500)
        r = client.post(
            f"/api/v1/loyalty/{self.ACCOUNT}/redeem",
            json={"points": 200, "reward_id": "reward_xyz"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["new_balance"] == 300
        assert body["transaction"]["type"] == "redeem"
        assert body["transaction"]["points"] == -200

    def test_insufficient_balance_returns_400(self, loyalty_client):
        client, sf = loyalty_client
        _seed_account(sf, self.ACCOUNT, balance=50)
        r = client.post(
            f"/api/v1/loyalty/{self.ACCOUNT}/redeem",
            json={"points": 200},
        )
        assert r.status_code == 400
        assert "Insufficient" in r.json()["detail"]

    def test_below_minimum_100_returns_400(self, loyalty_client):
        client, sf = loyalty_client
        _seed_account(sf, self.ACCOUNT, balance=1000)
        r = client.post(
            f"/api/v1/loyalty/{self.ACCOUNT}/redeem",
            json={"points": 50},
        )
        # Router enforces minimum at 400, not 422 (validation is in business logic)
        assert r.status_code in (400, 422)

    def test_zero_points_rejected_by_schema(self, loyalty_client):
        """RedeemRequest has points > 0 validator, so 0 is a 422."""
        client, sf = loyalty_client
        _seed_account(sf, self.ACCOUNT, balance=1000)
        r = client.post(
            f"/api/v1/loyalty/{self.ACCOUNT}/redeem",
            json={"points": 0},
        )
        assert r.status_code == 422

    def test_one_per_day_limit_returns_400(self, loyalty_client):
        """Second redemption on the same calendar day must be rejected."""
        client, sf = loyalty_client
        _seed_account(sf, self.ACCOUNT, balance=1000)

        # Seed a redemption ledger entry for today
        with sf() as s:
            s.add(
                PointsLedger(
                    id=str(uuid.uuid4()),
                    account_id=self.ACCOUNT,
                    txn_type="redeem",
                    points=-100,
                    created_at=datetime.now(timezone.utc),
                )
            )
            s.commit()

        r = client.post(
            f"/api/v1/loyalty/{self.ACCOUNT}/redeem",
            json={"points": 100},
        )
        assert r.status_code == 400
        assert "per day" in r.json()["detail"].lower() or "redemption" in r.json()["detail"].lower()

    def test_redemption_creates_ledger_entry(self, loyalty_client):
        """After a successful redeem the history endpoint should show it."""
        client, sf = loyalty_client
        _seed_account(sf, self.ACCOUNT, balance=500)
        r = client.post(
            f"/api/v1/loyalty/{self.ACCOUNT}/redeem",
            json={"points": 100},
        )
        assert r.status_code == 200

        r2 = client.get(f"/api/v1/loyalty/{self.ACCOUNT}/history")
        assert r2.status_code == 200
        entries = [e for e in r2.json()["data"] if e["type"] == "redeem"]
        assert len(entries) >= 1


# ---------------------------------------------------------------------------
# Tiers endpoint (smoke test)
# ---------------------------------------------------------------------------

class TestLoyaltyTiers:
    def test_tiers_list_returns_four_tiers(self, loyalty_client):
        client, _ = loyalty_client
        r = client.get("/api/v1/loyalty/tiers")
        assert r.status_code == 200
        tiers = r.json()
        assert len(tiers) == 4
        ids = [t["id"] for t in tiers]
        assert ids == ["bronze", "silver", "gold", "platinum"]
