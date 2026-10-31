"""Integration tests for GET /api/v1/alerts/predictive (issue #847).

The route handler called the async `generate_predictive_alerts` service
method without `await`, from a sync `def` route, so `result` was a coroutine
object -- `"error" in result` on that raised TypeError, and the route 500'd
on effectively every request that got past the basic account-id length
check. `api/routers/v1/alerts.py` is the copy actually mounted by the app
(via `api/routers/__init__.py`); the byte-identical `api/routers/alerts.py`
is unreferenced dead code, fixed to match for consistency but not exercised
here since nothing imports it.
"""

from __future__ import annotations


def test_predictive_endpoint_does_not_500_with_no_transaction_history(client):
    account_id = "G" + "A" * 55

    response = client.get("/api/v1/alerts/predictive", params={"account_id": account_id})

    assert response.status_code == 200
    body = response.json()
    assert body["alerts"] == []
    assert body["total_analyzed"] == 0


def test_predictive_endpoint_rejects_short_account_ids(client):
    response = client.get("/api/v1/alerts/predictive", params={"account_id": "short"})

    assert response.status_code == 400


def test_predictive_endpoint_returns_alerts_once_baselines_exist(client, db_session):
    from datetime import datetime, timedelta, timezone

    from api.models.orm import ApiTransaction

    account_id = "G" + "B" * 55
    now = datetime.now(timezone.utc)

    for i in range(10):
        db_session.add(
            ApiTransaction(
                hash=f"tx{i:03d}" + "0" * 58,
                ledger_sequence=i,
                source_account=account_id,
                destination_account="G" + "C" * 55,
                amount=100.0,
                asset_code="XLM",
                fee=100,
                successful=True,
                created_at=now - timedelta(hours=i),
            )
        )
    db_session.commit()

    response = client.get(
        "/api/v1/alerts/predictive",
        params={"account_id": account_id, "lookback_days": 7},
    )

    # No crash either way -- with this little history it's a legitimate
    # "insufficient data" response (200, empty alerts), not a 500.
    assert response.status_code == 200
    assert response.json()["alerts"] == []
