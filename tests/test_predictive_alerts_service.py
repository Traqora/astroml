"""Tests for PredictiveAlertService.generate_predictive_alerts (issue #847).

generate_predictive_alerts referenced an undefined name `result` instead of
the `learn_result` it had just assigned, right after awaiting
learn_behavior_from_transactions. Every call that reached that line raised a
NameError, which the method's own `except Exception` then swallowed into
`{"error": str(e)}` -- so the endpoint silently 500'd (via the router's
blanket "error" -> 500 mapping) on effectively every request, including the
common "account has no transaction history yet" case, which is not a server
error at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from api.services.predictive_alerts import PredictiveAlertService


@pytest.fixture
def service() -> PredictiveAlertService:
    return PredictiveAlertService()


class _FakeTransaction:
    def __init__(self, created_at, amount, hash_, destination_account):
        self.created_at = created_at
        self.amount = amount
        self.hash = hash_
        self.destination_account = destination_account


class _FakeQuery:
    def __init__(self, transactions):
        self._transactions = transactions

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return self._transactions


class _FakeDb:
    def __init__(self, transactions):
        self._transactions = transactions

    def query(self, model):
        return _FakeQuery(self._transactions)

    def close(self) -> None:
        pass


class TestLearnBehaviorFromTransactions:
    @pytest.mark.asyncio
    async def test_does_not_crash_on_transactions_with_a_destination_account(
        self, service: PredictiveAlertService
    ):
        """learn_behavior_from_transactions converted each ORM transaction into
        a plain dict (amount, hash) but then read `.destination_account` off
        that same dict as if it were still the ORM object -- an
        AttributeError on every account with any real transaction history,
        the thing the "no transaction data" message path accidentally hid.
        """
        now = datetime.now(timezone.utc)
        txs = [
            _FakeTransaction(now - timedelta(hours=i), 10.0 * i, f"hash{i}", "GDEST" + "A" * 51)
            for i in range(3)
        ]

        with patch(
            "api.services.predictive_alerts.get_sync_db",
            return_value=iter([_FakeDb(txs)]),
        ):
            result = await service.learn_behavior_from_transactions("GABC" + "A" * 52, 30)

        assert "error" not in result
        assert result["transactions_analyzed"] == 3


class TestGeneratePredictiveAlerts:
    @pytest.mark.asyncio
    async def test_does_not_crash_with_a_name_error_on_success_path(
        self, service: PredictiveAlertService
    ):
        """The bug reproduced unconditionally: any call that got past
        learn_behavior_from_transactions hit `if "error" in result` where
        `result` was never assigned, raising NameError every time.
        """
        with (
            patch.object(
                service,
                "learn_behavior_from_transactions",
                new=AsyncMock(return_value={"baselines_learned": 3}),
            ),
            patch.object(
                service,
                "_get_current_metrics",
                new=AsyncMock(return_value={"daily_transaction_count": 5.0}),
            ),
            patch.object(
                service.behavioral_learner,
                "get_all_baselines",
                return_value={
                    "daily_transaction_count": {
                        "mean": 3.0,
                        "std_dev": 1.0,
                        "min": 1.0,
                        "max": 6.0,
                        "sample_size": 10,
                        "last_updated": datetime.now(timezone.utc),
                    }
                },
            ),
        ):
            result = await service.generate_predictive_alerts(account_id="GABC" + "A" * 52)

        assert "error" not in result
        assert "alerts" in result

    @pytest.mark.asyncio
    async def test_propagates_a_genuine_learning_error_without_crashing(
        self, service: PredictiveAlertService
    ):
        with patch.object(
            service,
            "learn_behavior_from_transactions",
            new=AsyncMock(return_value={"error": "database unavailable"}),
        ):
            result = await service.generate_predictive_alerts(account_id="GABC" + "A" * 52)

        assert result == {"error": "database unavailable"}

    @pytest.mark.asyncio
    async def test_no_transaction_history_is_reported_as_a_message_not_an_error(
        self, service: PredictiveAlertService
    ):
        """An account with no transactions in the lookback window is a normal,
        client-facing "nothing to analyze yet" outcome -- not a failure.
        """
        with patch.object(
            service,
            "learn_behavior_from_transactions",
            new=AsyncMock(
                return_value={"message": "No transaction data found for learning period"}
            ),
        ):
            result = await service.generate_predictive_alerts(account_id="GABC" + "A" * 52)

        assert "error" not in result
        assert result["message"] == "No transaction data found for learning period"
        assert result["account_id"] == "GABC" + "A" * 52

    @pytest.mark.asyncio
    async def test_unexpected_exception_is_still_caught_and_reported_as_error(
        self, service: PredictiveAlertService
    ):
        with patch.object(
            service,
            "learn_behavior_from_transactions",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            result = await service.generate_predictive_alerts(account_id="GABC" + "A" * 52)

        assert result == {"error": "boom"}
