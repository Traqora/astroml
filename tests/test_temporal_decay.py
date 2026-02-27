import math
import pytest
from astroml.features.temporal_decay import TemporalDecayWeighter, compute_decay_weights
import numpy as np


class TestTemporalDecayWeighter:
    """Unit tests for TemporalDecayWeighter class."""

    def test_init_valid_lambda(self):
        """Test initialization with valid lambda parameter."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        assert weighter.lambda_param == 0.01

    def test_init_invalid_lambda(self):
        """Test initialization with negative lambda raises error."""
        with pytest.raises(ValueError):
            TemporalDecayWeighter(lambda_param=-0.01)

    def test_init_zero_lambda(self):
        """Test initialization with lambda=0 (no decay)."""
        weighter = TemporalDecayWeighter(lambda_param=0.0)
        assert weighter.lambda_param == 0.0

    def test_compute_decay_factor_zero_time(self):
        """Test decay factor at time zero."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        assert weighter.compute_decay_factor(0) == 1.0

    def test_compute_decay_factor_negative_time(self):
        """Test decay factor with negative time returns 1.0."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        assert weighter.compute_decay_factor(-10) == 1.0

    def test_compute_decay_factor_exponential(self):
        """Test exponential decay calculation."""
        weighter = TemporalDecayWeighter(lambda_param=0.1)
        time_delta = 10
        expected = math.exp(-0.1 * 10)
        result = weighter.compute_decay_factor(time_delta)
        assert abs(result - expected) < 1e-10

    def test_compute_decay_factor_no_decay(self):
        """Test no decay when lambda=0."""
        weighter = TemporalDecayWeighter(lambda_param=0.0)
        assert weighter.compute_decay_factor(100) == 1.0

    def test_compute_decay_factor_monotonic_decreasing(self):
        """Test that decay factor decreases with time."""
        weighter = TemporalDecayWeighter(lambda_param=0.05)
        factors = [weighter.compute_decay_factor(t) for t in range(0, 101, 10)]
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1]

    def test_weight_transactions_empty(self):
        """Test weighting empty transaction list."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        result = weighter.weight_transactions([], current_time=100.0)
        assert result == []

    def test_weight_transactions_single(self):
        """Test weighting single transaction."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        transactions = [{"timestamp": 100.0, "amount": 50.0}]
        result = weighter.weight_transactions(transactions, current_time=100.0)
        assert len(result) == 1
        assert result[0] == 1.0

    def test_weight_transactions_multiple(self):
        """Test weighting multiple transactions."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        transactions = [
            {"timestamp": 100.0},
            {"timestamp": 50.0},
            {"timestamp": 0.0}
        ]
        result = weighter.weight_transactions(transactions, current_time=100.0)
        assert len(result) == 3
        assert result[0] == 1.0  # time_delta = 0
        assert result[1] < result[0]  # time_delta = 50
        assert result[2] < result[1]  # time_delta = 100

    def test_weight_transactions_custom_timestamp_key(self):
        """Test weighting with custom timestamp key."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        transactions = [
            {"txn_time": 100.0},
            {"txn_time": 50.0}
        ]
        result = weighter.weight_transactions(
            transactions, current_time=100.0, timestamp_key="txn_time"
        )
        assert len(result) == 2
        assert result[0] == 1.0

    def test_weight_transactions_missing_timestamp(self):
        """Test weighting when timestamp is missing uses current_time."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        transactions = [{"amount": 50.0}]
        result = weighter.weight_transactions(transactions, current_time=100.0)
        assert result[0] == 1.0

    def test_apply_decay_to_amount_zero_delta(self):
        """Test applying decay with zero time delta."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        result = weighter.apply_decay_to_amount(100.0, 0)
        assert result == 100.0

    def test_apply_decay_to_amount_positive_delta(self):
        """Test applying decay with positive time delta."""
        weighter = TemporalDecayWeighter(lambda_param=0.1)
        amount = 100.0
        time_delta = 10.0
        expected = 100.0 * math.exp(-0.1 * 10)
        result = weighter.apply_decay_to_amount(amount, time_delta)
        assert abs(result - expected) < 1e-10

    def test_apply_decay_to_amount_zero_amount(self):
        """Test applying decay to zero amount."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        result = weighter.apply_decay_to_amount(0.0, 100.0)
        assert result == 0.0

    def test_aggregate_with_decay_empty(self):
        """Test aggregation with empty transaction list."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        result = weighter.aggregate_with_decay([], current_time=100.0)
        assert result == 0.0

    def test_aggregate_with_decay_sum(self):
        """Test sum aggregation with decay."""
        weighter = TemporalDecayWeighter(lambda_param=0.0)
        transactions = [
            {"timestamp": 100.0, "amount": 100.0},
            {"timestamp": 100.0, "amount": 50.0}
        ]
        result = weighter.aggregate_with_decay(
            transactions, current_time=100.0, aggregation="sum"
        )
        assert abs(result - 150.0) < 1e-10

    def test_aggregate_with_decay_mean(self):
        """Test mean aggregation with decay."""
        weighter = TemporalDecayWeighter(lambda_param=0.0)
        transactions = [
            {"timestamp": 100.0, "amount": 100.0},
            {"timestamp": 100.0, "amount": 50.0}
        ]
        result = weighter.aggregate_with_decay(
            transactions, current_time=100.0, aggregation="mean"
        )
        assert abs(result - 75.0) < 1e-10

    def test_aggregate_with_decay_weighted_mean(self):
        """Test weighted mean aggregation."""
        weighter = TemporalDecayWeighter(lambda_param=0.0)
        transactions = [
            {"timestamp": 100.0, "amount": 100.0},
            {"timestamp": 100.0, "amount": 50.0}
        ]
        result = weighter.aggregate_with_decay(
            transactions, current_time=100.0, aggregation="weighted_mean"
        )
        assert abs(result - 75.0) < 1e-10

    def test_aggregate_with_decay_invalid_aggregation(self):
        """Test that invalid aggregation method raises error."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        transactions = [{"timestamp": 100.0, "amount": 100.0}]
        with pytest.raises(ValueError):
            weighter.aggregate_with_decay(
                transactions, current_time=100.0, aggregation="invalid"
            )

    def test_aggregate_with_decay_custom_keys(self):
        """Test aggregation with custom timestamp and amount keys."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        transactions = [
            {"txn_time": 100.0, "value": 100.0},
            {"txn_time": 100.0, "value": 50.0}
        ]
        result = weighter.aggregate_with_decay(
            transactions,
            current_time=100.0,
            timestamp_key="txn_time",
            amount_key="value",
            aggregation="sum"
        )
        assert abs(result - 150.0) < 1e-10

    def test_aggregate_with_decay_missing_values(self):
        """Test aggregation handles missing amount values."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        transactions = [
            {"timestamp": 100.0, "amount": 100.0},
            {"timestamp": 100.0}
        ]
        result = weighter.aggregate_with_decay(
            transactions, current_time=100.0, aggregation="sum"
        )
        assert abs(result - 100.0) < 1e-10

    def test_compute_decay_weights_utility(self):
        """Test compute_decay_weights utility function."""
        transactions = [
            {"timestamp": 100.0},
            {"timestamp": 50.0},
            {"timestamp": 0.0}
        ]
        result = compute_decay_weights(transactions, current_time=100.0, lambda_param=0.01)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result[0] >= result[1] >= result[2]

    def test_lambda_effect_on_decay_rate(self):
        """Test that higher lambda values decay faster."""
        time_delta = 10.0
        weighter_slow = TemporalDecayWeighter(lambda_param=0.01)
        weighter_fast = TemporalDecayWeighter(lambda_param=0.1)
        slow_decay = weighter_slow.compute_decay_factor(time_delta)
        fast_decay = weighter_fast.compute_decay_factor(time_delta)
        assert slow_decay > fast_decay

    def test_realistic_daily_data(self):
        """Test with realistic daily financial data."""
        weighter = TemporalDecayWeighter(lambda_param=0.01)
        transactions = [
            {"timestamp": 1000000, "amount": 100.0},
            {"timestamp": 999900, "amount": 150.0},
            {"timestamp": 999500, "amount": 75.0}
        ]
        result = weighter.aggregate_with_decay(
            transactions,
            current_time=1000000,
            aggregation="sum"
        )
        assert result > 0
        assert result < sum(t["amount"] for t in transactions)
