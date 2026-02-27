import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from astroml.features.frequency import (
    _compute_burstiness,
    _extract_daily_counts,
    _validate_dataframe,
)


class TestExtractDailyCounts:
    """Unit tests for _extract_daily_counts helper function."""

    def test_empty_timestamps(self):
        """Test that empty timestamps return empty array."""
        timestamps = pd.Series([], dtype="datetime64[ns]")
        result = _extract_daily_counts(timestamps)
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_single_timestamp(self):
        """Test that single timestamp returns array [1]."""
        timestamps = pd.Series(pd.to_datetime(["2024-01-01"]))
        result = _extract_daily_counts(timestamps)
        np.testing.assert_array_equal(result, np.array([1]))

    def test_same_day_transactions(self):
        """Test multiple transactions on same day."""
        timestamps = pd.Series(pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-01 14:30:00",
            "2024-01-01 18:45:00",
        ]))
        result = _extract_daily_counts(timestamps)
        np.testing.assert_array_equal(result, np.array([3]))

    def test_consecutive_days(self):
        """Test transactions on consecutive days."""
        timestamps = pd.Series(pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ]))
        result = _extract_daily_counts(timestamps)
        np.testing.assert_array_equal(result, np.array([1, 1, 1]))

    def test_gaps_filled_with_zeros(self):
        """Test that missing days are filled with 0."""
        timestamps = pd.Series(pd.to_datetime([
            "2024-01-01",
            "2024-01-01",
            "2024-01-03",
        ]))
        result = _extract_daily_counts(timestamps)
        np.testing.assert_array_equal(result, np.array([2, 0, 1]))

    def test_larger_gap(self):
        """Test with larger gap between transactions."""
        timestamps = pd.Series(pd.to_datetime([
            "2024-01-01",
            "2024-01-05",
        ]))
        result = _extract_daily_counts(timestamps)
        expected = np.array([1, 0, 0, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_unordered_timestamps(self):
        """Test that timestamp order doesn't affect result."""
        timestamps = pd.Series(pd.to_datetime([
            "2024-01-03",
            "2024-01-01",
            "2024-01-02",
        ]))
        result = _extract_daily_counts(timestamps)
        np.testing.assert_array_equal(result, np.array([1, 1, 1]))

    def test_multiple_transactions_with_gaps(self):
        """Test realistic scenario with varying daily counts."""
        timestamps = pd.Series(pd.to_datetime([
            "2024-01-01",
            "2024-01-01",
            "2024-01-01",  # 3 transactions
            "2024-01-03",
            "2024-01-03",  # 2 transactions
            "2024-01-05",  # 1 transaction
        ]))
        result = _extract_daily_counts(timestamps)
        expected = np.array([3, 0, 2, 0, 1])
        np.testing.assert_array_equal(result, expected)


class TestComputeBurstiness:
    """Unit tests for _compute_burstiness helper function."""

    def test_known_value_equal_mean_std(self):
        """Test burstiness when mean equals std (should be 0)."""
        result = _compute_burstiness(mean=5.0, std=5.0)
        assert result == 0.0

    def test_known_value_high_std(self):
        """Test burstiness with high std relative to mean (bursty)."""
        result = _compute_burstiness(mean=2.0, std=8.0)
        expected = (8.0 - 2.0) / (8.0 + 2.0)  # 6/10 = 0.6
        assert result == pytest.approx(expected)
        assert result == pytest.approx(0.6)

    def test_known_value_low_std(self):
        """Test burstiness with low std relative to mean (regular)."""
        result = _compute_burstiness(mean=10.0, std=2.0)
        expected = (2.0 - 10.0) / (2.0 + 10.0)  # -8/12 ~= -0.667
        assert result == pytest.approx(expected)
        assert result == pytest.approx(-0.6666666666666666)

    def test_known_value_zero_std(self):
        """Test burstiness with zero std (perfectly regular)."""
        result = _compute_burstiness(mean=5.0, std=0.0)
        expected = (0.0 - 5.0) / (0.0 + 5.0)  # -5/5 = -1.0
        assert result == -1.0

    def test_edge_case_both_zero(self):
        """Test edge case: when both mean and std are 0, return 0.0."""
        result = _compute_burstiness(mean=0.0, std=0.0)
        assert result == 0.0

    def test_edge_case_zero_mean_nonzero_std(self):
        """Test edge case: zero mean with non-zero std."""
        result = _compute_burstiness(mean=0.0, std=3.0)
        expected = (3.0 - 0.0) / (3.0 + 0.0)  # 3/3 = 1.0
        assert result == 1.0

    def test_result_bounded_upper(self):
        """Test that result is bounded at upper limit (approaches 1)."""
        # When std >> mean, burstiness approaches 1
        result = _compute_burstiness(mean=1.0, std=1000.0)
        assert result < 1.0
        assert result > 0.99

    def test_result_bounded_lower(self):
        """Test that result is bounded at lower limit (approaches -1)."""
        # When std << mean, burstiness approaches -1
        result = _compute_burstiness(mean=1000.0, std=1.0)
        assert result > -1.0
        assert result < -0.99

    def test_symmetric_values(self):
        """Test that swapping mean and std produces opposite sign."""
        result1 = _compute_burstiness(mean=3.0, std=7.0)
        result2 = _compute_burstiness(mean=7.0, std=3.0)
        assert result1 == pytest.approx(-result2)


class TestComputeBurstinessProperties:
    """Property-based tests for _compute_burstiness using Hypothesis."""

    @given(
        mean=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        std=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_burstiness_always_bounded(self, mean, std):
        """Property: Burstiness is always in [-1, 1]."""
        result = _compute_burstiness(mean, std)
        assert -1.0 <= result <= 1.0, f"Burstiness {result} out of bounds for mean={mean}, std={std}"

    @given(
        mean=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        std=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_burstiness_formula_correctness(self, mean, std):
        """Property: Burstiness correctly implements (std - mean) / (std + mean)."""
        result = _compute_burstiness(mean, std)
        expected = (std - mean) / (std + mean)
        assert result == pytest.approx(expected), f"Formula mismatch for mean={mean}, std={std}"

    @given(
        value=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_equal_mean_std_gives_zero(self, value):
        """Property: When mean equals std, burstiness is 0."""
        result = _compute_burstiness(mean=value, std=value)
        assert result == pytest.approx(0.0)

    @given(
        mean=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_zero_std_gives_negative_one(self, mean):
        """Property: When std is 0 (perfectly regular), burstiness is -1."""
        result = _compute_burstiness(mean=mean, std=0.0)
        assert result == pytest.approx(-1.0)

    @given(
        std=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_zero_mean_gives_positive_one(self, std):
        """Property: When mean is 0 with non-zero std, burstiness is 1."""
        result = _compute_burstiness(mean=0.0, std=std)
        assert result == pytest.approx(1.0)

    @given(
        mean=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        std=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_std_greater_than_mean_positive_burstiness(self, mean, std):
        """Property: When std > mean, burstiness is positive (bursty)."""
        if std > mean:
            result = _compute_burstiness(mean, std)
            assert result > 0.0

    @given(
        mean=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        std=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_mean_greater_than_std_negative_burstiness(self, mean, std):
        """Property: When mean > std, burstiness is negative (regular)."""
        if mean > std:
            result = _compute_burstiness(mean, std)
            assert result < 0.0


class TestValidateDataFrame:
    """Unit tests for _validate_dataframe helper function."""

    def test_missing_required_columns(self):
        """Should fail when required columns are missing."""
        df = pd.DataFrame({"timestamp": ["2024-01-01"]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_dataframe(df, timestamp_col="timestamp", account_col="account")

    def test_null_timestamp_values(self):
        """Should fail when timestamp column contains null values."""
        df = pd.DataFrame({
            "timestamp": [None, "2024-01-01"],
            "account": ["acct-1", "acct-2"],
        })
        with pytest.raises(ValueError, match="contains null values"):
            _validate_dataframe(df, timestamp_col="timestamp", account_col="account")

    def test_null_account_values(self):
        """Should fail when account column contains null values."""
        df = pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-02"],
            "account": ["acct-1", None],
        })
        with pytest.raises(ValueError, match="contains null values"):
            _validate_dataframe(df, timestamp_col="timestamp", account_col="account")

    def test_string_timestamps_are_converted_to_datetime(self):
        """Should convert parseable string timestamps to datetime."""
        df = pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-02"],
            "account": ["acct-1", "acct-2"],
        })

        _validate_dataframe(df, timestamp_col="timestamp", account_col="account")

        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-01")

    def test_numeric_timestamps_are_converted_to_datetime(self):
        """Should convert numeric Unix timestamps to datetime."""
        df = pd.DataFrame({
            "timestamp": [1704067200, 1704153600],
            "account": ["acct-1", "acct-2"],
        })

        _validate_dataframe(df, timestamp_col="timestamp", account_col="account")

        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-01")

    def test_invalid_timestamps_raise_value_error(self):
        """Should fail when timestamp values are not parseable."""
        df = pd.DataFrame({
            "timestamp": ["not-a-date"],
            "account": ["acct-1"],
        })

        with pytest.raises(ValueError, match="must contain datetime values"):
            _validate_dataframe(df, timestamp_col="timestamp", account_col="account")
