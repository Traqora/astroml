import numpy as np
import pandas as pd
import pytest

from astroml.features.frequency import _extract_daily_counts


class TestExtractDailyCounts:
    """Unit tests for _extract_daily_counts helper function."""

    def test_empty_timestamps(self):
        """Test that empty timestamps return empty array."""
        timestamps = pd.Series([], dtype='datetime64[ns]')
        result = _extract_daily_counts(timestamps)
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_single_timestamp(self):
        """Test that single timestamp returns array [1]."""
        timestamps = pd.Series(pd.to_datetime(['2024-01-01']))
        result = _extract_daily_counts(timestamps)
        np.testing.assert_array_equal(result, np.array([1]))

    def test_same_day_transactions(self):
        """Test multiple transactions on same day."""
        timestamps = pd.Series(pd.to_datetime([
            '2024-01-01 10:00:00',
            '2024-01-01 14:30:00',
            '2024-01-01 18:45:00'
        ]))
        result = _extract_daily_counts(timestamps)
        np.testing.assert_array_equal(result, np.array([3]))

    def test_consecutive_days(self):
        """Test transactions on consecutive days."""
        timestamps = pd.Series(pd.to_datetime([
            '2024-01-01',
            '2024-01-02',
            '2024-01-03'
        ]))
        result = _extract_daily_counts(timestamps)
        np.testing.assert_array_equal(result, np.array([1, 1, 1]))

    def test_gaps_filled_with_zeros(self):
        """Test that missing days are filled with 0."""
        timestamps = pd.Series(pd.to_datetime([
            '2024-01-01',
            '2024-01-01',
            '2024-01-03'
        ]))
        result = _extract_daily_counts(timestamps)
        np.testing.assert_array_equal(result, np.array([2, 0, 1]))

    def test_larger_gap(self):
        """Test with larger gap between transactions."""
        timestamps = pd.Series(pd.to_datetime([
            '2024-01-01',
            '2024-01-05'
        ]))
        result = _extract_daily_counts(timestamps)
        expected = np.array([1, 0, 0, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_unordered_timestamps(self):
        """Test that timestamp order doesn't affect result."""
        timestamps = pd.Series(pd.to_datetime([
            '2024-01-03',
            '2024-01-01',
            '2024-01-02'
        ]))
        result = _extract_daily_counts(timestamps)
        np.testing.assert_array_equal(result, np.array([1, 1, 1]))

    def test_multiple_transactions_with_gaps(self):
        """Test realistic scenario with varying daily counts."""
        timestamps = pd.Series(pd.to_datetime([
            '2024-01-01', '2024-01-01', '2024-01-01',  # 3 transactions
            '2024-01-03', '2024-01-03',                 # 2 transactions
            '2024-01-05'                                # 1 transaction
        ]))
        result = _extract_daily_counts(timestamps)
        expected = np.array([3, 0, 2, 0, 1])
        np.testing.assert_array_equal(result, expected)
