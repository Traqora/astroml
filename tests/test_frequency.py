import numpy as np
import pandas as pd
import pytest

from astroml.features.frequency import (
    _extract_daily_counts,
    compute_frequency_metrics,
)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(accounts, dates, account_col="account_id", timestamp_col="created_at"):
    return pd.DataFrame(
        {
            account_col: accounts,
            timestamp_col: pd.to_datetime(dates),
        }
    )


# ---------------------------------------------------------------------------
# Tests for compute_frequency_metrics
# ---------------------------------------------------------------------------

class TestComputeFrequencyMetrics:
    """Unit tests for the public compute_frequency_metrics function."""

    _EXPECTED_COLUMNS = ["account", "mean_tx_per_day", "std_tx_per_day", "burstiness"]

    # ------------------------------------------------------------------
    # Schema / structure
    # ------------------------------------------------------------------

    def test_output_columns(self):
        """Output always has exactly the four expected columns."""
        df = _make_df(["A"], ["2024-01-01"])
        result = compute_frequency_metrics(df)
        assert list(result.columns) == self._EXPECTED_COLUMNS

    def test_empty_dataframe_returns_correct_schema(self):
        """Empty input returns an empty DataFrame with the correct columns."""
        df = pd.DataFrame(
            {
                "account_id": pd.Series([], dtype=str),
                "created_at": pd.Series([], dtype="datetime64[ns]"),
            }
        )
        result = compute_frequency_metrics(df)
        assert result.empty
        assert list(result.columns) == self._EXPECTED_COLUMNS

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_single_account_single_transaction(self):
        """Single transaction: mean=1, std=0, burstiness=-1."""
        df = _make_df(["alice"], ["2024-01-01"])
        result = compute_frequency_metrics(df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["account"] == "alice"
        assert row["mean_tx_per_day"] == pytest.approx(1.0)
        assert row["std_tx_per_day"] == pytest.approx(0.0)
        assert row["burstiness"] == pytest.approx(-1.0)

    def test_multi_account_one_row_each(self):
        """Each unique account produces exactly one output row."""
        df = _make_df(
            ["A", "A", "B", "B", "B"],
            ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-03"],
        )
        result = compute_frequency_metrics(df)
        assert len(result) == 2
        assert set(result["account"]) == {"A", "B"}

    def test_multi_account_values(self):
        """Metric values are correct for a two-account fixture.

        Account A: days=[1,1]  → mean=1.0, std(ddof=1)=0.0, B=-1.0
        Account B: days=[1,0,2] → mean=1.0, std(ddof=1)=1.0, B=0.0
        """
        df = _make_df(
            ["A", "A", "B", "B", "B"],
            ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-03"],
        )
        result = compute_frequency_metrics(df).set_index("account")

        assert result.loc["A", "mean_tx_per_day"] == pytest.approx(1.0)
        assert result.loc["A", "std_tx_per_day"] == pytest.approx(0.0)
        assert result.loc["A", "burstiness"] == pytest.approx(-1.0)

        assert result.loc["B", "mean_tx_per_day"] == pytest.approx(1.0)
        assert result.loc["B", "std_tx_per_day"] == pytest.approx(1.0)
        assert result.loc["B", "burstiness"] == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # Single-day window
    # ------------------------------------------------------------------

    def test_single_day_all_transactions(self):
        """All transactions on one day: std=0.0, burstiness=-1.0."""
        df = _make_df(
            ["A", "A", "A"],
            ["2024-01-01 10:00", "2024-01-01 14:00", "2024-01-01 18:00"],
        )
        result = compute_frequency_metrics(df)
        row = result.iloc[0]
        assert row["mean_tx_per_day"] == pytest.approx(3.0)
        assert row["std_tx_per_day"] == pytest.approx(0.0)
        assert row["burstiness"] == pytest.approx(-1.0)

    # ------------------------------------------------------------------
    # Gaps in activity
    # ------------------------------------------------------------------

    def test_gaps_affect_mean_and_std(self):
        """Zero-count days in the active window lower mean and raise std."""
        # days: [3, 0, 2, 0, 1]  mean=1.2, std(ddof=1)=np.std([3,0,2,0,1],ddof=1)
        df = _make_df(
            ["A"] * 6,
            [
                "2024-01-01", "2024-01-01", "2024-01-01",
                "2024-01-03", "2024-01-03",
                "2024-01-05",
            ],
        )
        result = compute_frequency_metrics(df)
        row = result.iloc[0]
        expected_mean = np.mean([3, 0, 2, 0, 1])
        expected_std = np.std([3, 0, 2, 0, 1], ddof=1)
        assert row["mean_tx_per_day"] == pytest.approx(expected_mean)
        assert row["std_tx_per_day"] == pytest.approx(expected_std)

    # ------------------------------------------------------------------
    # Custom column names
    # ------------------------------------------------------------------

    def test_custom_columns(self):
        """Non-default timestamp_col / account_col names are respected."""
        df = pd.DataFrame(
            {
                "wallet": ["X", "X"],
                "ts": pd.to_datetime(["2024-06-01", "2024-06-02"]),
            }
        )
        result = compute_frequency_metrics(df, timestamp_col="ts", account_col="wallet")
        assert len(result) == 1
        assert result.iloc[0]["account"] == "X"

    # ------------------------------------------------------------------
    # Unix timestamps
    # ------------------------------------------------------------------

    def test_unix_timestamps_treated_as_epoch_seconds(self):
        """Numeric Unix epoch seconds are converted correctly to dates."""
        # 2024-01-01 00:00:00 UTC = 1704067200
        # 2024-01-02 00:00:00 UTC = 1704153600
        df = pd.DataFrame(
            {
                "account_id": ["A", "A"],
                "created_at": [1704067200, 1704153600],
            }
        )
        result = compute_frequency_metrics(df)
        row = result.iloc[0]
        # Two consecutive days → mean=1.0, std=0.0
        assert row["mean_tx_per_day"] == pytest.approx(1.0)
        assert row["std_tx_per_day"] == pytest.approx(0.0)

    def test_unix_timestamps_match_datetime_equivalent(self):
        """Unix and datetime inputs produce identical results.

        Uses hardcoded epoch-second values to avoid resolution ambiguity
        across pandas versions (datetime64[ns] vs datetime64[s] etc.):
          2024-03-01 00:00:00 UTC → 1709251200 s
          2024-03-03 00:00:00 UTC → 1709424000 s
        """
        dates_str = ["2024-03-01", "2024-03-01", "2024-03-03"]
        unix_secs = [1709251200, 1709251200, 1709424000]

        df_dt = _make_df(["A", "A", "A"], dates_str)
        df_unix = pd.DataFrame({"account_id": ["A", "A", "A"], "created_at": unix_secs})

        r_dt = compute_frequency_metrics(df_dt).iloc[0]
        r_unix = compute_frequency_metrics(df_unix).iloc[0]

        assert r_dt["mean_tx_per_day"] == pytest.approx(r_unix["mean_tx_per_day"])
        assert r_dt["std_tx_per_day"] == pytest.approx(r_unix["std_tx_per_day"])
        assert r_dt["burstiness"] == pytest.approx(r_unix["burstiness"])

    # ------------------------------------------------------------------
    # Account identifier type preservation
    # ------------------------------------------------------------------

    def test_string_account_identifier_preserved(self):
        """String account IDs stay as strings in the output."""
        df = _make_df(["alice", "bob"], ["2024-01-01", "2024-01-01"])
        result = compute_frequency_metrics(df)
        for val in result["account"]:
            assert isinstance(val, str)

    def test_integer_account_identifier_preserved(self):
        """Integer account IDs stay as integers in the output."""
        df = pd.DataFrame(
            {
                "account_id": [1, 2, 1],
                "created_at": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            }
        )
        result = compute_frequency_metrics(df)
        for val in result["account"]:
            assert isinstance(val, (int, np.integer))

    # ------------------------------------------------------------------
    # Duplicate counting
    # ------------------------------------------------------------------

    def test_duplicate_rows_each_counted(self):
        """Two identical rows both count as separate transactions."""
        # Same timestamp twice on same day → daily_counts=[2]
        df = _make_df(["A", "A"], ["2024-01-01", "2024-01-01"])
        result = compute_frequency_metrics(df)
        assert result.iloc[0]["mean_tx_per_day"] == pytest.approx(2.0)

    # ------------------------------------------------------------------
    # Null handling (delegated to _validate_dataframe)
    # ------------------------------------------------------------------

    def test_null_in_timestamp_raises(self):
        """Null values in the timestamp column raise ValueError."""
        df = pd.DataFrame(
            {
                "account_id": ["A", "B"],
                "created_at": pd.to_datetime([None, "2024-01-01"]),
            }
        )
        with pytest.raises(ValueError, match="created_at"):
            compute_frequency_metrics(df)

    def test_null_in_account_raises(self):
        """Null values in the account column raise ValueError."""
        df = pd.DataFrame(
            {
                "account_id": ["A", None],
                "created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        )
        with pytest.raises(ValueError, match="account_id"):
            compute_frequency_metrics(df)

    def test_missing_timestamp_column_raises(self):
        """Missing timestamp column raises ValueError."""
        df = pd.DataFrame({"account_id": ["A"], "wrong_col": [1]})
        with pytest.raises(ValueError, match="created_at"):
            compute_frequency_metrics(df)

    def test_missing_account_column_raises(self):
        """Missing account column raises ValueError."""
        df = pd.DataFrame(
            {"wrong_col": ["A"], "created_at": pd.to_datetime(["2024-01-01"])}
        )
        with pytest.raises(ValueError, match="account_id"):
            compute_frequency_metrics(df)

    # ------------------------------------------------------------------
    # Statistical properties
    # ------------------------------------------------------------------

    def test_sample_std_ddof1(self):
        """Standard deviation uses ddof=1 (sample), not ddof=0 (population)."""
        # days: [1, 0, 1]  → ddof=1 std ≠ ddof=0 std
        df = _make_df(["A", "A"], ["2024-01-01", "2024-01-03"])
        result = compute_frequency_metrics(df)
        expected_std = np.std([1, 0, 1], ddof=1)
        assert result.iloc[0]["std_tx_per_day"] == pytest.approx(expected_std)

    def test_burstiness_bounded(self):
        """Burstiness is always in the closed interval [-1, 1]."""
        df = _make_df(
            ["A"] * 5,
            ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-04", "2024-01-07"],
        )
        result = compute_frequency_metrics(df)
        b = result.iloc[0]["burstiness"]
        assert -1.0 <= b <= 1.0

    def test_std_non_negative(self):
        """Standard deviation is always non-negative."""
        df = _make_df(
            ["A", "A", "B"],
            ["2024-01-01", "2024-01-03", "2024-01-01"],
        )
        result = compute_frequency_metrics(df)
        assert (result["std_tx_per_day"] >= 0.0).all()
