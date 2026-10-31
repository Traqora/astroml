import warnings

import numpy as np
import pandas as pd
import pytest

from astroml.validation import leakage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n=10):
    """Return a simple time-ordered DataFrame for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "amount": np.arange(n, dtype=float),
            "score": np.random.default_rng(42).random(n),
        }
    )


# ---------------------------------------------------------------------------
# temporal_train_test_split
# ---------------------------------------------------------------------------


def test_temporal_split_cutoff():
    df = _make_df(10)
    cutoff = pd.Timestamp("2024-01-06")
    train, test = leakage.temporal_train_test_split(df, "timestamp", cutoff=cutoff)

    assert (train["timestamp"] < cutoff).all()
    assert (test["timestamp"] >= cutoff).all()
    assert len(train) + len(test) == len(df)


def test_temporal_split_ratio():
    df = _make_df(10)
    train, test = leakage.temporal_train_test_split(df, "timestamp", train_ratio=0.8)

    assert len(train) == 8
    assert len(test) == 2
    # Train timestamps must precede test timestamps.
    assert train["timestamp"].max() < test["timestamp"].min()


def test_temporal_split_missing_col():
    df = _make_df(5)
    with pytest.raises(ValueError, match="not found"):
        leakage.temporal_train_test_split(df, "nonexistent")


def test_temporal_split_nulls_rejected():
    df = _make_df(5)
    df.loc[2, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="null"):
        leakage.temporal_train_test_split(df, "timestamp")


def test_temporal_split_empty_partition_warns():
    df = _make_df(5)
    # Cutoff after all data → empty test partition.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train, test = leakage.temporal_train_test_split(
            df, "timestamp", cutoff=pd.Timestamp("2099-01-01")
        )
    assert test.empty
    assert any("empty" in str(w.message).lower() for w in caught)


def test_temporal_split_invalid_ratio():
    df = _make_df(5)
    with pytest.raises(ValueError, match="train_ratio"):
        leakage.temporal_train_test_split(df, "timestamp", train_ratio=0.0)
    with pytest.raises(ValueError, match="train_ratio"):
        leakage.temporal_train_test_split(df, "timestamp", train_ratio=1.0)


# ---------------------------------------------------------------------------
# validate_temporal_split
# ---------------------------------------------------------------------------


def test_validate_split_clean():
    df = _make_df(10)
    train, test = leakage.temporal_train_test_split(df, "timestamp")
    assert leakage.validate_temporal_split(train, test, "timestamp") is True


def test_validate_split_overlap():
    df = _make_df(10)
    # Manually create overlapping partitions.
    train = df.iloc[:7].copy()
    test = df.iloc[5:].copy()  # rows 5-6 overlap
    with pytest.raises(leakage.LeakageError, match="overlap"):
        leakage.validate_temporal_split(train, test, "timestamp")


def test_validate_split_empty_partitions():
    df = _make_df(5)
    empty = df.iloc[:0]
    assert leakage.validate_temporal_split(empty, df, "timestamp") is True
    assert leakage.validate_temporal_split(df, empty, "timestamp") is True


def test_ratio_split_with_a_tied_timestamp_at_the_boundary_is_flagged_as_leakage():
    """Temporal split leakage guard (issue #837).

    The classic silent-leakage case this guard exists for: two rows share
    the exact same timestamp, and a plain index-based ratio split happens
    to put one in train and the other in test. Both rows carry the same
    time value, so the test set is not strictly "the future" relative to
    train — `train_max >= test_min` must catch this exactly like any other
    overlap, not treat a tie as a clean boundary.
    """
    df = pd.DataFrame(
        {
            # Index 4 and 5 share the same timestamp; a 0.5 ratio split on
            # 10 rows cuts exactly between them.
            "timestamp": [0, 1, 2, 3, 4, 4, 6, 7, 8, 9],
            "amount": np.arange(10, dtype=float),
        }
    )
    train, test = leakage.temporal_train_test_split(df, "timestamp", train_ratio=0.5)
    assert train["timestamp"].tolist() == [0, 1, 2, 3, 4]
    assert test["timestamp"].tolist() == [4, 6, 7, 8, 9]

    with pytest.raises(leakage.LeakageError, match="overlap"):
        leakage.validate_temporal_split(train, test, "timestamp")


def test_cutoff_split_with_a_row_exactly_at_the_cutoff_goes_to_test_and_is_clean():
    """Cutoff mode's `< cutoff` (strict) comparison means a row exactly at
    the cutoff value lands in test, not train — so, unlike ratio mode, an
    exact-cutoff tie is not leakage: train's rows are all strictly earlier.
    This is the deliberate complement to the ratio-mode tie case above."""
    df = pd.DataFrame({"timestamp": [0, 1, 2, 3, 4], "amount": np.arange(5, dtype=float)})
    train, test = leakage.temporal_train_test_split(df, "timestamp", cutoff=3)
    assert train["timestamp"].tolist() == [0, 1, 2]
    assert test["timestamp"].tolist() == [3, 4]
    assert leakage.validate_temporal_split(train, test, "timestamp") is True


# ---------------------------------------------------------------------------
# check_feature_leakage
# ---------------------------------------------------------------------------


def test_check_feature_leakage_unsorted():
    df = _make_df(10).sample(frac=1, random_state=0)  # shuffle
    results = leakage.check_feature_leakage(df, "timestamp")
    assert any(w.warning_type == "unsorted" for w in results)


def test_check_feature_leakage_constant_col():
    df = _make_df(10)
    df["const"] = 1.0
    results = leakage.check_feature_leakage(df, "timestamp")
    assert any(w.warning_type == "constant" and w.column == "const" for w in results)


def test_check_feature_leakage_clean():
    df = _make_df(10)
    results = leakage.check_feature_leakage(df, "timestamp")
    assert results == []


# ---------------------------------------------------------------------------
# check_target_leakage
# ---------------------------------------------------------------------------


def test_check_target_leakage_high_corr():
    df = _make_df(20)
    # Create a feature that is a near-perfect linear copy of the target.
    df["target"] = df["amount"]
    df["leaky"] = df["amount"] * 2.0 + 0.5
    results = leakage.check_target_leakage(df, "target", feature_cols=["leaky", "score"])
    assert any(w.column == "leaky" for w in results)


def test_check_target_leakage_clean():
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {
            "feature": rng.random(50),
            "target": rng.random(50),
        }
    )
    results = leakage.check_target_leakage(df, "target")
    assert results == []
