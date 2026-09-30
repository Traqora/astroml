"""Tests for astroml.training.temporal_split."""

import warnings
from dataclasses import dataclass

import pytest

from astroml.training.temporal_split import (
    GraphSplitResult,
    LeakageError,
    TemporalSplitter,
    temporal_graph_split,
    validate_graph_split,
)

# ---------------------------------------------------------------------------
# Minimal edge stub (mirrors astroml.features.graph.snapshot.Edge)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeEdge:
    src: str
    dst: str
    timestamp: int


def _make_edges(n: int = 10) -> list:
    """Return n sequential edges with timestamps 0..n-1."""
    return [FakeEdge(src=f"a{i}", dst=f"b{i}", timestamp=i) for i in range(n)]


# ---------------------------------------------------------------------------
# temporal_graph_split — ratio mode
# ---------------------------------------------------------------------------


class TestTemporalGraphSplitRatio:
    def test_basic_split(self):
        edges = _make_edges(10)
        result = temporal_graph_split(edges, train_ratio=0.8)

        assert len(result.train_edges) == 8
        assert len(result.test_edges) == 2

    def test_train_strictly_before_test(self):
        edges = _make_edges(10)
        result = temporal_graph_split(edges, train_ratio=0.7)

        train_max = max(e.timestamp for e in result.train_edges)
        test_min = min(e.timestamp for e in result.test_edges)
        assert train_max < test_min

    def test_shuffled_input_still_splits_temporally(self):
        import random

        edges = _make_edges(20)
        random.seed(0)
        random.shuffle(edges)

        result = temporal_graph_split(edges, train_ratio=0.8)

        train_max = max(e.timestamp for e in result.train_edges)
        test_min = min(e.timestamp for e in result.test_edges)
        assert train_max < test_min

    def test_invalid_ratio_raises(self):
        edges = _make_edges(5)
        with pytest.raises(ValueError, match="train_ratio"):
            temporal_graph_split(edges, train_ratio=0.0)
        with pytest.raises(ValueError, match="train_ratio"):
            temporal_graph_split(edges, train_ratio=1.0)

    def test_empty_edges_returns_empty(self):
        result = temporal_graph_split([], train_ratio=0.8)
        assert result.train_edges == []
        assert result.test_edges == []


# ---------------------------------------------------------------------------
# temporal_graph_split — cutoff mode
# ---------------------------------------------------------------------------


class TestTemporalGraphSplitCutoff:
    def test_cutoff_partitioning(self):
        edges = _make_edges(10)
        result = temporal_graph_split(edges, cutoff=5)

        assert all(e.timestamp < 5 for e in result.train_edges)
        assert all(e.timestamp >= 5 for e in result.test_edges)
        assert len(result.train_edges) + len(result.test_edges) == 10

    def test_cutoff_before_all_warns(self):
        edges = _make_edges(5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = temporal_graph_split(edges, cutoff=0)
        assert result.train_edges == []
        assert any("empty" in str(w.message).lower() for w in caught)

    def test_cutoff_after_all_warns(self):
        edges = _make_edges(5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = temporal_graph_split(edges, cutoff=999)
        assert result.test_edges == []
        assert any("empty" in str(w.message).lower() for w in caught)

    def test_resolved_cutoff_stored(self):
        edges = _make_edges(10)
        result = temporal_graph_split(edges, cutoff=7)
        assert result.cutoff == 7


# ---------------------------------------------------------------------------
# temporal_graph_split — edge validation
# ---------------------------------------------------------------------------


class TestTemporalGraphSplitValidation:
    def test_missing_time_attr_raises(self):
        @dataclass
        class BadEdge:
            src: str
            dst: str
            # no timestamp

        edges = [BadEdge(src="a", dst="b")]
        with pytest.raises(ValueError, match="no attribute"):
            temporal_graph_split(edges)

    def test_custom_time_attr(self):
        @dataclass(frozen=True)
        class TimedEdge:
            src: str
            dst: str
            created_at: int

        edges = [TimedEdge(src=f"a{i}", dst=f"b{i}", created_at=i) for i in range(10)]
        result = temporal_graph_split(edges, train_ratio=0.7, time_attr="created_at")

        assert len(result.train_edges) == 7
        assert len(result.test_edges) == 3


# ---------------------------------------------------------------------------
# validate_graph_split
# ---------------------------------------------------------------------------


class TestValidateGraphSplit:
    def test_clean_split_returns_true(self):
        edges = _make_edges(10)
        result = temporal_graph_split(edges, train_ratio=0.8)
        assert validate_graph_split(result) is True

    def test_overlap_raises_leakage_error(self):
        edges = _make_edges(10)
        # Manually construct an overlapping result.
        bad = GraphSplitResult(
            train_edges=edges[:7],  # timestamps 0-6
            test_edges=edges[5:],  # timestamps 5-9 — overlap at 5, 6
            cutoff=5,
        )
        with pytest.raises(LeakageError, match="overlap"):
            validate_graph_split(bad)

    def test_empty_partitions_are_valid(self):
        edges = _make_edges(5)
        empty_result = GraphSplitResult(train_edges=[], test_edges=edges, cutoff=0)
        assert validate_graph_split(empty_result) is True


# ---------------------------------------------------------------------------
# TemporalSplitter — DataFrame
# ---------------------------------------------------------------------------


class TestTemporalSplitterDataFrame:
    def test_dataframe_split(self):
        import numpy as np
        import pandas as pd

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
                "value": np.arange(10, dtype=float),
            }
        )
        splitter = TemporalSplitter(train_ratio=0.8, time_col="timestamp")
        train, test = splitter.split_dataframe(df)

        assert len(train) == 8
        assert len(test) == 2
        assert train["timestamp"].max() < test["timestamp"].min()

    def test_dataframe_cutoff(self):
        import numpy as np
        import pandas as pd

        df = pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-01", periods=10, freq="D"),
                "v": np.arange(10),
            }
        )
        cutoff = pd.Timestamp("2024-01-06")
        splitter = TemporalSplitter(cutoff=cutoff, time_col="ts")
        train, test = splitter.split_dataframe(df)

        assert (train["ts"] < cutoff).all()
        assert (test["ts"] >= cutoff).all()

    def test_dataframe_overlap_raises(self):
        """TemporalSplitter validates and re-raises LeakageError."""
        import numpy as np
        import pandas as pd

        from astroml.training.temporal_split import LeakageError, validate_temporal_split

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
                "v": np.arange(10),
            }
        )
        # Directly call validate_temporal_split with overlapping frames.
        train = df.iloc[:7].copy()
        test = df.iloc[5:].copy()
        with pytest.raises(LeakageError, match="overlap"):
            validate_temporal_split(train, test, "timestamp")


# ---------------------------------------------------------------------------
# TemporalSplitter — graph edges
# ---------------------------------------------------------------------------


class TestTemporalSplitterEdges:
    def test_edge_split_via_splitter(self):
        edges = _make_edges(20)
        splitter = TemporalSplitter(train_ratio=0.75)
        result = splitter.split_edges(edges)

        assert len(result.train_edges) == 15
        assert len(result.test_edges) == 5
        assert validate_graph_split(result) is True

    def test_splitter_invalid_ratio(self):
        with pytest.raises(ValueError, match="train_ratio"):
            TemporalSplitter(train_ratio=1.5)

    def test_no_leakage_guarantee(self):
        """Core property: no test edge timestamp precedes any train timestamp."""
        import random

        edges = _make_edges(100)
        random.seed(42)
        random.shuffle(edges)

        splitter = TemporalSplitter(train_ratio=0.8)
        result = splitter.split_edges(edges)

        train_timestamps = {e.timestamp for e in result.train_edges}
        test_timestamps = {e.timestamp for e in result.test_edges}
        # The maximum training timestamp must be strictly less than every test timestamp.
        assert max(train_timestamps) < min(test_timestamps)
