"""Property-based tests for feature computations using Hypothesis.

This module contains property-based tests that verify invariants and edge cases
for feature computation functions.

Properties tested:
- Feature values should be finite numbers (no NaN/Inf)
- Feature DataFrame should have consistent index
- Feature computation should be idempotent (same input = same output)
"""

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes

from astroml.features.frequency import (
    compute_burstiness,
    compute_daily_transaction_counts,
)
from astroml.features.structural_importance import (
    compute_degree_centrality,
    compute_pagerank,
)

# Hypothesis settings for property-based tests
hypothesis_settings = settings(
    max_examples=200,
    deadline=None,
    phases=[Phase.generate, Phase.shrink],
    suppress_health_check=[HealthCheck.too_slow],
)


class TestFeatureComputationProperties:
    """Property-based tests for feature computations."""

    @given(
        data_frames(
            columns=[
                column("account_id", st.text(min_size=1, max_size=10)),
                column(
                    "timestamp",
                    st.datetimes(
                        min_value=pd.Timestamp("2020-01-01"), max_value=pd.Timestamp("2024-12-31")
                    ),
                ),
            ],
            rows=st.integers(min_value=1, max_value=100),
            index=range_indexes(),
        )
    )
    @settings(hypothesis_settings)
    def test_daily_transaction_counts_finite(self, df):
        """Property: Daily transaction counts should always be finite numbers."""
        result = compute_daily_transaction_counts(df, "account_id", "timestamp")

        # Check that all values are finite
        assert np.all(np.isfinite(result.values)), "Daily transaction counts must be finite"

        # Check that values are non-negative
        assert np.all(result.values >= 0), "Daily transaction counts must be non-negative"

    @given(
        data_frames(
            columns=[
                column("account_id", st.text(min_size=1, max_size=10)),
                column(
                    "timestamp",
                    st.datetimes(
                        min_value=pd.Timestamp("2020-01-01"), max_value=pd.Timestamp("2024-12-31")
                    ),
                ),
            ],
            rows=st.integers(min_value=1, max_value=100),
            index=range_indexes(),
        )
    )
    @settings(hypothesis_settings)
    def test_daily_transaction_counts_consistent_index(self, df):
        """Property: Feature DataFrame should have consistent index (entity IDs)."""
        result = compute_daily_transaction_counts(df, "account_id", "timestamp")

        # Check that result index contains only account IDs from input
        unique_accounts = df["account_id"].unique()
        result_accounts = result.index.unique()

        # All result accounts should be in input accounts
        assert set(result_accounts).issubset(
            set(unique_accounts)
        ), "Result index should only contain entities from input"

    @given(
        data_frames(
            columns=[
                column("account_id", st.text(min_size=1, max_size=10)),
                column(
                    "timestamp",
                    st.datetimes(
                        min_value=pd.Timestamp("2020-01-01"), max_value=pd.Timestamp("2024-12-31")
                    ),
                ),
            ],
            rows=st.integers(min_value=1, max_value=100),
            index=range_indexes(),
        )
    )
    @settings(hypothesis_settings)
    def test_daily_transaction_counts_idempotent(self, df):
        """Property: Feature computation should be idempotent (same input = same output)."""
        result1 = compute_daily_transaction_counts(df, "account_id", "timestamp")
        result2 = compute_daily_transaction_counts(df, "account_id", "timestamp")

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    @given(
        st.lists(
            st.datetimes(
                min_value=pd.Timestamp("2020-01-01"), max_value=pd.Timestamp("2024-12-31")
            ),
            min_size=1,
            max_size=100,
        )
    )
    @settings(hypothesis_settings)
    def test_burstiness_finite(self, timestamps):
        """Property: Burstiness metric should always be finite."""
        timestamps_series = pd.Series(timestamps)
        result = compute_burstiness(timestamps_series)

        # Burstiness should be a finite number
        assert isinstance(result, (int, float)), "Burstiness should be numeric"
        assert np.isfinite(result), "Burstiness must be finite"

    @given(
        st.lists(
            st.datetimes(
                min_value=pd.Timestamp("2020-01-01"), max_value=pd.Timestamp("2024-12-31")
            ),
            min_size=1,
            max_size=100,
        )
    )
    @settings(hypothesis_settings)
    def test_burstiness_idempotent(self, timestamps):
        """Property: Burstiness computation should be idempotent."""
        timestamps_series = pd.Series(timestamps)
        result1 = compute_burstiness(timestamps_series)
        result2 = compute_burstiness(timestamps_series)

        # Results should be identical
        assert result1 == result2, "Burstiness computation should be idempotent"


class TestGraphFeatureProperties:
    """Property-based tests for graph feature computations."""

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.floats(min_value=0.1, max_value=100.0),
                st.integers(min_value=0, max_value=10000),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(hypothesis_settings)
    def test_degree_centrality_finite(self, edges_data):
        """Property: Degree centrality values should always be finite."""
        edges = [
            {"src": src, "dst": dst, "amount": amount, "timestamp": timestamp}
            for src, dst, amount, timestamp in edges_data
        ]

        result = compute_degree_centrality(edges, weighted=False)

        # All centrality values should be finite
        assert np.all(np.isfinite(result.values)), "Degree centrality must be finite"

        # Centrality values should be in [0, 1] for normalized version
        assert np.all(result.values >= 0), "Degree centrality must be non-negative"
        assert np.all(result.values <= 1), "Normalized degree centrality must be <= 1"

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.floats(min_value=0.1, max_value=100.0),
                st.integers(min_value=0, max_value=10000),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(hypothesis_settings)
    def test_degree_centrality_consistent_index(self, edges_data):
        """Property: Graph feature index should contain all nodes from edges."""
        edges = [
            {"src": src, "dst": dst, "amount": amount, "timestamp": timestamp}
            for src, dst, amount, timestamp in edges_data
        ]

        result = compute_degree_centrality(edges, weighted=False)

        # Collect all unique nodes from edges
        nodes = set()
        for edge in edges:
            nodes.add(edge["src"])
            nodes.add(edge["dst"])

        # Result index should contain all nodes
        assert set(result.index) == nodes, "Result index should contain all graph nodes"

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.floats(min_value=0.1, max_value=100.0),
                st.integers(min_value=0, max_value=10000),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(hypothesis_settings)
    def test_pagerank_properties(self, edges_data):
        """Property: PageRank should sum to 1 and all values should be positive."""
        edges = [
            {"src": src, "dst": dst, "amount": amount, "timestamp": timestamp}
            for src, dst, amount, timestamp in edges_data
        ]

        result = compute_pagerank(edges)

        # All PageRank values should be positive
        assert np.all(result.values > 0), "PageRank values must be positive"

        # PageRank values should sum to 1 (or very close due to floating point)
        assert np.abs(result.sum() - 1.0) < 1e-10, "PageRank应该 sum to 1"

        # All values should be finite
        assert np.all(np.isfinite(result.values)), "PageRank must be finite"

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.floats(min_value=0.1, max_value=100.0),
                st.integers(min_value=0, max_value=10000),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(hypothesis_settings)
    def test_graph_features_idempotent(self, edges_data):
        """Property: Graph feature computation should be idempotent."""
        edges = [
            {"src": src, "dst": dst, "amount": amount, "timestamp": timestamp}
            for src, dst, amount, timestamp in edges_data
        ]

        result1 = compute_degree_centrality(edges, weighted=False)
        result2 = compute_degree_centrality(edges, weighted=False)

        # Results should be identical
        pd.testing.assert_series_equal(result1, result2)


class TestGraphBuildingProperties:
    """Property-based tests for graph building invariants."""

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.floats(min_value=0.1, max_value=100.0),
                st.integers(min_value=0, max_value=10000),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(hypothesis_settings)
    def test_node_count_ge_edge_count(self, edges_data):
        """Property: Graph should always have node_count >= edge_count after deduplication."""
        edges = [
            {"src": src, "dst": dst, "amount": amount, "timestamp": timestamp}
            for src, dst, amount, timestamp in edges_data
        ]

        # Count unique nodes
        nodes = set()
        for edge in edges:
            nodes.add(edge["src"])
            nodes.add(edge["dst"])

        # Count unique edges (considering direction)
        unique_edges = set((e["src"], e["dst"]) for e in edges)

        # Node count should be >= edge count for simple graphs
        # (This is a basic graph theory property)
        assert (
            len(nodes) >= len(unique_edges) or len(unique_edges) == 0
        ), "Node count should be >= edge count for simple graphs"

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.floats(min_value=0.1, max_value=100.0),
                st.integers(min_value=0, max_value=10000),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(hypothesis_settings)
    def test_no_orphan_edges(self, edges_data):
        """Property: No orphan edges (src and dst must exist in node set)."""
        edges = [
            {"src": src, "dst": dst, "amount": amount, "timestamp": timestamp}
            for src, dst, amount, timestamp in edges_data
        ]

        # Collect all nodes
        nodes = set()
        for edge in edges:
            nodes.add(edge["src"])
            nodes.add(edge["dst"])

        # Verify all edge endpoints are in node set
        for edge in edges:
            assert edge["src"] in nodes, f"Edge source {edge['src']} not in node set"
            assert edge["dst"] in nodes, f"Edge destination {edge['dst']} not in node set"

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.text(min_size=1, max_size=5, alphabet="ABC"),
                st.floats(min_value=0.1, max_value=100.0),
                st.integers(min_value=0, max_value=10000),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(hypothesis_settings)
    def test_timestamps_monotonic_when_sorted(self, edges_data):
        """Property: Timestamps should be monotonically increasing when sorted."""
        edges = [
            {"src": src, "dst": dst, "amount": amount, "timestamp": timestamp}
            for src, dst, amount, timestamp in edges_data
        ]

        # Sort edges by timestamp
        sorted_edges = sorted(edges, key=lambda e: e["timestamp"])

        # Verify monotonic increase
        for i in range(1, len(sorted_edges)):
            assert (
                sorted_edges[i]["timestamp"] >= sorted_edges[i - 1]["timestamp"]
            ), "Timestamps should be monotonically increasing after sorting"
