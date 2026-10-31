"""Bias tests for link-prediction negative sampling (#899).

``sample_negative_edges`` draws (u, v) uniformly from non-edges. These tests
check that the draw does not lean toward particular nodes, that it never
returns positives or self-loops, and that it stays reproducible with a seed.
"""

from __future__ import annotations

import random
from collections import Counter

from astroml.tasks.link_prediction_task import sample_negative_edges


def _max_relative_deviation(counts: Counter, num_nodes: int, total: int) -> float:
    expected = total / num_nodes
    return max(abs(counts.get(n, 0) - expected) / expected for n in range(num_nodes))


def test_src_and_dst_are_close_to_uniform_on_sparse_graph():
    num_nodes = 20
    rng = random.Random(7)
    # Sparse positives, so nearly every pair is a valid negative.
    positives = {(i, (i + 1) % num_nodes) for i in range(num_nodes)}
    negatives = sample_negative_edges(num_nodes, positives, num_samples=300, rng=rng)
    assert len(negatives) == 300

    src_counts = Counter(u for u, _ in negatives)
    dst_counts = Counter(v for _, v in negatives)
    # 300 samples over 20 nodes -> ~15 each; allow generous noise.
    assert _max_relative_deviation(src_counts, num_nodes, 300) < 0.8
    assert _max_relative_deviation(dst_counts, num_nodes, 300) < 0.8


def test_hub_with_many_positives_is_not_oversampled():
    """A hub's existing edges are excluded, so it should appear less, not more."""
    num_nodes = 30
    hub = 0
    positives = {(hub, v) for v in range(1, num_nodes)}
    rng = random.Random(3)
    negatives = sample_negative_edges(num_nodes, positives, num_samples=400, rng=rng)

    # The hub has no outgoing non-edges left, so it can never be a source.
    assert all(u != hub for u, _ in negatives)
    # It can still be a destination, at roughly the uniform rate.
    dst_counts = Counter(v for _, v in negatives)
    assert dst_counts[hub] < 2 * (len(negatives) / num_nodes)


def test_no_positives_self_loops_or_duplicates():
    num_nodes = 10
    positives = {(0, 1), (1, 2), (2, 3), (5, 5)}
    negatives = sample_negative_edges(num_nodes, positives, num_samples=50, rng=random.Random(1))
    assert not set(negatives) & positives
    assert all(u != v for u, v in negatives)
    assert len(negatives) == len(set(negatives))


def test_dense_graph_returns_only_what_exists():
    num_nodes = 4
    all_pairs = {(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v}
    positives = set(list(sorted(all_pairs))[:-2])  # leave two non-edges
    negatives = sample_negative_edges(
        num_nodes, positives, num_samples=10, max_attempts=50, rng=random.Random(0)
    )
    assert set(negatives) == all_pairs - positives


def test_same_seed_same_sample():
    positives = {(0, 1), (2, 3)}
    a = sample_negative_edges(15, positives, 40, rng=random.Random(42))
    b = sample_negative_edges(15, positives, 40, rng=random.Random(42))
    c = sample_negative_edges(15, positives, 40, rng=random.Random(43))
    assert a == b
    assert a != c
