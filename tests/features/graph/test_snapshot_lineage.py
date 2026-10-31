"""Completeness tests for the snapshot lineage API — issue #888.

``snapshot_lineage`` is the auditable record of *which* slice of data produced
a snapshot window.  These tests pin every field an audit/debugging tool needs
(source, bounds, counts, ordering contract, temporal validity), the empty-window
edge case, deterministic ids, and parity between the function and the
``SnapshotWindow.lineage`` convenience method.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from astroml.features.graph.snapshot import (
    SNAPSHOT_ORDER_COLUMNS,
    SNAPSHOT_SOURCE,
    Edge,
    SnapshotLineage,
    SnapshotWindow,
    snapshot_lineage,
)

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = T0 + timedelta(hours=1)


def _window(edges: list[Edge] | None = None, *, index: int = 0) -> SnapshotWindow:
    edges = (
        edges
        if edges is not None
        else [
            Edge(src="alice", dst="bob", timestamp=1_000),
            Edge(src="bob", dst="carol", timestamp=1_100),
        ]
    )
    nodes = {e.src for e in edges} | {e.dst for e in edges}
    return SnapshotWindow(index=index, start=T0, end=T1, edges=edges, nodes=nodes)


def test_lineage_contains_the_complete_provenance_record() -> None:
    lineage = snapshot_lineage(_window())

    assert isinstance(lineage, SnapshotLineage)
    assert lineage.snapshot_id
    assert lineage.source == SNAPSHOT_SOURCE == "normalized_transactions"
    assert lineage.window_index == 0
    assert lineage.start == T0
    assert lineage.end == T1
    assert lineage.edge_count == 2
    assert lineage.node_count == 3
    assert lineage.first_timestamp == 1_000
    assert lineage.last_timestamp == 1_100
    assert lineage.order_by == SNAPSHOT_ORDER_COLUMNS
    assert lineage.temporal_order_valid is True


def test_lineage_to_dict_exposes_every_field() -> None:
    payload = snapshot_lineage(_window()).to_dict()

    assert set(payload) == {
        "snapshot_id",
        "source",
        "window_index",
        "start",
        "end",
        "edge_count",
        "node_count",
        "first_timestamp",
        "last_timestamp",
        "order_by",
        "temporal_order_valid",
        "seed",
        "git_commit",
    }
    assert payload["start"] == T0.isoformat()
    assert payload["end"] == T1.isoformat()
    assert payload["order_by"] == list(SNAPSHOT_ORDER_COLUMNS)


def test_lineage_for_empty_window_is_valid() -> None:
    lineage = snapshot_lineage(_window(edges=[]))

    assert lineage.edge_count == 0
    assert lineage.node_count == 0
    assert lineage.first_timestamp is None
    assert lineage.last_timestamp is None
    assert lineage.temporal_order_valid is True


def test_lineage_flags_a_broken_temporal_order() -> None:
    lineage = snapshot_lineage(
        _window(
            [
                Edge(src="a", dst="b", timestamp=2_000),
                Edge(src="c", dst="d", timestamp=1_000),
            ]
        )
    )

    assert lineage.temporal_order_valid is False


def test_default_snapshot_id_is_deterministic() -> None:
    first = snapshot_lineage(_window())
    rebuilt = snapshot_lineage(_window())

    assert first.snapshot_id == rebuilt.snapshot_id
    assert "0" in first.snapshot_id


def test_custom_snapshot_id_and_source_are_preserved() -> None:
    lineage = snapshot_lineage(_window(index=4), snapshot_id="snap-4", source="custom_table")

    assert lineage.snapshot_id == "snap-4"
    assert lineage.source == "custom_table"
    assert lineage.window_index == 4


def test_window_lineage_method_matches_function() -> None:
    window = _window(index=2)

    assert window.lineage() == snapshot_lineage(window)
    assert window.lineage(snapshot_id="explicit").snapshot_id == "explicit"


def test_seed_defaults_to_none_and_is_recorded_when_supplied() -> None:
    """GitHub Issue #840: a result's lineage must be able to trace back to
    the RNG seed that produced it. seed is purely a caller-supplied value,
    not derived from the window, so it defaults to None and is recorded
    verbatim whenever one is passed."""
    unspecified = snapshot_lineage(_window())
    assert unspecified.seed is None

    with_seed = snapshot_lineage(_window(), seed=42)
    assert with_seed.seed == 42
    assert with_seed.to_dict()["seed"] == 42


def test_window_lineage_method_forwards_seed() -> None:
    window = _window(index=2)
    assert window.lineage(seed=7).seed == 7


def test_git_commit_is_captured_automatically(monkeypatch) -> None:
    """GitHub Issue #840: every lineage record carries the git commit of
    the working tree at run time, via detect_git_commit(), so a result can
    be traced back to the exact code revision that produced it."""
    import astroml.features.graph.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "detect_git_commit", lambda: "abc123deadbeef")

    lineage = snapshot_lineage(_window())
    assert lineage.git_commit == "abc123deadbeef"
    assert lineage.to_dict()["git_commit"] == "abc123deadbeef"


def test_git_commit_is_none_when_unavailable(monkeypatch) -> None:
    """detect_git_commit() returning None (not a git checkout, git not
    installed, etc.) must not break lineage construction (#840)."""
    import astroml.features.graph.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "detect_git_commit", lambda: None)

    lineage = snapshot_lineage(_window())
    assert lineage.git_commit is None
