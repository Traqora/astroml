"""Tests for RFC 7807 error conformance in graph snapshot windowing — issue #949.

`astroml/features/graph/snapshot.py` previously raised bare `ValueError`s
with only a free-text message for its three validation failures (invalid
window bounds, invalid day count, unknown window unit). That gives an API
boundary nothing structured to render as `application/problem+json` — it
can only forward `str(exc)`.

`SnapshotWindowError` (a `ValueError` subclass) now carries a `ProblemDetail`
with the RFC 7807 member set (`type`, `title`, `status`, `detail`,
`instance`). These tests cover the `ProblemDetail`/`SnapshotWindowError`
mechanics directly and each of the three call sites that raise them,
including that `except ValueError` callers (the pattern used by every
existing caller of `window_snapshot`) keep working unchanged.

See `tests/features/graph/conftest.py` for why this module is loaded via
`importlib` (through the `graph_snapshot_module` fixture) instead of a plain
`from astroml.features.graph.snapshot import ...` — plain import currently
fails repo-wide due to an unrelated, pre-existing SyntaxError in
`astroml/cache/graph_cache.py`.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def snapshot(graph_snapshot_module):
    return graph_snapshot_module


# ---------------------------------------------------------------------------
# ProblemDetail / SnapshotWindowError mechanics
# ---------------------------------------------------------------------------


def test_problem_detail_to_dict_has_rfc7807_member_set(snapshot) -> None:
    problem = snapshot.ProblemDetail(
        type="https://example.com/problems/x",
        title="Example problem",
        status=400,
        detail="something went wrong",
    )
    payload = problem.to_dict()

    assert set(payload.keys()) == {"type", "title", "status", "detail", "instance"}
    assert payload["type"] == "https://example.com/problems/x"
    assert payload["title"] == "Example problem"
    assert payload["status"] == 400
    assert payload["detail"] == "something went wrong"


def test_problem_detail_generates_a_unique_instance_by_default(snapshot) -> None:
    p1 = snapshot.ProblemDetail(type="t", title="T", status=400, detail="d1")
    p2 = snapshot.ProblemDetail(type="t", title="T", status=400, detail="d2")
    assert p1.instance != p2.instance
    assert p1.instance.startswith("urn:uuid:")


def test_snapshot_window_error_is_a_value_error(snapshot) -> None:
    problem = snapshot.ProblemDetail(type="t", title="T", status=400, detail="boom")
    err = snapshot.SnapshotWindowError(problem)
    assert isinstance(err, ValueError)
    assert str(err) == "boom"


def test_snapshot_window_error_to_problem_detail_matches_the_problem(snapshot) -> None:
    problem = snapshot.ProblemDetail(type="t", title="T", status=422, detail="boom")
    err = snapshot.SnapshotWindowError(problem)
    assert err.to_problem_detail() == problem.to_dict()


# ---------------------------------------------------------------------------
# window_snapshot: start_ts > end_ts
# ---------------------------------------------------------------------------


def test_window_snapshot_invalid_bounds_raises_snapshot_window_error(snapshot) -> None:
    with pytest.raises(snapshot.SnapshotWindowError) as exc_info:
        snapshot.window_snapshot([], start_ts=100, end_ts=50)

    err = exc_info.value
    problem = err.to_problem_detail()
    assert problem["status"] == 400
    assert problem["type"] == "https://astroml.dev/problems/graph-snapshot/invalid-window-bounds"
    assert "start_ts=100" in problem["detail"]
    assert "end_ts=50" in problem["detail"]


def test_window_snapshot_invalid_bounds_still_catchable_as_value_error(snapshot) -> None:
    """Existing callers only ever caught (or let propagate) plain ValueError;
    the new type must not break that contract."""
    with pytest.raises(ValueError):
        snapshot.window_snapshot([], start_ts=5, end_ts=1)


def test_window_snapshot_valid_bounds_do_not_raise(snapshot) -> None:
    nodes, edges = snapshot.window_snapshot([], start_ts=1, end_ts=1)
    assert nodes == set()
    assert edges == []


# ---------------------------------------------------------------------------
# snapshot_last_n_days: days <= 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("days", [0, -1, -30])
def test_snapshot_last_n_days_invalid_days_raises_snapshot_window_error(
    snapshot, days: int
) -> None:
    with pytest.raises(snapshot.SnapshotWindowError) as exc_info:
        snapshot.snapshot_last_n_days([], now_ts=1_000_000, days=days)

    problem = exc_info.value.to_problem_detail()
    assert problem["status"] == 400
    assert problem["type"] == "https://astroml.dev/problems/graph-snapshot/invalid-day-count"
    assert f"days={days}" in problem["detail"]


def test_snapshot_last_n_days_invalid_days_still_catchable_as_value_error(snapshot) -> None:
    with pytest.raises(ValueError):
        snapshot.snapshot_last_n_days([], now_ts=1_000_000, days=0)


def test_snapshot_last_n_days_valid_days_does_not_raise(snapshot) -> None:
    nodes, edges = snapshot.snapshot_last_n_days([], now_ts=1_000_000, days=1)
    assert nodes == set()
    assert edges == []


# ---------------------------------------------------------------------------
# _parse_window_size: unknown unit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", ["7x", "24w", "100m"])
def test_parse_window_size_unknown_unit_raises_snapshot_window_error(snapshot, window: str) -> None:
    with pytest.raises(snapshot.SnapshotWindowError) as exc_info:
        snapshot._parse_window_size(window)

    problem = exc_info.value.to_problem_detail()
    assert problem["status"] == 400
    assert problem["type"] == "https://astroml.dev/problems/graph-snapshot/unknown-window-unit"
    assert window in problem["detail"]


def test_parse_window_size_unknown_unit_still_catchable_as_value_error(snapshot) -> None:
    with pytest.raises(ValueError):
        snapshot._parse_window_size("7x")


@pytest.mark.parametrize("window", ["bad", "", "d", "7.5d"])
def test_parse_window_size_malformed_spec_raises_snapshot_window_error(
    snapshot, window: str
) -> None:
    """#949 — a spec whose numeric prefix fails to parse (including one that
    ends in a character that happens to look like a valid unit, e.g. "bad")
    previously escaped as a bare, unstructured `ValueError` straight out of
    `int()` instead of going through the RFC 7807 problem-detail path."""
    with pytest.raises(snapshot.SnapshotWindowError) as exc_info:
        snapshot._parse_window_size(window)

    problem = exc_info.value.to_problem_detail()
    assert problem["status"] == 400
    assert problem["type"] == "https://astroml.dev/problems/graph-snapshot/malformed-window-spec"


def test_parse_window_size_malformed_spec_still_catchable_as_value_error(snapshot) -> None:
    with pytest.raises(ValueError):
        snapshot._parse_window_size("bad")


@pytest.mark.parametrize(
    ("window", "expected_seconds"),
    [
        ("7d", 7 * 86400),
        ("24h", 24 * 3600),
        ("3600s", 3600),
    ],
)
def test_parse_window_size_valid_units_do_not_raise(
    snapshot, window: str, expected_seconds: int
) -> None:
    delta = snapshot._parse_window_size(window)
    assert delta.total_seconds() == expected_seconds


# ---------------------------------------------------------------------------
# Each failure kind has a distinct, stable `type` URI
# ---------------------------------------------------------------------------


def test_each_problem_kind_has_a_distinct_type_uri(snapshot) -> None:
    try:
        snapshot.window_snapshot([], start_ts=2, end_ts=1)
    except snapshot.SnapshotWindowError as exc:
        bounds_type = exc.problem.type

    try:
        snapshot.snapshot_last_n_days([], now_ts=1, days=0)
    except snapshot.SnapshotWindowError as exc:
        days_type = exc.problem.type

    try:
        snapshot._parse_window_size("1x")
    except snapshot.SnapshotWindowError as exc:
        unit_type = exc.problem.type

    assert len({bounds_type, days_type, unit_type}) == 3
