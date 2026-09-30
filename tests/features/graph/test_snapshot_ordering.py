"""Temporal-window ordering in snapshot construction — issue #732.

A snapshot is consumed by a temporal model as a *sequence*, so two properties
have to hold for training to be reproducible:

* the edges of a window come back in the same order every time, and
* that order is the blockchain's, not an artefact of the query plan.

``ORDER BY timestamp`` alone gave neither.  A Stellar ledger closes all of its
operations in the same second, so a window is full of rows sharing a timestamp
and the database was free to return them in any order it liked — a different
plan, a different statistics snapshot or a different SQLite/Postgres version
produced a different sequence from unchanged data.

These tests pin the total order the builders now use,
``(timestamp, ledger_sequence, operation_id, hop_index, id)``, the validation
that rejects a broken ``presorted`` promise, and the index that lets the
planner produce that order without a sort.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from astroml.db.models import Base, NormalizedTransaction
from astroml.features.graph.snapshot import (
    Edge,
    SnapshotOrderingError,
    _snapshot_ordering,
    iter_db_snapshot_edges,
    iter_db_snapshots,
    window_snapshot,
)

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)


@pytest.fixture()
def engine():
    """A fresh SQLite database built from the ORM models."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = factory()
    yield db
    db.close()


def _row(ledger: int, hop: int, src: str, dst: str, ts: datetime, op: int | None = None):
    """A normalized activity row keyed the way ingestion writes one.

    The operation id is a toid, so ``ledger_sequence`` and ``operation_id``
    agree with each other exactly as they do in the real table.
    """
    return NormalizedTransaction(
        transaction_hash="a" * 64,
        ledger_sequence=ledger,
        operation_id=op if op is not None else ledger << 32,
        hop_index=hop,
        sender=src,
        receiver=dst,
        asset="native",
        amount=1.0,
        timestamp=ts,
    )


def _edges(*timestamps: int) -> list[Edge]:
    return [Edge(src=f"s{i}", dst=f"d{i}", timestamp=ts) for i, ts in enumerate(timestamps)]


class TestPresortedContract:
    """``presorted=True`` is a promise the caller makes, and it is checked."""

    def test_sorted_edges_are_accepted(self):
        edges = _edges(100, 200, 300)

        assert window_snapshot(edges, 100, 300, presorted=True)[1] == edges

    def test_ties_are_not_a_violation(self):
        """Every operation in a ledger shares that ledger's timestamp."""
        edges = _edges(100, 100, 100)

        assert len(window_snapshot(edges, 100, 100, presorted=True)[1]) == 3

    def test_out_of_order_edges_are_rejected(self):
        """The bisect below only works on sorted input, so an unsorted list
        used to return a window that was quietly wrong."""
        with pytest.raises(SnapshotOrderingError):
            window_snapshot(_edges(100, 300, 200), 100, 300, presorted=True)

    def test_the_error_is_a_value_error(self):
        """Callers already catching ValueError for bad bounds keep working."""
        assert issubclass(SnapshotOrderingError, ValueError)

    def test_the_error_names_the_offending_index(self):
        with pytest.raises(SnapshotOrderingError, match=r"index 1 is at 300"):
            window_snapshot(_edges(100, 300, 200), 100, 300, presorted=True)

    def test_unsorted_edges_are_still_handled_when_not_promised(self):
        """``presorted=False`` means "sort it for me", not an error."""
        shuffled = _edges(100, 200, 300)
        random.Random(7).shuffle(shuffled)

        _, window = window_snapshot(shuffled, 100, 300, presorted=False)

        assert [e.timestamp for e in window] == [100, 200, 300]


class TestCanonicalTieOrder:
    """With ``presorted=False`` the result depends on the edges, not the input."""

    def test_ties_are_ordered_by_endpoints(self):
        edges = [
            Edge(src="b", dst="x", timestamp=100),
            Edge(src="a", dst="z", timestamp=100),
            Edge(src="a", dst="y", timestamp=100),
        ]

        _, window = window_snapshot(edges, 100, 100, presorted=False)

        assert [(e.src, e.dst) for e in window] == [("a", "y"), ("a", "z"), ("b", "x")]

    def test_any_input_order_yields_the_same_sequence(self):
        """Sorting on the timestamp alone left ties in the caller's order."""
        edges = [
            Edge(src="b", dst="x", timestamp=100),
            Edge(src="a", dst="z", timestamp=100),
            Edge(src="c", dst="w", timestamp=200),
            Edge(src="a", dst="y", timestamp=100),
        ]
        expected = None

        for seed in range(8):
            shuffled = list(edges)
            random.Random(seed).shuffle(shuffled)
            _, window = window_snapshot(shuffled, 100, 200, presorted=False)
            sequence = [(e.timestamp, e.src, e.dst) for e in window]
            if expected is None:
                expected = sequence
            assert sequence == expected


class TestDatabaseTotalOrder:
    """The DB builders sort by the chain's order, not by timestamp alone."""

    def test_rows_sharing_a_timestamp_come_back_in_ledger_order(self, session):
        """Insertion order contradicts the natural key, so the test can tell
        them apart: a plain ``ORDER BY timestamp`` would return rowid order."""
        session.add_all(
            [
                _row(ledger=30, hop=0, src="r30", dst="x", ts=T0),
                _row(ledger=10, hop=0, src="r10", dst="x", ts=T0),
                _row(ledger=20, hop=0, src="r20", dst="x", ts=T0),
            ]
        )
        session.commit()

        windows = list(iter_db_snapshots("1h", t0=T0, t_now=T1, session=session, chunk_size=100))

        assert [e.src for e in windows[0].edges] == ["r10", "r20", "r30"]

    def test_hops_of_one_operation_are_ordered_by_hop_index(self, session):
        session.add_all(
            [
                _row(ledger=10, hop=2, src="hop2", dst="x", ts=T0),
                _row(ledger=10, hop=0, src="hop0", dst="x", ts=T0),
                _row(ledger=10, hop=1, src="hop1", dst="x", ts=T0),
            ]
        )
        session.commit()

        windows = list(iter_db_snapshots("1h", t0=T0, t_now=T1, session=session, chunk_size=100))

        assert [e.src for e in windows[0].edges] == ["hop0", "hop1", "hop2"]

    def test_a_window_is_identical_across_runs(self, engine):
        """The replay case: a second process, a second session, same sequence.

        This is the property training reproducibility actually rests on — the
        same window rebuilt later has to be the same sequence.
        """
        factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        first = factory()
        first.add_all(
            [
                _row(ledger=i, hop=0, src=f"r{i}", dst=f"r{(i + 1) % 7}", ts=T0)
                for i in (9, 3, 7, 1, 5)
            ]
        )
        first.commit()
        first.close()

        def build():
            db = factory()
            try:
                windows = list(iter_db_snapshots("1h", t0=T0, t_now=T1, session=db, chunk_size=2))
                return [(e.src, e.dst, e.timestamp) for e in windows[0].edges]
            finally:
                db.close()

        assert build() == build()

    def test_the_streaming_builder_uses_the_same_order(self, session):
        """``iter_db_snapshot_edges`` and ``iter_db_snapshots`` must not
        disagree — a caller switching between them for memory reasons would
        otherwise get a different sequence for the same window."""
        session.add_all(
            [
                _row(ledger=30, hop=0, src="r30", dst="x", ts=T0),
                _row(ledger=10, hop=0, src="r10", dst="x", ts=T0),
            ]
        )
        session.commit()

        materialized = next(
            iter_db_snapshots("1h", t0=T0, t_now=T1, session=session, chunk_size=100)
        )
        _, streamed = next(
            iter_db_snapshot_edges("1h", t0=T0, t_now=T1, session=session, chunk_size=100)
        )

        assert (
            [e.src for e in streamed]
            == [e.src for e in materialized.edges]
            == [
                "r10",
                "r30",
            ]
        )

    def test_every_window_is_time_ordered(self, session):
        session.add_all(
            _row(ledger=i, hop=0, src=f"r{i}", dst=f"r{i + 1}", ts=T0 + timedelta(minutes=i))
            for i in range(6)
        )
        session.commit()

        windows = list(
            iter_db_snapshots(
                "1h",
                t0=T0,
                t_now=T0 + timedelta(minutes=5),
                step="120s",
                session=session,
                chunk_size=2,
            )
        )

        assert len(windows) > 1
        for window in windows:
            stamps = [e.timestamp for e in window.edges]
            assert stamps == sorted(stamps)


class TestOrderingIsIndexBacked:
    """Performance — the total order is walked, not sorted for each window."""

    def test_the_total_order_is_the_documented_one(self):
        columns = [column.name for column in _snapshot_ordering()]

        assert columns == [
            "timestamp",
            "ledger_sequence",
            "operation_id",
            "hop_index",
            "id",
        ]

    def test_an_index_covers_the_order_and_the_range(self, engine):
        """A sort node is where the non-determinism came from; an index that
        matches the ORDER BY removes it and keeps the range scan cheap."""
        indexes = {
            index["name"]: index for index in inspect(engine).get_indexes("normalized_transactions")
        }

        assert "ix_normalized_transactions_timestamp_order" in indexes
        assert indexes["ix_normalized_transactions_timestamp_order"]["column_names"] == [
            "timestamp",
            "ledger_sequence",
            "operation_id",
            "hop_index",
        ]

    def test_the_index_leads_on_timestamp(self, engine):
        """``timestamp`` had no index at all, so the window range predicate was
        a sequential scan even before the ordering was a problem."""
        indexes = inspect(engine).get_indexes("normalized_transactions")
        leading = {columns["column_names"][0] for columns in indexes}

        assert "timestamp" in leading

    def test_the_index_is_defined_on_the_model_and_not_just_in_the_database(self):
        """The migration and the ORM model have to agree, or a database built
        by ``create_all`` and one built by ``alembic upgrade`` diverge."""
        names = {index.name for index in NormalizedTransaction.__table__.indexes}

        assert "ix_normalized_transactions_timestamp_order" in names
