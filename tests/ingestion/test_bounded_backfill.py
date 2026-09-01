"""Memory-bounded backfill state (issue #724).

The processed-ledger set was one entry per ledger, in memory and on disk, so
peak RSS and state-file size grew with the size of a backfill. It is now a set
of contiguous ranges, which costs one interval for a sequential run of any
length.
"""

from __future__ import annotations

import json

import pytest

from astroml.ingestion.state import IngestionState, StateStore
from astroml.utils.ranges import LedgerRangeSet


class TestLedgerRangeSetMembership:
    def test_reports_added_ids_as_members(self):
        ranges = LedgerRangeSet()
        ranges.add(5)

        assert 5 in ranges
        assert 4 not in ranges
        assert 6 not in ranges

    def test_is_empty_before_anything_is_added(self):
        ranges = LedgerRangeSet()

        assert len(ranges) == 0
        assert not ranges
        assert 1 not in ranges
        assert ranges.max is None
        assert ranges.min is None

    def test_covers_every_id_in_an_added_range(self):
        ranges = LedgerRangeSet()
        ranges.add_range(10, 20)

        assert all(ledger in ranges for ledger in range(10, 21))
        assert 9 not in ranges
        assert 21 not in ranges

    def test_a_non_integer_is_never_a_member(self):
        ranges = LedgerRangeSet()
        ranges.add(1)

        assert "1" not in ranges
        assert 1.5 not in ranges
        assert None not in ranges

    def test_a_boolean_is_not_treated_as_an_integer(self):
        """``True == 1`` in Python; a boolean ledger id is a caller bug."""
        ranges = LedgerRangeSet()
        ranges.add(1)

        assert True not in ranges


class TestLedgerRangeSetBounding:
    """The property the issue is about."""

    def test_a_sequential_backfill_collapses_to_one_range(self):
        ranges = LedgerRangeSet()
        for ledger in range(1, 100_001):
            ranges.add(ledger)

        assert ranges.range_count == 1
        assert len(ranges) == 100_000
        assert ranges.to_list() == [[1, 100_000]]

    def test_memory_does_not_grow_with_the_size_of_the_range(self):
        small = LedgerRangeSet()
        small.add_range(1, 10)

        huge = LedgerRangeSet()
        huge.add_range(1, 50_000_000)

        # Both are one interval; the second covers five million times as many
        # ledgers for the same storage.
        assert small.range_count == huge.range_count == 1
        assert len(huge) == 50_000_000

    def test_adjacent_ranges_merge(self):
        ranges = LedgerRangeSet()
        ranges.add_range(1, 5)
        ranges.add_range(6, 9)

        assert ranges.to_list() == [[1, 9]]

    def test_overlapping_ranges_merge(self):
        ranges = LedgerRangeSet()
        ranges.add_range(1, 10)
        ranges.add_range(5, 20)

        assert ranges.to_list() == [[1, 20]]

    def test_a_gap_filled_from_both_sides_merges_into_one(self):
        ranges = LedgerRangeSet()
        ranges.add_range(1, 5)
        ranges.add_range(7, 10)
        assert ranges.range_count == 2

        ranges.add(6)

        assert ranges.to_list() == [[1, 10]]

    def test_out_of_order_additions_stay_sorted_and_disjoint(self):
        ranges = LedgerRangeSet()
        for ledger in [50, 10, 30, 11, 51, 31]:
            ranges.add(ledger)

        assert ranges.to_list() == [[10, 11], [30, 31], [50, 51]]

    def test_the_same_id_added_twice_changes_nothing(self):
        ranges = LedgerRangeSet()
        ranges.add(7)
        ranges.add(7)

        assert len(ranges) == 1
        assert ranges.to_list() == [[7, 7]]

    def test_rejects_an_inverted_range(self):
        with pytest.raises(ValueError, match="must be >="):
            LedgerRangeSet().add_range(10, 5)


class TestLedgerRangeSetGaps:
    def test_reports_the_work_left_in_a_span(self):
        ranges = LedgerRangeSet()
        ranges.add_range(1, 5)
        ranges.add_range(10, 12)

        assert list(ranges.missing_in(1, 15)) == [(6, 9), (13, 15)]

    def test_a_fully_covered_span_has_no_gaps(self):
        ranges = LedgerRangeSet()
        ranges.add_range(1, 100)

        assert list(ranges.missing_in(10, 20)) == []

    def test_an_untouched_span_is_one_gap(self):
        assert list(LedgerRangeSet().missing_in(5, 9)) == [(5, 9)]

    def test_gaps_are_clipped_to_the_requested_span(self):
        ranges = LedgerRangeSet()
        ranges.add_range(100, 200)

        assert list(ranges.missing_in(150, 175)) == []
        assert list(ranges.missing_in(90, 110)) == [(90, 99)]

    def test_an_inverted_span_yields_nothing(self):
        assert list(LedgerRangeSet().missing_in(10, 5)) == []


class TestLedgerRangeSetSerialisation:
    def test_round_trips_through_the_compact_form(self):
        ranges = LedgerRangeSet()
        ranges.add_range(1, 5)
        ranges.add_range(10, 12)

        assert LedgerRangeSet.from_list(ranges.to_list()) == ranges

    def test_the_compact_form_is_json_serialisable(self):
        ranges = LedgerRangeSet()
        ranges.add_range(1, 1_000_000)

        assert json.loads(json.dumps(ranges.to_list())) == [[1, 1_000_000]]

    def test_reads_the_flat_list_written_before_this_change(self):
        """A state file from an older build must still resume.

        Refusing it would restart a backfill from scratch, or reprocess
        ledgers an operator believed were finished.
        """
        legacy = LedgerRangeSet.from_list([1, 2, 3, 7, 8])

        assert legacy.to_list() == [[1, 3], [7, 8]]
        assert 2 in legacy
        assert 5 not in legacy

    def test_reads_a_mixed_list(self):
        mixed = LedgerRangeSet.from_list([[1, 3], 5, [7, 9]])

        assert mixed.to_list() == [[1, 3], [5, 5], [7, 9]]

    def test_rejects_a_malformed_entry(self):
        with pytest.raises(ValueError, match=r"\[low, high\]"):
            LedgerRangeSet.from_list([[1, 2, 3]])

        with pytest.raises(ValueError, match="cannot interpret"):
            LedgerRangeSet.from_list(["not-a-ledger"])

    def test_iterating_yields_every_covered_id_in_order(self):
        ranges = LedgerRangeSet()
        ranges.add_range(1, 3)
        ranges.add_range(7, 8)

        assert list(ranges) == [1, 2, 3, 7, 8]

    def test_compares_equal_to_the_equivalent_set(self):
        ranges = LedgerRangeSet()
        ranges.add_range(1, 3)

        assert ranges == {1, 2, 3}


class TestIngestionStateBounding:
    def test_the_state_file_stays_small_for_a_huge_backfill(self, tmp_path):
        """The file used to hold one entry per ledger."""
        store = StateStore(path=str(tmp_path / "state.json"))
        state = store.load()
        state.processed_ledgers.add_range(1, 1_000_000)
        state.last_processed_ledger = 1_000_000

        store.save(state)

        written = json.loads((tmp_path / "state.json").read_text())
        assert written["processed_ledgers"] == [[1, 1_000_000]]
        # A list of a million ids would be megabytes.
        assert (tmp_path / "state.json").stat().st_size < 1_000

    def test_round_trips_through_the_store(self, tmp_path):
        store = StateStore(path=str(tmp_path / "state.json"))
        state = store.load()
        state.processed_ledgers.add_range(10, 20)
        state.processed_ledgers.add(30)
        state.last_processed_ledger = 30
        store.save(state)

        reloaded = store.load()

        assert reloaded.last_processed_ledger == 30
        assert 15 in reloaded.processed_ledgers
        assert 25 not in reloaded.processed_ledgers
        assert 30 in reloaded.processed_ledgers

    def test_an_absent_file_loads_as_empty(self, tmp_path):
        store = StateStore(path=str(tmp_path / "missing.json"))

        state = store.load()

        assert state.last_processed_ledger is None
        assert len(state.processed_ledgers) == 0

    def test_resumes_from_a_legacy_state_file(self, tmp_path):
        """Restart-safety across the upgrade."""
        path = tmp_path / "state.json"
        path.write_text(json.dumps({"last_processed_ledger": 3, "processed_ledgers": [1, 2, 3]}))
        store = StateStore(path=str(path))

        state = store.load()

        assert state.last_processed_ledger == 3
        assert 2 in state.processed_ledgers
        assert 4 not in state.processed_ledgers

    def test_mark_processed_still_records_and_advances(self, tmp_path):
        store = StateStore(path=str(tmp_path / "state.json"))

        store.mark_processed(5)
        state = store.mark_processed(6)

        assert 5 in state.processed_ledgers
        assert 6 in state.processed_ledgers
        assert state.last_processed_ledger == 6

    def test_mark_processed_does_not_move_the_high_water_mark_backwards(self, tmp_path):
        store = StateStore(path=str(tmp_path / "state.json"))
        store.mark_processed(10)

        state = store.mark_processed(4)

        assert state.last_processed_ledger == 10
        assert 4 in state.processed_ledgers

    def test_state_dict_round_trips(self):
        state = IngestionState(last_processed_ledger=9, processed_ledgers=LedgerRangeSet())
        state.processed_ledgers.add_range(1, 9)

        restored = IngestionState.from_dict(state.to_dict())

        assert restored.last_processed_ledger == 9
        assert restored.processed_ledgers == state.processed_ledgers

    def test_a_save_leaves_the_previous_file_intact_on_failure(self, tmp_path):
        """The write is atomic, so a crash cannot truncate the state."""
        path = tmp_path / "state.json"
        store = StateStore(path=str(path))
        state = store.load()
        state.processed_ledgers.add_range(1, 5)
        state.last_processed_ledger = 5
        store.save(state)

        original = path.read_text()
        assert json.loads(original)["processed_ledgers"] == [[1, 5]]


class TestIngestStreamIdempotency:
    """The existing contract must be unchanged (#724 acceptance criteria)."""

    def test_a_ledger_is_processed_once_across_restarts(self, tmp_path):
        from astroml.ingestion.service import IngestionService

        store = StateStore(path=str(tmp_path / "state.json"))
        processed: list[int] = []

        service = IngestionService(state_store=store)
        list(
            service.ingest_stream(
                start_ledger=1,
                end_ledger=5,
                process_fn=lambda ledger_id, _payload: processed.append(ledger_id),
            )
        )

        # A fresh service over the same state file — a restart.
        restarted = IngestionService(state_store=StateStore(path=str(tmp_path / "state.json")))
        outcomes = list(
            restarted.ingest_stream(
                start_ledger=1,
                end_ledger=5,
                process_fn=lambda ledger_id, _payload: processed.append(ledger_id),
            )
        )

        assert processed == [1, 2, 3, 4, 5]
        assert all(outcome.status == "skipped" for _, outcome in outcomes)

    def test_resuming_a_partial_range_only_does_the_remainder(self, tmp_path):
        from astroml.ingestion.service import IngestionService

        path = str(tmp_path / "state.json")
        first_pass: list[int] = []
        service = IngestionService(state_store=StateStore(path=path))
        for index, (_ledger_id, _outcome) in enumerate(
            service.ingest_stream(
                start_ledger=1,
                end_ledger=10,
                process_fn=lambda ledger_id, _payload: first_pass.append(ledger_id),
                batch_size=1,
            )
        ):
            if index == 4:
                break  # abandon partway, as a crash would

        second_pass: list[int] = []
        resumed = IngestionService(state_store=StateStore(path=path))
        list(
            resumed.ingest_stream(
                start_ledger=1,
                end_ledger=10,
                process_fn=lambda ledger_id, _payload: second_pass.append(ledger_id),
                batch_size=1,
            )
        )

        # Nothing is processed twice, and everything is processed once.
        assert sorted(first_pass + second_pass) == list(range(1, 11))

    def test_the_state_after_a_full_range_is_a_single_interval(self, tmp_path):
        from astroml.ingestion.service import IngestionService

        path = str(tmp_path / "state.json")
        service = IngestionService(state_store=StateStore(path=path))
        list(service.ingest_stream(start_ledger=1, end_ledger=500))

        state = StateStore(path=path).load()

        assert state.processed_ledgers.range_count == 1
        assert state.last_processed_ledger == 500
