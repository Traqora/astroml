from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from astroml.utils.ranges import LedgerRangeSet

DEFAULT_STATE_DIR = os.path.join(os.getcwd(), ".astroml_state")
DEFAULT_STATE_FILE = os.path.join(DEFAULT_STATE_DIR, "ingestion_state.json")


@dataclass
class IngestionState:
    """Which ledgers have been processed, and how far ingestion has reached.

    ``processed_ledgers`` is a :class:`~astroml.utils.ranges.LedgerRangeSet`
    rather than a ``set`` (issue #724). It supports ``add``, ``in``, ``len``
    and iteration, so existing call sites are unchanged, but it stores
    contiguous runs instead of individual ids: a sequential million-ledger
    backfill costs one interval in memory and one pair on disk, where the set
    cost a million of each.
    """

    last_processed_ledger: int | None
    processed_ledgers: LedgerRangeSet = field(default_factory=LedgerRangeSet)

    def to_dict(self) -> dict:
        return {
            "last_processed_ledger": self.last_processed_ledger,
            # Compact ``[[low, high], ...]`` form. Bounded by the number of
            # gaps rather than the number of ledgers, so the state file does
            # not grow with the size of the backfill.
            "processed_ledgers": self.processed_ledgers.to_list(),
        }

    @staticmethod
    def from_dict(data: dict) -> IngestionState:
        # ``from_list`` accepts both the compact form and the flat list of ids
        # written before #724, so an in-progress backfill resumes across the
        # upgrade instead of starting over.
        return IngestionState(
            last_processed_ledger=data.get("last_processed_ledger"),
            processed_ledgers=LedgerRangeSet.from_list(data.get("processed_ledgers", [])),
        )


class StateStore:
    """File-based state store to track processed ledgers.

    Properties:
      - Idempotency: processed ledger ids are retained and checked before processing
      - Incremental: ``last_processed_ledger`` lets a range resume efficiently
      - Bounded: ids are stored as contiguous ranges, so both the in-memory
        footprint and the file size scale with the number of gaps rather than
        the size of the backfill (issue #724)

    The file is replaced atomically via ``os.replace``, so a crash mid-write
    leaves the previous state intact rather than a truncated file that would
    read as "nothing processed".
    """

    def __init__(self, path: str = DEFAULT_STATE_FILE) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load(self) -> IngestionState:
        if not os.path.exists(self.path):
            return IngestionState(last_processed_ledger=None, processed_ledgers=LedgerRangeSet())
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        return IngestionState.from_dict(data)

    def save(self, state: IngestionState) -> None:
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
        os.replace(tmp_path, self.path)

    def mark_processed(self, ledger_id: int) -> IngestionState:
        state = self.load()
        state.processed_ledgers.add(ledger_id)
        if state.last_processed_ledger is None:
            state.last_processed_ledger = ledger_id
        else:
            state.last_processed_ledger = max(state.last_processed_ledger, ledger_id)
        self.save(state)
        return state


class StreamStateManager:
    """Manages cursors for multiple streams."""

    def __init__(self, path: str = DEFAULT_STATE_FILE) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._cursors: dict[str, str] = self._load_cursors()

    def _load_cursors(self) -> dict[str, str]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
                return data.get("cursors", {})
        except (OSError, json.JSONDecodeError):
            return {}

    def save_cursor(self, stream_id: str, cursor: str) -> None:
        self._cursors[stream_id] = cursor
        self._save()

    def get_cursor(self, stream_id: str) -> str | None:
        return self._cursors.get(stream_id)

    def _save(self) -> None:
        data = {"cursors": self._cursors}
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)
