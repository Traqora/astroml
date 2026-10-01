from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from astroml.utils.ranges import LedgerRangeSet

DEFAULT_STATE_DIR = os.path.join(os.getcwd(), ".astroml_state")
DEFAULT_STATE_FILE = os.path.join(DEFAULT_STATE_DIR, "ingestion_state.json")


def utc_now_iso() -> str:
    """Current time as an ISO-8601 UTC timestamp.

    Written verbatim into the state file, so keep the ``+00:00`` offset form
    rather than ``Z``: ``datetime.fromisoformat`` only learned to read ``Z`` in
    Python 3.11 and this project still supports 3.10.
    """
    return datetime.now(timezone.utc).isoformat()


def resolve_state_path(path: str | None = None) -> str:
    """Resolve the ingestion state file: explicit path, env, then default.

    ``INGESTION_STATE_FILE`` lets a worker and the API's ingestion health probe
    agree on one shared volume without threading a path through both call
    sites.
    """
    if path:
        return path
    return os.environ.get("INGESTION_STATE_FILE") or DEFAULT_STATE_FILE


@dataclass
class IngestionState:
    """Which ledgers have been processed, and how far ingestion has reached.

    ``processed_ledgers`` is a :class:`~astroml.utils.ranges.LedgerRangeSet`
    rather than a ``set`` (issue #724). It supports ``add``, ``in``, ``len``
    and iteration, so existing call sites are unchanged, but it stores
    contiguous runs instead of individual ids: a sequential million-ledger
    backfill costs one interval in memory and one pair on disk, where the set
    cost a million of each.

    ``last_processed_at`` is the ingestion heartbeat: an ISO-8601 UTC timestamp
    refreshed every time a ledger is processed. It is what
    :func:`astroml.observability.ingestion.check_ingestion_heartbeat` compares
    against the wall clock to decide whether data has gone stale. It is
    optional so state files written before the field existed still load.
    """

    last_processed_ledger: int | None
    processed_ledgers: LedgerRangeSet = field(default_factory=LedgerRangeSet)
    last_processed_at: str | None = None

    def record_processed(self, ledger_id: int, *, processed_at: str | None = None) -> None:
        """Mark ``ledger_id`` processed, advance the high-water mark, stamp time.

        The single place that keeps the processed set, the high-water mark and
        the heartbeat timestamp in agreement, so :meth:`StateStore.mark_processed`
        and the batched flush inside ``IngestionService.ingest_stream`` cannot
        drift apart.

        Args:
            ledger_id: Ledger that was just processed.
            processed_at: Override for the heartbeat timestamp (tests, replay).
                Defaults to now.
        """
        self.processed_ledgers.add(ledger_id)
        if self.last_processed_ledger is None:
            self.last_processed_ledger = ledger_id
        else:
            self.last_processed_ledger = max(self.last_processed_ledger, ledger_id)
        self.last_processed_at = processed_at or utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the on-disk form written by :meth:`StateStore.save`.

        Returns:
            Mapping of the state's fields, with ``processed_ledgers`` in the
                compact ``[[low, high], ...]`` range form.
        """
        return {
            "last_processed_ledger": self.last_processed_ledger,
            # Compact ``[[low, high], ...]`` form. Bounded by the number of
            # gaps rather than the number of ledgers, so the state file does
            # not grow with the size of the backfill.
            "processed_ledgers": self.processed_ledgers.to_list(),
            "last_processed_at": self.last_processed_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> IngestionState:
        """Rebuild a state from the mapping produced by :meth:`to_dict`.

        Missing keys are tolerated rather than raised: a partially written or
        older file should resume the work it describes, not fail the process
        that is trying to pick up where it left off.

        Args:
            data: Parsed contents of the state file.

        Returns:
            The equivalent :class:`IngestionState`.
        """
        # ``from_list`` accepts both the compact form and the flat list of ids
        # written before #724, so an in-progress backfill resumes across the
        # upgrade instead of starting over.
        return IngestionState(
            last_processed_ledger=data.get("last_processed_ledger"),
            processed_ledgers=LedgerRangeSet.from_list(data.get("processed_ledgers", [])),
            # Absent in state files written before the heartbeat existed.
            last_processed_at=data.get("last_processed_at"),
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

    Every write also stamps ``last_processed_at`` (see
    :meth:`IngestionState.record_processed`), which is the heartbeat the
    ingestion staleness probe reads.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = resolve_state_path(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load(self) -> IngestionState:
        """Read the state file.

        A file that does not exist yet is an empty state, not an error: first
        run and "nothing has been recorded" are the same situation for a
        resume, and the caller should not have to special-case them.

        Returns:
            The persisted :class:`IngestionState`, or a fresh empty one.

        Raises:
            json.JSONDecodeError: If the file exists but is not valid JSON. The
                file is written atomically, so this means genuine corruption —
                silently starting over would discard the processed set and
                re-ingest the whole range.
        """
        if not os.path.exists(self.path):
            return IngestionState(last_processed_ledger=None, processed_ledgers=LedgerRangeSet())
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        return IngestionState.from_dict(data)

    def save(self, state: IngestionState) -> None:
        """Write ``state`` to disk, replacing the previous file atomically.

        Side effects: creates ``<path>.tmp`` and then renames it over ``path``.
        The rename is what keeps a crash mid-write from leaving a truncated
        file that :meth:`load` would read as corruption.

        Args:
            state: The state to persist.
        """
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
        os.replace(tmp_path, self.path)

    def mark_processed(self, ledger_id: int) -> IngestionState:
        """Record ``ledger_id`` as processed and persist the result.

        Read-modify-write on every call: the store holds no cache, so two
        writers cannot diverge from the file between calls. Cost is one full
        serialisation per ledger, which is why a large backfill uses
        :class:`~astroml.ingestion.memory_efficient.ChunkedStateStore` instead.

        Args:
            ledger_id: Ledger that finished processing.

        Returns:
            The state as just written, including the new heartbeat timestamp.
        """
        state = self.load()
        state.record_processed(ledger_id)
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
        """Record ``stream_id``'s resume point and persist it immediately.

        Each call writes the whole cursor map, so a stream that checkpoints
        per record turns into per-record file writes. Callers should checkpoint
        at a boundary, not on every event.

        Args:
            stream_id: Stream the cursor belongs to.
            cursor: Horizon paging token to resume from.
        """
        self._cursors[stream_id] = cursor
        self._save()

    def get_cursor(self, stream_id: str) -> str | None:
        """Return where ``stream_id`` should resume, or ``None`` if unseen.

        Args:
            stream_id: Stream to look up.

        Returns:
            The last saved cursor, or ``None`` — which means start fresh, not
                that there is no saved position to trust.
        """
        return self._cursors.get(stream_id)

    def _save(self) -> None:
        data = {"cursors": self._cursors}
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)
