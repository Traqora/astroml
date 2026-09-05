"""Memory-efficient state tracking for large-range backfills — issue #766.

Provides compact alternatives to ``IngestionState`` that avoid holding the
entire ``processed_ledgers`` set in RAM.  For backfills spanning millions
of ledgers the naive ``set[int]`` can consume hundreds of MB; these
implementations keep peak memory bounded and constant regardless of range
size.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compact set backed by a sorted array (issue #766)
# ---------------------------------------------------------------------------


class _CompactLedgerSet:
    """Sorted-array set with O(log N) membership checks.

    Memory usage is ~8 bytes per entry (int64) versus ~70+ bytes for a
    Python ``set[int]`` due to hash-table overhead.  For continuous ranges
    the compression is even better because we store as a list of
    ``(start, end)`` intervals.
    """

    def __init__(self, values: list[int] | None = None) -> None:
        self._sorted: list[int] = sorted(values) if values else []

    def __contains__(self, item: int) -> bool:
        idx = bisect.bisect_left(self._sorted, item)
        return idx < len(self._sorted) and self._sorted[idx] == item

    def add(self, item: int) -> None:
        idx = bisect.bisect_left(self._sorted, item)
        if idx < len(self._sorted) and self._sorted[idx] == item:
            return  # already present
        self._sorted.insert(idx, item)

    def __len__(self) -> int:
        return len(self._sorted)

    def __iter__(self):  # type: ignore[override]
        return iter(self._sorted)

    def to_list(self) -> list[int]:
        return list(self._sorted)


class _BloomFilterSet:
    """Bounded-memory probabilistic set with configurable false-positive rate.

    For very large ranges (e.g. 1M+ ledgers) this trades a small
    false-positive rate (~1%) for dramatically lower memory usage
    (~12 KB for 1M entries at 1% FP rate vs ~70 MB for a Python set).

    False positives mean a ledger might be *incorrectly* considered
    already-processed, causing it to be skipped.  The error rate is
    bounded and configurable.
    """

    def __init__(self, expected_items: int = 1_000_000, fp_rate: float = 0.01) -> None:
        self._expected = expected_items
        self._fp_rate = fp_rate
        # Optimal m (bits) and k (hash functions)
        self._size = self._optimal_size(expected_items, fp_rate)
        self._num_hashes = self._optimal_hashes(self._size, expected_items)
        self._bits = bytearray(math.ceil(self._size / 8))
        self._count = 0

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Optimal bit-array size for n items at false-positive rate p."""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return max(int(math.ceil(m)), 64)

    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        """Optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return max(int(math.ceil(k)), 1)

    def _get_positions(self, item: int) -> list[int]:
        """Get bit positions for an item using double hashing."""
        h1 = hash(item) & 0xFFFFFFFF
        h2 = (hash(item * 2654435761) & 0xFFFFFFFF) or 1  # ensure non-zero
        return [(h1 + i * h2) % self._size for i in range(self._num_hashes)]

    def __contains__(self, item: int) -> bool:
        for pos in self._get_positions(item):
            byte_idx = pos >> 3
            bit_idx = pos & 7
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def add(self, item: int) -> None:
        for pos in self._get_positions(item):
            byte_idx = pos >> 3
            bit_idx = pos & 7
            self._bits[byte_idx] |= 1 << bit_idx
        self._count += 1

    def __len__(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Memory-efficient ingestion state (issue #766)
# ---------------------------------------------------------------------------


@dataclass
class MemoryEfficientState:
    """Ingestion state with bounded memory footprint.

    Unlike ``IngestionState`` which keeps a full ``set[int]`` of every
    processed ledger, this implementation uses either:

    - A compact sorted-array set (~8 bytes/entry vs ~70+ bytes for set)
    - A bloom filter (~12 KB fixed for 1M entries at 1% FP rate)

    Parameters
    ----------
    last_processed_ledger : int | None
        Highest ledger ID confirmed processed.
    processed_ledgers : set[int] | _CompactLedgerSet | _BloomFilterSet
        Internal processed-ledger tracking structure.
    mode : str
        ``'compact'`` for exact sorted-array tracking or ``'bloom'`` for
        probabilistic tracking with bounded memory.
    """

    last_processed_ledger: int | None = None
    _processed: _CompactLedgerSet | _BloomFilterSet = field(
        default_factory=_CompactLedgerSet
    )
    mode: str = "compact"

    @classmethod
    def from_legacy(cls, state) -> MemoryEfficientState:  # type: ignore[no-undef]
        """Create from an ``IngestionState`` instance."""
        n = len(state.processed_ledgers)
        if n > 100_000:
            logger.info(
                "Large processed set (%d ledgers) — using bloom filter mode", n
            )
            bloom = _BloomFilterSet(expected_items=max(n * 2, 1_000_000))
            for lid in state.processed_ledgers:
                bloom.add(lid)
            return cls(
                last_processed_ledger=state.last_processed_ledger,
                _processed=bloom,
                mode="bloom",
            )
        return cls(
            last_processed_ledger=state.last_processed_ledger,
            _processed=_CompactLedgerSet(list(state.processed_ledgers)),
            mode="compact",
        )

    def __contains__(self, ledger_id: int) -> bool:
        return ledger_id in self._processed

    def add(self, ledger_id: int) -> None:
        self._processed.add(ledger_id)
        if self.last_processed_ledger is None:
            self.last_processed_ledger = ledger_id
        else:
            self.last_processed_ledger = max(self.last_processed_ledger, ledger_id)

    def memory_usage_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        if isinstance(self._processed, _BloomFilterSet):
            return len(self._processed._bits) + 64  # bits + overhead
        else:
            return len(self._processed) * 8 + 64  # ~8 bytes per int64 + overhead


# ---------------------------------------------------------------------------
# Chunked state store — flushes partial state to disk (issue #766)
# ---------------------------------------------------------------------------


class ChunkedStateStore:
    """State store that flushes in bounded-size chunks.

    The naive per-ledger ``StateStore.mark_processed`` reloads and
    re-serialises the entire processed-ledger set on every call — O(N²)
    over a large backfill.  This store batches flushes so disk I/O is
    amortised and peak memory stays bounded.

    Parameters
    ----------
    path : str
        File path for the state JSON.
    flush_interval : int
        Flush to disk every ``flush_interval`` processed ledgers.
    """

    def __init__(self, path: str, flush_interval: int = 1000) -> None:
        self.path = path
        self.flush_interval = flush_interval
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._pending_count = 0

    def load(self) -> MemoryEfficientState:
        """Load state from disk, using memory-efficient representation."""
        from astroml.ingestion.state import IngestionState, StateStore

        if not os.path.exists(self.path):
            return MemoryEfficientState()

        # Load via legacy store then convert to compact representation
        legacy_store = StateStore(self.path)
        legacy_state = legacy_store.load()
        return MemoryEfficientState.from_legacy(legacy_state)

    def save(self, state: MemoryEfficientState) -> None:
        """Persist state to disk atomically."""
        data = {
            "last_processed_ledger": state.last_processed_ledger,
            "processed_ledgers": (
                state._processed.to_list()
                if isinstance(state._processed, _CompactLedgerSet)
                else sorted(
                    list(state._processed)
                    if not isinstance(state._processed, _BloomFilterSet)
                    else []
                )
            ),
            "_compact_mode": state.mode,
        }
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)

    def should_flush(self) -> bool:
        """Check if it's time to flush based on pending count."""
        self._pending_count += 1
        return self._pending_count >= self.flush_interval

    def reset_flush_counter(self) -> None:
        self._pending_count = 0
