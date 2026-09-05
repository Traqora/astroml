"""Memory-bounded tracking of processed integer ids (issue #724).

Ingestion records which ledgers it has already handled so a restart can skip
them. That set was held — and persisted — as one entry per ledger, which is
fine for a few thousand and is the reason peak RSS grows with the size of a
backfill: a million-ledger range holds a million-element ``set`` in memory and
rewrites a multi-megabyte JSON list on every state flush.

A backfill processes ledgers in order, so that set is almost always a single
contiguous run. :class:`LedgerRangeSet` stores the runs instead of the
members: ``[[1, 1_000_000]]`` rather than a million integers. Memory is
proportional to the number of *gaps*, not to the number of ledgers, so a
sequential backfill of any length costs one interval.

The API is deliberately a subset of ``set``'s — ``add``, ``in``, ``len``,
iteration — so it drops into the existing call sites unchanged.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Iterator

__all__ = ["LedgerRangeSet"]


class LedgerRangeSet:
    """A set of integers stored as sorted, disjoint, non-adjacent ranges.

    Ranges are inclusive on both ends and kept normalised: after any mutation
    the internal list is sorted, no two ranges overlap, and no two are
    adjacent (``[1, 5]`` and ``[6, 9]`` become ``[1, 9]``). That invariant is
    what keeps a sequential backfill at exactly one interval.

    Example:
        >>> processed = LedgerRangeSet()
        >>> for ledger in range(1, 1_000_001):
        ...     processed.add(ledger)
        >>> processed.range_count
        1
        >>> 999_999 in processed
        True
    """

    __slots__ = ("_ranges",)

    def __init__(self, ranges: Iterable[tuple[int, int]] | None = None) -> None:
        self._ranges: list[list[int]] = []
        for low, high in ranges or ():
            self.add_range(low, high)

    # -- membership ---------------------------------------------------------

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, int) or isinstance(value, bool):
            return False

        # Binary search for the last range starting at or below `value`; only
        # that one can contain it. O(log r) in the number of ranges, so a
        # membership check stays cheap however long the backfill runs.
        index = bisect.bisect_right(self._ranges, [value, float("inf")]) - 1
        if index < 0:
            return False
        low, high = self._ranges[index]
        return low <= value <= high

    def __len__(self) -> int:
        """Number of integers covered, not the number of ranges."""
        return sum(high - low + 1 for low, high in self._ranges)

    def __iter__(self) -> Iterator[int]:
        """Iterate every covered integer, ascending.

        Materialises nothing: this is what makes a `sorted(...)` over a huge
        span expensive rather than impossible. Prefer :meth:`to_list` when
        writing the state to disk.
        """
        for low, high in self._ranges:
            yield from range(low, high + 1)

    def __bool__(self) -> bool:
        return bool(self._ranges)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LedgerRangeSet):
            return self._ranges == other._ranges
        if isinstance(other, (set, frozenset)):
            return set(self) == other
        return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LedgerRangeSet({self.to_list()!r})"

    # -- mutation -----------------------------------------------------------

    def add(self, value: int) -> None:
        """Add a single integer."""
        self.add_range(value, value)

    def add_range(self, low: int, high: int) -> None:
        """Add every integer in ``[low, high]`` inclusive.

        Raises:
            ValueError: If ``high < low``.
        """
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")

        # Merge with any range that overlaps or merely touches the new one.
        # Touching counts: [1,5] + [6,9] must collapse to [1,9], or a
        # sequential backfill would accumulate one interval per ledger and
        # reproduce the very growth this class exists to avoid.
        start = bisect.bisect_left(self._ranges, [low - 1])
        if start > 0 and self._ranges[start - 1][1] >= low - 1:
            start -= 1

        end = start
        merged_low, merged_high = low, high
        while end < len(self._ranges) and self._ranges[end][0] <= high + 1:
            merged_low = min(merged_low, self._ranges[end][0])
            merged_high = max(merged_high, self._ranges[end][1])
            end += 1

        if start < len(self._ranges):
            merged_low = min(merged_low, self._ranges[start][0]) if start < end else merged_low

        self._ranges[start:end] = [[merged_low, merged_high]]

    def update(self, values: Iterable[int]) -> None:
        """Add every integer in ``values``."""
        for value in values:
            self.add(value)

    # -- inspection ---------------------------------------------------------

    @property
    def range_count(self) -> int:
        """Number of stored ranges — the thing that bounds memory."""
        return len(self._ranges)

    @property
    def max(self) -> int | None:
        """Highest covered integer, or ``None`` when empty."""
        return self._ranges[-1][1] if self._ranges else None

    @property
    def min(self) -> int | None:
        """Lowest covered integer, or ``None`` when empty."""
        return self._ranges[0][0] if self._ranges else None

    def missing_in(self, low: int, high: int) -> Iterator[tuple[int, int]]:
        """Yield the gaps within ``[low, high]`` that are not covered.

        What a backfill actually needs in order to resume: the work left,
        rather than a per-ledger membership test across the whole span.
        """
        if high < low:
            return

        cursor = low
        for range_low, range_high in self._ranges:
            if range_high < cursor:
                continue
            if range_low > high:
                break
            if range_low > cursor:
                yield cursor, min(range_low - 1, high)
            cursor = max(cursor, range_high + 1)
            if cursor > high:
                return

        if cursor <= high:
            yield cursor, high

    # -- serialisation ------------------------------------------------------

    def to_list(self) -> list[list[int]]:
        """Compact, JSON-friendly form: a list of ``[low, high]`` pairs."""
        return [list(pair) for pair in self._ranges]

    @classmethod
    def from_list(cls, data: Iterable[object]) -> LedgerRangeSet:
        """Rebuild from either the compact form or a plain list of integers.

        Both are accepted so a state file written before this change still
        loads: an existing deployment's ``processed_ledgers`` is a flat list of
        ids, and refusing it would silently restart a backfill from scratch —
        or, worse, reprocess ledgers an operator believed were done.
        """
        result = cls()
        for entry in data or ():
            if isinstance(entry, (list, tuple)):
                if len(entry) != 2:
                    raise ValueError(f"range entry must be [low, high], got {entry!r}")
                result.add_range(int(entry[0]), int(entry[1]))
            elif isinstance(entry, int) and not isinstance(entry, bool):
                result.add(entry)
            else:
                raise ValueError(f"cannot interpret {entry!r} as a ledger id or range")
        return result
