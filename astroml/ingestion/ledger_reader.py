"""Streaming and paginated reads for large ledger files.

Provides memory-efficient access to ledger data stored on disk:
- Streaming reads: iterate records without loading entire files into memory
- Paginated reads: skip/limit access for API-style pagination
- Multi-file streaming: treat a directory of ledgers as a single stream
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Iterator, Optional

logger = logging.getLogger("astroml.ingestion.ledger_reader")


@dataclass
class Page:
    """A page of ledger records with pagination metadata."""

    records: list[dict[str, Any]]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class LedgerReader:
    """Memory-efficient reader for ledger files.

    Supports:
    - Streaming individual files (line-delimited JSON or JSON arrays)
    - Paginated access with skip/limit
    - Multi-file directory streaming with sort order
    """

    def __init__(self, data_dir: str = "data/ledgers") -> None:
        self._data_dir = pathlib.Path(data_dir)

    @property
    def data_dir(self) -> pathlib.Path:
        return self._data_dir

    def list_ledger_files(
        self,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None,
    ) -> list[pathlib.Path]:
        """List ledger files in the directory, optionally filtered by sequence range.

        Returns files sorted by ledger sequence number.
        """
        if not self._data_dir.exists():
            return []

        files = []
        for f in self._data_dir.rglob("ledger_*.json"):
            if not f.is_file():
                continue
            seq = self._extract_sequence(f.name)
            if seq is None:
                continue
            if start_seq is not None and seq < start_seq:
                continue
            if end_seq is not None and seq > end_seq:
                continue
            files.append(f)

        files.sort(key=lambda f: self._extract_sequence(f.name))  # type: ignore[arg-type,return-value]
        return files

    def stream_file(self, file_path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
        """Stream records from a single ledger file.

        Supports:
        - JSON array files: ``[{...}, {...}, ...]``
        - Line-delimited JSON (JSONL): one JSON object per line
        - Single JSON object files: ``{...}``

        Yields one record at a time for memory efficiency.
        """
        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ledger file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            content = f.read(2048).lstrip()
            f.seek(0)

            if not content:
                return

            if content[0] == "[":
                yield from self._stream_json_array(f)
            elif content[0] == "{":
                f.seek(0)
                raw = f.read()
                f.seek(0)
                stripped = raw.strip()
                if stripped.count("\n") == 0:
                    yield json.loads(raw)
                else:
                    for line in stripped.split("\n"):
                        line = line.strip()
                        if line:
                            yield json.loads(line)
            else:
                yield from self._stream_jsonl(f)

    def _stream_json_array(self, f: Any) -> Iterator[dict[str, Any]]:
        """Stream records from a JSON array file.

        Attempts incremental parsing with ijson if available; otherwise
        falls back to loading the full array. For very large arrays,
        consider converting to JSONL format first.
        """
        try:
            import ijson  # noqa: PLC0415

            for record in ijson.items(f, "item"):
                yield record
        except ImportError:
            f.seek(0)
            data = json.load(f)
            yield from data

    def _stream_jsonl(self, f: Any) -> Iterator[dict[str, Any]]:
        """Stream records from a line-delimited JSON file."""
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

    def stream_range(
        self,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream all ledger records in a sequence range.

        Reads files one at a time, yielding records in sequence order.
        """
        files = self.list_ledger_files(start_seq, end_seq)
        for f in files:
            yield from self.stream_file(f)

    def stream_all(self) -> Iterator[dict[str, Any]]:
        """Stream all ledger records from the data directory."""
        yield from self.stream_range()

    def read_page(
        self,
        page: int = 1,
        page_size: int = 100,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None,
    ) -> Page:
        """Read a specific page of ledger records.

        Args:
            page: Page number (1-indexed).
            page_size: Number of records per page.
            start_seq: Optional start ledger sequence filter.
            end_seq: Optional end ledger sequence filter.

        Returns:
            Page object with records and pagination metadata.
        """
        if page < 1:
            raise ValueError("Page number must be >= 1")
        if page_size < 1:
            raise ValueError("Page size must be >= 1")

        skip = (page - 1) * page_size
        records: list[dict[str, Any]] = []
        total = 0

        for record in self.stream_range(start_seq, end_seq):
            total += 1
            if total > skip and len(records) < page_size:
                records.append(record)

        remaining = total - skip - len(records)
        has_next = remaining > 0
        has_prev = page > 1

        return Page(
            records=records,
            total=total,
            page=page,
            page_size=page_size,
            has_next=has_next,
            has_prev=has_prev,
        )

    def count(
        self,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None,
    ) -> int:
        """Count total ledger records in the given range."""
        return sum(1 for _ in self.stream_range(start_seq, end_seq))

    def read_ledger(self, sequence: int) -> Optional[dict[str, Any]]:
        """Read a single ledger by sequence number.

        Streams the file to find the matching record without loading
        the entire directory.
        """
        file_path = self._find_ledger_file(sequence)
        if file_path is None:
            return None
        for record in self.stream_file(file_path):
            if record.get("sequence") == sequence:
                return record
        return None

    def _find_ledger_file(self, sequence: int) -> Optional[pathlib.Path]:
        """Find a ledger file by sequence, supporting both flat and partitioned layouts."""
        # Fast path: current default partitioning scheme
        bucket_start = (sequence // 10_000) * 10_000
        partitioned = self._data_dir / f"ledger_bucket_{bucket_start:08d}" / f"ledger_{sequence}.json"
        if partitioned.exists():
            return partitioned

        # Fallback: legacy flat layout or custom partitioning
        for f in self._data_dir.rglob(f"ledger_{sequence}.json"):
            if f.is_file():
                return f
        return None

    @staticmethod
    def _extract_sequence(filename: str) -> Optional[int]:
        """Extract the ledger sequence number from a filename."""
        try:
            for suffix in (".jsonl", ".json"):
                if filename.endswith(suffix):
                    base = filename[: -len(suffix)]
                    break
            else:
                return None
            if not base.startswith("ledger_"):
                return None
            return int(base[len("ledger_"):])
        except (ValueError, AttributeError):
            return None


class LedgerBatchReader:
    """Batch reader that processes ledger files in configurable chunk sizes.

    Useful for ETL pipelines that need to process large numbers of ledgers
    without loading everything into memory at once.
    """

    def __init__(
        self,
        data_dir: str = "data/ledgers",
        batch_size: int = 1000,
    ) -> None:
        self._reader = LedgerReader(data_dir)
        self._batch_size = batch_size

    def iter_batches(
        self,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Iterate over ledger records in batches.

        Each batch contains up to ``batch_size`` records.
        """
        batch: list[dict[str, Any]] = []
        for record in self._reader.stream_range(start_seq, end_seq):
            batch.append(record)
            if len(batch) >= self._batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
