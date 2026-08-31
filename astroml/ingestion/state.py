from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from astroml.ingestion.metrics import INGESTION_LAST_PROCESSED_AT

DEFAULT_STATE_DIR = os.path.join(os.getcwd(), ".astroml_state")
DEFAULT_STATE_FILE = os.path.join(DEFAULT_STATE_DIR, "ingestion_state.json")


@dataclass
class IngestionState:
    last_processed_ledger: int | None
    processed_ledgers: set[int]
    last_processed_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "last_processed_ledger": self.last_processed_ledger,
            # store as sorted list for readability
            "processed_ledgers": sorted(self.processed_ledgers),
            "last_processed_at": self.last_processed_at,
        }

    @staticmethod
    def from_dict(data: dict) -> IngestionState:
        return IngestionState(
            last_processed_ledger=data.get("last_processed_ledger"),
            processed_ledgers=set(data.get("processed_ledgers", [])),
            last_processed_at=data.get("last_processed_at"),
        )


class StateStore:
    """File-based state store to track processed ledgers.

    Properties:
      - Idempotency: we retain a set of processed ledger ids and check before processing
      - Incremental: we track last_processed_ledger to resume ranges efficiently
    """

    def __init__(self, path: str = DEFAULT_STATE_FILE) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load(self) -> IngestionState:
        if not os.path.exists(self.path):
            return IngestionState(last_processed_ledger=None, processed_ledgers=set())
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
        state.last_processed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            INGESTION_LAST_PROCESSED_AT.set(datetime.now(timezone.utc).timestamp())
        except Exception:
            pass  # metrics registry may not be initialised in tests/CLI
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
