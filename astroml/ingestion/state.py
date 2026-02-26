from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Set


DEFAULT_STATE_DIR = os.path.join(os.getcwd(), ".astroml_state")
DEFAULT_STATE_FILE = os.path.join(DEFAULT_STATE_DIR, "ingestion_state.json")


@dataclass
class IngestionState:
    last_processed_ledger: Optional[int]
    processed_ledgers: Set[int]

    def to_dict(self) -> dict:
        return {
            "last_processed_ledger": self.last_processed_ledger,
            # store as sorted list for readability
            "processed_ledgers": sorted(self.processed_ledgers),
        }

    @staticmethod
    def from_dict(data: dict) -> "IngestionState":
        return IngestionState(
            last_processed_ledger=data.get("last_processed_ledger"),
            processed_ledgers=set(data.get("processed_ledgers", [])),
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
        with open(self.path, "r", encoding="utf-8") as f:
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
