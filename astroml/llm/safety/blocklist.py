"""Blocklist manager — manage and persist term blocklists.

Resolves #455: In-memory blocklist with file-based persistence and
dynamic term addition/removal.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Default set of high-confidence harmful terms
_DEFAULT_TERMS: frozenset[str] = frozenset(
    [
        "make a bomb",
        "build explosives",
        "synthesize meth",
        "child exploitation",
        "cp ",
        "csam",
        "how to kill",
        "suicide methods",
        "how to hack banking",
    ]
)


class BlocklistManager:
    """Thread-safe in-memory blocklist with optional file persistence.

    Example::

        mgr = BlocklistManager()
        mgr.add_terms(["bad phrase"])
        blocked, term = mgr.contains("I want to bad phrase you")
        # blocked == True, term == "bad phrase"
    """

    def __init__(
        self,
        terms: Iterable[str] | None = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._terms: set[str] = set(_DEFAULT_TERMS)
        if terms:
            self._terms.update(t.lower() for t in terms)
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load_from_file()

    def contains(self, text: str) -> tuple[bool, str]:
        """Check if *text* contains any blocklisted term.

        Returns:
            (is_blocked, matched_term_or_empty_string)
        """
        lower = text.lower()
        with self._lock:
            for term in self._terms:
                if term in lower:
                    return True, term
        return False, ""

    def add_terms(self, terms: Iterable[str]) -> None:
        """Add *terms* to the blocklist."""
        with self._lock:
            self._terms.update(t.lower().strip() for t in terms)
        self._save_to_file()

    def remove_terms(self, terms: Iterable[str]) -> None:
        """Remove *terms* from the blocklist."""
        with self._lock:
            for t in terms:
                self._terms.discard(t.lower().strip())
        self._save_to_file()

    def all_terms(self) -> list[str]:
        """Return a sorted list of all blocklisted terms."""
        with self._lock:
            return sorted(self._terms)

    def _save_to_file(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._persist_path.open("w") as f:
                json.dump(sorted(self._terms), f, indent=2)
        except OSError:
            logger.warning("Could not persist blocklist to %s", self._persist_path)

    def _load_from_file(self) -> None:
        try:
            with self._persist_path.open() as f:  # type: ignore[arg-type]
                data = json.load(f)
            if isinstance(data, list):
                self._terms.update(data)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not load blocklist from %s", self._persist_path)
