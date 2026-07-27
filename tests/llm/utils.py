"""Testing utilities for LLM tests.

Resolves #458: Helpers for snapshot testing, response assertion,
async test support, VCR cassette management, and performance benchmarking.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable


# ─── Snapshot testing ────────────────────────────────────────────────────────

class SnapshotStore:
    """Simple snapshot store for LLM output regression testing.

    Snapshots are stored as JSON files in the ``snapshots/`` directory.
    On the first run they are written; on subsequent runs they are compared.

    Example::

        store = SnapshotStore()
        store.assert_matches("prompt_001", actual_response)
    """

    def __init__(self, snapshot_dir: str | Path = "tests/llm/snapshots") -> None:
        self._dir = Path(snapshot_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def assert_matches(self, key: str, value: Any) -> None:
        """Assert that *value* matches the stored snapshot for *key*.

        If no snapshot exists, saves *value* as the new snapshot.
        """
        path = self._dir / f"{key}.json"
        if not path.exists():
            self._save(path, value)
            return
        stored = self._load(path)
        assert stored == value, (
            f"Snapshot mismatch for key={key!r}.\n"
            f"Expected: {stored!r}\n"
            f"Actual  : {value!r}"
        )

    def update(self, key: str, value: Any) -> None:
        """Force-update the snapshot for *key* with *value*."""
        self._save(self._dir / f"{key}.json", value)

    @staticmethod
    def _save(path: Path, value: Any) -> None:
        with path.open("w") as f:
            json.dump(value, f, indent=2)

    @staticmethod
    def _load(path: Path) -> Any:
        with path.open() as f:
            return json.load(f)


# ─── Response assertion helpers ───────────────────────────────────────────────

def assert_valid_generate_response(response: dict[str, Any]) -> None:
    """Assert that *response* has the required fields of a GenerateResponse."""
    required = {"id", "model", "content", "usage", "cost", "latency_ms"}
    missing = required - response.keys()
    assert not missing, f"Missing keys in generate response: {missing}"
    assert isinstance(response["content"], str), "content must be a string"
    assert response["latency_ms"] >= 0, "latency_ms must be non-negative"
    usage = response["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage


def assert_valid_embed_response(response: dict[str, Any], expected_dims: int = 1536) -> None:
    """Assert that *response* has the required fields of an EmbedResponse."""
    assert "embeddings" in response
    assert isinstance(response["embeddings"], list)
    for vec in response["embeddings"]:
        assert isinstance(vec, list), "Each embedding must be a list"
        assert len(vec) == expected_dims, f"Expected {expected_dims} dims, got {len(vec)}"


def assert_valid_rag_response(response: dict[str, Any]) -> None:
    """Assert that *response* has the required fields of a RAGQueryResponse."""
    required = {"id", "query", "answer", "documents"}
    missing = required - response.keys()
    assert not missing, f"Missing keys in RAG response: {missing}"
    assert isinstance(response["documents"], list)


def assert_safety_blocked(guard_result: Any) -> None:
    """Assert that *guard_result* represents a blocked decision."""
    assert hasattr(guard_result, "is_blocked"), "Result must have is_blocked property"
    assert guard_result.is_blocked, f"Expected blocked but got: {guard_result.decision}"


# ─── Async test runner ────────────────────────────────────────────────────────

def run_async(coro) -> Any:
    """Run an async coroutine synchronously — useful in non-async test contexts."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── Latency benchmark helper ─────────────────────────────────────────────────

class LatencyTimer:
    """Context manager for timing test operations.

    Example::

        with LatencyTimer(max_ms=200) as t:
            result = provider.generate("test")
        # Raises AssertionError if latency > 200ms
    """

    def __init__(self, max_ms: float | None = None) -> None:
        self.max_ms = max_ms
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "LatencyTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000
        if self.max_ms is not None and self.elapsed_ms > self.max_ms:
            raise AssertionError(
                f"Latency {self.elapsed_ms:.1f}ms exceeded limit {self.max_ms}ms"
            )


# ─── VCR cassette helpers ─────────────────────────────────────────────────────

def vcr_cassette_path(test_name: str) -> Path:
    """Return the cassette file path for *test_name*."""
    return Path("tests/llm/cassettes") / f"{test_name}.yaml"


def ensure_cassette_dir() -> Path:
    """Create the VCR cassettes directory if it doesn't exist."""
    cassette_dir = Path("tests/llm/cassettes")
    cassette_dir.mkdir(parents=True, exist_ok=True)
    return cassette_dir
