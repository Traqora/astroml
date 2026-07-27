"""Memory subsystems for the LLM Agent Framework.

Three tiers of memory are provided:

* :class:`ShortTermMemory` — a sliding-window buffer of recent messages
  (analogous to a conversation context window).
* :class:`LongTermMemory` — a key-value store of facts and knowledge
  that persists across conversations and can be queried by similarity.
* :class:`EpisodicMemory` — a log of completed task episodes, used for
  learning from past successes and failures.

:class:`MemoryManager` orchestrates all three and provides a unified
interface to the agent.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import MemoryConfig
from .exceptions import MemoryError as AgentMemoryError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in the agent's conversation history.

    Attributes:
        role: ``"system"``, ``"user"``, ``"assistant"``, or ``"tool"``.
        content: The message text.
        timestamp: Unix timestamp when the message was created.
        metadata: Optional dict of extra information (e.g. tool_call_id).
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MemoryEntry:
    """An entry in long-term memory.

    Attributes:
        key: Unique identifier for the entry.
        value: The stored value (can be any JSON-serialisable object).
        tags: Optional list of tags for categorisation.
        created_at: Unix timestamp.
        access_count: How many times this entry has been retrieved.
    """

    key: str
    value: Any
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "tags": self.tags,
            "created_at": self.created_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            key=data["key"],
            value=data["value"],
            tags=data.get("tags", []),
            created_at=data.get("created_at", time.time()),
            access_count=data.get("access_count", 0),
        )


@dataclass
class Episode:
    """A record of a completed task episode.

    Attributes:
        task: The task description.
        steps: List of step descriptions and their outcomes.
        success: Whether the task was completed successfully.
        result: Summary of the final result.
        started_at: Unix timestamp when the episode started.
        completed_at: Unix timestamp when the episode completed.
        metadata: Optional extra information.
    """

    task: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    result: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "steps": self.steps,
            "success": self.success,
            "result": self.result,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        return cls(
            task=data["task"],
            steps=data.get("steps", []),
            success=data.get("success", False),
            result=data.get("result", ""),
            started_at=data.get("started_at", time.time()),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Short-term memory (sliding window)
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """Sliding-window buffer of recent conversation messages.

    When the buffer exceeds ``limit`` messages, the oldest entries are
    evicted (FIFO).  This mirrors the behaviour of a transformer's
    context window.
    """

    def __init__(self, limit: int = 20) -> None:
        if limit < 1:
            raise AgentMemoryError(f"Short-term memory limit must be >= 1, got {limit}")
        self.limit = limit
        self._messages: List[Message] = []

    def add(self, message: Message) -> None:
        """Append a message, evicting the oldest if over capacity."""
        self._messages.append(message)
        while len(self._messages) > self.limit:
            self._messages.pop(0)

    def add_many(self, messages: Sequence[Message]) -> None:
        """Append multiple messages."""
        for msg in messages:
            self.add(msg)

    def get_all(self) -> List[Message]:
        """Return a copy of all stored messages."""
        return list(self._messages)

    def get_context(self) -> List[Dict[str, Any]]:
        """Return messages as plain dicts for LLM API payloads."""
        return [msg.to_dict() for msg in self._messages]

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, index: int) -> Message:
        return self._messages[index]


# ---------------------------------------------------------------------------
# Long-term memory (key-value store with tagging)
# ---------------------------------------------------------------------------

class LongTermMemory:
    """Persistent key-value store for facts and knowledge.

    Entries are indexed by key and can be tagged for categorisation.
    Supports simple prefix-based lookup and tag-based filtering.
    """

    def __init__(self, limit: int = 1000) -> None:
        if limit < 1:
            raise AgentMemoryError(f"Long-term memory limit must be >= 1, got {limit}")
        self.limit = limit
        self._entries: Dict[str, MemoryEntry] = {}

    def store(
        self,
        key: str,
        value: Any,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store or overwrite a value under *key*."""
        if key in self._entries:
            # Update existing entry
            entry = self._entries[key]
            entry.value = value
            if tags:
                entry.tags = list(tags)
            entry.access_count += 1
        else:
            # Evict oldest if at capacity
            if len(self._entries) >= self.limit:
                oldest_key = min(
                    self._entries,
                    key=lambda k: self._entries[k].created_at,
                )
                del self._entries[oldest_key]
            self._entries[key] = MemoryEntry(
                key=key,
                value=value,
                tags=tags or [],
            )

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by exact key. Returns ``None`` if not found."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        entry.access_count += 1
        return entry.value

    def retrieve_by_tag(self, tag: str) -> List[MemoryEntry]:
        """Return all entries tagged with *tag*."""
        return [e for e in self._entries.values() if tag in e.tags]

    def search(self, query: str) -> List[MemoryEntry]:
        """Simple substring search across keys and values.

        This is a lightweight alternative to vector similarity search
        suitable for small-scale use.  For production, replace with
        a vector database or embedding-based retrieval.
        """
        query_lower = query.lower()
        results: List[MemoryEntry] = []
        for entry in self._entries.values():
            # Search in key
            if query_lower in entry.key.lower():
                results.append(entry)
                continue
            # Search in value (if string)
            if isinstance(entry.value, str) and query_lower in entry.value.lower():
                results.append(entry)
                continue
            # Search in tags
            if any(query_lower in tag.lower() for tag in entry.tags):
                results.append(entry)
        return results

    def delete(self, key: str) -> bool:
        """Delete an entry by key. Returns ``True`` if it existed."""
        return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    def keys(self) -> List[str]:
        """Return all stored keys."""
        return list(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self._entries.items()}

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load entries from a serialised dict."""
        self._entries = {
            k: MemoryEntry.from_dict(v) for k, v in data.items()
        }


# ---------------------------------------------------------------------------
# Episodic memory (task history)
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """Log of completed task episodes.

    Stores :class:`Episode` objects for later analysis and learning.
    When at capacity, the oldest episodes are evicted.
    """

    def __init__(self, limit: int = 100) -> None:
        if limit < 1:
            raise AgentMemoryError(f"Episode limit must be >= 1, got {limit}")
        self.limit = limit
        self._episodes: List[Episode] = []

    def add(self, episode: Episode) -> None:
        """Append an episode, evicting the oldest if over capacity."""
        self._episodes.append(episode)
        while len(self._episodes) > self.limit:
            self._episodes.pop(0)

    def get_all(self) -> List[Episode]:
        """Return a copy of all episodes."""
        return list(self._episodes)

    def get_recent(self, n: int = 5) -> List[Episode]:
        """Return the *n* most recent episodes."""
        return self._episodes[-n:]

    def get_successful(self) -> List[Episode]:
        """Return all successful episodes."""
        return [e for e in self._episodes if e.success]

    def get_failed(self) -> List[Episode]:
        """Return all failed episodes."""
        return [e for e in self._episodes if not e.success]

    def clear(self) -> None:
        """Remove all episodes."""
        self._episodes.clear()

    def __len__(self) -> int:
        return len(self._episodes)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._episodes]

    def from_dict(self, data: List[Dict[str, Any]]) -> None:
        """Load episodes from a serialised list."""
        self._episodes = [Episode.from_dict(d) for d in data]


# ---------------------------------------------------------------------------
# Memory manager (orchestrator)
# ---------------------------------------------------------------------------

class MemoryManager:
    """Orchestrates short-term, long-term, and episodic memory.

    This is the main interface the agent uses to interact with memory.
    It provides convenience methods that delegate to the appropriate
    subsystem.

    Example::

        manager = MemoryManager(MemoryConfig())
        manager.add_message(Message(role="user", content="Hello"))
        manager.store_fact("capital_of_france", "Paris", tags=["geography"])
        print(manager.search_facts("capital"))
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.short_term = ShortTermMemory(limit=config.short_term_limit)
        self.long_term = LongTermMemory(limit=config.long_term_limit)
        self.episodic = EpisodicMemory(limit=config.episode_limit)

    # ------------------------------------------------------------------
    # Short-term memory
    # ------------------------------------------------------------------

    def add_message(self, message: Message) -> None:
        self.short_term.add(message)

    def add_messages(self, messages: Sequence[Message]) -> None:
        self.short_term.add_many(messages)

    def get_context(self) -> List[Dict[str, Any]]:
        return self.short_term.get_context()

    def clear_short_term(self) -> None:
        self.short_term.clear()

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------

    def store_fact(
        self,
        key: str,
        value: Any,
        tags: Optional[List[str]] = None,
    ) -> None:
        self.long_term.store(key, value, tags)

    def retrieve_fact(self, key: str) -> Optional[Any]:
        return self.long_term.retrieve(key)

    def search_facts(self, query: str) -> List[MemoryEntry]:
        return self.long_term.search(query)

    def retrieve_by_tag(self, tag: str) -> List[MemoryEntry]:
        return self.long_term.retrieve_by_tag(tag)

    def clear_long_term(self) -> None:
        self.long_term.clear()

    # ------------------------------------------------------------------
    # Episodic memory
    # ------------------------------------------------------------------

    def add_episode(self, episode: Episode) -> None:
        self.episodic.add(episode)

    def get_recent_episodes(self, n: int = 5) -> List[Episode]:
        return self.episodic.get_recent(n)

    def clear_episodic(self) -> None:
        self.episodic.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist long-term memory and episodes to a JSON file."""
        path = Path(path)
        data = {
            "long_term": self.long_term.to_dict(),
            "episodic": self.episodic.to_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Memory persisted to %s", path)

    def load(self, path: str | Path) -> None:
        """Load long-term memory and episodes from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Memory file not found: %s — starting fresh", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.long_term.from_dict(data.get("long_term", {}))
        self.episodic.from_dict(data.get("episodic", []))
        logger.info("Memory loaded from %s (%d facts, %d episodes)",
                    path, len(self.long_term), len(self.episodic))

    def reset(self) -> None:
        """Clear all memory subsystems."""
        self.short_term.clear()
        self.long_term.clear()
        self.episodic.clear()
