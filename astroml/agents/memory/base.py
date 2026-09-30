"""Base memory store implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """Represents a message in conversation history."""

    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class MemoryStore:
    """Base class for memory storage."""

    def __init__(self) -> None:
        """Initialize memory store."""
        self._messages: List[Message] = []

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to memory.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )
        self._messages.append(message)

    def get_messages(self) -> List[Message]:
        """Get all messages.

        Returns:
            List of all messages
        """
        return self._messages.copy()

    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get most recent n messages.

        Args:
            n: Number of recent messages to retrieve

        Returns:
            List of recent messages
        """
        return self._messages[-n:] if n > 0 else []

    def clear(self) -> None:
        """Clear all messages from memory."""
        self._messages.clear()

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Search messages by content (simple keyword match).

        Args:
            query: Search query
            limit: Optional limit on results

        Returns:
            List of matching messages
        """
        query_lower = query.lower()
        matches = [
            msg for msg in self._messages if query_lower in msg.content.lower()
        ]
        return matches[:limit] if limit else matches

    def get_context_window(
        self,
        max_tokens: int = 4000,
        avg_tokens_per_char: float = 0.25,
    ) -> List[Message]:
        """Get messages that fit within token limit.

        Args:
            max_tokens: Maximum tokens in context window
            avg_tokens_per_char: Average tokens per character

        Returns:
            List of messages fitting in token limit
        """
        result = []
        total_chars = 0
        max_chars = int(max_tokens / avg_tokens_per_char)

        for msg in reversed(self._messages):
            msg_chars = len(msg.content)
            if total_chars + msg_chars > max_chars:
                break
            result.insert(0, msg)
            total_chars += msg_chars

        return result
