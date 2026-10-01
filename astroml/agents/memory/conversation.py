"""Conversation memory with role-based message tracking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import MemoryStore, Message


class ConversationMemory(MemoryStore):
    """Conversation memory with role-based organization and summarization."""

    def __init__(self, max_history: int = 100) -> None:
        """Initialize conversation memory.

        Args:
            max_history: Maximum number of messages to keep
        """
        super().__init__()
        self.max_history = max_history

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add message with history limit.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        super().add_message(role, content, metadata)

        # Enforce history limit
        if len(self._messages) > self.max_history:
            # Keep the most recent messages
            self._messages = self._messages[-self.max_history:]

    def get_user_messages(self) -> List[Message]:
        """Get all user messages.

        Returns:
            List of user messages
        """
        return [msg for msg in self._messages if msg.role == "user"]

    def get_assistant_messages(self) -> List[Message]:
        """Get all assistant messages.

        Returns:
            List of assistant messages
        """
        return [msg for msg in self._messages if msg.role == "assistant"]

    def get_system_messages(self) -> List[Message]:
        """Get all system messages.

        Returns:
            List of system messages
        """
        return [msg for msg in self._messages if msg.role == "system"]

    def summarize(self, max_length: int = 500) -> str:
        """Generate a summary of the conversation.

        Args:
            max_length: Maximum length of summary

        Returns:
            Conversation summary
        """
        if not self._messages:
            return "No conversation history."

        summary_parts = []
        total_length = 0

        for msg in self._messages:
            msg_summary = f"{msg.role}: {msg.content[:100]}..."
            if total_length + len(msg_summary) > max_length:
                break
            summary_parts.append(msg_summary)
            total_length += len(msg_summary)

        return "\n".join(summary_parts)

    def get_dialogue_pairs(self) -> List[tuple[Message, Optional[Message]]]:
        """Get conversation as user-assistant pairs.

        Returns:
            List of (user_message, assistant_message) tuples
        """
        pairs = []
        i = 0
        while i < len(self._messages):
            if self._messages[i].role == "user":
                user_msg = self._messages[i]
                assistant_msg = None
                if i + 1 < len(self._messages) and self._messages[i + 1].role == "assistant":
                    assistant_msg = self._messages[i + 1]
                    i += 1
                pairs.append((user_msg, assistant_msg))
            i += 1
        return pairs
