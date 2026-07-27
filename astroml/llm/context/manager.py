"""Context manager for LLM conversations."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class MessageRole(str, Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Single message in conversation."""

    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
        }


class ContextManager:
    """Manages conversation context with token budgeting and pruning."""

    def __init__(
        self,
        model: str = "gpt-4",
        max_tokens: int = 8192,
        reserve_tokens: int = 500,
        pruning_strategy: str = "sliding_window",
    ):
        """Initialize context manager.

        Args:
            model: LLM model name
            max_tokens: Maximum tokens in context window
            reserve_tokens: Reserved tokens for response
            pruning_strategy: Strategy for pruning: 'sliding_window', 'summarization', 'importance'
        """
        self.model = model
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.pruning_strategy = pruning_strategy
        self.messages: List[Message] = []
        self.system_prompt: Optional[str] = None
        self.context_budget = max_tokens - reserve_tokens

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt (never pruned)."""
        self.system_prompt = prompt

    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict] = None) -> None:
        """Add message to conversation.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        token_count = self._estimate_tokens(content)
        message = Message(
            role=role,
            content=content,
            token_count=token_count,
            metadata=metadata or {},
        )

        self.messages.append(message)
        self._ensure_budget()

    def get_context(self) -> str:
        """Get formatted context for LLM.

        Returns:
            Formatted context string with system prompt and messages
        """
        context_parts = []

        if self.system_prompt:
            context_parts.append(f"<system>\n{self.system_prompt}\n</system>\n")

        for msg in self.messages:
            if msg.role == MessageRole.SYSTEM:
                continue
            context_parts.append(f"<{msg.role.value}>\n{msg.content}\n</{msg.role.value}>\n")

        return "\n".join(context_parts)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get messages in format for API calls."""
        messages = []

        for msg in self.messages:
            messages.append(msg.to_dict())

        return messages

    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics."""
        system_tokens = self._estimate_tokens(self.system_prompt) if self.system_prompt else 0
        message_tokens = sum(msg.token_count for msg in self.messages)
        total = system_tokens + message_tokens

        return {
            "system": system_tokens,
            "messages": message_tokens,
            "total": total,
            "remaining": self.context_budget - total,
        }

    def can_add_message(self, content: str) -> bool:
        """Check if message can be added within budget."""
        token_count = self._estimate_tokens(content)
        current_usage = sum(msg.token_count for msg in self.messages)
        system_tokens = self._estimate_tokens(self.system_prompt) if self.system_prompt else 0

        return current_usage + system_tokens + token_count <= self.context_budget

    def _ensure_budget(self) -> None:
        """Apply pruning strategy if over budget."""
        usage = self.get_token_usage()

        if usage["total"] <= self.context_budget:
            return

        if self.pruning_strategy == "sliding_window":
            self._prune_sliding_window()
        elif self.pruning_strategy == "importance":
            self._prune_importance()
        else:
            self._prune_sliding_window()

    def _prune_sliding_window(self, keep_recent: int = 10) -> None:
        """Keep only most recent messages."""
        if len(self.messages) <= keep_recent:
            return

        self.messages = self.messages[-keep_recent:]

    def _prune_importance(self) -> None:
        """Keep most important messages based on scoring."""
        if not self.messages:
            return

        scores = []
        for i, msg in enumerate(self.messages):
            score = self._score_message(msg, i)
            scores.append((score, i))

        scores.sort(reverse=True)
        keep_indices = sorted([idx for _, idx in scores[: len(self.messages) // 2]])
        self.messages = [self.messages[i] for i in keep_indices]

    def _score_message(self, msg: Message, position: int) -> float:
        """Score message for importance."""
        score = 0.0

        if msg.role == MessageRole.ASSISTANT:
            score += 0.7
        else:
            score += 0.3

        if "important" in msg.metadata:
            score += 1.0

        recency_factor = 1.0 + (position / len(self.messages))
        score *= recency_factor

        return score

    @staticmethod
    def _estimate_tokens(text: Optional[str]) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        if not text:
            return 0
        return len(text) // 4 + 1

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []

    def export_history(self) -> List[Dict[str, Any]]:
        """Export full conversation history."""
        return [msg.to_dict() for msg in self.messages]
