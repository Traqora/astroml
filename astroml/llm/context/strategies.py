"""Pruning strategies for context management."""

from enum import Enum
from typing import List, Callable, Optional
from .manager import Message


class PruningStrategy(str, Enum):
    """Available context pruning strategies."""

    SLIDING_WINDOW = "sliding_window"
    SUMMARIZATION = "summarization"
    IMPORTANCE = "importance"
    HYBRID = "hybrid"


class WindowPruner:
    """Sliding window pruning strategy."""

    def __init__(self, window_size: int = 10):
        """Initialize with window size."""
        self.window_size = window_size

    def prune(self, messages: List[Message]) -> List[Message]:
        """Keep only recent messages."""
        if len(messages) <= self.window_size:
            return messages
        return messages[-self.window_size :]


class ImportancePruner:
    """Importance-based pruning strategy."""

    def __init__(self, scorer: Optional[Callable] = None, retain_ratio: float = 0.5):
        """Initialize with optional custom scorer.

        Args:
            scorer: Function to score messages
            retain_ratio: Fraction of messages to retain
        """
        self.scorer = scorer or self._default_scorer
        self.retain_ratio = retain_ratio

    def prune(self, messages: List[Message]) -> List[Message]:
        """Keep high-importance messages."""
        if not messages:
            return messages

        scored = [(self.scorer(msg, i), i) for i, msg in enumerate(messages)]
        scored.sort(reverse=True)

        keep_count = max(1, int(len(messages) * self.retain_ratio))
        keep_indices = sorted([idx for _, idx in scored[:keep_count]])

        return [messages[i] for i in keep_indices]

    @staticmethod
    def _default_scorer(msg: Message, position: int) -> float:
        """Default scoring function."""
        score = 0.0

        if msg.role.value == "assistant":
            score += 1.0

        if "important" in msg.metadata:
            score += 2.0

        recency = 1.0 + (position / max(1, len([msg])))
        score *= recency

        return score


class SummarizationPruner:
    """Summarization-based pruning (requires summarizer function)."""

    def __init__(self, summarizer: Optional[Callable] = None, summary_ratio: float = 0.5):
        """Initialize with optional custom summarizer.

        Args:
            summarizer: Function to summarize text
            summary_ratio: Target compression ratio
        """
        self.summarizer = summarizer
        self.summary_ratio = summary_ratio

    def prune(self, messages: List[Message], keep_recent: int = 5) -> List[Message]:
        """Summarize old messages, keep recent verbatim."""
        if len(messages) <= keep_recent:
            return messages

        recent = messages[-keep_recent:]
        old = messages[:-keep_recent]

        if self.summarizer and old:
            old_text = "\n".join([msg.content for msg in old])
            summary = self.summarizer(old_text)

            summary_msg = Message(
                role=old[0].role,
                content=f"[Summary]\n{summary}",
                metadata={"summarized": True},
            )
            return [summary_msg] + recent
        else:
            return recent


class HybridPruner:
    """Hybrid pruning combining multiple strategies."""

    def __init__(
        self,
        window_size: int = 10,
        importance_ratio: float = 0.3,
        summarize_old: bool = True,
    ):
        """Initialize hybrid pruner."""
        self.window_pruner = WindowPruner(window_size)
        self.importance_pruner = ImportancePruner(retain_ratio=importance_ratio)
        self.summarization_pruner = SummarizationPruner()
        self.summarize_old = summarize_old

    def prune(self, messages: List[Message]) -> List[Message]:
        """Apply hybrid pruning."""
        # First apply window
        windowed = self.window_pruner.prune(messages)

        # Then apply importance pruning
        important = self.importance_pruner.prune(windowed)

        if self.summarize_old and len(important) > 5:
            important = self.summarization_pruner.prune(important, keep_recent=3)

        return important
