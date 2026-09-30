"""Conversation memory for agent runs.

The executor is stateless with respect to history: it asks a
:class:`Memory` for the messages to send on every turn and writes each new
message back.  That keeps "how much context do we keep?" a policy decision
rather than a hard-coded loop detail.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional

from .types import Message, Role


class Memory(ABC):
    """Minimal message store interface used by the agent loop."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Append *message* to the store."""

    @abstractmethod
    def messages(self) -> List[Message]:
        """Return the messages to send to the model, oldest first."""

    def extend(self, messages: Iterable[Message]) -> None:
        """Append several messages in order."""
        for message in messages:
            self.add(message)

    def extend_parallel(
        self,
        messages: Iterable[Message],
        *,
        max_workers: Optional[int] = None,
        threshold: int = 10,
    ) -> None:
        """Append several messages with parallel processing for validation.

        Uses ThreadPoolExecutor to parallelize message validation and type checking.
        Beneficial for bulk insertion of large message batches.

        Args:
            messages: Iterable of messages to append.
            max_workers: Maximum number of worker threads.
            threshold: Minimum number of messages to enable parallel processing.
        """
        message_list = list(messages)
        if len(message_list) < threshold:
            for message in message_list:
                self.add(message)
            return

        def validate_and_add(message: Message) -> None:
            """Validate and add a single message."""
            if not isinstance(message, Message):
                raise TypeError("ConversationMemory only stores Message instances")
            if message.role is Role.SYSTEM:
                self._system.append(message)
            else:
                self._history.append(message)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # First validate all messages in parallel
            futures = [executor.submit(validate_and_add, msg) for msg in message_list]
            for future in as_completed(futures):
                future.result()  # Raise any validation errors

        # Then enforce the max_messages constraint sequentially to maintain order
        while len(self._history) > self.max_messages:
            dropped = self._history.pop(0)
            self.evicted.append(dropped)
            if self.on_evict is not None:
                self.on_evict(dropped)

    def clear(self) -> None:
        """Drop all stored messages."""
        raise NotImplementedError("clear() is not implemented by this memory")


class ConversationMemory(Memory):
    """Bounded, system-message aware conversation buffer.

    System messages are pinned (never evicted) and always returned first.
    The remaining messages form a FIFO buffer capped at ``max_messages``.
    ``on_evict`` is invoked with every evicted message, which is the intended
    hook for rolling summarisation.

    Args:
        messages: Optional seed messages (may include system messages).
        max_messages: Maximum number of non-system messages retained.
        on_evict: Callback invoked once per evicted message.
    """

    def __init__(
        self,
        messages: Optional[Iterable[Message]] = None,
        *,
        max_messages: int = 40,
        on_evict: Optional[Callable[[Message], None]] = None,
    ) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self.max_messages = max_messages
        self.on_evict = on_evict
        self._system: List[Message] = []
        self._history: List[Message] = []
        #: Messages dropped from the buffer, oldest first.
        self.evicted: List[Message] = []
        if messages:
            self.extend(messages)

    def add(self, message: Message) -> None:
        """Store *message*, evicting the oldest history entry if needed."""
        if not isinstance(message, Message):
            raise TypeError("ConversationMemory only stores Message instances")
        if message.role is Role.SYSTEM:
            self._system.append(message)
            return

        self._history.append(message)
        while len(self._history) > self.max_messages:
            dropped = self._history.pop(0)
            self.evicted.append(dropped)
            if self.on_evict is not None:
                self.on_evict(dropped)

    def messages(self) -> List[Message]:
        """Pinned system messages followed by the rolling history."""
        return [*self._system, *self._history]

    def clear(self) -> None:
        """Drop all messages, including pinned system messages."""
        self._system.clear()
        self._history.clear()

    @property
    def history(self) -> List[Message]:
        """Read-only view of the non-system messages."""
        return list(self._history)

    def filter_parallel(
        self,
        predicate: Callable[[Message], bool],
        *,
        max_workers: Optional[int] = None,
        threshold: int = 20,
    ) -> List[Message]:
        """Filter messages using a predicate with parallel processing.

        Uses ThreadPoolExecutor to parallelize predicate evaluation across messages.
        Beneficial for complex filtering operations on large message sets.

        Args:
            predicate: Function that returns True for messages to keep.
            max_workers: Maximum number of worker threads.
            threshold: Minimum number of messages to enable parallel processing.

        Returns:
            List of messages that satisfy the predicate, in original order.
        """
        all_messages = self.messages()
        if len(all_messages) < threshold:
            return [msg for msg in all_messages if predicate(msg)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Evaluate predicate in parallel
            futures = {
                executor.submit(predicate, msg): (idx, msg)
                for idx, msg in enumerate(all_messages)
            }
            results = []
            for future in as_completed(futures):
                idx, msg = futures[future]
                if future.result():
                    results.append((idx, msg))

        # Sort by original index to maintain order
        results.sort(key=lambda x: x[0])
        return [msg for _, msg in results]

    def __len__(self) -> int:
        return len(self._system) + len(self._history)

    def __repr__(self) -> str:
        return (
            f"ConversationMemory(system={len(self._system)}, "
            f"history={len(self._history)}, max_messages={self.max_messages})"
        )
