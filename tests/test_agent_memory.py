"""Tests for agent conversation memory."""
from __future__ import annotations

import pytest

from astroml.agent.memory import ConversationMemory, Memory
from astroml.agent.types import Message


class TestConversationMemory:
    def test_cannot_be_used_as_a_plain_interface(self):
        with pytest.raises(TypeError):
            Memory()

    def test_rejects_non_message_values(self):
        memory = ConversationMemory()
        with pytest.raises(TypeError):
            memory.add("not a message")

    def test_rejects_invalid_max_messages(self):
        with pytest.raises(ValueError):
            ConversationMemory(max_messages=0)

    def test_messages_preserve_insertion_order(self):
        memory = ConversationMemory()
        memory.add(Message.user("one"))
        memory.add(Message.assistant("two"))
        assert [message.content for message in memory.messages()] == ["one", "two"]

    def test_system_messages_are_pinned_first(self):
        memory = ConversationMemory(max_messages=2)
        memory.add(Message.user("early"))
        memory.add(Message.system("rules"))
        memory.add(Message.user("later"))
        memory.add(Message.assistant("answer"))
        contents = [message.content for message in memory.messages()]
        assert contents[0] == "rules"
        assert contents == ["rules", "later", "answer"]

    def test_history_is_bounded_and_evictions_are_recorded(self):
        memory = ConversationMemory(max_messages=2)
        for index in range(4):
            memory.add(Message.user(f"m{index}"))
        assert [message.content for message in memory.history] == ["m2", "m3"]
        assert [message.content for message in memory.evicted] == ["m0", "m1"]

    def test_on_evict_callback_receives_dropped_messages(self):
        dropped = []
        memory = ConversationMemory(max_messages=1, on_evict=dropped.append)
        memory.add(Message.user("first"))
        memory.add(Message.user("second"))
        assert [message.content for message in dropped] == ["first"]

    def test_extend_adds_messages_in_order(self):
        memory = ConversationMemory()
        memory.extend([Message.user("a"), Message.assistant("b")])
        assert [message.content for message in memory.messages()] == ["a", "b"]

    def test_seed_messages_are_stored(self):
        memory = ConversationMemory([Message.system("s"), Message.user("u")])
        assert [message.role.value for message in memory.messages()] == ["system", "user"]

    def test_len_counts_system_and_history(self):
        memory = ConversationMemory(
            [Message.system("s"), Message.system("s2"), Message.user("u")]
        )
        assert len(memory) == 3

    def test_clear_removes_everything(self):
        memory = ConversationMemory([Message.system("s"), Message.user("u")])
        memory.clear()
        assert memory.messages() == []
        assert len(memory) == 0

    def test_repr_reports_sizes(self):
        memory = ConversationMemory(max_messages=5)
        assert "max_messages=5" in repr(memory)
