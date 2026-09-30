"""Tests for tool audit log."""

from astroml.llm.tools.audit import ToolAuditLog


class TestToolAuditLog:
    def setup_method(self):
        self.audit = ToolAuditLog()

    def test_record_and_retrieve(self):
        self.audit.record(
            tool_name="test_tool",
            params={"input": "test"},
            result={"output": "done"},
            user_id="user_1",
            duration=0.5,
        )
        entries = self.audit.get_entries()
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "test_tool"
        assert entries[0]["result"] == {"output": "done"}
        assert entries[0]["error"] is None

    def test_record_error(self):
        self.audit.record(
            tool_name="test_tool",
            params={"input": "test"},
            result=None,
            user_id="user_1",
            duration=0.3,
            error="Something went wrong",
        )
        entries = self.audit.get_entries()
        assert entries[0]["error"] == "Something went wrong"
        assert entries[0]["result"] is None

    def test_clear(self):
        self.audit.record("t1", {}, {}, "u1")
        self.audit.clear()
        assert len(self.audit.get_entries()) == 0

    def test_limit(self):
        for i in range(50):
            self.audit.record(f"t{i}", {}, {}, "u1")
        entries = self.audit.get_entries(limit=10)
        assert len(entries) == 10
