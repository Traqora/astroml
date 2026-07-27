"""Tests for tool executor."""

import pytest
from astroml.llm.tools import (
    ToolRegistry,
    ToolExecutor,
    ToolExecutionError,
    PermissionChecker,
    ToolAuditLog,
)
from astroml.llm.tools.definitions import BaseTool
from astroml.llm.tools.validators import ValidationError


class FastTool(BaseTool):
    name = "fast_tool"
    description = "A fast tool"
    parameters = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}

    async def execute(self, params):
        return {"result": params["x"] * 2}


class FailingTool(BaseTool):
    name = "failing_tool"
    description = "A tool that always fails"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, params):
        raise ValueError("Intentional failure")


class TestToolExecutor:
    def setup_method(self):
        self.registry = ToolRegistry()
        self.registry.register(FastTool())
        self.registry.register(FailingTool())
        self.executor = ToolExecutor(self.registry)

    @pytest.mark.asyncio
    async def test_execute_success(self):
        result = await self.executor.execute("fast_tool", {"x": 5})
        assert result == {"result": 10}

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        with pytest.raises(ToolExecutionError, match="Unknown tool"):
            await self.executor.execute("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_invalid_params(self):
        with pytest.raises(ValidationError):
            await self.executor.execute("fast_tool", {})

    @pytest.mark.asyncio
    async def test_execute_retry_on_failure(self):
        with pytest.raises(ToolExecutionError):
            await self.executor.execute("failing_tool", {})

    @pytest.mark.asyncio
    async def test_executor_with_permissions(self):
        checker = PermissionChecker()
        checker.allow("fast_tool", "allowed_user")
        executor = ToolExecutor(self.registry, permission_checker=checker)
        result = await executor.execute("fast_tool", {"x": 3}, user_id="allowed_user")
        assert result == {"result": 6}

    @pytest.mark.asyncio
    async def test_executor_with_audit(self):
        audit = ToolAuditLog()
        executor = ToolExecutor(self.registry, audit_log=audit)
        await executor.execute("fast_tool", {"x": 1}, user_id="u1")
        entries = audit.get_entries()
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "fast_tool"
        assert entries[0]["error"] is None

    @pytest.mark.asyncio
    async def test_executor_audit_on_failure(self):
        audit = ToolAuditLog()
        executor = ToolExecutor(self.registry, audit_log=audit)
        with pytest.raises(ToolExecutionError):
            await executor.execute("failing_tool", {})
        entries = audit.get_entries()
        assert len(entries) == 1
        assert entries[0]["error"] is not None
