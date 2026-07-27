"""Tests for tool registry."""

import pytest
from astroml.llm.tools import ToolRegistry, get_global_registry, reset_registry
from astroml.llm.tools.definitions import BaseTool


class MockTool(BaseTool):
    name = "mock_tool"
    description = "A mock tool for testing"
    parameters = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }

    async def execute(self, params):
        return {"result": f"processed {params.get('input')}"}


class TestToolRegistry:
    def setup_method(self):
        reset_registry()

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        assert registry.get("mock_tool") is tool

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockTool())

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        assert registry.list_tools() == ["mock_tool"]

    def test_get_unknown_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_get_openai_tools(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        schemas = registry.get_openai_tools()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "mock_tool"

    def test_get_anthropic_tools(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        schemas = registry.get_anthropic_tools()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "mock_tool"
        assert "input_schema" in schemas[0]

    def test_global_registry_singleton(self):
        r1 = get_global_registry()
        r2 = get_global_registry()
        assert r1 is r2
