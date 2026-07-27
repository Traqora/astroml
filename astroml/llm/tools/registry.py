"""Tool registration and discovery."""

from typing import Any
from .definitions import BaseTool


class ToolRegistry:
    """Registry for discovering and managing AstroML tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [t.get_openai_schema() for t in self._tools.values()]

    def get_anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.get_anthropic_schema() for t in self._tools.values()]

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return self.get_openai_tools()


_registry: ToolRegistry | None = None


def get_global_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = None
