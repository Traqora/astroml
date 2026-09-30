"""Base tool system for agent function calling."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata or {},
        }


class Tool(ABC):
    """Base class for agent tools."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize tool.

        Args:
            name: Tool name
            description: Tool description
            parameters: Parameter schema
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    @abstractmethod
    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute the tool.

        Args:
            input_data: Input parameters
            context: Execution context

        Returns:
            ToolResult with output
        """
        raise NotImplementedError("Subclasses must implement execute")

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate input against parameter schema.

        Args:
            input_data: Input to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Simple validation - check required parameters
        for param_name, param_schema in self.parameters.items():
            if param_schema.get("required", False) and param_name not in input_data:
                return False, f"Missing required parameter: {param_name}"

            # Check type if specified
            if param_name in input_data and "type" in param_schema:
                expected_type = param_schema["type"]
                actual_value = input_data[param_name]

                if expected_type == "string" and not isinstance(actual_value, str):
                    return False, f"Parameter {param_name} must be a string"
                elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                    return False, f"Parameter {param_name} must be a number"
                elif expected_type == "boolean" and not isinstance(actual_value, bool):
                    return False, f"Parameter {param_name} must be a boolean"
                elif expected_type == "array" and not isinstance(actual_value, list):
                    return False, f"Parameter {param_name} must be an array"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        """Initialize tool registry.

        Args:
            tools: Initial list of tools
        """
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool to register
        """
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool.

        Args:
            tool_name: Name of tool to unregister
        """
        if tool_name in self._tools:
            del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name.

        Args:
            tool_name: Name of tool

        Returns:
            Tool if found, None otherwise
        """
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools.

        Returns:
            List of all tools
        """
        return list(self._tools.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "tools": [tool.to_dict() for tool in self._tools.values()],
            "count": len(self._tools),
        }


class FunctionTool(Tool):
    """Tool that wraps a Python function."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize function tool.

        Args:
            name: Tool name
            description: Tool description
            func: Function to execute
            parameters: Parameter schema
        """
        super().__init__(name, description, parameters)
        self.func = func

    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute the wrapped function.

        Args:
            input_data: Input parameters
            context: Execution context

        Returns:
            ToolResult with function output
        """
        is_valid, error = self.validate_input(input_data)
        if not is_valid:
            return ToolResult(success=False, data=None, error=error)

        try:
            result = self.func(**input_data)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
