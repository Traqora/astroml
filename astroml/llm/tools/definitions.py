"""Tool schema definitions and base tool class."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for all AstroML tools."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def get_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters.get("properties", {}),
                    "required": self.parameters.get("required", []),
                },
            },
        }

    def get_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters.get("properties", {}),
                "required": self.parameters.get("required", []),
            },
        }

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> Any:
        """Execute the tool with the given parameters."""
        pass
