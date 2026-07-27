"""Output parsers for structured LLM responses."""
import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OutputParser(ABC):
    """Base class for output parsers."""

    @abstractmethod
    def parse(self, text: str, schema: Type[T]) -> T:
        """Parse text into structured output."""
        pass


class JSONParser(OutputParser):
    """Parser for JSON-formatted LLM outputs."""

    def parse(self, text: str, schema: Type[T]) -> T:
        """Extract and parse JSON from text.

        Args:
            text: Raw LLM response text
            schema: Pydantic model class

        Returns:
            Parsed and validated schema instance

        Raises:
            ValidationError: If JSON doesn't match schema
            ValueError: If no valid JSON found
        """
        json_str = self._extract_json(text)
        if not json_str:
            raise ValueError("No JSON found in response")

        try:
            data = json.loads(json_str)
            return schema(**data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Invalid JSON: {e}")
        except ValidationError as e:
            logger.error(f"Schema validation error: {e}")
            raise

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling markdown code blocks."""
        # Try to find JSON in code blocks first
        code_block_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        match = re.search(code_block_pattern, text, re.MULTILINE)
        if match:
            return match.group(1)

        # Try to find raw JSON object
        json_pattern = r"\{[\s\S]*\}"
        match = re.search(json_pattern, text)
        if match:
            return match.group(0)

        return None


class PydanticParser(OutputParser):
    """Parser with automatic type coercion and correction."""

    def __init__(self, enable_coercion: bool = True):
        self.enable_coercion = enable_coercion
        self.json_parser = JSONParser()

    def parse(self, text: str, schema: Type[T]) -> T:
        """Parse with type coercion support.

        Args:
            text: Raw LLM response text
            schema: Pydantic model class

        Returns:
            Parsed and validated schema instance
        """
        json_str = self.json_parser._extract_json(text)
        if not json_str:
            raise ValueError("No JSON found in response")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        # Apply type coercion if enabled
        if self.enable_coercion:
            data = self._coerce_types(data, schema)

        return schema(**data)

    def _coerce_types(self, data: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
        """Apply type coercion to match schema.

        Handles:
        - String to int/float conversion
        - String to bool conversion
        - String to list conversion (comma-separated)
        """
        if not isinstance(data, dict):
            return data

        coerced = {}
        annotations = schema.model_fields

        for key, value in data.items():
            if key not in annotations:
                coerced[key] = value
                continue

            field = annotations[key]
            target_type = field.annotation

            # Handle Optional types
            if hasattr(target_type, "__origin__") and target_type.__origin__ is type(None) or "Optional" in str(target_type):
                if hasattr(target_type, "__args__"):
                    target_type = target_type.__args__[0]

            try:
                if target_type == int and isinstance(value, str):
                    coerced[key] = int(float(value))  # Handle "1.0" -> 1
                elif target_type == float and isinstance(value, str):
                    coerced[key] = float(value)
                elif target_type == bool and isinstance(value, str):
                    coerced[key] = value.lower() in ("true", "yes", "1")
                elif target_type == list and isinstance(value, str):
                    coerced[key] = [item.strip() for item in value.split(",")]
                else:
                    coerced[key] = value
            except (ValueError, TypeError):
                logger.warning(f"Failed to coerce {key}={value} to {target_type}")
                coerced[key] = value

        return coerced
