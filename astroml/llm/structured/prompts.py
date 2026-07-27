"""Schema-aware prompt augmentation."""
import json
from typing import Any, Dict, List, Type
from pydantic import BaseModel


class PromptAugmenter:
    """Augments prompts with schema information for better structured outputs."""

    def augment(self, prompt: str, schema: Type[BaseModel], examples: List[Dict[str, Any]] = None) -> str:
        """Add schema information to prompt.

        Args:
            prompt: Original user prompt
            schema: Target Pydantic schema
            examples: Optional few-shot examples

        Returns:
            Augmented prompt with schema instructions
        """
        schema_json = schema.model_json_schema()
        schema_description = self._format_schema_description(schema_json)

        augmented = f"""{prompt}

Please respond with valid JSON matching the following schema:

{schema_description}

"""

        if examples:
            examples_str = self._format_examples(examples)
            augmented += f"""Examples:

{examples_str}

"""

        augmented += "Respond with ONLY the JSON object, no additional text."

        return augmented

    def _format_schema_description(self, schema: Dict[str, Any]) -> str:
        """Format schema into readable description."""
        lines = []
        lines.append("```json")
        lines.append(json.dumps(schema, indent=2))
        lines.append("```")
        return "\n".join(lines)

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format few-shot examples."""
        lines = []
        for i, example in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append("```json")
            lines.append(json.dumps(example, indent=2))
            lines.append("```")
            lines.append("")
        return "\n".join(lines)

    def get_json_mode_message(self, provider: str) -> str:
        """Get provider-specific JSON mode instructions."""
        if provider == "openai":
            return 'Using OpenAI JSON mode. Set response_format={"type": "json_object"}'
        elif provider == "anthropic":
            return "Using Anthropic structured output with XML tags"
        elif provider == "local":
            return "Using constrained decoding for structured output"
        return ""
