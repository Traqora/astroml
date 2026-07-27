"""Structured generation orchestrator."""
import time
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

from ..providers.factory import get_llm_provider
from .parser import PydanticParser
from .validator import OutputValidator
from .correction import AutoCorrector
from .prompts import PromptAugmenter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredGenerator:
    """Orchestrates structured output generation with validation and correction."""

    def __init__(
        self,
        provider_name: str = None,
        max_retries: int = 3,
        enable_auto_correction: bool = True,
    ):
        """Initialize structured generator.

        Args:
            provider_name: LLM provider to use
            max_retries: Maximum correction attempts
            enable_auto_correction: Whether to auto-correct validation failures
        """
        self.provider = get_llm_provider(provider_name)
        self.parser = PydanticParser(enable_coercion=True)
        self.validator = OutputValidator()
        self.corrector = AutoCorrector()
        self.augmenter = PromptAugmenter()
        self.max_retries = max_retries
        self.enable_auto_correction = enable_auto_correction

    def generate_structured(
        self,
        prompt: str,
        output_schema: Type[T],
        examples: List[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> T:
        """Generate structured output matching the schema.

        Args:
            prompt: User prompt
            output_schema: Pydantic model class
            examples: Optional few-shot examples
            **kwargs: Additional provider parameters

        Returns:
            Validated instance of output_schema

        Raises:
            ValidationError: If output cannot be validated after all retries
        """
        start_time = time.time()

        # Augment prompt with schema
        augmented_prompt = self.augmenter.augment(prompt, output_schema, examples)

        # Add JSON mode hint for provider
        provider_name = self.provider.__class__.__name__.lower().replace("provider", "")
        kwargs.setdefault("temperature", 0.3)  # Lower temperature for structured output

        # Provider-specific JSON mode
        if provider_name == "openai":
            kwargs["response_format"] = {"type": "json_object"}

        attempt = 0
        last_error = None

        while attempt <= self.max_retries:
            try:
                # Generate response
                response_text = self.provider.generate(augmented_prompt, **kwargs)

                # Parse and validate
                parsed = self.parser.parse(response_text, output_schema)

                latency_ms = (time.time() - start_time) * 1000
                logger.info(f"Structured output generated successfully in {latency_ms:.0f}ms (attempt {attempt + 1})")

                return parsed

            except (ValidationError, ValueError) as e:
                last_error = e
                attempt += 1

                if attempt > self.max_retries:
                    logger.error(f"Failed to generate valid structured output after {self.max_retries} retries")
                    raise

                logger.warning(f"Validation failed (attempt {attempt}/{self.max_retries}): {e}")

                if self.enable_auto_correction and isinstance(e, ValidationError):
                    # Try auto-correction
                    try:
                        # Extract data from error
                        invalid_data = self._extract_data_from_response(response_text)
                        corrected_data = self.corrector.correct(invalid_data, output_schema)

                        # Validate corrected data
                        result = self.validator.validate(corrected_data, output_schema)
                        if result.valid:
                            logger.info("Auto-correction successful")
                            return result.data

                        # Generate correction prompt for next attempt
                        augmented_prompt = self.corrector.generate_correction_prompt(
                            corrected_data, output_schema, result.errors
                        )
                    except Exception as correction_error:
                        logger.warning(f"Auto-correction failed: {correction_error}")

        # Should not reach here, but just in case
        raise last_error or RuntimeError("Failed to generate structured output")

    def _extract_data_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract dictionary from response text."""
        import json
        import re

        # Try to extract JSON
        code_block_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        match = re.search(code_block_pattern, response_text, re.MULTILINE)
        if match:
            return json.loads(match.group(1))

        json_pattern = r"\{[\s\S]*\}"
        match = re.search(json_pattern, response_text)
        if match:
            return json.loads(match.group(0))

        return {}

    async def generate_structured_async(
        self,
        prompt: str,
        output_schema: Type[T],
        examples: List[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> T:
        """Async version of generate_structured.

        Note: Current implementation wraps sync call. For true async,
        provider would need async generate support.
        """
        import asyncio
        return await asyncio.to_thread(
            self.generate_structured,
            prompt,
            output_schema,
            examples,
            **kwargs,
        )
