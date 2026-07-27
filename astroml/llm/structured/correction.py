"""Auto-correction of invalid structured outputs."""
import logging
from typing import Any, Dict, Type, TypeVar
from pydantic import BaseModel, ValidationError
from .validator import OutputValidator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AutoCorrector:
    """Automatic correction of validation failures."""

    def __init__(self):
        self.validator = OutputValidator()

    def correct(self, data: Dict[str, Any], schema: Type[T]) -> Dict[str, Any]:
        """Apply automatic corrections to data.

        Corrections applied:
        - Add default values for missing fields
        - Clamp numeric values to valid ranges
        - Convert types where possible
        - Remove unexpected fields

        Args:
            data: Invalid data dictionary
            schema: Target Pydantic schema

        Returns:
            Corrected data dictionary
        """
        corrected = dict(data)

        # Add defaults for missing required fields
        for name, field in schema.model_fields.items():
            if name not in corrected:
                if not field.is_required():
                    corrected[name] = field.get_default()
                elif field.annotation in (str, int, float, bool, list, dict):
                    corrected[name] = self._get_type_default(field.annotation)

        # Clamp numeric values
        for name, field in schema.model_fields.items():
            if name not in corrected:
                continue

            value = corrected[name]
            constraints = field.metadata

            # Check for ge/le constraints
            if hasattr(field, "ge") and field.ge is not None:
                if isinstance(value, (int, float)) and value < field.ge:
                    corrected[name] = field.ge
                    logger.warning(f"Clamped {name} from {value} to {field.ge}")

            if hasattr(field, "le") and field.le is not None:
                if isinstance(value, (int, float)) and value > field.le:
                    corrected[name] = field.le
                    logger.warning(f"Clamped {name} from {value} to {field.le}")

        # Remove unexpected fields
        valid_fields = set(schema.model_fields.keys())
        for key in list(corrected.keys()):
            if key not in valid_fields:
                logger.debug(f"Removing unexpected field: {key}")
                del corrected[key]

        return corrected

    def _get_type_default(self, annotation: Type) -> Any:
        """Get default value for a type."""
        if annotation == str:
            return ""
        elif annotation == int:
            return 0
        elif annotation == float:
            return 0.0
        elif annotation == bool:
            return False
        elif annotation == list:
            return []
        elif annotation == dict:
            return {}
        return None

    def generate_correction_prompt(self, data: Dict[str, Any], schema: Type[T], errors: list[str]) -> str:
        """Generate prompt for LLM to self-correct invalid output.

        Args:
            data: Invalid data
            schema: Target schema
            errors: Validation error messages

        Returns:
            Correction prompt for LLM
        """
        schema_str = schema.model_json_schema()
        errors_str = "\n".join(f"- {err}" for err in errors)

        return f"""The previous response did not match the required schema. Please fix the following errors and return ONLY valid JSON:

Schema:
{schema_str}

Errors:
{errors_str}

Invalid data:
{data}

Provide the corrected JSON output that matches the schema exactly."""
