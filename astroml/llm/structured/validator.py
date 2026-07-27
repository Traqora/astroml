"""Validation logic for structured outputs."""
import logging
from typing import Any, Dict, List, Type, TypeVar
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ValidationResult:
    """Result of output validation."""

    def __init__(self, valid: bool, errors: List[str] = None, data: Any = None):
        self.valid = valid
        self.errors = errors or []
        self.data = data

    def __bool__(self) -> bool:
        return self.valid


class OutputValidator:
    """Validator for structured outputs with detailed error messages."""

    def validate(self, data: Dict[str, Any], schema: Type[T]) -> ValidationResult:
        """Validate data against schema.

        Args:
            data: Dictionary to validate
            schema: Pydantic model class

        Returns:
            ValidationResult with detailed error messages
        """
        try:
            instance = schema(**data)
            return ValidationResult(valid=True, data=instance)
        except ValidationError as e:
            errors = self._format_errors(e)
            return ValidationResult(valid=False, errors=errors, data=None)

    def _format_errors(self, error: ValidationError) -> List[str]:
        """Format validation errors into readable messages."""
        messages = []
        for err in error.errors():
            field = ".".join(str(x) for x in err["loc"])
            msg = err["msg"]
            messages.append(f"Field '{field}': {msg}")
        return messages

    def get_missing_fields(self, data: Dict[str, Any], schema: Type[T]) -> List[str]:
        """Identify missing required fields."""
        required = set()
        for name, field in schema.model_fields.items():
            if field.is_required():
                required.add(name)

        present = set(data.keys())
        return list(required - present)

    def get_invalid_fields(self, data: Dict[str, Any], schema: Type[T]) -> Dict[str, str]:
        """Identify fields with invalid values."""
        invalid = {}
        try:
            schema(**data)
        except ValidationError as e:
            for err in error.errors():
                field = ".".join(str(x) for x in err["loc"])
                invalid[field] = err["msg"]
        return invalid
