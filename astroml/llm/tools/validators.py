"""Input/output validation for tool parameters and results."""

from typing import Any


class ValidationError(Exception):
    """Raised when tool parameters or output fail validation."""
    pass


def validate_parameters(params: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate tool parameters against the schema."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for field in required:
        if field not in params:
            raise ValidationError(f"Missing required parameter: '{field}'")

    for field, value in params.items():
        if field not in properties:
            raise ValidationError(f"Unknown parameter: '{field}'")
        prop = properties[field]
        _validate_value(field, value, prop)


def _validate_value(field: str, value: Any, prop: dict[str, Any]) -> None:
    expected_type = prop.get("type", "string")
    if expected_type == "string" and not isinstance(value, str):
        raise ValidationError(f"Parameter '{field}' must be a string")
    elif expected_type == "integer" and not isinstance(value, int):
        if isinstance(value, float) and value == int(value):
            return
        if isinstance(value, str):
            try:
                int(value)
                return
            except ValueError:
                pass
        raise ValidationError(f"Parameter '{field}' must be an integer")
    elif expected_type == "number" and not isinstance(value, (int, float)):
        raise ValidationError(f"Parameter '{field}' must be a number")
    elif expected_type == "boolean" and not isinstance(value, bool):
        raise ValidationError(f"Parameter '{field}' must be a boolean")
    elif expected_type == "array" and not isinstance(value, list):
        raise ValidationError(f"Parameter '{field}' must be an array")
    elif expected_type == "object" and not isinstance(value, dict):
        raise ValidationError(f"Parameter '{field}' must be an object")


def validate_output_size(result: Any, max_bytes: int = 100_000) -> None:
    """Validate that tool output does not exceed max size."""
    import json
    serialized = json.dumps(result)
    if len(serialized) > max_bytes:
        raise ValidationError(
            f"Tool output exceeds maximum size of {max_bytes} bytes"
        )
