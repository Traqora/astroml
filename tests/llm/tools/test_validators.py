"""Tests for tool validators."""

import pytest
from astroml.llm.tools.validators import validate_parameters, ValidationError


SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "count"],
}


class TestValidators:
    def test_valid_params_pass(self):
        validate_parameters({"name": "test", "count": 5}, SIMPLE_SCHEMA)

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError, match="Missing required parameter"):
            validate_parameters({"name": "test"}, SIMPLE_SCHEMA)

    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError, match="Unknown parameter"):
            validate_parameters({"name": "test", "count": 5, "extra": "bad"}, SIMPLE_SCHEMA)

    def test_wrong_type_raises(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_parameters({"name": "test", "count": "not_a_number"}, SIMPLE_SCHEMA)

    def test_boolean_type_valid(self):
        validate_parameters({"name": "test", "count": 5, "active": True}, SIMPLE_SCHEMA)
