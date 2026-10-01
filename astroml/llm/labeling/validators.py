from astroml.utils.exceptions import AstroMLError
"""Label validation for LLM-based data labeling (issue #475)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .schemas import Label, LabelDefinition

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """A validation rule for labels.

    Attributes:
        name: Rule name
        description: Rule description
        validator: Validation function
        severity: Severity of violation (error, warning, info)
    """

    name: str
    description: str
    validator: Callable[[Label, Any], bool]
    severity: str = "error"

    def validate(self, label: Label, context: Any) -> tuple[bool, Optional[str]]:
        """Validate a label against this rule.

        Args:
            label: Label to validate
            context: Additional context

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            is_valid = self.validator(label, context)
            return is_valid, None if is_valid else f"Rule '{self.name}' violated"
        except AstroMLError as e:
            return False, f"Rule '{self.name}' error: {str(e)}"


class LabelValidator:
    """Validator for generated labels."""

    def __init__(self) -> Any:
        """Initialize label validator."""
        self.rules: List[ValidationRule] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        # Confidence range check
        self.add_rule(
            ValidationRule(
                name="confidence_range",
                description="Confidence must be between 0 and 1",
                validator=lambda l, c: 0 <= l.confidence <= 1,
                severity="error",
            )
        )

        # Value not empty
        self.add_rule(
            ValidationRule(
                name="value_not_empty",
                description="Label value must not be empty",
                validator=lambda l, c: l.value is not None and l.value != "",
                severity="error",
            )
        )

        # Label name not empty
        self.add_rule(
            ValidationRule(
                name="label_name_not_empty",
                description="Label name must not be empty",
                validator=lambda l, c: l.label_name is not None and l.label_name != "",
                severity="error",
            )
        )

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.

        Args:
            rule: Validation rule to add
        """
        self.rules.append(rule)

    def validate_label(
        self,
        label: Label,
        definition: Optional[LabelDefinition] = None,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Validate a single label.

        Args:
            label: Label to validate
            definition: Optional label definition for additional validation
            context: Additional context

        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        info = []

        for rule in self.rules:
            is_valid, message = rule.validate(label, context)
            if not is_valid:
                if rule.severity == "error":
                    errors.append(message)
                elif rule.severity == "warning":
                    warnings.append(message)
                else:
                    info.append(message)

        # Validate against definition if provided
        if definition:
            def_errors = self._validate_against_definition(label, definition)
            errors.extend(def_errors)

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "info": info,
        }

    def _validate_against_definition(
        self,
        label: Label,
        definition: LabelDefinition,
    ) -> List[str]:
        """Validate label against its definition.

        Args:
            label: Label to validate
            definition: Label definition

        Returns:
            List of error messages
        """
        errors = []

        # Check label name matches
        if label.label_name != definition.name:
            errors.append(
                f"Label name '{label.label_name}' does not match definition '{definition.name}'"
            )

        # Check allowed values
        if definition.allowed_values and label.value not in definition.allowed_values:
            errors.append(
                f"Label value '{label.value}' not in allowed values: {definition.allowed_values}"
            )

        # Check confidence threshold
        if label.confidence < definition.confidence_threshold:
            errors.append(
                f"Label confidence {label.confidence} below threshold {definition.confidence_threshold}"
            )

        return errors

    def validate_labels(
        self,
        labels: List[Label],
        definitions: Optional[Dict[str, LabelDefinition]] = None,
        context: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Validate multiple labels.

        Args:
            labels: Labels to validate
            definitions: Optional label definitions
            context: Additional context

        Returns:
            List of validation results
        """
        results = []
        for label in labels:
            definition = definitions.get(label.label_name) if definitions else None
            result = self.validate_label(label, definition, context)
            results.append(result)
        return results
