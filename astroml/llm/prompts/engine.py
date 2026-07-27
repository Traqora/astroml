"""Template rendering engine for prompt templates."""

from typing import Any, Dict, Optional
import json
from jinja2 import Environment, Template, TemplateError, UndefinedError
from pydantic import BaseModel, ValidationError


class TemplateVariable(BaseModel):
    """Definition of a template variable."""

    name: str
    type: str = "str"
    required: bool = True
    default: Optional[Any] = None
    description: Optional[str] = None


class PromptTemplate(BaseModel):
    """Prompt template definition."""

    name: str
    version: str
    variables: list[TemplateVariable]
    template: str
    variants: Dict[str, str] = {}
    ab_test: Optional[Dict[str, float]] = None


class TemplateEngine:
    """Jinja2-based template rendering engine."""

    def __init__(self):
        """Initialize Jinja2 environment."""
        self.env = Environment(autoescape=False)
        self._cache = {}

    def validate_variables(
        self, template_def: PromptTemplate, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and type-convert variables against template definition."""
        validated = {}

        for var_def in template_def.variables:
            var_name = var_def.name

            if var_name in variables:
                value = variables[var_name]

                if var_def.type == "int":
                    validated[var_name] = int(value)
                elif var_def.type == "float":
                    validated[var_name] = float(value)
                elif var_def.type == "bool":
                    if isinstance(value, bool):
                        validated[var_name] = value
                    validated[var_name] = value in (True, "true", "True", 1, "1")
                else:  # str
                    validated[var_name] = str(value)
            elif var_def.default is not None:
                validated[var_name] = var_def.default
            elif var_def.required:
                raise ValueError(f"Required variable '{var_name}' not provided")

        return validated

    def render(
        self,
        template_def: PromptTemplate,
        variables: Dict[str, Any],
        variant: Optional[str] = None,
    ) -> str:
        """Render template with variables.

        Args:
            template_def: Template definition
            variables: Variable values
            variant: Optional variant name to use

        Returns:
            Rendered prompt string
        """
        # Validate variables
        validated_vars = self.validate_variables(template_def, variables)

        # Select template source
        if variant and variant in template_def.variants:
            template_source = template_def.variants[variant]
        else:
            template_source = template_def.template

        # Check cache
        cache_key = f"{template_def.name}:{template_def.version}:{variant or 'base'}"
        if cache_key not in self._cache:
            try:
                self._cache[cache_key] = self.env.from_string(template_source)
            except TemplateError as e:
                raise ValueError(f"Template syntax error: {e}")

        template = self._cache[cache_key]

        try:
            return template.render(**validated_vars)
        except UndefinedError as e:
            raise ValueError(f"Undefined variable in template: {e}")

    def render_string(self, template_string: str, variables: Dict[str, Any]) -> str:
        """Render a raw template string."""
        try:
            template = self.env.from_string(template_string)
            return template.render(**variables)
        except (TemplateError, UndefinedError) as e:
            raise ValueError(f"Template rendering error: {e}")

    def clear_cache(self):
        """Clear template cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {"size": len(self._cache), "cached_templates": list(self._cache.keys())}
