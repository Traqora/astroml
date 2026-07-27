"""Versioned prompt template registry."""

from typing import Any, Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path
import random

from .engine import PromptTemplate, TemplateEngine, TemplateVariable


class PromptRegistry:
    """Manages versioned prompt templates with A/B testing support."""

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize prompt registry.

        Args:
            storage_path: Path to directory for storing prompt definitions
        """
        self.engine = TemplateEngine()
        self.storage_path = Path(storage_path or "prompts")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, List[PromptTemplate]] = {}
        self._ab_variants: Dict[str, Dict[str, float]] = {}

    def register(self, template_def: PromptTemplate) -> None:
        """Register a new template or version.

        Args:
            template_def: Template to register
        """
        if template_def.name not in self.templates:
            self.templates[template_def.name] = []

        self.templates[template_def.name].append(template_def)
        self.templates[template_def.name].sort(
            key=lambda t: self._parse_version(t.version), reverse=True
        )

        if template_def.ab_test:
            self._ab_variants[template_def.name] = template_def.ab_test

    def get_latest(self, name: str) -> Optional[PromptTemplate]:
        """Get the latest version of a template."""
        if name not in self.templates or not self.templates[name]:
            return None
        return self.templates[name][0]

    def get_version(self, name: str, version: str) -> Optional[PromptTemplate]:
        """Get a specific version of a template."""
        if name not in self.templates:
            return None

        for template in self.templates[name]:
            if template.version == version:
                return template
        return None

    def get_for_ab_test(self, name: str) -> Optional[PromptTemplate]:
        """Get template variant based on A/B testing configuration.

        Args:
            name: Template name

        Returns:
            Template (base or variant based on random distribution)
        """
        template = self.get_latest(name)
        if not template or name not in self._ab_variants:
            return template

        variants = self._ab_variants[name]
        rand = random.random()
        cumulative = 0.0

        for variant_name, weight in variants.items():
            cumulative += weight
            if rand < cumulative:
                return self._get_variant(template, variant_name)

        return template

    def _get_variant(self, template: PromptTemplate, variant_name: str) -> PromptTemplate:
        """Get template with specific variant selected."""
        variant_template = PromptTemplate(**template.dict())
        variant_template.template = template.variants.get(
            variant_name, template.template
        )
        return variant_template

    def list_templates(self) -> Dict[str, str]:
        """List all registered templates with their latest versions."""
        return {name: versions[0].version for name, versions in self.templates.items()}

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a template."""
        if name not in self.templates:
            return []
        return [t.version for t in self.templates[name]]

    def save(self, template_def: PromptTemplate) -> None:
        """Save template definition to disk."""
        file_path = self.storage_path / f"{template_def.name}_{template_def.version}.json"
        with open(file_path, "w") as f:
            json.dump(template_def.dict(), f, indent=2)

    def load(self, name: str, version: Optional[str] = None) -> None:
        """Load template from disk.

        Args:
            name: Template name
            version: Specific version to load (loads latest if not specified)
        """
        if version:
            file_path = self.storage_path / f"{name}_{version}.json"
        else:
            # Find latest version file
            matching_files = sorted(
                self.storage_path.glob(f"{name}_*.json"), reverse=True
            )
            if not matching_files:
                raise FileNotFoundError(f"No templates found for '{name}'")
            file_path = matching_files[0]

        with open(file_path) as f:
            data = json.load(f)
            template = PromptTemplate(**data)
            self.register(template)

    def render(
        self, name: str, variables: Dict[str, Any], version: Optional[str] = None
    ) -> str:
        """Render a template by name.

        Args:
            name: Template name
            variables: Variable values
            version: Specific version (uses latest if not specified)

        Returns:
            Rendered prompt
        """
        if version:
            template = self.get_version(name, version)
        else:
            template = self.get_latest(name)

        if not template:
            raise ValueError(f"Template '{name}' not found")

        return self.engine.render(template, variables)

    def render_ab(
        self, name: str, variables: Dict[str, Any]
    ) -> tuple[str, Optional[str]]:
        """Render template with A/B test variant selection.

        Returns:
            Tuple of (rendered_prompt, variant_name_or_None)
        """
        template = self.get_for_ab_test(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")

        variant_name = None
        if name in self._ab_variants:
            variants = self._ab_variants[name]
            rand = random.random()
            cumulative = 0.0
            for v_name, weight in variants.items():
                cumulative += weight
                if rand < cumulative:
                    variant_name = v_name
                    break

        rendered = self.engine.render(template, variables, variant=variant_name)
        return rendered, variant_name

    @staticmethod
    def _parse_version(version: str) -> tuple:
        """Parse semantic version for sorting."""
        try:
            return tuple(map(int, version.split(".")))
        except (ValueError, AttributeError):
            return (0, 0, 0)
