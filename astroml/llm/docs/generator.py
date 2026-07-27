"""
Documentation generation orchestrator.

This module provides the main documentation generation functionality,
coordinating code analysis, writing, validation, and updating.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Set

from astroml.llm.docs.code_analyzer import CodeAnalyzer, CodeElement, ElementType
from astroml.llm.docs.writers import (
    BaseWriter,
    MarkdownWriter,
    RstWriter,
    HtmlWriter,
    WriterConfig,
)
from astroml.llm.docs.validator import DocumentationValidator, ValidationResult
from astroml.llm.docs.updater import DocumentationUpdater, UpdateResult


class DocType(Enum):
    """Types of documentation to generate."""

    API = "api"
    CODE = "code"
    ARCHITECTURE = "architecture"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"
    README = "README"


class OutputFormat(Enum):
    """Output formats for documentation."""

    MARKDOWN = "markdown"
    RST = "rst"
    HTML = "html"


@dataclass
class GenerationConfig:
    """
    Configuration for documentation generation.

    Attributes:
        doc_type: Type of documentation to generate
        output_format: Output format
        output_dir: Directory to write documentation
        include_private: Include private members
        include_internal: Include internal members
        include_examples: Include code examples
        include_type_hints: Include type hints
        validate_after_generation: Validate generated docs
        update_existing: Update existing documentation
        preserve_manual_edits: Preserve manual edits when updating
    """

    doc_type: DocType = DocType.CODE
    output_format: OutputFormat = OutputFormat.MARKDOWN
    output_dir: str = "docs"
    include_private: bool = False
    include_internal: bool = False
    include_examples: bool = True
    include_type_hints: bool = True
    validate_after_generation: bool = True
    update_existing: bool = False
    preserve_manual_edits: bool = True


@dataclass
class GenerationResult:
    """
    Result of documentation generation.

    Attributes:
        success: Whether generation was successful
        files_generated: List of generated file paths
        files_updated: List of updated file paths
        validation_result: Validation result if validation was performed
        duration_seconds: Time taken for generation
        error: Error message if generation failed
    """

    success: bool
    files_generated: List[str] = field(default_factory=list)
    files_updated: List[str] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None


class DocumentationGenerator:
    """
    Main documentation generation orchestrator.

    Coordinates:
    - Code analysis
    - Documentation writing
    - Validation
    - Updates
    """

    def __init__(self, config: GenerationConfig = None):
        """
        Initialize the documentation generator.

        Args:
            config: Generation configuration
        """
        self.config = config or GenerationConfig()
        self.analyzer = CodeAnalyzer()
        self.validator = DocumentationValidator()
        self.updater = DocumentationUpdater()

    def generate_from_directory(
        self, source_dir: str, output_dir: str = None
    ) -> GenerationResult:
        """
        Generate documentation from a directory of source files.

        Args:
            source_dir: Directory containing source files
            output_dir: Optional output directory (overrides config)

        Returns:
            GenerationResult with generation status
        """
        start_time = time.time()
        result = GenerationResult(success=True)

        try:
            output_path = Path(output_dir or self.config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Analyze source code
            elements = self.analyzer.analyze_directory(source_dir)

            if not elements:
                result.success = False
                result.error = "No code elements found in source directory"
                return result

            # Select appropriate writer
            writer = self._get_writer()

            # Generate documentation
            output_file = output_path / f"{self.config.doc_type.value}.{self._get_extension()}"
            writer.write(elements, str(output_file))
            result.files_generated.append(str(output_file))

            # Validate if configured
            if self.config.validate_after_generation:
                result.validation_result = self.validator.validate_documentation(
                    str(output_file), elements
                )

            result.duration_seconds = time.time() - start_time

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.duration_seconds = time.time() - start_time

        return result

    def generate_from_file(
        self, source_file: str, output_file: str = None
    ) -> GenerationResult:
        """
        Generate documentation from a single source file.

        Args:
            source_file: Path to the source file
            output_file: Optional output file path

        Returns:
            GenerationResult with generation status
        """
        start_time = time.time()
        result = GenerationResult(success=True)

        try:
            # Analyze source code
            elements = self.analyzer.analyze_file(source_file)

            if not elements:
                result.success = False
                result.error = "No code elements found in source file"
                return result

            # Determine output path
            if output_file is None:
                output_path = (
                    Path(self.config.output_dir)
                    / f"{Path(source_file).stem}.{self._get_extension()}"
                )
            else:
                output_path = Path(output_file)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Select appropriate writer
            writer = self._get_writer()

            # Generate documentation
            writer.write(elements, str(output_path))
            result.files_generated.append(str(output_path))

            # Validate if configured
            if self.config.validate_after_generation:
                result.validation_result = self.validator.validate_documentation(
                    str(output_path), elements
                )

            result.duration_seconds = time.time() - start_time

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.duration_seconds = time.time() - start_time

        return result

    def generate_api_docs(
        self, api_file: str, output_file: str = None
    ) -> GenerationResult:
        """
        Generate API documentation from FastAPI routes.

        Args:
            api_file: Path to the file containing FastAPI routes
            output_file: Optional output file path

        Returns:
            GenerationResult with generation status
        """
        start_time = time.time()
        result = GenerationResult(success=True)

        try:
            # Extract API endpoints
            endpoints = self.analyzer.extract_api_endpoints(api_file)

            if not endpoints:
                result.success = False
                result.error = "No API endpoints found in file"
                return result

            # Generate API documentation
            output_path = Path(output_file or self.config.output_dir) / "api.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            content = self._generate_api_content(endpoints, api_file)
            output_path.write_text(content, encoding="utf-8")

            result.files_generated.append(str(output_path))
            result.duration_seconds = time.time() - start_time

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.duration_seconds = time.time() - start_time

        return result

    def generate_changelog(
        self, repo_path: str, output_file: str = None
    ) -> GenerationResult:
        """
        Generate changelog from git history.

        Args:
            repo_path: Path to the git repository
            output_file: Optional output file path

        Returns:
            GenerationResult with generation status
        """
        start_time = time.time()
        result = GenerationResult(success=True)

        try:
            # This would integrate with git to extract commit history
            # For now, return a placeholder result
            output_path = Path(output_file or self.config.output_dir) / "CHANGELOG.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            content = "# Changelog\n\nThis changelog is automatically generated from git history.\n"
            output_path.write_text(content, encoding="utf-8")

            result.files_generated.append(str(output_path))
            result.duration_seconds = time.time() - start_time

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.duration_seconds = time.time() - start_time

        return result

    def generate_readme_sections(
        self, project_root: str, output_file: str = None
    ) -> GenerationResult:
        """
        Generate README sections from project structure.

        Args:
            project_root: Path to the project root
            output_file: Optional output file path

        Returns:
            GenerationResult with generation status
        """
        start_time = time.time()
        result = GenerationResult(success=True)

        try:
            project_path = Path(project_root)

            # Analyze project structure
            sections = {
                "Installation": self._generate_installation_section(project_path),
                "Usage": self._generate_usage_section(project_path),
                "Project Structure": self._generate_structure_section(project_path),
            }

            # Generate README
            output_path = Path(output_file or project_root) / "README.md"
            existing_content = ""

            if output_path.exists():
                existing_content = output_path.read_text(encoding="utf-8")

            content = self._merge_readme_sections(existing_content, sections)
            output_path.write_text(content, encoding="utf-8")

            result.files_generated.append(str(output_path))
            result.duration_seconds = time.time() - start_time

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.duration_seconds = time.time() - start_time

        return result

    def update_documentation(
        self, doc_path: str, source_paths: List[str]
    ) -> GenerationResult:
        """
        Update existing documentation.

        Args:
            doc_path: Path to the documentation file
            source_paths: List of source file paths

        Returns:
            GenerationResult with update status
        """
        start_time = time.time()
        result = GenerationResult(success=True)

        try:
            update_result = self.updater.update_documentation(
                doc_path,
                source_paths,
                preserve_manual_edits=self.config.preserve_manual_edits,
            )

            result.success = update_result.success
            result.files_updated = update_result.updated_files
            result.files_generated = update_result.updated_files
            result.error = "\n".join(update_result.errors) if update_result.errors else None

            result.duration_seconds = time.time() - start_time

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.duration_seconds = time.time() - start_time

        return result

    def _get_writer(self) -> BaseWriter:
        """Get the appropriate writer based on configuration."""
        config = WriterConfig(
            include_private=self.config.include_private,
            include_internal=self.config.include_internal,
            include_examples=self.config.include_examples,
            include_type_hints=self.config.include_type_hints,
        )

        if self.config.output_format == OutputFormat.MARKDOWN:
            return MarkdownWriter(config)
        elif self.config.output_format == OutputFormat.RST:
            return RstWriter(config)
        elif self.config.output_format == OutputFormat.HTML:
            return HtmlWriter(config)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")

    def _get_extension(self) -> str:
        """Get file extension for the output format."""
        extensions = {
            OutputFormat.MARKDOWN: "md",
            OutputFormat.RST: "rst",
            OutputFormat.HTML: "html",
        }
        return extensions.get(self.config.output_format, "md")

    def _generate_api_content(self, endpoints: List[Dict], api_file: str) -> str:
        """Generate API documentation content."""
        lines = ["# API Documentation\n"]
        lines.append(f"Generated from `{api_file}`\n")

        # Group by path
        paths = {}
        for endpoint in endpoints:
            path = endpoint["path"]
            if path not in paths:
                paths[path] = []
            paths[path].append(endpoint)

        # Generate documentation for each path
        for path in sorted(paths.keys()):
            lines.append(f"## {path}\n")

            for endpoint in paths[path]:
                lines.append(f"### {endpoint['method']} {endpoint['name']}\n")

                if endpoint.get("docstring"):
                    lines.append(endpoint["docstring"])
                    lines.append("")

                if endpoint.get("parameters"):
                    lines.append("**Parameters:**")
                    for param in endpoint["parameters"]:
                        param_line = f"- `{param['name']}`"
                        if param.get("type"):
                            param_line += f": `{param['type']}`"
                        lines.append(param_line)
                    lines.append("")

                if endpoint.get("returns"):
                    lines.append(f"**Returns:** `{endpoint['returns']}`")
                    lines.append("")

        return "\n".join(lines)

    def _generate_installation_section(self, project_path: Path) -> str:
        """Generate installation section."""
        lines = ["## Installation\n"]

        # Check for requirements.txt
        if (project_path / "requirements.txt").exists():
            lines.append("```bash")
            lines.append("pip install -r requirements.txt")
            lines.append("```")
        elif (project_path / "pyproject.toml").exists():
            lines.append("```bash")
            lines.append("pip install -e .")
            lines.append("```")
        else:
            lines.append("```bash")
            lines.append("pip install astroml")
            lines.append("```")

        lines.append("")
        return "\n".join(lines)

    def _generate_usage_section(self, project_path: Path) -> str:
        """Generate usage section."""
        lines = ["## Usage\n"]

        # Look for examples directory
        examples_dir = project_path / "examples"
        if examples_dir.exists():
            lines.append("See the [examples](examples/) directory for usage examples.\n")
        else:
            lines.append("```python")
            lines.append("import astroml")
            lines.append("")
            lines.append("# Your code here")
            lines.append("```")

        lines.append("")
        return "\n".join(lines)

    def _generate_structure_section(self, project_path: Path) -> str:
        """Generate project structure section."""
        lines = ["## Project Structure\n"]
        lines.append("```")

        # Generate tree structure
        def generate_tree(path: Path, prefix: str = ""):
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{item.name}")
                if item.is_dir() and not item.name.startswith("."):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    generate_tree(item, new_prefix)

        generate_tree(project_path)
        lines.append("```")
        lines.append("")
        return "\n".join(lines)

    def _merge_readme_sections(
        self, existing_content: str, new_sections: Dict[str, str]
    ) -> str:
        """Merge new sections into existing README."""
        lines = existing_content.split("\n")
        merged = []
        skip_until = None

        for line in lines:
            if skip_until:
                if line.startswith(skip_until):
                    skip_until = None
                continue

            section_found = False
            for section_name in new_sections.keys():
                if line.startswith(f"## {section_name}"):
                    merged.append(new_sections[section_name])
                    skip_until = f"## "
                    section_found = True
                    break

            if not section_found:
                merged.append(line)

        # Add any sections that weren't found
        for section_name, section_content in new_sections.items():
            if f"## {section_name}" not in existing_content:
                merged.append(section_content)

        return "\n".join(merged)
