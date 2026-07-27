"""
Documentation writers for different output formats.

This module provides writers for generating documentation in various formats:
- Markdown (MD)
- reStructuredText (RST)
- HTML
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from astroml.llm.docs.code_analyzer import CodeElement, ElementType


@dataclass
class WriterConfig:
    """Configuration for documentation writers."""

    include_private: bool = False
    include_internal: bool = False
    include_examples: bool = True
    include_type_hints: bool = True
    include_source_links: bool = True
    base_url: Optional[str] = None
    toc_depth: int = 3


class BaseWriter(ABC):
    """Base class for documentation writers."""

    def __init__(self, config: WriterConfig = None):
        """Initialize the writer with configuration."""
        self.config = config or WriterConfig()

    @abstractmethod
    def write(self, elements: List[CodeElement], output_path: str) -> None:
        """
        Write documentation for the given elements.

        Args:
            elements: List of CodeElement objects
            output_path: Path to write the documentation
        """
        pass

    @abstractmethod
    def write_element(self, element: CodeElement) -> str:
        """
        Write documentation for a single element.

        Args:
            element: CodeElement to document

        Returns:
            Formatted documentation string
        """
        pass


class MarkdownWriter(BaseWriter):
    """Writer for Markdown documentation."""

    def write(self, elements: List[CodeElement], output_path: str) -> None:
        """Write Markdown documentation."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        content = self._generate_content(elements)
        output.write_text(content, encoding="utf-8")

    def _generate_content(self, elements: List[CodeElement]) -> str:
        """Generate complete Markdown content."""
        lines = []

        # Group by module
        modules = {}
        for element in elements:
            if element.element_type == ElementType.MODULE:
                modules[element.name] = element

        # Generate table of contents
        lines.append("# Table of Contents\n")
        for module_name in sorted(modules.keys()):
            lines.append(f"- [{module_name}](#{module_name.lower().replace(' ', '-')})")
        lines.append("")

        # Generate documentation for each module
        for module_name in sorted(modules.keys()):
            module = modules[module_name]
            lines.append(self.write_element(module))
            lines.append("")

            # Add children
            for child in module.children:
                if self._should_include(child):
                    lines.append(self.write_element(child))
                    lines.append("")

                    # Add grandchildren (methods, etc.)
                    for grandchild in child.children:
                        if self._should_include(grandchild):
                            lines.append(self.write_element(grandchild))
                            lines.append("")

        return "\n".join(lines)

    def write_element(self, element: CodeElement) -> str:
        """Write documentation for a single element."""
        lines = []

        # Header based on element type
        level = self._get_header_level(element.element_type)
        header = f"{'#' * level} {element.name}"
        lines.append(header)
        lines.append("")

        # Signature for functions/methods
        if element.signature and element.element_type in [
            ElementType.FUNCTION,
            ElementType.METHOD,
        ]:
            lines.append("```python")
            lines.append(element.signature)
            lines.append("```")
            lines.append("")

        # Docstring
        if element.docstring:
            lines.append(element.docstring)
            lines.append("")

        # Type hints
        if self.config.include_type_hints and element.type_hints:
            lines.append("**Type Hints:**")
            for name, type_hint in element.type_hints.items():
                lines.append(f"- `{name}`: `{type_hint}`")
            lines.append("")

        # Parameters
        if element.parameters:
            lines.append("**Parameters:**")
            for param in element.parameters:
                param_line = f"- `{param['name']}`"
                if param.get("type"):
                    param_line += f" (`{param['type']}`)"
                if param.get("default"):
                    param_line += f" = {param['default']}"
                lines.append(param_line)
            lines.append("")

        # Returns
        if element.returns:
            lines.append(f"**Returns:** `{element.returns}`")
            lines.append("")

        # Raises
        if element.raises:
            lines.append("**Raises:**")
            for exc in element.raises:
                lines.append(f"- `{exc}`")
            lines.append("")

        # Examples
        if self.config.include_examples and element.examples:
            lines.append("**Examples:**")
            for example in element.examples:
                lines.append("```python")
                lines.append(example)
                lines.append("```")
            lines.append("")

        # Source link
        if self.config.include_source_links and element.file_path:
            lines.append(f"[Source]({element.file_path}#L{element.line_number})")
            lines.append("")

        return "\n".join(lines)

    def _get_header_level(self, element_type: ElementType) -> int:
        """Get header level for element type."""
        levels = {
            ElementType.MODULE: 1,
            ElementType.CLASS: 2,
            ElementType.FUNCTION: 3,
            ElementType.METHOD: 4,
        }
        return levels.get(element_type, 3)

    def _should_include(self, element: CodeElement) -> bool:
        """Check if element should be included in documentation."""
        if not self.config.include_private and element.name.startswith("_"):
            return False
        if not self.config.include_internal and element.name.startswith("__"):
            return False
        return True


class RstWriter(BaseWriter):
    """Writer for reStructuredText documentation."""

    def write(self, elements: List[CodeElement], output_path: str) -> None:
        """Write RST documentation."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        content = self._generate_content(elements)
        output.write_text(content, encoding="utf-8")

    def _generate_content(self, elements: List[CodeElement]) -> str:
        """Generate complete RST content."""
        lines = []

        # Group by module
        modules = {}
        for element in elements:
            if element.element_type == ElementType.MODULE:
                modules[element.name] = element

        # Generate documentation for each module
        for module_name in sorted(modules.keys()):
            module = modules[module_name]
            lines.append(self.write_element(module))
            lines.append("")

            # Add children
            for child in module.children:
                if self._should_include(child):
                    lines.append(self.write_element(child))
                    lines.append("")

                    # Add grandchildren
                    for grandchild in child.children:
                        if self._should_include(grandchild):
                            lines.append(self.write_element(grandchild))
                            lines.append("")

        return "\n".join(lines)

    def write_element(self, element: CodeElement) -> str:
        """Write documentation for a single element in RST format."""
        lines = []

        # Header based on element type
        level = self._get_header_level(element.element_type)
        char = self._get_header_char(level)
        lines.append(element.name)
        lines.append(char * len(element.name))
        lines.append("")

        # Signature
        if element.signature and element.element_type in [
            ElementType.FUNCTION,
            ElementType.METHOD,
        ]:
            lines.append(".. code-block:: python")
            lines.append("")
            lines.append(f"    {element.signature}")
            lines.append("")

        # Docstring
        if element.docstring:
            lines.append(element.docstring)
            lines.append("")

        # Type hints
        if self.config.include_type_hints and element.type_hints:
            lines.append("**Type Hints:**")
            for name, type_hint in element.type_hints.items():
                lines.append(f"- :py:data:`{name}`: :py:class:`{type_hint}`")
            lines.append("")

        # Parameters
        if element.parameters:
            lines.append("**Parameters:**")
            for param in element.parameters:
                param_line = f"- :py:data:`{param['name']}`"
                if param.get("type"):
                    param_line += f" (:py:class:`{param['type']}`)"
                if param.get("default"):
                    param_line += f" = {param['default']}"
                lines.append(param_line)
            lines.append("")

        # Returns
        if element.returns:
            lines.append(f"**Returns:** :py:class:`{element.returns}`")
            lines.append("")

        # Examples
        if self.config.include_examples and element.examples:
            lines.append("**Examples:**")
            for example in element.examples:
                lines.append(".. code-block:: python")
                lines.append("")
                for line in example.split("\n"):
                    lines.append(f"    {line}")
                lines.append("")

        return "\n".join(lines)

    def _get_header_level(self, element_type: ElementType) -> int:
        """Get header level for element type."""
        levels = {
            ElementType.MODULE: 1,
            ElementType.CLASS: 2,
            ElementType.FUNCTION: 3,
            ElementType.METHOD: 4,
        }
        return levels.get(element_type, 3)

    def _get_header_char(self, level: int) -> str:
        """Get RST header character for level."""
        chars = ["=", "-", "~", "`"]
        return chars[min(level - 1, len(chars) - 1)]

    def _should_include(self, element: CodeElement) -> bool:
        """Check if element should be included in documentation."""
        if not self.config.include_private and element.name.startswith("_"):
            return False
        if not self.config.include_internal and element.name.startswith("__"):
            return False
        return True


class HtmlWriter(BaseWriter):
    """Writer for HTML documentation."""

    def write(self, elements: List[CodeElement], output_path: str) -> None:
        """Write HTML documentation."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        content = self._generate_content(elements)
        output.write_text(content, encoding="utf-8")

    def _generate_content(self, elements: List[CodeElement]) -> str:
        """Generate complete HTML content."""
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>API Documentation</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
            "h1, h2, h3, h4 { color: #333; }",
            "pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }",
            "code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }",
            ".parameter { margin: 5px 0; }",
            ".example { margin: 20px 0; }",
            "</style>",
            "</head>",
            "<body>",
        ]

        # Group by module
        modules = {}
        for element in elements:
            if element.element_type == ElementType.MODULE:
                modules[element.name] = element

        # Generate documentation for each module
        for module_name in sorted(modules.keys()):
            module = modules[module_name]
            lines.append(self.write_element(module))

            # Add children
            for child in module.children:
                if self._should_include(child):
                    lines.append(self.write_element(child))

                    # Add grandchildren
                    for grandchild in child.children:
                        if self._should_include(grandchild):
                            lines.append(self.write_element(grandchild))

        lines.extend(["</body>", "</html>"])
        return "\n".join(lines)

    def write_element(self, element: CodeElement) -> str:
        """Write documentation for a single element in HTML format."""
        lines = []

        # Header based on element type
        level = self._get_header_level(element.element_type)
        lines.append(f"<h{level}>{element.name}</h{level}>")

        # Signature
        if element.signature and element.element_type in [
            ElementType.FUNCTION,
            ElementType.METHOD,
        ]:
            lines.append("<pre><code>")
            lines.append(self._escape_html(element.signature))
            lines.append("</code></pre>")

        # Docstring
        if element.docstring:
            lines.append(f"<p>{self._escape_html(element.docstring)}</p>")

        # Type hints
        if self.config.include_type_hints and element.type_hints:
            lines.append("<h4>Type Hints:</h4>")
            lines.append("<ul>")
            for name, type_hint in element.type_hints.items():
                lines.append(f"<li><code>{name}</code>: <code>{type_hint}</code></li>")
            lines.append("</ul>")

        # Parameters
        if element.parameters:
            lines.append("<h4>Parameters:</h4>")
            lines.append("<ul>")
            for param in element.parameters:
                param_line = f"<li class='parameter'><code>{param['name']}</code>"
                if param.get("type"):
                    param_line += f" (<code>{param['type']}</code>)"
                if param.get("default"):
                    param_line += f" = {param['default']}"
                param_line += "</li>"
                lines.append(param_line)
            lines.append("</ul>")

        # Returns
        if element.returns:
            lines.append(f"<h4>Returns:</h4><p><code>{element.returns}</code></p>")

        # Examples
        if self.config.include_examples and element.examples:
            lines.append("<h4>Examples:</h4>")
            for example in element.examples:
                lines.append("<div class='example'>")
                lines.append("<pre><code>")
                lines.append(self._escape_html(example))
                lines.append("</code></pre>")
                lines.append("</div>")

        return "\n".join(lines)

    def _get_header_level(self, element_type: ElementType) -> int:
        """Get header level for element type."""
        levels = {
            ElementType.MODULE: 1,
            ElementType.CLASS: 2,
            ElementType.FUNCTION: 3,
            ElementType.METHOD: 4,
        }
        return levels.get(element_type, 3)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _should_include(self, element: CodeElement) -> bool:
        """Check if element should be included in documentation."""
        if not self.config.include_private and element.name.startswith("_"):
            return False
        if not self.config.include_internal and element.name.startswith("__"):
            return False
        return True
