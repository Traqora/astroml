"""
LLM-powered documentation generation system for astroml.

This module provides automated documentation generation capabilities including:
- API documentation from FastAPI endpoints
- Code documentation from docstrings and type hints
- Architecture documentation from code structure
- Tutorial generation from examples
- Changelog generation from git history
- README section generation
"""

from astroml.llm.docs.generator import DocumentationGenerator
from astroml.llm.docs.code_analyzer import CodeAnalyzer, CodeElement
from astroml.llm.docs.writers import MarkdownWriter, RstWriter, HtmlWriter
from astroml.llm.docs.validator import DocumentationValidator, ValidationResult
from astroml.llm.docs.updater import DocumentationUpdater

__all__ = [
    "DocumentationGenerator",
    "CodeAnalyzer",
    "CodeElement",
    "MarkdownWriter",
    "RstWriter",
    "HtmlWriter",
    "DocumentationValidator",
    "ValidationResult",
    "DocumentationUpdater",
]

__version__ = "0.1.0"
