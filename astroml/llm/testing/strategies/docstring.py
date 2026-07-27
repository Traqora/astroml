"""Docstring-based test generation strategy.

Parses function docstrings to extract test examples,
parameter descriptions, and usage patterns.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DocstringStrategy:
    """Strategy that generates tests by parsing docstrings for examples."""

    def __init__(self, docstring: str):
        self.docstring = docstring

    def extract_examples(self) -> List[Dict[str, Any]]:
        """Extract usage examples from the docstring."""
        examples = []

        code_blocks = re.findall(
            r'```(?:python)?\n(.*?)```',
            self.docstring,
            re.DOTALL,
        )
        for block in code_blocks:
            examples.append({
                "type": "code_block",
                "code": block.strip(),
            })

        example_lines = re.findall(
            r'(?:>>>|\.\.\.)\s*(.*)',
            self.docstring,
        )
        if example_lines:
            examples.append({
                "type": "doctest",
                "code": "\n".join(example_lines),
            })

        return examples

    def extract_params(self) -> List[Dict[str, str]]:
        """Extract parameter descriptions from the docstring."""
        params = []
        patterns = [
            r":param\s+(\w+):\s*(.*)",
            r"Args:\n((?:\s+\w+.*\n)*)",
            r"Parameters\n[-]+\n((?:\s+\w+.*\n)*)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, self.docstring, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    params.append({"name": match[0], "description": match[1]})
                else:
                    for line in match.strip().split("\n"):
                        line = line.strip()
                        m = re.match(r"(\w+):\s*(.*)", line)
                        if m:
                            params.append({"name": m.group(1), "description": m.group(2)})
        return params

    def extract_return_description(self) -> Optional[str]:
        """Extract return value description."""
        patterns = [
            r":return:\s*(.*)",
            r":returns:\s*(.*)",
            r"Returns:\n((?:\s+.*\n)*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, self.docstring)
            if match:
                return match.group(1).strip() if match.lastindex else match.group(0).strip()
        return None

    def extract_raises(self) -> List[str]:
        """Extract exception information from the docstring."""
        raises = []
        pattern = r":raises\s+(\w+):\s*(.*)"
        for match in re.finditer(pattern, self.docstring):
            raises.append(match.group(1))
        return raises

    def has_doctests(self) -> bool:
        """Check if the docstring contains embedded doctests."""
        return bool(re.search(r'>>>\s', self.docstring))

    def get_description(self) -> str:
        """Extract the first paragraph as a brief description."""
        lines = self.docstring.strip().split("\n")
        description = []
        for line in lines:
            line = line.strip()
            if not line:
                break
            if line.startswith(":") or line.startswith("Args") or line.startswith("Returns"):
                break
            description.append(line)
        return " ".join(description)
