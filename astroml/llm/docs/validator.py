"""
Documentation quality validator.

This module provides validation capabilities for generated documentation,
including link checking, example validation, completeness scoring, and
readability metrics.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import ast


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """
    Represents a validation issue found in documentation.

    Attributes:
        severity: Severity level of the issue
        message: Description of the issue
        location: Location in the documentation (file, line)
        suggestion: Suggested fix
    """

    severity: ValidationSeverity
    message: str
    location: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Result of documentation validation.

    Attributes:
        is_valid: Overall validity status
        issues: List of validation issues
        completeness_score: Completeness score (0-100)
        readability_score: Readability score (0-100)
        broken_links: List of broken links
        invalid_examples: List of invalid code examples
    """

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    completeness_score: float = 0.0
    readability_score: float = 0.0
    broken_links: List[str] = field(default_factory=list)
    invalid_examples: List[str] = field(default_factory=list)

    def add_issue(
        self,
        severity: ValidationSeverity,
        message: str,
        location: str,
        suggestion: Optional[str] = None,
    ) -> None:
        """Add a validation issue."""
        self.issues.append(
            ValidationIssue(severity=severity, message=message, location=location, suggestion=suggestion)
        )

    def get_summary(self) -> str:
        """Get a summary of validation results."""
        error_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO)

        summary = f"Validation Results:\n"
        summary += f"- Valid: {self.is_valid}\n"
        summary += f"- Completeness Score: {self.completeness_score:.1f}/100\n"
        summary += f"- Readability Score: {self.readability_score:.1f}/100\n"
        summary += f"- Errors: {error_count}\n"
        summary += f"- Warnings: {warning_count}\n"
        summary += f"- Info: {info_count}\n"
        summary += f"- Broken Links: {len(self.broken_links)}\n"
        summary += f"- Invalid Examples: {len(self.invalid_examples)}\n"

        return summary


class DocumentationValidator:
    """
    Validator for documentation quality.

    Performs various quality checks:
    - Broken link detection
    - Code example validation
    - Completeness scoring
    - Readability metrics
    - Consistency checks
    """

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the validator.

        Args:
            base_url: Base URL for link validation
        """
        self.base_url = base_url

    def validate_documentation(
        self, doc_path: str, code_elements: List = None
    ) -> ValidationResult:
        """
        Validate documentation file.

        Args:
            doc_path: Path to the documentation file
            code_elements: Optional list of code elements for consistency checks

        Returns:
            ValidationResult with validation findings
        """
        result = ValidationResult(is_valid=True)

        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for broken links
        broken_links = self._check_links(content, doc_path)
        result.broken_links.extend(broken_links)
        for link in broken_links:
            result.add_issue(
                severity=ValidationSeverity.WARNING,
                message=f"Broken or invalid link: {link}",
                location=doc_path,
                suggestion="Verify the link is correct and accessible",
            )

        # Validate code examples
        invalid_examples = self._validate_examples(content, doc_path)
        result.invalid_examples.extend(invalid_examples)
        for example in invalid_examples:
            result.add_issue(
                severity=ValidationSeverity.ERROR,
                message=f"Invalid code example",
                location=doc_path,
                suggestion="Fix syntax errors in the code example",
            )

        # Calculate completeness score
        result.completeness_score = self._calculate_completeness(content, code_elements)

        # Calculate readability score
        result.readability_score = self._calculate_readability(content)

        # Check for consistency with code
        if code_elements:
            consistency_issues = self._check_consistency(content, code_elements, doc_path)
            for issue in consistency_issues:
                result.add_issue(**issue)

        # Determine overall validity
        result.is_valid = (
            len(result.broken_links) == 0
            and len(result.invalid_examples) == 0
            and all(i.severity != ValidationSeverity.ERROR for i in result.issues)
        )

        return result

    def _check_links(self, content: str, doc_path: str) -> List[str]:
        """
        Check for broken or invalid links in documentation.

        Args:
            content: Documentation content
            doc_path: Path to the documentation file

        Returns:
            List of broken links
        """
        broken_links = []

        # Extract markdown links
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        for text, url in markdown_links:
            # Check for local file references
            if url.startswith("./") or url.startswith("../"):
                local_path = Path(doc_path).parent / url
                if not local_path.exists():
                    broken_links.append(url)

            # Check for http/https links (basic validation)
            elif url.startswith(("http://", "https://")):
                if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url):
                    broken_links.append(url)

        return broken_links

    def _validate_examples(self, content: str, doc_path: str) -> List[str]:
        """
        Validate code examples in documentation.

        Args:
            content: Documentation content
            doc_path: Path to the documentation file

        Returns:
            List of invalid examples
        """
        invalid_examples = []

        # Extract code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)

        for code in code_blocks:
            try:
                ast.parse(code)
            except SyntaxError:
                invalid_examples.append(code[:50] + "...")

        return invalid_examples

    def _calculate_completeness(self, content: str, code_elements: List = None) -> float:
        """
        Calculate completeness score for documentation.

        Args:
            content: Documentation content
            code_elements: Optional code elements to compare against

        Returns:
            Completeness score (0-100)
        """
        score = 0.0
        total_checks = 0

        # Check for title
        if re.search(r'^#\s+.+$', content, re.MULTILINE):
            score += 10
        total_checks += 1

        # Check for description
        if len(content) > 100:
            score += 10
        total_checks += 1

        # Check for code examples
        if re.search(r'```', content):
            score += 20
        total_checks += 1

        # Check for parameter documentation
        if re.search(r'parameter|arg|argument', content, re.IGNORECASE):
            score += 15
        total_checks += 1

        # Check for return value documentation
        if re.search(r'return|returns', content, re.IGNORECASE):
            score += 15
        total_checks += 1

        # Check for exception documentation
        if re.search(r'raise|raises|exception', content, re.IGNORECASE):
            score += 10
        total_checks += 1

        # Check for type information
        if re.search(r'type|:py:class:|:py:data:', content, re.IGNORECASE):
            score += 10
        total_checks += 1

        # Check for links
        if re.search(r'\[.*\]\(.*\)', content):
            score += 10
        total_checks += 1

        return (score / total_checks * 100) if total_checks > 0 else 0.0

    def _calculate_readability(self, content: str) -> float:
        """
        Calculate readability score for documentation.

        Args:
            content: Documentation content

        Returns:
            Readability score (0-100)
        """
        score = 0.0
        total_checks = 0

        # Check sentence length (average)
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_length < 25:  # Good sentence length
                score += 20
            elif avg_length < 35:
                score += 10
            total_checks += 1

        # Check paragraph length
        paragraphs = content.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        if paragraphs:
            avg_para_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            if avg_para_length < 100:  # Good paragraph length
                score += 20
            elif avg_para_length < 150:
                score += 10
            total_checks += 1

        # Check for excessive jargon
        jargon_words = ['implement', 'utilize', 'leverage', 'facilitate', 'optimize']
        jargon_count = sum(1 for word in jargon_words if word in content.lower())
        if jargon_count < 3:
            score += 20
        elif jargon_count < 5:
            score += 10
        total_checks += 1

        # Check for active voice (simple heuristic)
        passive_indicators = ['is used', 'are used', 'was used', 'were used']
        passive_count = sum(1 for indicator in passive_indicators if indicator in content.lower())
        if passive_count < 2:
            score += 20
        elif passive_count < 4:
            score += 10
        total_checks += 1

        # Check for formatting consistency
        if re.search(r'#{1,6}\s', content):  # Has headers
            score += 20
        total_checks += 1

        return (score / total_checks * 100) if total_checks > 0 else 0.0

    def _check_consistency(
        self, content: str, code_elements: List, doc_path: str
    ) -> List[Dict]:
        """
        Check consistency between documentation and code.

        Args:
            content: Documentation content
            code_elements: List of code elements
            doc_path: Path to documentation

        Returns:
            List of consistency issues
        """
        issues = []

        if not code_elements:
            return issues

        # Check if all public elements are documented
        documented_names = set(re.findall(r'#{1,6}\s+([^\n]+)', content))

        for element in code_elements:
            if not element.name.startswith("_"):  # Public element
                if element.name not in documented_names:
                    issues.append({
                        "severity": ValidationSeverity.WARNING,
                        "message": f"Public element '{element.name}' not documented",
                        "location": doc_path,
                        "suggestion": f"Add documentation for {element.name}",
                    })

        # Check for outdated signatures
        for element in code_elements:
            if element.signature and element.element_type.value in ["function", "method"]:
                # Check if signature is mentioned in docs
                if element.name in content:
                    # Simple check: does the doc mention the function name
                    pass  # More sophisticated checking could be added

        return issues

    def validate_example_runnable(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a code example is runnable.

        Args:
            code: Code example to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check syntax
            ast.parse(code)

            # Check for common issues
            if "TODO" in code or "FIXME" in code:
                return False, "Example contains TODO/FIXME comments"

            if "import" not in code:
                return False, "Example may be missing imports"

            return True, None

        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

    def check_outdated_docs(
        self, doc_path: str, code_path: str
    ) -> List[ValidationIssue]:
        """
        Check if documentation is outdated compared to code.

        Args:
            doc_path: Path to documentation
            code_path: Path to corresponding code

        Returns:
            List of outdated documentation issues
        """
        issues = []

        try:
            doc_mtime = Path(doc_path).stat().st_mtime
            code_mtime = Path(code_path).stat().st_mtime

            if code_mtime > doc_mtime:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Documentation may be outdated (code modified since doc generation)",
                        location=doc_path,
                        suggestion="Regenerate documentation from latest code",
                    )
                )

        except FileNotFoundError:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"File not found: {code_path or doc_path}",
                    location=doc_path,
                    suggestion="Ensure both code and documentation files exist",
                )
            )

        return issues
