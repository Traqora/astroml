#!/usr/bin/env python
"""
CLI tool for documentation generation.

This script provides a command-line interface for generating,
updating, and validating documentation.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from astroml.llm.docs.generator import (
    DocumentationGenerator,
    GenerationConfig,
    DocType,
    OutputFormat,
)
from astroml.llm.docs.validator import DocumentationValidator
from astroml.llm.docs.updater import DocumentationUpdater


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate and manage documentation for astroml"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate documentation")
    generate_parser.add_argument(
        "source", help="Source file or directory to document"
    )
    generate_parser.add_argument(
        "-o", "--output", help="Output directory or file"
    )
    generate_parser.add_argument(
        "-f", "--format",
        choices=["markdown", "rst", "html"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    generate_parser.add_argument(
        "-t", "--type",
        choices=["api", "code", "architecture", "tutorial", "changelog", "readme"],
        default="code",
        help="Type of documentation (default: code)"
    )
    generate_parser.add_argument(
        "--include-private", action="store_true",
        help="Include private members"
    )
    generate_parser.add_argument(
        "--include-internal", action="store_true",
        help="Include internal members"
    )
    generate_parser.add_argument(
        "--no-examples", action="store_true",
        help="Exclude code examples"
    )
    generate_parser.add_argument(
        "--no-type-hints", action="store_true",
        help="Exclude type hints"
    )
    generate_parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip validation"
    )

    # Update command
    update_parser = subparsers.add_parser("update", help="Update existing documentation")
    update_parser.add_argument("doc_path", help="Path to documentation file")
    update_parser.add_argument(
        "source_paths", nargs="+",
        help="Source file paths"
    )
    update_parser.add_argument(
        "--no-preserve-edits", action="store_true",
        help="Do not preserve manual edits"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate documentation")
    validate_parser.add_argument("doc_path", help="Path to documentation file")
    validate_parser.add_argument(
        "--code-dir", help="Directory containing source code for consistency checks"
    )

    # Check outdated command
    check_parser = subparsers.add_parser("check-outdated", help="Check for outdated documentation")
    check_parser.add_argument(
        "doc_dir", help="Directory containing documentation"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        handle_generate(args)
    elif args.command == "update":
        handle_update(args)
    elif args.command == "validate":
        handle_validate(args)
    elif args.command == "check-outdated":
        handle_check_outdated(args)


def handle_generate(args):
    """Handle the generate command."""
    config = GenerationConfig(
        doc_type=DocType(args.type),
        output_format=OutputFormat(args.format),
        output_dir=args.output or "docs",
        include_private=args.include_private,
        include_internal=args.include_internal,
        include_examples=not args.no_examples,
        include_type_hints=not args.no_type_hints,
        validate_after_generation=not args.no_validate,
    )

    generator = DocumentationGenerator(config)

    source_path = Path(args.source)
    if source_path.is_file():
        result = generator.generate_from_file(str(source_path), args.output)
    elif source_path.is_dir():
        result = generator.generate_from_directory(str(source_path), args.output)
    else:
        print(f"Error: Source path does not exist: {args.source}")
        sys.exit(1)

    if result.success:
        print(f"✓ Documentation generated successfully")
        print(f"  Files generated: {len(result.files_generated)}")
        for file in result.files_generated:
            print(f"    - {file}")
        print(f"  Duration: {result.duration_seconds:.2f}s")

        if result.validation_result:
            print(f"\nValidation Results:")
            print(f"  Valid: {result.validation_result.is_valid}")
            print(f"  Completeness Score: {result.validation_result.completeness_score:.1f}/100")
            print(f"  Readability Score: {result.validation_result.readability_score:.1f}/100")
            print(f"  Issues: {len(result.validation_result.issues)}")

            if result.validation_result.issues:
                print("\n  Issues found:")
                for issue in result.validation_result.issues:
                    print(f"    - [{issue.severity.value.upper()}] {issue.message}")
                    if issue.suggestion:
                        print(f"      Suggestion: {issue.suggestion}")
    else:
        print(f"✗ Documentation generation failed")
        print(f"  Error: {result.error}")
        sys.exit(1)


def handle_update(args):
    """Handle the update command."""
    updater = DocumentationUpdater()

    result = updater.update_documentation(
        args.doc_path,
        args.source_paths,
        preserve_manual_edits=not args.no_preserve_edits,
    )

    if result.success:
        print(f"✓ Documentation updated successfully")
        print(f"  Files updated: {len(result.updated_files)}")
        for file in result.updated_files:
            print(f"    - {file}")
        print(f"  Files skipped: {len(result.skipped_files)}")
        print(f"  Changes: {result.changes_made}")
    else:
        print(f"✗ Documentation update failed")
        for error in result.errors:
            print(f"  Error: {error}")
        sys.exit(1)


def handle_validate(args):
    """Handle the validate command."""
    validator = DocumentationValidator()

    code_elements = None
    if args.code_dir:
        from astroml.llm.docs.code_analyzer import CodeAnalyzer
        analyzer = CodeAnalyzer()
        code_elements = analyzer.analyze_directory(args.code_dir)

    result = validator.validate_documentation(args.doc_path, code_elements)

    print(result.get_summary())

    if result.issues:
        print("\nDetailed Issues:")
        for issue in result.issues:
            print(f"  [{issue.severity.value.upper()}] {issue.message}")
            print(f"    Location: {issue.location}")
            if issue.suggestion:
                print(f"    Suggestion: {issue.suggestion}")

    sys.exit(0 if result.is_valid else 1)


def handle_check_outdated(args):
    """Handle the check-outdated command."""
    updater = DocumentationUpdater()

    outdated = updater.detect_outdated_docs(args.doc_dir)

    if outdated:
        print(f"Found {len(outdated)} outdated documentation file(s):")
        for doc in outdated:
            print(f"  - {doc}")
        sys.exit(1)
    else:
        print("✓ All documentation is up to date")
        sys.exit(0)


if __name__ == "__main__":
    main()
