#!/usr/bin/env python3
"""CLI entry point for LLM-powered test generation.

Usage:
    python -m tools.test_generator.cli --source path/to/module.py
    python -m tools.test_generator.cli --spec path/to/api_spec.yaml
    python -m tools.test_generator.cli --function "def add(a, b): return a + b"
"""

from __future__ import annotations

import argparse
import sys
import textwrap


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-powered test generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s --source module.py
              %(prog)s --source module.py --framework unittest
              %(prog)s --spec api.yaml --output tests/test_api.py
              %(prog)s --function "def add(a, b): return a + b"
        """),
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source", "-s",
        help="Path to source file for test generation",
    )
    source_group.add_argument(
        "--spec", "-p",
        help="Path to API spec file (OpenAPI, etc.)",
    )
    source_group.add_argument(
        "--function", "-f",
        help="Inline function definition to generate tests for",
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--framework",
        choices=["pytest", "unittest"],
        default="pytest",
        help="Test framework to use (default: pytest)",
    )
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "property", "edge_case", "regression"],
        default="unit",
        help="Type of tests to generate (default: unit)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="LLM model to use (default: gpt-4)",
    )
    parser.add_argument(
        "--max-tests",
        type=int,
        default=5,
        help="Maximum tests per function (default: 5)",
    )
    parser.add_argument(
        "--include-edge-cases",
        action="store_true",
        default=True,
        help="Include edge case tests",
    )
    parser.add_argument(
        "--no-edge-cases",
        action="store_false",
        dest="include_edge_cases",
        help="Skip edge case tests",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        print(f"Test Generation Configuration:")
        print(f"  Framework: {args.framework}")
        print(f"  Test type: {args.test_type}")
        print(f"  Model: {args.model}")
        print()

    try:
        source_code = ""
        source_name = ""

        if args.source:
            with open(args.source) as f:
                source_code = f.read()
            source_name = args.source
        elif args.function:
            source_code = args.function
            source_name = "inline_function"
        elif args.spec:
            with open(args.spec) as f:
                source_code = f.read()
            source_name = args.spec

        from astroml.llm.testing.generator import (
            TestGenerator,
            TestGenerationConfig,
            TestType,
        )

        config = TestGenerationConfig(
            framework=args.framework,
            test_type=TestType(args.test_type),
            max_tests_per_function=args.max_tests,
            include_edge_cases=args.include_edge_cases,
            model=args.model,
        )
        generator = TestGenerator(config)

        print(f"Generating {args.test_type} tests...")
        tests = generator.generate_from_module(source_code, source_name)

        output = generator.format_test_file(tests, source_name)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Tests written to {args.output}")
        else:
            print(output)

        if args.verbose and tests:
            print(f"\nGenerated {len(tests)} tests")

    except ImportError as e:
        print(f"Error: Missing dependencies - {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
