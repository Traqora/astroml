# Contributing to AstroML

Thank you for your interest in contributing to AstroML! This document provides guidelines and standards for contributions.

## Complexity Budget

New functions should stay under a McCabe cyclomatic complexity of **10**. CI will fail if any function exceeds **15**. Maintainability is preferred over cleverness — if a function is complex, consider splitting it.

- **Soft limit (10):** Reported as a warning in CI. New functions should aim for this.
- **Hard limit (15):** CI failure. Existing functions above this should be refactored when touched.

Use `python -m astroml.ci.complexity_check astroml api` locally before submitting.

## Logging Standards

All modules must use the structured logger from `astroml.utils.logging`. Avoid `print()` in library code; use `logger.info()` or `logger.debug()` instead. Critical paths (ingestion, training, API entrypoints) must include structured log emission.

- `DEBUG` for verbose diagnostic telemetry
- `INFO` for normal operational events
- `WARNING` for recoverable anomalies
- `ERROR` for failures that do not stop the process
- `CRITICAL` for unrecoverable failures

Use `logger.exception(...)` inside `except` blocks to capture tracebacks.

## Type Annotations

Use built-in parameterized generics where possible (`dict[str, Any]`, `list[str]`). We target Python 3.10+. Public API signatures should be fully type-annotated.

## Style

- Run `black`, `ruff`, and type checks before submitting.
- Document public classes, methods, and functions. Run `make lint-docs` to
  enforce the repository's docstring coverage floor before submitting.
- Keep functions small and testable.

## Pull Request Size & Exemption Policy

Pull requests should be kept reviewable: **≤ 1000 lines changed** and **≤ 10 files changed**. An automated soft check evaluates each PR and posts a warning comment if exceeded (CI is not failed).

- **Large refactors**: Planned mechanical refactors may use the `refactor:large` label to raise the file limit to 50 files.
- **Title exemption**: In cases where a cohesive subsystem, third-party vendoring, or database baseline cannot be split without breaking atomic changes, authors may include `[large PR]` in the PR title along with an architectural justification in the description.
- See full details in [docs/PR_SIZE_LIMITS.md](docs/PR_SIZE_LIMITS.md).

## Contributor Roadmap

New contributor? Start with [docs/contributor-roadmap.md](docs/contributor-roadmap.md):
how issues are labelled by difficulty, how Stellar Wave rewards work, and
what response times to expect. Claim an issue by commenting before you start.
