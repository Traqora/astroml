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

## PR Size Guidelines

AstroML values **small, focused pull requests** to maintain code quality and speed up review cycles.

### Size Categories

| Category | Lines Changed | Action Required |
|----------|---------------|-----------------|
| **Small** | < 200 | ✅ Ideal, no special handling |
| **Medium** | 200-500 | ✅ Good, no special handling |
| **Large** | 500-1000 | ⚠️ Justify or consider splitting |
| **Very Large** | > 1000 | ❌ Must justify or split |

*Lines changed = additions + deletions (as shown in GitHub PR header)*

### When Large PRs Are Acceptable

Large PRs are acceptable when:

1. **The change is inherently cohesive** - Cannot be meaningfully split into functional units
2. **Comprehensive test coverage** - Includes thorough tests for all changes
3. **Breaking changes** - API/schema changes requiring coordinated updates
4. **Performance optimizations** - Algorithm changes with before/after benchmarks

### Automatic Exemptions

- Documentation-only changes (`.md` files with no code modifications)
- Dependabot updates (automated dependency updates)
- Database migrations (Alembic files with model changes)
- Configuration changes (YAML, JSON, TOML files)
- Test additions for previously untested code

### Justifying a Large PR

If your PR exceeds 500 lines:

1. Add a justification section to your PR description:
   ```markdown
   ## PR Size Justification
   - **Category:** [Major feature / Breaking change / Performance optimization]
   - **Reason:** [Explain why the change is inherently cohesive]
   - **Testing:** [Confirm comprehensive test coverage]
   ```

2. Label the PR with `pr-size-exempt`
3. Tag a maintainer for approval

### Splitting Large PRs

Consider these strategies:
- **Phased implementation:** Foundation → Implementation → Integration → Documentation
- **Feature flags:** Implement behind a flag, enable in follow-up PR
- **Abstraction first:** Extract interfaces, then implement new feature
- **Independent components:** Split into non-dependent sub-features


