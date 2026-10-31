# Pull Request Size Limits

AstroML enforces **soft** size limits on pull requests (Issue #557). Oversized
PRs receive a warning comment from the `PR Size Limit` workflow. **CI is never
failed because of size.**

## Thresholds

| Metric | Limit | Notes |
| --- | ---: | --- |
| Lines changed (additions + deletions) | 1000 | Counted from the GitHub PR event payload |
| Files changed | 10 | Default ceiling |
| Files changed (`refactor:large`) | 50 | Planned, mechanical refactors only |

## Rationale

Review quality degrades sharply as diff size grows:

- Reviewers skim instead of reading, so defects reach `main`.
- Time-to-first-review grows, which stalls the author.
- Reverts get riskier, because unrelated changes are bundled into one commit.
- Merge conflicts multiply while the PR waits.

Small PRs are reviewed faster *and* more thoroughly, so the total time to land a
feature is usually lower when it is split.

## How to stay under the limits

- **Stack the work.** Land the interface, then the implementation, then the
  callers — each as its own PR.
- **Separate refactors from behaviour changes.** A pure rename in one PR, the
  logic change in the next.
- **Use feature flags.** Merge incomplete work behind a disabled flag rather
  than holding a long-lived branch open.
- **Isolate generated content.** Lockfiles, migrations, vendored code, and
  fixtures belong in their own PR.

## Exceptions and Exemption Policy

AstroML recognizes that certain structural changes cannot realistically be kept under the 10-file or 1000-line limits without breaking atomic correctness. Two escape hatches exist to handle these cases cleanly:

| Escape hatch | Mechanism | Effect |
| --- | --- | --- |
| Planned Large Refactor | `refactor:large` label | Raises the file ceiling from 10 to 50 files (line count limit still applies) |
| Title Exemption | `[large PR]` in PR title | Completely skips the automated PR size check comment |

### Acceptable Justifications for Exemption

The `[large PR]` exemption token must be used sparingly. Acceptable scenarios include:

1. **Vendoring or Dependency Ingestion**: Adding or upgrading third-party vendored packages, generated client stubs, or pinned schemas.
2. **Repository-Wide Mechanical Refactors**: Global renames, import organization across the entire project, or framework version upgrades touching dozens of modules without semantic logic changes.
3. **Comprehensive New Architectural Subsystems**: Introducing complete, cohesive components (such as the autonomous LLM agent framework, multi-provider integrations, or major pipeline additions) where partial delivery would leave the codebase broken or untestable.
4. **Data Schemas, Migrations & Test Fixtures**: Comprehensive database schema baselines, Alembic migrations, or golden test datasets (JSON/CSV) that naturally exceed line count thresholds.

### PR Description & Justification Requirement

Whenever using the `[large PR]` title token or requesting the `refactor:large` label, the author must explicitly document in the PR description:
- **Why the change cannot be split**: A clear architectural explanation detailing why intermediate splits would break backwards compatibility, pipeline integrity, or CI verification.
- **Review Strategy / Reading Guide**: Suggested order of review (e.g., core abstractions first, followed by concrete implementations, then tests/configs) to reduce cognitive load on reviewers.
- **Verification Evidence**: Comprehensive local test results and static checks demonstrating zero regressions across unaffected subsystems.

### Review Expectations for Oversized PRs

Because review quality naturally declines with diff size, oversized PRs are subject to:
- **Extended Review Latency**: Maintainers prioritize review-friendly PRs first; large PRs require dedicated multi-hour review slots.
- **Enhanced Verification Rigor**: Reviewers will require 100% green CI passes, rigorous docstring coverage, and full unit/regression test suites.

## Implementation

- Policy logic: [`astroml/ci/pr_size.py`](../astroml/ci/pr_size.py) — fully typed
  and unit tested in `tests/test_pr_size_limit.py`.
- Workflow: [`.github/workflows/pr-size-limit.yml`](../.github/workflows/pr-size-limit.yml).

The workflow runs on `pull_request_target` so it can comment on fork PRs. It
checks out the **base** commit and never executes contributor code; it only
reads the event payload.

The bot maintains a single comment (identified by the
`<!-- astroml:pr-size-limit -->` marker) and updates it in place — pushing more
commits will not spam the thread, and shrinking the PR flips the comment to a
pass state.

To adjust the thresholds, change `SizeThresholds` defaults in
`astroml/ci/pr_size.py` and update this document.
