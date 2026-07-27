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

## Exceptions

| Escape hatch | Effect |
| --- | --- |
| `refactor:large` label | Raises the file ceiling to 50 files |
| `[large PR]` in the PR title | Skips the check entirely |

Use `[large PR]` sparingly, and explain in the description why the change cannot
be split.

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
