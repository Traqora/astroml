## Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context.

Fixes # (issue)

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## Size

Keep pull requests reviewable: **≤ 1000 lines changed** and **≤ 10 files changed**.
An automated check comments (but never fails CI) when a PR goes over.

- [ ] This PR is within the size limits, **or** it is a planned large refactor
      labelled `refactor:large` (≤ 50 files), **or** the title contains
      `[large PR]` with a justification in the description.

If it is too big: split it into a stack of smaller PRs, land refactors separately
from behaviour changes, use a feature flag to merge incomplete work
incrementally, and isolate generated/vendored files. See
[docs/PR_SIZE_LIMITS.md](../docs/PR_SIZE_LIMITS.md).

## Checklist:

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules
