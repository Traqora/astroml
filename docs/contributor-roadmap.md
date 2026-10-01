# Contributor Roadmap and Labels Guide

New here? This page maps the whole contributor journey: how open effort is
labelled, what the difficulty and reward labels mean, how Stellar Wave
rewards work, and what to expect after you pick something up.

## How issues are labelled

Every contributor-ready issue carries three kinds of labels:

| Label family | Examples | Meaning |
|---|---|---|
| Difficulty | `Easy`, `Medium`, `Hard` (older issues: `good first issue`, `trivial`) | Expected scope. Pick `Easy` first if you are new. |
| Effort signal | `help wanted`, `contributors` | Maintainers want community pickup. |
| Program | `Stellar Wave`, `drips-wave` | Counts toward Stellar Wave rewards (see below). |
| Area | `tests`, `testing`, `database`, `documentation`, `integration-tests`, `community` | Where the work lives; use it to find your niche. |

### Difficulty and points

Difficulty maps to reward points. The table below reflects the values used
across recent Stellar Wave issues; the exact number on your issue is the
source of truth — it is printed right in the issue body (e.g.
`Complexity: **Complex (200 points)**`).

| Difficulty | Points | Typical scope |
|---|---|---|
| Trivial / Easy (`good first issue`) | 100 | One focused file: a test, a doc page, a small fix. |
| Medium | 150 | Several files, needs a design decision or two. |
| Hard / Complex | 200 | Spans packages, real design trade-offs, heavy test burden. |

## How Stellar Wave rewards work

1. **Claim first.** Comment on the issue to be assigned *before* starting.
   Unassigned PRs may duplicate someone else's work.
2. **Build it.** Follow the issue's suggested files, keep edits minimal on
   shared files, and rebase onto `main` before opening the PR.
3. **Green CI.** Build, tests, and lint must pass. Database-backed tests
   need their service running (see each issue's Testing section) — a green
   local run without it does not prove the covered paths work.
4. **Open the PR** against `main` with `Closes #NNN` and a description that
   states your design decisions and why. Review follows; points are awarded
   on merge.

## What to expect (response times)

There is no hard SLA — maintainers review in batches around Wave cycles.
What you can rely on:

- **Claim comments** are acknowledged by assignment on the issue.
- **PRs with green CI and a filled-in description** are reviewed before
  PRs that need CI fixes or lack context. Keep your branch rebased; a
  stale branch is the most common reason a review stalls.
- **If you are blocked**, ask on the issue rather than guessing. A
  question costs less than a rewritten PR, and it bumps the thread for
  reviewers.

If a discussion goes quiet for more than a few days, a short comment
restating your question is the right nudge — never open a duplicate issue
or PR.
