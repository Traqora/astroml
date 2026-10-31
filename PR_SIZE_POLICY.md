# PR Size Policy

This document provides detailed guidelines on pull request size limits and exemption policies for the AstroML project.

## Overview

AstroML encourages small, focused pull requests to maintain code quality, speed up review cycles, and keep the git history clean. However, we recognize that some changes cannot be reasonably split without compromising functionality.

## Size Limits

### Standard Limits

| Category | Lines Changed | Description |
|----------|---------------|-------------|
| **Ideal** | 200-400 | Most PRs should fall in this range |
| **Acceptable** | 400-600 | Medium-sized features or refactors |
| **Large** | 600+ | Requires justification or exemption |

**Lines changed** = additions + deletions (as reported by GitHub)

### Why Size Matters

**Benefits of Small PRs:**
- ✅ Faster review turnaround (reviewers can complete in one session)
- ✅ Higher code quality (easier to spot bugs and logic errors)
- ✅ Reduced merge conflicts (less likely to conflict with other work)
- ✅ Better git history (easier to bisect for debugging)
- ✅ Easier to roll back if issues arise
- ✅ More incremental feedback loop for contributors

**Risks of Large PRs:**
- ❌ Review fatigue (reviewers may miss subtle issues)
- ❌ Longer time to merge (blocked on reviewer availability)
- ❌ Higher chance of merge conflicts
- ❌ Difficult to test thoroughly
- ❌ Harder to understand the intent of each change

## Automatic Exemptions

The following PR types are automatically exempt from size limits:

### 1. Documentation Updates
- Changes to `.md` files only
- No code modifications
- Examples: README updates, API documentation, tutorial additions

### 2. Dependabot Updates
- PRs created by GitHub Dependabot
- Grouped dependency updates as configured in `.github/dependabot.yml`
- These are already grouped to reduce noise

### 3. Database Migrations
- Alembic migration files in `alembic/versions/`
- Schema changes that require coordinated updates
- Must include corresponding model changes

### 4. Configuration Changes
- YAML, JSON, TOML configuration files
- Environment variable updates
- Docker configuration changes

### 5. Comprehensive Refactoring
- Large-scale refactoring with full test coverage
- Must include tests covering all refactored paths
- No reduction in test coverage

## Exemptions Requiring Approval

The following PR types require maintainer approval to exceed size limits:

### 1. Major Feature Additions
**When to use:**
- The feature is inherently large and cohesive
- Splitting would create incomplete, non-functional code
- The changes represent a single logical unit

**Approval criteria:**
- Feature is well-scoped and documented
- Includes comprehensive tests
- Backwards compatible (or breaking changes are documented)
- Maintainer reviews and approves the size

**Example:**
> "Adding GraphSAGE inductive learning - this requires model architecture, training pipeline, feature extraction, and evaluation scripts. Splitting would leave incomplete implementation."

### 2. Performance Optimizations
**When to use:**
- Optimizations requiring extensive code changes
- Algorithm swaps or data structure changes
- Performance-critical path modifications

**Approval criteria:**
- Includes before/after benchmarks
- Tests verify performance improvement
- No regression in functionality
- Maintainer reviews and approves

**Example:**
> "Replaced networkx graph storage with sparse matrix format for 10x memory reduction. This touches all graph construction and feature extraction code."

### 3. Breaking Changes
**When to use:**
- API changes requiring updates across the codebase
- Database schema migrations with cascading changes
- Configuration format changes

**Approval criteria:**
- Migration guide provided
- All affected code updated in same PR
- Tests verify migration path
- Maintainer reviews and approves

**Example:**
> "Changed transaction graph storage from adjacency list to edge list format. Requires updates to ingestion, features, and models modules."

### 4. Test Suite Additions
**When to use:**
- Adding comprehensive tests for previously untested code
- Integration test suites for pipeline stages
- Property-based testing frameworks

**Approval criteria:**
- Tests cover previously untested critical paths
- Test quality is high (not just coverage boosting)
- Maintainer reviews and approves

**Example:**
> "Added integration tests for the full ingestion pipeline from Stellar Horizon to database. Tests multiple ledger ranges and edge cases."

### 5. Dependency Upgrades
**When to use:**
- Major version bumps of core dependencies
- Updates requiring extensive code changes
- Security patches affecting multiple modules

**Approval criteria:**
- Upgrade is necessary (security, EOL, critical bug)
- All compatibility issues addressed
- Tests verify upgrade doesn't break functionality
- Maintainer reviews and approves

**Example:**
> "Upgrading PyTorch from 1.x to 2.x requires changes to model definitions, training loops, and checkpoint loading logic."

## Requesting an Exemption

### Process

1. **Label the PR**
   - Add the `pr-size-exempt` label to your PR
   - This signals to reviewers that size has been considered

2. **Justify in PR Description**
   - Add a section explaining why the PR is large
   - Explain why splitting isn't feasible
   - Reference the exemption category from this document

   ```markdown
   ## PR Size Exemption
   
   This PR exceeds 600 lines because:
   
   - Category: Major Feature Addition
   - Reason: GraphSAGE implementation requires coordinated changes across models, training, and evaluation
   - Splitting would leave incomplete, non-functional code
   - Full test coverage included
   ```

3. **Tag Maintainer**
   - Mention a maintainer in a comment for explicit approval
   - Wait for approval before merging

### Approval Workflow

```
Contributor opens large PR
    ↓
Labels with `pr-size-exempt`
    ↓
Adds justification in description
    ↓
Tags maintainer for review
    ↓
Maintainer reviews justification
    ↓
Approved → PR proceeds normally
Rejected → Contributor splits PR or addresses concerns
```

## Splitting Large PRs

When a PR is large and doesn't qualify for exemption, consider these strategies:

### Strategy 1: Phased Implementation

**Part 1: Foundation**
- Core data structures and interfaces
- Basic functionality with minimal tests
- No integration with existing systems

**Part 2: Integration**
- Connect to existing pipeline
- Add integration tests
- Update documentation

**Part 3: Polish**
- Add advanced features
- Performance optimizations
- Examples and tutorials

### Strategy 2: Feature Flags

- Implement new feature behind a feature flag
- Merge incrementally with flag disabled
- Enable flag in follow-up PR after validation

### Strategy 3: Abstraction First

**Part 1: Abstraction**
- Extract common interfaces
- Refactor existing code to use abstractions
- Tests verify no behavior change

**Part 2: Implementation**
- Implement new feature using abstractions
- Minimal changes to existing code

### Strategy 4: Independent Sub-features

If a feature has multiple independent components:
- Sub-feature A (e.g., data ingestion)
- Sub-feature B (e.g., feature extraction)
- Sub-feature C (e.g., model training)

Each can be its own PR if they don't depend on each other.

## Enforcement

### Automated Checks

Currently, PR size limits are enforced through:
- Manual review (maintainers flag oversized PRs)
- Contributor self-policing (refer to this document)

### Future Automation

Potential automation to consider:
- GitHub Action that checks PR size and adds comments
- Size limit enforcement via branch protection rules
- Automatic labeling based on line count

### Reviewer Responsibilities

Reviewers should:
- Check PR size and flag violations
- Request splits when appropriate
- Approve exemptions when justified
- Provide guidance on splitting strategies

## Examples

### Example 1: Small PR (Ideal)

```
feat(features): add transaction frequency feature

Lines changed: 120
Description: Adds a feature that calculates transaction frequency per account
Status: Ideal size, no exemption needed
```

### Example 2: Medium PR (Acceptable)

```
fix(ingestion): handle Horizon API rate limiting

Lines changed: 450
Description: Implements exponential backoff and retry logic for Horizon API calls
Status: Acceptable size, no exemption needed
```

### Example 3: Large PR with Exemption

```
feat(models): implement GraphSAGE inductive learning

Lines changed: 1200
Exemption: Major Feature Addition
Justification: GraphSAGE requires model architecture, sampling strategy, training pipeline, and evaluation - cannot be split into functional units
Approval: @maintainer approved
Status: Exempt and approved
```

### Example 4: Large PR Requiring Split

```
refactor(graph): migrate to edge list storage

Lines changed: 800
Exemption: None (not approved)
Reviewer feedback: Please split into:
  1. Add edge list storage alongside adjacency list
  2. Migrate ingestion to use edge list
  3. Remove adjacency list storage
Status: Needs to be split
```

## Questions?

If you're unsure whether your PR qualifies for an exemption:

1. Check the exemption categories above
2. Consider if the PR can be reasonably split
3. Open a draft PR and ask for feedback from maintainers
4. Tag with `pr-size-exempt` and explain your reasoning

## Related Documents

- [CONTRIBUTING.md](CONTRIBUTING.md) - General contribution guidelines
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community guidelines
- [Pull Request Process](CONTRIBUTING.md#pr-process) - PR workflow details
