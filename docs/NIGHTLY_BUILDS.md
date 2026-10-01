# Nightly Builds (Issue #558)

## Overview

Nightly builds validate compatibility with the latest dependencies and ensure the full test suite passes daily.

## Schedule

- **Time**: 2:00 AM UTC daily
- **Trigger**: Scheduled cron job
- **Manual Trigger**: Available via GitHub Actions UI

## Purpose

Nightly builds serve to:
1. Test compatibility with latest dependency versions
2. Run full test suite with maximum coverage
3. Detect dependency conflicts early
4. Validate system stability with unpinned versions
5. Generate daily coverage reports

## Configuration

### Dependency Installation

Nightly builds install dependencies without version pins:

```bash
pip install --upgrade --no-cache-dir \
  fastapi \
  uvicorn[standard] \
  sqlalchemy \
  alembic \
  pandas \
  numpy \
  scikit-learn \
  torch \
  torch-geometric \
  networkx \
  pytest \
  pytest-cov \
  pytest-asyncio \
  httpx \
  redis \
  celery
```

This ensures we catch breaking changes in upstream dependencies early.

### Test Execution

Full test suite with coverage:

```bash
pytest tests/ -v --cov=astroml --cov-report=xml --cov-report=html
```

## Notifications

### Slack Notification

Results are posted to Slack with:
- Build status (success/failure)
- Commit SHA
- Timestamp
- Workflow name

### GitHub Issue on Failure

If the nightly build fails, a GitHub issue is automatically created with:
- Title: "Nightly build failed - [date]"
- Workflow run link
- Labels: `nightly-build`, `ci-failure`

### Daily Coverage Report

On success, coverage summary is posted to a tracking issue.

## Checking Results

### GitHub Actions

1. Go to Actions tab
2. Select "Nightly Builds" workflow
3. View latest run

### Slack

Check the configured Slack channel for nightly build notifications.

### Coverage Reports

Coverage reports are available as artifacts for 7 days:
- `coverage.xml` - XML coverage data
- `htmlcov/` - HTML coverage report
- `coverage_summary.md` - Markdown summary

## Manual Trigger

To manually trigger a nightly build:

1. Go to Actions tab
2. Select "Nightly Builds"
3. Click "Run workflow"
4. Select branch (default: main)
5. Click "Run workflow"

## Troubleshooting

### Dependency Conflicts

**Cause**: Upstream package introduced breaking changes

**Resolution**:
1. Check the workflow logs for specific package
2. Pin problematic version in requirements.txt
3. Open issue with upstream package if needed

### Test Failures

**Cause**: Test incompatibility with new dependencies

**Resolution**:
1. Review test failure logs
2. Update test if behavior changed legitimately
3. Fix test if it's a genuine bug

### Build Timeout

**Cause**: Full test suite taking too long

**Resolution**:
1. Review slow tests
2. Consider parallel test execution
3. Optimize test performance

## Maintenance

### Updating Dependencies

To add new dependencies to nightly builds:

1. Add to `.github/workflows/nightly.yml` install step
2. Update this documentation
3. Test locally first

### Adjusting Schedule

To change the nightly build schedule:

1. Edit `.github/workflows/nightly.yml`
2. Modify the cron expression
3. Current: `0 2 * * *` (2:00 AM UTC)

### Notification Channels

To change notification channels:

1. Update `SLACK_WEBHOOK_URL` secret
2. Modify Slack configuration in workflow
3. Update tracking issue number for coverage reports

## Best Practices

1. **Review Daily**: Check nightly build results each morning
2. **Act Quickly**: Address failures promptly to prevent accumulation
3. **Document Changes**: Note any dependency updates in changelog
4. **Test Locally**: Run with latest deps before committing
5. **Monitor Coverage**: Track coverage trends over time

## Related Documentation

- [CI/CD Pipeline](../.github/workflows/ci.yml)
- [Test Documentation](SMOKE_TESTS.md)
- [Dependency Management](../requirements.txt)
