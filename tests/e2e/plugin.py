"""Pytest plugin for E2E test reporting."""
import time
import pytest
from tests.e2e.reporter import E2ETestReporter


class E2EReportPlugin:
    """Pytest plugin for E2E test reporting."""

    def __init__(self):
        self.reporter = E2ETestReporter()
        self.test_start = None

    def pytest_runtest_setup(self, item):
        """Mark test start."""
        self.test_start = time.time()

    def pytest_runtest_logreport(self, report):
        """Capture test results."""
        if report.when == "call":
            duration = time.time() - self.test_start if self.test_start else 0
            
            if report.passed:
                status = "passed"
                error = None
            elif report.failed:
                status = "failed"
                error = report.longrepr or str(report.longreprtext)
            elif report.skipped:
                status = "skipped"
                error = report.wasxfail
            else:
                status = "unknown"
                error = None

            self.reporter.add_test_result(
                name=report.nodeid,
                status=status,
                duration=duration,
                error=error,
            )

    def pytest_sessionfinish(self, session):
        """Generate reports on session finish."""
        self.reporter.save_json_report()
        self.reporter.save_html_report()


def pytest_configure(config):
    """Configure pytest with E2E plugin."""
    plugin = E2EReportPlugin()
    config.pluginmanager.register(plugin)
