"""E2E test report generation."""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class E2ETestReporter:
    """Generate E2E test reports with metrics and analysis."""

    def __init__(self, report_dir: str = "test-results"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        self.start_time = time.time()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "duration": 0,
            }
        }

    def add_test_result(
        self,
        name: str,
        status: str,
        duration: float,
        error: str = None,
        metadata: Dict = None
    ):
        """Add a test result."""
        result = {
            "name": name,
            "status": status,
            "duration": duration,
            "error": error,
            "metadata": metadata or {}
        }
        self.results["tests"].append(result)

        # Update summary
        self.results["summary"]["total"] += 1
        if status == "passed":
            self.results["summary"]["passed"] += 1
        elif status == "failed":
            self.results["summary"]["failed"] += 1
        elif status == "skipped":
            self.results["summary"]["skipped"] += 1

    def detect_flaky_tests(self, threshold: float = 0.5) -> List[str]:
        """Detect potentially flaky tests."""
        # Analyze test patterns for instability
        flaky = []
        for test in self.results["tests"]:
            if test["status"] == "failed" and test.get("error"):
                # Tests with timeout or connection errors are often flaky
                if any(keyword in test["error"].lower() for keyword in ["timeout", "connection", "reset"]):
                    flaky.append(test["name"])
        return flaky

    def generate_report(self) -> Dict[str, Any]:
        """Generate complete report."""
        self.results["summary"]["duration"] = time.time() - self.start_time
        
        # Add flaky test detection
        flaky_tests = self.detect_flaky_tests()
        self.results["summary"]["flaky_tests"] = flaky_tests
        
        # Calculate pass rate
        total = self.results["summary"]["total"]
        if total > 0:
            self.results["summary"]["pass_rate"] = (
                self.results["summary"]["passed"] / total * 100
            )
        
        return self.results

    def save_json_report(self):
        """Save report as JSON."""
        report_path = self.report_dir / "e2e-report.json"
        with open(report_path, "w") as f:
            json.dump(self.generate_report(), f, indent=2)
        return report_path

    def save_html_report(self):
        """Save report as HTML."""
        report_data = self.generate_report()
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>E2E Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .passed {{ color: #28a745; }}
                .failed {{ color: #dc3545; }}
                .skipped {{ color: #ffc107; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #0066cc; color: white; }}
                .flaky-warning {{ background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>E2E API Test Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Timestamp:</strong> {report_data['timestamp']}</p>
                <p><strong>Total Tests:</strong> {report_data['summary']['total']}</p>
                <p class="passed"><strong>Passed:</strong> {report_data['summary']['passed']}</p>
                <p class="failed"><strong>Failed:</strong> {report_data['summary']['failed']}</p>
                <p class="skipped"><strong>Skipped:</strong> {report_data['summary']['skipped']}</p>
                <p><strong>Duration:</strong> {report_data['summary']['duration']:.2f}s</p>
                <p><strong>Pass Rate:</strong> {report_data['summary'].get('pass_rate', 0):.1f}%</p>
            </div>

            {self._flaky_section(report_data['summary'].get('flaky_tests', []))}

            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Duration (s)</th>
                    <th>Error</th>
                </tr>
                {''.join(
                    f"<tr><td>{t['name']}</td><td class='{t['status']}'>"
                    f"{t['status']}</td><td>{t['duration']:.3f}</td>"
                    f"<td>{t.get('error', '')}</td></tr>"
                    for t in report_data['tests'][:50]
                )}
            </table>
        </body>
        </html>
        """
        
        report_path = self.report_dir / "e2e-report.html"
        with open(report_path, "w") as f:
            f.write(html_template)
        return report_path

    @staticmethod
    def _flaky_section(flaky_tests: List[str]) -> str:
        """Generate flaky tests warning section."""
        if not flaky_tests:
            return ""
        
        return f"""
            <div class="flaky-warning">
                <h3>⚠️ Potentially Flaky Tests Detected</h3>
                <ul>
                    {''.join(f"<li>{test}</li>" for test in flaky_tests)}
                </ul>
            </div>
        """
