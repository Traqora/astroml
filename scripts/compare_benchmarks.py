"""Benchmark comparison script for issue #559.

Compares current benchmark results against baseline and detects regressions.
"""

from __future__ import annotations

import json
import argparse
from typing import Dict, Any, List
from pathlib import Path


def load_benchmark_results(file_path: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file.

    Args:
        file_path: Path to benchmark results file

    Returns:
        Dictionary with benchmark results
    """
    with open(file_path, "r") as f:
        return json.load(f)


def compare_benchmarks(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    threshold_warning: float = 0.10,
    threshold_fail: float = 0.20,
) -> Dict[str, Any]:
    """Compare current benchmarks against baseline.

    Args:
        current: Current benchmark results
        baseline: Baseline benchmark results
        threshold_warning: Warning threshold (10% regression)
        threshold_fail: Failure threshold (20% regression)

    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "benchmarks": [],
        "regressions": [],
        "improvements": [],
        "has_regression": False,
    }

    current_benchmarks = current.get("benchmarks", {})
    baseline_benchmarks = baseline.get("benchmarks", {})

    for name, current_data in current_benchmarks.items():
        if name not in baseline_benchmarks:
            comparison["benchmarks"].append(
                {
                    "name": name,
                    "status": "new",
                    "current": current_data,
                }
            )
            continue

        baseline_data = baseline_benchmarks[name]
        current_time = current_data.get("stats", {}).get("mean", 0)
        baseline_time = baseline_data.get("stats", {}).get("mean", 0)

        if baseline_time == 0:
            continue

        # Calculate percentage change
        change = (current_time - baseline_time) / baseline_time

        benchmark_info = {
            "name": name,
            "current_time": current_time,
            "baseline_time": baseline_time,
            "change_percent": change * 100,
            "status": "stable",
        }

        if change > threshold_fail:
            benchmark_info["status"] = "regression"
            benchmark_info["severity"] = "critical"
            comparison["regressions"].append(benchmark_info)
            comparison["has_regression"] = True
        elif change > threshold_warning:
            benchmark_info["status"] = "regression"
            benchmark_info["severity"] = "warning"
            comparison["regressions"].append(benchmark_info)
        elif change < -threshold_warning:
            benchmark_info["status"] = "improvement"
            comparison["improvements"].append(benchmark_info)

        comparison["benchmarks"].append(benchmark_info)

    return comparison


def generate_markdown_report(comparison: Dict[str, Any]) -> str:
    """Generate markdown report from comparison results.

    Args:
        comparison: Comparison results

    Returns:
        Markdown formatted report
    """
    lines = [
        "## Performance Benchmark Comparison",
        "",
        "### Summary",
        "",
    ]

    regressions = comparison.get("regressions", [])
    improvements = comparison.get("improvements", [])

    if not regressions:
        lines.append("✅ No performance regressions detected.")
    else:
        lines.append(f"⚠️ {len(regressions)} performance regression(s) detected.")

    if improvements:
        lines.append(f"🚀 {len(improvements)} improvement(s) detected.")

    lines.append("")

    if regressions:
        lines.append("### Regressions")
        lines.append("")
        lines.append("| Benchmark | Current | Baseline | Change | Severity |")
        lines.append("|-----------|---------|----------|--------|----------|")

        for reg in regressions:
            lines.append(
                f"| {reg['name']} | {reg['current_time']:.4f}s | "
                f"{reg['baseline_time']:.4f}s | {reg['change_percent']:+.2f}% | "
                f"{reg['severity'].upper()} |"
            )

        lines.append("")

    if improvements:
        lines.append("### Improvements")
        lines.append("")
        lines.append("| Benchmark | Current | Baseline | Change |")
        lines.append("|-----------|---------|----------|--------|")

        for imp in improvements:
            lines.append(
                f"| {imp['name']} | {imp['current_time']:.4f}s | "
                f"{imp['baseline_time']:.4f}s | {imp['change_percent']:+.2f}% |"
            )

        lines.append("")

    lines.append("### All Benchmarks")
    lines.append("")
    lines.append("| Benchmark | Status | Current | Baseline | Change |")
    lines.append("|-----------|--------|---------|----------|--------|")

    for bench in comparison.get("benchmarks", []):
        if bench["status"] == "new":
            lines.append(
                f"| {bench['name']} | NEW | {bench['current']['stats']['mean']:.4f}s | N/A | N/A |"
            )
        else:
            lines.append(
                f"| {bench['name']} | {bench['status'].upper()} | "
                f"{bench['current_time']:.4f}s | {bench['baseline_time']:.4f}s | "
                f"{bench['change_percent']:+.2f}% |"
            )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by performance regression checks (issue #559)*")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--current", required=True, help="Current benchmark results file")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark results file")
    parser.add_argument(
        "--threshold-warning", type=float, default=0.10, help="Warning threshold (default: 0.10)"
    )
    parser.add_argument(
        "--threshold-fail", type=float, default=0.20, help="Failure threshold (default: 0.20)"
    )
    parser.add_argument("--output", help="Output file for comparison report")
    parser.add_argument(
        "--fail-on-regression", action="store_true", help="Exit with error if regression detected"
    )

    args = parser.parse_args()

    # Load results
    current = load_benchmark_results(args.current)
    baseline = load_benchmark_results(args.baseline)

    # Compare
    comparison = compare_benchmarks(
        current,
        baseline,
        args.threshold_warning,
        args.threshold_fail,
    )

    # Generate report
    report = generate_markdown_report(comparison)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
    else:
        print(report)

    # Exit with error if regression detected
    if args.fail_on_regression and comparison["has_regression"]:
        critical_regressions = [r for r in comparison["regressions"] if r["severity"] == "critical"]
        if critical_regressions:
            print(
                "ERROR: Critical performance regressions detected!", file=__import__("sys").stderr
            )
            exit(1)


if __name__ == "__main__":
    main()
