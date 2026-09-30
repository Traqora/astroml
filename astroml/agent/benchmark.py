"""Performance benchmarking utilities for parallel agent operations.

This module provides tools to measure and compare the performance of
sequential vs parallel implementations across the agent framework.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    duration_s: float
    iterations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def avg_duration_s(self) -> float:
        """Average duration per iteration."""
        return self.duration_s / self.iterations

    @property
    def ops_per_second(self) -> float:
        """Operations per second."""
        if self.avg_duration_s == 0:
            return float("inf")
        return 1.0 / self.avg_duration_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_s": round(self.duration_s, 6),
            "iterations": self.iterations,
            "avg_duration_s": round(self.avg_duration_s, 6),
            "ops_per_second": round(self.ops_per_second, 2),
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.duration_s:.4f}s "
            f"({self.iterations} iterations, "
            f"{self.avg_duration_s:.4f}s avg, "
            f"{self.ops_per_second:.2f} ops/s)"
        )


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark runs."""

    baseline: BenchmarkResult
    comparison: BenchmarkResult
    speedup: float
    improvement_pct: float

    @property
    def is_faster(self) -> bool:
        """True if the comparison is faster than baseline."""
        return self.speedup > 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "comparison": self.comparison.to_dict(),
            "speedup": round(self.speedup, 4),
            "improvement_pct": round(self.improvement_pct, 2),
            "is_faster": self.is_faster,
        }

    def __str__(self) -> str:
        direction = "faster" if self.is_faster else "slower"
        return (
            f"{self.comparison.name} is {self.speedup:.2f}x {direction} than "
            f"{self.baseline.name} ({self.improvement_pct:+.1f}% improvement)"
        )


@contextmanager
def timer(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for timing a code block.

    Example::

        with timer("compression") as result:
            compressor.compress(messages)
        print(result)
    """
    start = time.perf_counter()
    metadata = metadata or {}
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        result = BenchmarkResult(
            name=name, duration_s=duration, metadata=metadata
        )
        print(result)


def benchmark(
    func: Callable,
    name: str,
    iterations: int = 1,
    warmup: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> BenchmarkResult:
    """Run a function multiple times and return benchmark results.

    Args:
        func: Function to benchmark.
        name: Name for the benchmark.
        iterations: Number of times to run the function.
        warmup: Number of warmup runs (not timed).
        metadata: Optional metadata to attach to the result.

    Returns:
        BenchmarkResult with timing information.
    """
    metadata = metadata or {}

    # Warmup runs
    for _ in range(warmup):
        func()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    duration = time.perf_counter() - start

    return BenchmarkResult(
        name=name, duration_s=duration, iterations=iterations, metadata=metadata
    )


def compare_benchmarks(
    baseline: BenchmarkResult,
    comparison: BenchmarkResult,
) -> ComparisonResult:
    """Compare two benchmark results.

    Args:
        baseline: The baseline benchmark result.
        comparison: The comparison benchmark result.

    Returns:
        ComparisonResult with speedup and improvement metrics.
    """
    if baseline.avg_duration_s == 0:
        speedup = float("inf")
    else:
        speedup = baseline.avg_duration_s / comparison.avg_duration_s

    improvement_pct = ((baseline.avg_duration_s - comparison.avg_duration_s) / baseline.avg_duration_s) * 100

    return ComparisonResult(
        baseline=baseline,
        comparison=comparison,
        speedup=speedup,
        improvement_pct=improvement_pct,
    )


def benchmark_parallel_vs_sequential(
    sequential_func: Callable,
    parallel_func: Callable,
    name: str,
    iterations: int = 1,
    warmup: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> ComparisonResult:
    """Benchmark sequential vs parallel implementations.

    Args:
        sequential_func: Sequential implementation to benchmark.
        parallel_func: Parallel implementation to benchmark.
        name: Base name for the benchmarks.
        iterations: Number of iterations per benchmark.
        warmup: Number of warmup runs.
        metadata: Optional metadata to attach.

    Returns:
        ComparisonResult showing the speedup of parallel over sequential.
    """
    metadata = metadata or {}

    baseline = benchmark(
        sequential_func,
        name=f"{name}_sequential",
        iterations=iterations,
        warmup=warmup,
        metadata={**metadata, "implementation": "sequential"},
    )

    comparison = benchmark(
        parallel_func,
        name=f"{name}_parallel",
        iterations=iterations,
        warmup=warmup,
        metadata={**metadata, "implementation": "parallel"},
    )

    return compare_benchmarks(baseline, comparison)


class BenchmarkSuite:
    """Suite for running multiple related benchmarks."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparisonResult] = []

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)

    def add_comparison(self, comparison: ComparisonResult) -> None:
        """Add a comparison result to the suite."""
        self.comparisons.append(comparison)

    def run(
        self,
        func: Callable,
        name: str,
        iterations: int = 1,
        warmup: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run a benchmark and add it to the suite."""
        result = benchmark(func, name, iterations, warmup, metadata)
        self.add_result(result)
        return result

    def compare(
        self,
        sequential_func: Callable,
        parallel_func: Callable,
        name: str,
        iterations: int = 1,
        warmup: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComparisonResult:
        """Run a comparison and add it to the suite."""
        comparison = benchmark_parallel_vs_sequential(
            sequential_func, parallel_func, name, iterations, warmup, metadata
        )
        self.add_comparison(comparison)
        return comparison

    def summary(self) -> str:
        """Generate a summary of all benchmarks and comparisons."""
        lines = [f"Benchmark Suite: {self.name}", "=" * 50]

        if self.results:
            lines.append("\nIndividual Benchmarks:")
            lines.append("-" * 50)
            for result in self.results:
                lines.append(f"  {result}")

        if self.comparisons:
            lines.append("\nComparisons:")
            lines.append("-" * 50)
            for comparison in self.comparisons:
                lines.append(f"  {comparison}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export all results as a dictionary."""
        return {
            "name": self.name,
            "results": [result.to_dict() for result in self.results],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }

    def __str__(self) -> str:
        return self.summary()


__all__ = [
    "BenchmarkResult",
    "ComparisonResult",
    "benchmark",
    "benchmark_parallel_vs_sequential",
    "compare_benchmarks",
    "BenchmarkSuite",
    "timer",
]
