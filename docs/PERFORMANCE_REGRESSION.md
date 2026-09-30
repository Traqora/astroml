# Performance Regression CI Checks

## Overview

Automated performance regression checks ensure that code changes do not degrade system performance beyond acceptable thresholds (issue #559).

## Performance Budgets

### Defined Thresholds

| Benchmark | Threshold | Status |
|-----------|-----------|--------|
| Graph building (10k edges) | <5s | ✅ |
| Feature computation (1000 nodes) | <2s | ✅ |
| Database query (1000 records) | <500ms | ✅ |
| Model inference (batch of 100) | <100ms | ✅ |
| Graph validation (10k nodes) | <5s | ✅ |

### Regression Thresholds

- **Warning**: 10% regression → Comment on PR
- **Block**: 20% regression → Block merge

## Running Benchmarks

### Locally

```bash
# Run all benchmarks
pytest tests/performance/test_benchmarks.py --benchmark-only

# Run specific benchmark group
pytest tests/performance/test_benchmarks.py --benchmark-only -k "graph-building"

# Generate detailed report
pytest tests/performance/test_benchmarks.py --benchmark-only --benchmark-html=report.html
```

### CI/CD

Benchmarks run automatically on:
- Every pull request to main/develop
- Every push to main branch

## Baseline Management

### Baseline Storage

Baselines are stored in `benchmarks/baseline.json` in the repository.

### Updating Baseline

Baselines are automatically updated on main branch pushes. To manually update:

```bash
# Run benchmarks
pytest tests/performance/test_benchmarks.py --benchmark-only --benchmark-json=benchmark_results.json

# Update baseline
cp benchmark_results.json benchmarks/baseline.json
git add benchmarks/baseline.json
git commit -m "Update performance baseline"
git push
```

## Comparison Results

### PR Comments

Performance comparison results are automatically posted as comments on PRs:

```
## Performance Benchmark Comparison

### Summary
⚠️ 1 performance regression(s) detected.

### Regressions
| Benchmark | Current | Baseline | Change | Severity |
|-----------|---------|----------|--------|----------|
| graph_building_10k_edges | 5.2341s | 4.5000s | +16.31% | WARNING |

### All Benchmarks
| Benchmark | Status | Current | Baseline | Change |
|-----------|--------|---------|----------|--------|
| graph_building_10k_edges | REGRESSION | 5.2341s | 4.5000s | +16.31% |
| feature_computation_1000_nodes | STABLE | 1.8234s | 1.8500s | -1.44% |
```

### Manual Comparison

```bash
python scripts/compare_benchmarks.py \
  --current benchmark_results.json \
  --baseline benchmarks/baseline.json \
  --threshold-warning 0.10 \
  --threshold-fail 0.20 \
  --output comparison_report.md
```

## Adding New Benchmarks

1. Add benchmark test to `tests/performance/test_benchmarks.py`:

```python
@pytest.mark.benchmark(group="new-feature")
def test_new_feature_benchmark(benchmark):
    """Benchmark new feature operation.
    
    Threshold: should complete in <1s
    """
    result = benchmark(your_function, args)
    assert result is not None
```

2. Update performance budget documentation
3. Run locally to establish baseline
4. Commit baseline update

## Troubleshooting

### Flaky Benchmarks

**Cause**: External dependencies or resource contention

**Solutions**:
1. Increase number of runs for stability
2. Use `--benchmark-min-rounds` for more iterations
3. Isolate from external dependencies

### False Positives

**Cause**: Normal variance in performance

**Solutions**:
1. Adjust thresholds if appropriate
2. Use statistical significance testing
3. Increase sample size

### Missing Baseline

**Cause**: First run or baseline file deleted

**Solutions**:
1. Run benchmarks on main branch to establish baseline
2. Manually create baseline from current results

## Best Practices

1. **Keep Tests Fast**: Benchmarks should complete within reasonable time
2. **Isolate Dependencies**: Avoid external API calls in benchmarks
3. **Use Realistic Data**: Benchmark with production-like data sizes
4. **Document Thresholds**: Keep performance budgets updated
5. **Review Regressions**: Investigate all regressions before accepting

## Related Documentation

- [Benchmark Tests](../tests/performance/test_benchmarks.py)
- [Performance Workflow](../.github/workflows/performance.yml)
- [Comparison Script](../scripts/compare_benchmarks.py)
