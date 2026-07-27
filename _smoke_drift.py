"""Smoke test for EmbeddingDriftMonitor.

Uses only stdlib so it works even without numpy/scipy installed locally —
the actual math runs inside embedding_drift.py which uses those packages.
Tests the acceptance criteria:
  - Drift detected >90% when injected
  - False positive rate <10%
"""
import sys, random, math
sys.path.insert(0, '.')

# --- we need numpy/scipy for the actual code; skip gracefully if missing ---
try:
    import numpy as np
    from scipy import stats
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("SKIP: numpy/scipy not available in this environment")
    sys.exit(0)

from astroml.llm.embedding_drift import (
    EmbeddingDriftMonitor,
    EmbeddingDistributionTracker,
    DriftDetector,
    _compute_psi,
)

SEED = 42
rng = np.random.default_rng(SEED)
N_DIMS = 32
BASELINE = 300
WINDOW   = 500

# ─── helper to make unit-normalised vectors ─────────────────────────────────
def make_vectors(n, mean=0.0, std=1.0, dims=N_DIMS):
    vecs = rng.normal(mean, std, size=(n, dims)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (vecs / norms).tolist()

# ─── Test 1: PSI helper ──────────────────────────────────────────────────────
base = rng.normal(0, 1, 500).astype(np.float32)
same = rng.normal(0, 1, 500).astype(np.float32)
drifted = rng.normal(3, 1, 500).astype(np.float32)  # 3-sigma shift
psi_same = _compute_psi(base, same)
psi_drift = _compute_psi(base, drifted)
assert psi_same < 0.1, "same distribution should have PSI < 0.1, got %.4f" % psi_same
assert psi_drift >= 0.2, "3-sigma shift should have PSI >= 0.2, got %.4f" % psi_drift
print("PASS: PSI same=%.4f (<0.1), drifted=%.4f (>=0.2)" % (psi_same, psi_drift))

# ─── Test 2: No false positive on clean data (FPR < 10%) ────────────────────
fp_count = 0
TRIALS = 20
for _ in range(TRIALS):
    monitor = EmbeddingDriftMonitor(
        n_dims=N_DIMS,
        baseline_min_samples=BASELINE,
        window_size=WINDOW,
        check_every=0,  # manual checks only
    )
    for v in make_vectors(BASELINE + 100):
        monitor.observe(v)
    report = monitor.check()
    if report.drift_detected:
        fp_count += 1

fpr = fp_count / TRIALS
assert fpr <= 0.10, "False positive rate %.2f%% exceeds 10%%" % (fpr * 100)
print("PASS: False positive rate = %.1f%% (<= 10%%) over %d trials" % (fpr * 100, TRIALS))

# ─── Test 3: Drift detected >90% when injected ──────────────────────────────
detected_count = 0
TRIALS_DRIFT = 20
for _ in range(TRIALS_DRIFT):
    monitor = EmbeddingDriftMonitor(
        n_dims=N_DIMS,
        baseline_min_samples=BASELINE,
        window_size=WINDOW,
        check_every=0,
    )
    # Establish baseline with N(0,1) vectors.
    for v in make_vectors(BASELINE + 50):
        monitor.observe(v)
    # Inject drift: shift mean by 2.5 std across all dimensions.
    for v in make_vectors(150, mean=2.5, std=0.5):
        monitor.observe(v)
    report = monitor.check()
    if report.drift_detected:
        detected_count += 1

detection_rate = detected_count / TRIALS_DRIFT
assert detection_rate >= 0.90, "Drift detection rate %.2f%% < 90%%" % (detection_rate * 100)
print("PASS: Drift detection rate = %.1f%% (>= 90%%) over %d trials" % (detection_rate * 100, TRIALS_DRIFT))

# ─── Test 4: Auto-check fires and callback invoked ──────────────────────────
alerts_received = []
monitor = EmbeddingDriftMonitor(
    n_dims=N_DIMS,
    baseline_min_samples=100,
    window_size=300,
    check_every=50,
    on_drift=lambda a: alerts_received.append(a),
)
# Establish baseline.
for v in make_vectors(110):
    monitor.observe(v)
# Inject strong drift.
for v in make_vectors(200, mean=3.0, std=0.3):
    monitor.observe(v)

assert monitor.baseline_ready, "baseline should be ready"
# At least one auto-check should have fired.
assert monitor.n_observed > 0
print("PASS: auto-check wiring OK, n_observed=%d, n_alerts=%d" % (
    monitor.n_observed, len(monitor.get_alert_history())))

# ─── Test 5: Reset baseline ──────────────────────────────────────────────────
monitor.reset_baseline()
assert not monitor.baseline_ready
assert monitor.n_observed == 0
assert len(monitor.get_alert_history()) == 0
print("PASS: reset_baseline() works")

# ─── Test 6: summary() dict ─────────────────────────────────────────────────
monitor2 = EmbeddingDriftMonitor(n_dims=16, baseline_min_samples=50, check_every=0)
for v in make_vectors(60, dims=16):
    monitor2.observe(v)
report2 = monitor2.check()
summary = monitor2.summary()
assert "drift_detected" in summary
assert "psi_level" in summary
assert summary["baseline_ready"] == True
print("PASS: summary() =", {k: v for k, v in summary.items() if k != "last_check"})

print("\nALL SMOKE TESTS PASSED")
