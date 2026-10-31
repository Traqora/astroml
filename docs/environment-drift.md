# Environment Drift Reference

**Purpose**

This document defines what "environment drift" means for AstroML, which sources
of truth are authoritative, and how to detect and report drift between a host's
actual configuration and the declared contract.  It is intended for operators,
CI pipelines, and contributors who need to ensure that the same code produces
the same results regardless of deployment host.

---

## Sources of truth

| Source | Scope | Stability | Notes |
|--------|-------|-----------|-------|
| ``.env.example`` | Local developer / CI contract | Version‑controlled (tracked in Git) | Contains *placeholders* (``your_…_here``).  Do **not** commit real secrets here. |
| ``.env`` (local, untracked) | Personal / ad‑hoc overrides | Unstable — gitignored | Merges with ``.env.example``; any key present here that is missing from ``.env.example`` is a *drift addition*. |
| ``docker-env-guide.md`` | Docker / Kubernetes deployment | Semi‑stable (updated on infrastructure change) | Maps compose variables and Docker secrets. |
| ``docs/CONFIGURATION.md`` | Documentation / policy | Stable (reviewed on PR) | Records long‑term policy; may reference env vars for completeness. |
| ``config/*.yaml`` / ``config/*.py`` | Feature‑store, MLflow, etc. | Stable (part of release) | Should be consistent with ``.env.example`` values. |
| K8s ConfigMaps / Secrets | Production cluster | Unstable (cluster‑admin changes) | External to repo; tracked separately. |
| ``ASTROML_DATABASE_URL`` / ``POSTGRES_DB`` / etc. | Runtime database access | Unstable (set at start‑up) | Checked by snapshot builders and the ingestion service. |

> **Key rule** – the canonical contract is ``.env.example``.  Any key that
> appears in another source **but not** in ``.env.example`` is a drift addition;
> any key present in ``.env.example`` that is absent (or overridden to a
> different value) in another source is a *drift removal / value change*.

---

## Drift categories

| Category | What to look for | Detection command |
|----------|------------------|-------------------|
| **Missing required keys** | A key listed in ``.env.example`` is absent from the runtime environment. | `comm -23 <(grep -E '^[A-Z0-9_]+=' .env.example | cut -d= -f1 | sort -u) <(grep -E '^[A-Z0-9_]+=' .env | cut -d= -f1 | sort -u)` |
| **Stale extras** | Keys present in the runtime environment that are not in ``.env.example``. | `comm -13 <(grep -E '^[A-Z0-9_]+=' .env.example | cut -d= -f1 | sort -u) <(grep -E '^[A-Z0-9_]+=' .env | cut -d= -f1 | sort -u)` |
| **Duplicate keys (same key defined twice with different values)** | The same variable defined in multiple places (e.g. ``.env.example`` and a runtime override). | `grep -n "^KEY=" .env.example .env 2>/dev/null` |
| **Value drift** | A key exists in both sources but with different values. | `grep "^KEY=" .env.example .env 2>/dev/null` |
| **Secret drift** | A secret / credential value has changed. Look for `is_secret: true` in baseline audit. | `make secrets-scan` |
| **Version drift** | Dependencies or library versions differ across hosts (e.g. ``torch`` version). | `pip freeze --local | sort > /tmp/versions.txt` and compare. |

---

## Reference matrix (critical variables)

The following variables are flagged as *critical* for snapshot / training drift.
If any of these change between environments, **re‑run** the affected pipeline and
compare the resulting ``SnapshotLineage`` records (issue \#888).

| Variable | Expected source | Typical drift risk |
|----------|----------------|-------------------|
| ``ASTROML_DATABASE_URL`` | ``.env.example``, ``config/database.yaml`` | High – drives DB‑backed windows |
| ``APP_ENV`` / ``ASTROML_ENV`` | ``.env.example`` | Medium – affects feature flags |
| ``DEBUG`` / ``LOG_LEVEL`` | ``.env.example`` | Low – logging only |
| ``TRAINING_DEVICE`` / ``CUDA_VISIBLE_DEVICES`` | ``.env.example`` | Medium – hardware‑dependent |
| ``REDIS_URL`` / ``POSTGRES_DB`` | ``.env.example`` | High – ingestion cache / state |
| ``MLFLOW_TRACKING_URI`` | ``.env.example`` | Medium – experiment tracking |
| ``SOROBAN_SECRET_KEY`` | ``.env.example`` (placeholder) | Very high – secrets |
| ``SECRET_KEY`` / ``JWT_SECRET_KEY`` | ``.env.example`` | Very high – security |
| ``ENABLE_STREAMING`` / ``ENABLE_MONITORING`` / ``ENABLE_GPU_TRAINING`` / ``ENABLE_SOROBAN_CONTRACTS`` | ``.env.example`` / Feature flags | Low – toggles |
| ``DATA_PATH`` / ``MODELS_PATH`` / ``LOGS_PATH`` | ``.env.example`` / K8s ConfigMap | Medium – path drift |
| ``NETWORK_TIMEOUT`` / ``RETRY_COUNT`` / ``RETRY_DELAY`` | ``.env.example`` | Low – network policy |
| ``FEATURE_STORE_CACHE_STRATEGY`` / ``FEATURE_STORE_STORAGE_FORMAT`` / ``FEATURE_STORAGE_COMPRESSION`` / ``FEATURE_STORE_VERSIONING`` | ``.env.example`` (these four are now **single‑defined** after the fix in \#937) | Low – after fix |

The above table was generated from ``.env.example`` after removing the duplicate
blocks that caused false‑positive drift reports (see the .env.example edit in
issue \#937).

---

## Detecting drift in CI

Add a CI step that runs the ``comm`` checks above and fails the job when
any drift is found.  A minimal GitHub Action snippet:

```yaml
- name: Check environment drift
  run: |
    MISSING=$(comm -23 \
      <(grep -E '^[A-Z0-9_]+=' .env.example | cut -d= -f1 | sort -u) \
      <(grep -E '^[A-Z0-9_]+=' .env | cut -d= -f1 | sort -u))
    if [ -n "$MISSING" ]; then
      echo "❌ Missing required env vars:$MISSING"
      exit 1
    fi

    EXTRAS=$(comm -13 \
      <(grep -E '^[A-Z0-9_]+=' .env.example | cut -d= -f1 | sort -u) \
      <(grep -E '^[A-Z0-9_]+=' .env | cut -d= -f1 | sort -u))
    if [ -n "$EXTRAS" ]; then
      echo "❌ Stale env extras:$EXTRAS"
      exit 1
    fi
```

Also invoke ``make secrets-scan`` and assert ``no true positives`` in the
audited baseline.

---

## Known duplicates that were fixed (issue \#937)

The ``.env.example`` file previously contained duplicated configuration blocks.
These have been **removed** in the current version:

| Removed block | Former location | Reason |
|---------------|----------------|--------|
| Feature Store Advanced Configuration | after ``Network Configuration`` (line 18‑24) | Duplicate key definitions; kept only the first instance. |
| Production Configuration header | line 35‑39 (triple `# …`) | Malformed header causing parser confusion; collapsed to a single header. |
| Docker Configuration block | after ``Feature Flags`` (line 51‑71) | Duplicate of the canonical block later in the file; removed. |
| Data Configuration block | after Docker (line 59‑66) | Duplicate of later block; removed. |
| Network Configuration block | after Data (line 67‑72) | Duplicate of earlier block; removed. |

After these removals the reference matrix above is accurate; no entry appears
twice with different values.

---

## Remediation checklist

1. **Identify** the drift category using the detection commands.
2. **If a key is missing** from ``.env.example``, add it (as a placeholder or
   documented value) so the contract is complete.
3. **If an extra key exists**, decide whether it should be whitelisted in
   ``.env.example`` or removed from the runtime environment.
4. **If a value has drifted**, reconcile the two values and commit the
   ``.env.example`` change.
5. **If a secret has changed**, rotate the credential, update the baseline
   with ``detect-secrets audit .secrets.baseline``, and re‑run ``make secrets-scan``.
6. **Commit** the ``.env.example`` update (do **not** commit real secrets).
7. **Run** ``make secrets-scan`` and ``make env-drift`` (once the CI step is added)
   to confirm the build passes.
8. **Add** a regression test (e.g. ``tests/docs/test_environment_drift_reference.py``)
   that the drift‑check step runs on every PR.

---

## Reference checklist for reviewers

- ``.env.example`` contains all keys listed in the **Reference matrix** with
  placeholder values (`your_…_here`). No real secrets.
- The **Drift categories** table is reviewed; no “Missing required keys” or
  “Stale extras” are present on the PR head.
- ``make secrets-scan`` exits 0 and the ``.secrets.baseline`` audit contains no
  `is_secret: true` entries.
- The CI workflow includes the environment‑drift check (or the equivalent
  manual commands have been run locally).
- ``SnapshotLineage`` records from the PR head can be compared against the base
  branch to confirm no unintended DB or cache changes.