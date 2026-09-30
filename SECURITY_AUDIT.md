# Security Audit Checklist — AstroML / Fraud Registry

## 1. Smart Contract (Soroban / Rust)

### 1.1 Access Control
- [x] Admin-only functions (`register_validator`, `update_config`, `deactivate_validator`, `update_validator_reputation`) verify the caller matches the stored admin address
- [x] Non-admin callers receive `Error::Unauthorized`
- [x] **FIXED (SC-1):** `initialize` now has a guard against re-initialization using `env.storage().instance().has(&DATA_KEY)` check
- [ ] Admin key rotation mechanism is not implemented; document the operational runbook for key compromise

### 1.2 Input Validation
- [x] `confidence` and `reputation` values > 100 are rejected with `Error::InvalidInput`
- [x] Boundary values 0 and 100 are accepted as valid
- [x] **FIXED (SC-3):** Empty `reason` string is now rejected with `Error::InvalidInput`
- [x] **FIXED (SC-2):** `consensus_threshold` of 0 is rejected with `Error::InvalidInput` in `update_config`

### 1.3 Replay / Duplicate Prevention
- [x] Duplicate reports from the same validator for the same account are blocked via `Error::AlreadyReported`
- [x] Unique validator counting in `is_fraudulent` prevents a single address inflating consensus

### 1.4 Sybil Resistance
- [x] Reputation minimum enforced before accepting reports
- [x] Configurable `consensus_threshold` requires independent validators
- [ ] **REVIEW:** Admin can register unlimited validators and immediately set high reputations — document trusted-setup assumption or add a time-lock

### 1.5 Integer Safety
- [x] `u8` arithmetic for reputation/confidence cannot overflow standard addition since values are validated to ≤ 100
- [x] `u64` counters (`report_count`, `accurate_reports`) use saturating Soroban semantics
- [ ] Confirm `consensus_threshold` comparison (`validator_count >= data.consensus_threshold`) uses matching integer types to avoid sign-extension issues

### 1.6 Storage
- [ ] TTL / expiry of instance storage not configured — very old fraud reports persist indefinitely; consider archival strategy
- [x] Single `DATA_KEY` storage is atomic per ledger operation; no partial-write risk

### 1.7 Denial of Service
- [x] **FIXED (SC-4):** `get_active_validators` now accepts an optional `limit` parameter (default 100) to prevent unbounded iteration
- [ ] `get_fraud_reports` iterates all reports per account — same concern for heavily-targeted accounts

---

## 2. Python ML Pipeline

### 2.1 Injection Attacks
- [ ] All raw SQL queries must use parameterised statements (SQLAlchemy ORM or `%s` placeholders); audit `astroml/db/` for string-formatted queries
- [ ] Graph construction paths that accept external filenames must be validated against a whitelist of allowed directories

### 2.2 Secrets Management
- [x] `config/database.yaml` is listed in `.gitignore` (verify)
- [ ] Ensure no credentials are hard-coded in source files (run `git grep -n "password\|secret\|api_key"`)
- [ ] Database passwords should be read from environment variables, not YAML files checked into VCS

### 2.3 Dependency Security
- [ ] Run `pip-audit` against `requirements.txt` to identify known CVEs
- [ ] Pin all dependency versions and maintain a lock file (`pip-compile`)
- [ ] Rust dependencies: run `cargo audit` against `Cargo.lock`

#### pip-audit CI job (issue #531)

A GitHub Actions job runs `pip-audit` on every push and pull request.
The step is defined in `.github/workflows/security.yml`:

```yaml
- name: Audit Python dependencies
  run: pip-audit --strict
```

`--strict` causes the job to fail on any vulnerability with a fix available.
Run locally with `make security-audit`.

#### CVE handling process (issue #531)

1. **Triage** — `pip-audit` or Dependabot surfaces a new CVE.
2. **Assess** — Determine whether the vulnerable code path is reachable in
   production. Document findings in this file under *Remediation Tracker*.
3. **Remediate** — Update the pinned version in the relevant
   `requirements*.txt` file and run `pip-compile` to regenerate the lock file.
4. **Verify** — Re-run `pip-audit` locally to confirm no outstanding issues.
5. **Ship** — Open a PR with the version bump. Reference the CVE ID in the PR
   description (e.g., `CVE-2024-XXXXX`).

Critical CVEs (CVSS ≥ 9.0) must be remediated within **24 hours** of
discovery. High CVEs (CVSS 7–9) within **7 days**. Others within **30 days**.

### 2.4 Secrets Scanning

#### detect-secrets pre-commit hook (issue #531)

`detect-secrets` is installed as a pre-commit hook to prevent credentials from
being committed to the repository.

Install the hooks once:
```bash
pip install detect-secrets pre-commit
pre-commit install
```

The `.pre-commit-config.yaml` hook entry:
```yaml
- repo: https://github.com/Yelp/detect-secrets
  rev: v1.4.0
  hooks:
    - id: detect-secrets
      args: ["--baseline", ".secrets.baseline"]
```

To update the baseline after an intentional change:
```bash
detect-secrets scan --baseline .secrets.baseline
```

Run the scan manually with `make secrets-scan`.

#### Secret handling review (issue #531)

- API keys, database passwords, and JWT secrets must only be set via
  environment variables or a secrets manager — never hard-coded in source.
- The `.env.example` file must contain placeholder values only (e.g.
  `SECRET_KEY=change_me`). Never commit a real `.env` file.
- Review `config/database.yaml` and `configs/` on each PR to confirm no
  credentials are present (the `.secrets.baseline` scan covers this).
- Model artefacts (`*.pt`, `*.pkl`) must not contain embedded credentials or
  PII. Training outputs stored in `benchmark_results/` should be reviewed
  before sharing externally.

### 2.4 Deserialization
- [ ] Pickle-based model serialisation (`torch.save` / `torch.load`) must only load files from trusted paths; never load user-supplied model files directly

### 2.5 Data Leakage
- [ ] Training labels must not be visible to the model during inference evaluation (covered by `tests/test_leakage.py`)
- [ ] Logged metrics / artefacts must not contain PII from Stellar account addresses in plaintext

### 2.6 Configuration Security
- [ ] Hydra / YAML configs must validate types and ranges on load; reject unknown keys
- [ ] `consensus_threshold` and other thresholds in `configs/` should have documented acceptable ranges

---

## 3. Infrastructure

### 3.1 Docker
- [ ] Base images pinned to digest, not floating tags
- [ ] Container does not run as root (`USER` directive set in `Dockerfile`)
- [ ] No secrets in `docker-compose.yml` environment blocks in plaintext

### 3.2 CI/CD
- [ ] Add `cargo audit` step to CI pipeline
- [ ] Add `pip-audit` or `safety check` step to CI pipeline
- [ ] Secret scanning (e.g., `git-secrets` or GitHub secret scanning) enabled on the repository

---

## 4. Remediation Tracker

| ID   | Severity | Finding                                      | Status   |
|------|----------|----------------------------------------------|----------|
| SC-1 | High     | `__init__` can be called again, overwriting admin | Resolved |
| SC-2 | Medium   | `consensus_threshold = 0` marks all accounts fraudulent | Resolved |
| SC-3 | Low      | Empty `reason` string accepted               | Resolved |
| SC-4 | Medium   | `get_active_validators` unbounded iteration  | Resolved |
| PY-1 | High     | Confirm no hard-coded credentials in source  | Open |
| PY-2 | High     | Run `pip-audit`; remediate CVE findings      | Open |
| PY-3 | Medium   | Pickle load from untrusted path              | Open |
| IN-1 | Medium   | Docker base image tags not pinned to digest  | Open |
