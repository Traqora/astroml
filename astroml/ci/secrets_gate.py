"""Secret-scan gate for CI (issue #720).

`detect-secrets scan --baseline` silently *rewrites* the baseline and exits 0,
so it cannot fail a build on a newly leaked secret.  This gate is the missing
half: it compares a fresh scan against the reviewed baseline and fails when

  * a secret appears that the baseline does not whitelist,
  * a whitelisted secret was never audited, or was audited as a real secret,
  * the scanner itself is missing or errors out (exit 2 — a broken scan gate
    must not read as "no secrets").

Runs wherever ``detect-secrets`` is on PATH (the pre-commit CI job installs
it), so it is wired in via ``.pre-commit-config.yaml`` rather than a dedicated
workflow step.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

EXIT_OK = 0
EXIT_FOUND_SECRETS = 1
EXIT_SCAN_ERROR = 2


@dataclass(frozen=True)
class SecretLocation:
    filename: str
    hashed_secret: str

    def __str__(self) -> str:
        return f"{self.filename} ({self.hashed_secret})"


@dataclass
class GateReport:
    new_secrets: list[SecretLocation] = field(default_factory=list)
    unaudited_secrets: list[SecretLocation] = field(default_factory=list)
    true_positive_secrets: list[SecretLocation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.new_secrets or self.unaudited_secrets or self.true_positive_secrets)

    def format(self) -> str:
        if self.ok:
            return "secrets gate: PASS — no new secrets, baseline fully audited"
        lines = ["secrets gate: FAIL"]
        for loc in self.new_secrets:
            lines.append(f"  new secret not in baseline: {loc}")
        for loc in self.unaudited_secrets:
            lines.append(f"  baseline secret has no audit decision: {loc}")
        for loc in self.true_positive_secrets:
            lines.append(f"  baseline secret audited as REAL secret: {loc}")
        lines.append(
            "  Remediation: run `detect-secrets audit .secrets.baseline` to review,"
            " remove genuine credentials and rotate them."
        )
        return "\n".join(lines)


def _as_results(raw: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return the {filename: [secret-entry, ...]} map from either baseline format."""
    if isinstance(raw.get("results"), list):
        # Hand-edited v0.x baselines can carry an empty list; no entries to key.
        return {}
    if "results" in raw:
        return raw["results"]
    state = raw.get("state") or {}
    return state.get("secrets") or {}


def _as_audit(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Audit section, or None when the baseline predates auditing (v0.x)."""
    if "state" in raw:
        return raw["state"].get("audit_data")
    if "audit" in raw:
        return raw["audit"]
    return None


def _audit_decision(audit: dict[str, Any], loc: SecretLocation) -> Optional[bool]:
    per_file = audit.get(loc.filename) or {}
    entries = per_file.get(loc.hashed_secret)
    if not entries:
        return None
    is_secret = entries[-1].get("is_secret")
    return None if is_secret is None else bool(is_secret)


def evaluate(scan_output: dict[str, Any], baseline: dict[str, Any]) -> GateReport:
    report = GateReport()
    baseline_results = _as_results(baseline)
    whitelisted = {
        SecretLocation(filename, entry["hashed_secret"])
        for filename, entries in baseline_results.items()
        for entry in entries
    }
    scanned = {
        SecretLocation(filename, entry["hashed_secret"])
        for filename, entries in _as_results(scan_output).items()
        for entry in entries
    }
    report.new_secrets = sorted(scanned - whitelisted, key=str)

    audit = _as_audit(baseline)
    for loc in sorted(whitelisted, key=str):
        if audit is None:
            report.unaudited_secrets.append(loc)
            continue
        decision = _audit_decision(audit, loc)
        if decision is None:
            report.unaudited_secrets.append(loc)
        elif decision:
            report.true_positive_secrets.append(loc)
    return report


def run_detect_secrets(paths: Optional[Sequence[str]], root: Path) -> dict[str, Any]:
    if shutil.which("detect-secrets") is None:
        raise RuntimeError("detect-secrets executable not found on PATH")
    cmd = ["detect-secrets", "scan", "--exclude-files", r"\.secrets\.baseline$"]
    if paths:
        cmd += list(paths)
    elif not (root / ".git").exists():
        cmd.append("--all-files")  # non-git checkout: scan recursively
    result = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"detect-secrets failed ({result.returncode}): {result.stderr.strip()}")
    return json.loads(result.stdout)


def load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"baseline not found: {path}")
    return json.loads(path.read_text())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="secrets-gate", description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", default=".secrets.baseline", type=Path)
    parser.add_argument(
        "--root",
        default=Path.cwd(),
        type=Path,
        help="repository root to scan (default: cwd)",
    )
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        default=None,
        metavar="FILE",
        help="scan only these paths (default: all tracked files)",
    )
    args = parser.parse_args(argv)

    try:
        baseline = load_baseline(args.baseline)
        scan_output = run_detect_secrets(args.paths, args.root)
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"secrets gate: ERROR — {exc}", file=sys.stderr)
        return EXIT_SCAN_ERROR

    report = evaluate(scan_output, baseline)
    print(report.format())
    return EXIT_OK if report.ok else EXIT_FOUND_SECRETS


if __name__ == "__main__":
    sys.exit(main())
