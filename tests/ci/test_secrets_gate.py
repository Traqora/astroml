"""Tests for the secrets-scan CI gate (issue #720)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from astroml.ci.secrets_gate import (
    EXIT_FOUND_SECRETS,
    EXIT_OK,
    EXIT_SCAN_ERROR,
    SecretLocation,
    evaluate,
    main,
)

AWS_FAKE_SECRET = (
    'aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"\n'
    'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"\n'
)
GITHUB_FAKE_SECRET = 'GITHUB_TOKEN = "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"\n'


def _scan(secret_files: dict[str, str]) -> dict:
    """Build a scan-result document from {filename: hashed_secret} pairs."""
    return {
        "results": {
            name: [{"type": "AWS Key Detector", "hashed_secret": digest, "is_verified": False}]
            for name, digest in secret_files.items()
        }
    }


def _baseline(secrets: dict[str, str], audits: dict[str, bool] | None) -> dict:
    doc = _scan(secrets)
    if audits is not None:
        doc["audit"] = {
            name: {secrets[name]: [{"is_secret": is_secret}]} for name, is_secret in audits.items()
        }
    return doc


class TestEvaluate:
    def test_clean_scan_and_empty_baseline_passes(self):
        report = evaluate(_scan({}), _baseline({}, audits={}))
        assert report.ok

    def test_secret_missing_from_baseline_is_reported_as_new(self):
        report = evaluate(
            _scan({"app/settings.py": "aaaa"}),
            _baseline({}, audits={}),
        )
        assert SecretLocation("app/settings.py", "aaaa") in report.new_secrets
        assert not report.ok

    def test_whitelisted_and_audited_false_positive_passes(self):
        scan = _scan({"tests/fixtures/keys.py": "bbbb"})
        base = _baseline(
            {"tests/fixtures/keys.py": "bbbb"}, audits={"tests/fixtures/keys.py": False}
        )
        assert evaluate(scan, base).ok

    def test_whitelisted_but_unaudited_fails(self):
        scan = _scan({"seeds.py": "cccc"})
        base = _baseline({"seeds.py": "cccc"}, audits=None)  # legacy v0.x baseline, no audit data
        report = evaluate(scan, base)
        assert SecretLocation("seeds.py", "cccc") in report.unaudited_secrets
        assert not report.ok

    def test_audit_entry_absent_for_specific_secret_fails(self):
        scan = _scan({"a.py": "1", "b.py": "2"})
        base = _baseline(
            {"a.py": "1", "b.py": "2"},
            audits={"a.py": False},
        )
        report = evaluate(scan, base)
        assert [s.filename for s in report.unaudited_secrets] == ["b.py"]

    def test_baseline_secret_audited_as_real_secret_fails(self):
        scan = _scan({})
        base = _baseline({"prod.env": "dddd"}, audits={"prod.env": True})
        report = evaluate(scan, base)
        assert SecretLocation("prod.env", "dddd") in report.true_positive_secrets
        assert not report.ok

    def test_state_style_baseline_is_supported(self):
        scan = _scan({"x.py": "ee"})
        base = {
            "state": {
                "secrets": scan["results"],
                "audit_data": {"x.py": {"ee": [{"is_secret": False}]}},
            }
        }
        assert evaluate(scan, base).ok


class TestCli:
    def test_missing_baseline_exits_scan_error(self, tmp_path):
        code = main(["--baseline", str(tmp_path / "nope.baseline"), "--root", str(tmp_path)])
        assert code == EXIT_SCAN_ERROR

    def test_bad_repo_path_exits_scan_error(self, tmp_path):
        baseline = tmp_path / ".secrets.baseline"
        baseline.write_text("{not json")
        code = main(["--baseline", str(baseline), "--root", str(tmp_path)])
        assert code == EXIT_SCAN_ERROR

    @pytest.mark.skipif(
        shutil.which("detect-secrets") is None, reason="detect-secrets CLI required"
    )
    def test_real_scan_of_leaked_aws_key_fails_gate(self, tmp_path):
        (tmp_path / "leak.py").write_text(GITHUB_FAKE_SECRET)
        baseline = tmp_path / ".secrets.baseline"
        baseline.write_text(json.dumps({"results": {}, "audit": {}}))
        code = main(["--baseline", str(baseline), "--root", str(tmp_path), "--path", "leak.py"])
        assert code == EXIT_FOUND_SECRETS

    @pytest.mark.skipif(
        shutil.which("detect-secrets") is None, reason="detect-secrets CLI required"
    )
    def test_real_scan_of_clean_file_passes_gate(self, tmp_path):
        (tmp_path / "clean.py").write_text("PORT = 8080\n")
        baseline = tmp_path / ".secrets.baseline"
        baseline.write_text(json.dumps({"results": {}, "audit": {}}))
        code = main(["--baseline", str(baseline), "--root", str(tmp_path), "--path", "clean.py"])
        assert code == EXIT_OK

    @pytest.mark.skipif(
        shutil.which("detect-secrets") is None, reason="detect-secrets CLI required"
    )
    def test_gate_does_not_mutate_baseline(self, tmp_path):
        (tmp_path / "leak.py").write_text(AWS_FAKE_SECRET)
        baseline = tmp_path / ".secrets.baseline"
        original = json.dumps({"results": {}, "audit": {}})
        baseline.write_text(original)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "astroml.ci.secrets_gate",
                "--baseline",
                str(baseline),
                "--root",
                str(tmp_path),
                "--path",
                "leak.py",
            ],
            cwd=str(Path(__file__).resolve().parents[2]),
            check=False,
        )
        assert baseline.read_text() == original
