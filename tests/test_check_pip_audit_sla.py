"""Tests for scripts/check_pip_audit_sla.py — issue #864.

This checker gates CI on pip-audit findings, so its pass/fail logic needs
pinning. All OSV lookups are mocked (via OSVClient.get) so these tests
never make a real network call.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_pip_audit_sla.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_pip_audit_sla", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


class FakeOSVClient:
    """Stand-in for OSVClient that serves canned OSV records instead of
    making HTTP calls. ``records`` maps an id to the raw OSV JSON dict
    that id would return (or omits it, to simulate a 404 / unknown id)."""

    def __init__(self, records: dict[str, dict]):
        self.records = records
        self.calls: list[str] = []

    def get(self, vuln_id: str):
        self.calls.append(vuln_id)
        return self.records.get(vuln_id)


def _audit_json(dependencies: list[dict]) -> str:
    return json.dumps({"dependencies": dependencies, "fixes": []})


class TestLoadFindings:
    def test_empty_dependencies_yields_no_findings(self):
        raw = _audit_json([])
        assert checker.load_findings(raw) == []

    def test_dependency_with_no_vulns_is_skipped(self):
        raw = _audit_json([{"name": "numpy", "version": "2.0.0", "vulns": []}])
        assert checker.load_findings(raw) == []

    def test_skipped_dependency_is_ignored(self):
        raw = _audit_json([{"name": "weird-pkg", "skip_reason": "could not resolve"}])
        assert checker.load_findings(raw) == []

    def test_finding_fields_are_parsed(self):
        raw = _audit_json(
            [
                {
                    "name": "fastapi",
                    "version": "0.104.1",
                    "vulns": [
                        {
                            "id": "PYSEC-2024-38",
                            "fix_versions": ["0.109.1"],
                            "aliases": ["GHSA-qf9m-vfgh-m389", "CVE-2024-24762"],
                        }
                    ],
                }
            ]
        )
        findings = checker.load_findings(raw)
        assert len(findings) == 1
        f = findings[0]
        assert f.package == "fastapi"
        assert f.version == "0.104.1"
        assert f.vuln_id == "PYSEC-2024-38"
        assert f.fix_versions == ["0.109.1"]
        assert f.aliases == ["GHSA-qf9m-vfgh-m389", "CVE-2024-24762"]
        assert f.severity == checker.UNKNOWN_SEVERITY

    def test_multiple_vulns_on_one_dependency_each_become_a_finding(self):
        raw = _audit_json(
            [
                {
                    "name": "starlette",
                    "version": "0.27.0",
                    "vulns": [
                        {"id": "PYSEC-1", "fix_versions": [], "aliases": []},
                        {"id": "PYSEC-2", "fix_versions": [], "aliases": []},
                    ],
                }
            ]
        )
        assert len(checker.load_findings(raw)) == 2


class TestResolveSeverity:
    def test_severity_on_primary_id_is_used_directly(self):
        client = FakeOSVClient(
            {"GHSA-aaaa": {"database_specific": {"severity": "HIGH"}, "aliases": []}}
        )
        severity, source = checker.resolve_severity("GHSA-aaaa", [], client)
        assert severity == "HIGH"
        assert source == "GHSA-aaaa"

    def test_severity_is_uppercased(self):
        client = FakeOSVClient(
            {"GHSA-aaaa": {"database_specific": {"severity": "high"}, "aliases": []}}
        )
        severity, _ = checker.resolve_severity("GHSA-aaaa", [], client)
        assert severity == "HIGH"

    def test_falls_through_to_osvs_own_alias_when_primary_lacks_severity(self):
        # Mirrors the real PYSEC-2025-185 case: the primary PYSEC record has
        # no severity, but OSV's own record for it lists a GHSA alias that
        # pip-audit's local `aliases` output did not include, and that GHSA
        # record does carry a severity.
        client = FakeOSVClient(
            {
                "PYSEC-2025-185": {"aliases": ["CVE-2024-29370", "GHSA-h4pw-wxh7-4vjj"]},
                "GHSA-h4pw-wxh7-4vjj": {"database_specific": {"severity": "MODERATE"}},
            }
        )
        severity, source = checker.resolve_severity("PYSEC-2025-185", ["CVE-2024-29370"], client)
        assert severity == "MODERATE"
        assert source == "GHSA-h4pw-wxh7-4vjj"

    def test_falls_through_to_locally_reported_alias_when_primary_unreachable(self):
        # The primary id 404s on OSV, but a locally (pip-audit) reported
        # alias resolves and carries severity.
        client = FakeOSVClient(
            {"GHSA-bbbb": {"database_specific": {"severity": "CRITICAL"}, "aliases": []}}
        )
        severity, source = checker.resolve_severity("PYSEC-unreachable", ["GHSA-bbbb"], client)
        assert severity == "CRITICAL"
        assert source == "GHSA-bbbb"

    def test_unknown_when_no_record_has_severity(self):
        client = FakeOSVClient(
            {
                "PYSEC-x": {"aliases": ["CVE-x"]},
                "CVE-x": {"aliases": []},
            }
        )
        severity, source = checker.resolve_severity("PYSEC-x", [], client)
        assert severity == checker.UNKNOWN_SEVERITY
        assert source is None

    def test_unknown_when_nothing_resolves_on_osv_at_all(self):
        client = FakeOSVClient({})
        severity, source = checker.resolve_severity("PYSEC-ghost", ["CVE-ghost"], client)
        assert severity == checker.UNKNOWN_SEVERITY
        assert source is None

    def test_does_not_infinite_loop_on_circular_aliases(self):
        # A <-> B aliasing each other, neither with severity, must terminate.
        client = FakeOSVClient(
            {
                "GHSA-a": {"aliases": ["GHSA-b"]},
                "GHSA-b": {"aliases": ["GHSA-a"]},
            }
        )
        severity, source = checker.resolve_severity("GHSA-a", [], client)
        assert severity == checker.UNKNOWN_SEVERITY
        assert source is None


class TestCheckSlaIntegration:
    """End-to-end: load_findings -> check_sla -> main()'s pass/fail decision."""

    def _run(self, dependencies, records, json_mode=False, tmp_path=None):
        raw = _audit_json(dependencies)
        audit_file = tmp_path / "audit.json"
        audit_file.write_text(raw)

        client = FakeOSVClient(records)
        # Patch OSVClient so main() uses our fake instead of making real
        # HTTP calls.
        checker.OSVClient = lambda timeout=10.0: client  # type: ignore[assignment]

        argv = [str(audit_file)]
        if json_mode:
            argv.append("--json")
        return checker.main(argv)

    def test_exit_zero_when_no_vulnerabilities(self, tmp_path, capsys):
        exit_code = self._run(
            [{"name": "numpy", "version": "2.0.0", "vulns": []}], {}, tmp_path=tmp_path
        )
        assert exit_code == 0
        assert "no known vulnerabilities" in capsys.readouterr().out

    def test_exit_zero_when_only_low_and_moderate(self, tmp_path):
        dependencies = [
            {
                "name": "pkg-a",
                "version": "1.0.0",
                "vulns": [{"id": "GHSA-low", "fix_versions": [], "aliases": []}],
            },
            {
                "name": "pkg-b",
                "version": "2.0.0",
                "vulns": [{"id": "GHSA-mod", "fix_versions": [], "aliases": []}],
            },
        ]
        records = {
            "GHSA-low": {"database_specific": {"severity": "LOW"}, "aliases": []},
            "GHSA-mod": {"database_specific": {"severity": "MODERATE"}, "aliases": []},
        }
        exit_code = self._run(dependencies, records, tmp_path=tmp_path)
        assert exit_code == 0

    def test_exit_one_when_high_present(self, tmp_path):
        dependencies = [
            {
                "name": "pkg-a",
                "version": "1.0.0",
                "vulns": [{"id": "GHSA-high", "fix_versions": [], "aliases": []}],
            },
        ]
        records = {"GHSA-high": {"database_specific": {"severity": "HIGH"}, "aliases": []}}
        exit_code = self._run(dependencies, records, tmp_path=tmp_path)
        assert exit_code == 1

    def test_exit_one_when_critical_present_among_mixed_severities(self, tmp_path):
        dependencies = [
            {
                "name": "pkg-a",
                "version": "1.0.0",
                "vulns": [{"id": "GHSA-low", "fix_versions": [], "aliases": []}],
            },
            {
                "name": "pkg-b",
                "version": "2.0.0",
                "vulns": [{"id": "GHSA-crit", "fix_versions": [], "aliases": []}],
            },
        ]
        records = {
            "GHSA-low": {"database_specific": {"severity": "LOW"}, "aliases": []},
            "GHSA-crit": {"database_specific": {"severity": "CRITICAL"}, "aliases": []},
        }
        exit_code = self._run(dependencies, records, tmp_path=tmp_path)
        assert exit_code == 1

    def test_exit_one_when_severity_unresolvable(self, tmp_path):
        # A finding that cannot be scored at all must fail closed, not pass.
        dependencies = [
            {
                "name": "pkg-a",
                "version": "1.0.0",
                "vulns": [{"id": "PYSEC-ghost", "fix_versions": [], "aliases": []}],
            },
        ]
        exit_code = self._run(dependencies, {}, tmp_path=tmp_path)
        assert exit_code == 1

    def test_json_output_reports_failing_flag(self, tmp_path, capsys):
        dependencies = [
            {
                "name": "pkg-a",
                "version": "1.0.0",
                "vulns": [{"id": "GHSA-high", "fix_versions": ["2.0.0"], "aliases": []}],
            },
        ]
        records = {"GHSA-high": {"database_specific": {"severity": "HIGH"}, "aliases": []}}
        exit_code = self._run(dependencies, records, json_mode=True, tmp_path=tmp_path)
        assert exit_code == 1

        output = json.loads(capsys.readouterr().out)
        assert output["failing"] is True
        assert len(output["findings"]) == 1
        assert output["findings"][0]["severity"] == "HIGH"
        assert output["findings"][0]["package"] == "pkg-a"

    def test_stdin_input(self, tmp_path, monkeypatch, capsys):
        dependencies = [{"name": "numpy", "version": "2.0.0", "vulns": []}]
        raw = _audit_json(dependencies)
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO(raw))
        checker.OSVClient = lambda timeout=10.0: FakeOSVClient({})  # type: ignore[assignment]
        exit_code = checker.main(["-"])
        assert exit_code == 0
