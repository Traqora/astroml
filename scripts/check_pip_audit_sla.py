#!/usr/bin/env python3
"""Enforce a critical/high severity SLA on pip-audit findings (issue #864).

pip-audit (checked against 2.10.1) has no CLI flag to filter findings by
severity: there is no ``--severity-filter`` option, and ``pip-audit``'s own
JSON output (``pip_audit._format.json.JsonFormat``) only ever carries
``id``, ``fix_versions``, and optionally ``aliases``/``description`` per
vulnerability -- there is no severity field anywhere in pip-audit's data
model, because its upstream sources (PyPI's advisory feed, OSV) do not
normalize severity consistently across every record.

What OSV (https://osv.dev) *does* carry reliably is
``database_specific.severity`` on GitHub Security Advisory (``GHSA-*``)
records specifically -- a plain ``LOW``/``MODERATE``/``HIGH``/``CRITICAL``
string. pip-audit's primary identifier for a finding is usually a PYSEC or
CVE id (see ``pip_audit._service.interface.VulnerabilityResult.create``,
which explicitly prioritizes those over GHSA when choosing the primary
id), and those record *types* do not carry severity even when OSV hosts
them -- only their GHSA alias does. So this script:

1. Reads pip-audit's ``--format json --aliases on`` output.
2. For each finding, resolves severity by querying OSV's ``GET
   /v1/vulns/{id}`` for the finding's own id, then -- if that record has no
   severity -- for each alias OSV's *own* record reports (which is often
   more complete than pip-audit's local alias set) until a GHSA-prefixed
   record with ``database_specific.severity`` is found.
3. Exits non-zero only if any CRITICAL or HIGH severity finding exists.
   MEDIUM/LOW findings are printed for visibility but do not fail the
   build. A finding whose severity genuinely cannot be resolved (no GHSA
   alias at all, or OSV is unreachable) is treated as a failure too --
   silently waving through an unscored finding would defeat the point of
   a severity gate.

Usage::

    pip-audit -r requirements.txt --format json --aliases on -o audit.json
    python scripts/check_pip_audit_sla.py audit.json

    # or read from stdin:
    pip-audit -r requirements.txt --format json --aliases on | \
        python scripts/check_pip_audit_sla.py -

Exit status is 0 when every finding is MEDIUM/LOW (or there are none), and
1 if any finding is CRITICAL/HIGH or unresolvable, so it drops straight
into CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{id}"

#: Severities that fail the build. Matches OSV's database_specific.severity
#: vocabulary (LOW/MODERATE/HIGH/CRITICAL); pip-audit/OSV both use "HIGH"
#: and "CRITICAL" verbatim.
FAILING_SEVERITIES = frozenset({"CRITICAL", "HIGH"})

#: Severities that are visible but do not fail the build.
PASSING_SEVERITIES = frozenset({"MODERATE", "MEDIUM", "LOW"})

UNKNOWN_SEVERITY = "UNKNOWN"

# How many OSV alias hops to follow looking for a severity-bearing record,
# before giving up. pip-audit findings observed in practice never need more
# than one hop (id -> its GHSA alias), but a small bound keeps this from
# ever looping indefinitely on a pathological alias graph.
_MAX_ALIAS_HOPS = 5


@dataclass
class Finding:
    """One package + vulnerability pair, with resolved severity."""

    package: str
    version: str
    vuln_id: str
    aliases: list[str]
    fix_versions: list[str]
    severity: str = UNKNOWN_SEVERITY
    severity_source: str | None = None  # which id the severity came from

    def as_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "version": self.version,
            "vuln_id": self.vuln_id,
            "aliases": self.aliases,
            "fix_versions": self.fix_versions,
            "severity": self.severity,
            "severity_source": self.severity_source,
        }


class OSVClient:
    """Thin, cached client for OSV's vuln-by-id lookup."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._cache: dict[str, dict[str, Any] | None] = {}

    def get(self, vuln_id: str) -> dict[str, Any] | None:
        """Return the raw OSV record for ``vuln_id``, or None if not found."""
        if vuln_id in self._cache:
            return self._cache[vuln_id]
        request = urllib.request.Request(  # noqa: S310 - fixed, trusted OSV API host
            OSV_VULN_URL.format(id=vuln_id),
            headers={"User-Agent": "astroml-pip-audit-sla-check"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                record: dict[str, Any] = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                self._cache[vuln_id] = None
                return None
            raise
        self._cache[vuln_id] = record
        return record


def resolve_severity(
    vuln_id: str, local_aliases: list[str], client: OSVClient
) -> tuple[str, str | None]:
    """Resolve a severity for one finding, hopping through OSV aliases.

    Returns ``(severity, source_id)``. ``source_id`` is the OSV record id
    the severity was actually read from (which may differ from
    ``vuln_id`` if a hop through an alias was needed), or None if no
    severity could be determined.
    """
    seen: set[str] = set()
    # Seed the queue with the finding's own id, then its locally-known
    # aliases (from pip-audit's own output) as a fallback if OSV's record
    # for the primary id is unreachable.
    queue = [vuln_id, *local_aliases]

    for _ in range(_MAX_ALIAS_HOPS):
        if not queue:
            break
        candidate = queue.pop(0)
        if candidate in seen:
            continue
        seen.add(candidate)

        record = client.get(candidate)
        if record is None:
            continue

        severity = record.get("database_specific", {}).get("severity")
        if severity:
            return severity.upper(), candidate

        # OSV's own aliases for this record are often more complete than
        # pip-audit's locally reported set (e.g. a PYSEC id's OSV record
        # listing a GHSA alias pip-audit itself did not surface).
        for alias in record.get("aliases", []):
            if alias not in seen:
                queue.append(alias)

    return UNKNOWN_SEVERITY, None


def load_findings(raw: str) -> list[Finding]:
    """Parse pip-audit's ``--format json`` output into a flat Finding list."""
    data = json.loads(raw)
    findings: list[Finding] = []
    for dep in data.get("dependencies", []):
        if dep.get("skip_reason"):
            continue
        for vuln in dep.get("vulns", []):
            findings.append(
                Finding(
                    package=dep["name"],
                    version=dep.get("version", "?"),
                    vuln_id=vuln["id"],
                    aliases=list(vuln.get("aliases", [])),
                    fix_versions=list(vuln.get("fix_versions", [])),
                )
            )
    return findings


def check_sla(findings: list[Finding], client: OSVClient) -> list[Finding]:
    """Resolve severity for every finding in place; return the same list."""
    for finding in findings:
        severity, source = resolve_severity(finding.vuln_id, finding.aliases, client)
        finding.severity = severity
        finding.severity_source = source
    return findings


def _print_report(findings: list[Finding]) -> None:
    by_severity: dict[str, list[Finding]] = {}
    for finding in findings:
        by_severity.setdefault(finding.severity, []).append(finding)

    order = ["CRITICAL", "HIGH", "MODERATE", "MEDIUM", "LOW", UNKNOWN_SEVERITY]
    for severity in order:
        group = by_severity.get(severity, [])
        if not group:
            continue
        label = "UNRESOLVED SEVERITY" if severity == UNKNOWN_SEVERITY else severity
        print(f"\n{label} ({len(group)}):")
        for f in group:
            fix = f", fix: {', '.join(f.fix_versions)}" if f.fix_versions else ""
            src = (
                f" [via {f.severity_source}]"
                if f.severity_source and f.severity_source != f.vuln_id
                else ""
            )
            print(f"  {f.package} {f.version}: {f.vuln_id}{src}{fix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "audit_json",
        help="path to pip-audit's --format json output, or '-' to read stdin",
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable JSON report")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="per-request timeout for OSV lookups (seconds)"
    )
    args = parser.parse_args(argv)

    if args.audit_json == "-":
        raw = sys.stdin.read()
    else:
        try:
            with open(args.audit_json, encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            print(f"error: could not read {args.audit_json!r}: {exc}", file=sys.stderr)
            return 1

    try:
        findings = load_findings(raw)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        # A malformed/empty audit file usually means pip-audit itself failed
        # for a reason other than "found vulnerabilities" (e.g. a network
        # error reaching its vulnerability service) -- fail closed rather
        # than silently reporting zero findings.
        print(
            f"error: could not parse pip-audit JSON from {args.audit_json!r}: {exc}",
            file=sys.stderr,
        )
        return 1
    if not findings:
        if args.json:
            print(json.dumps({"findings": [], "failing": False}, indent=2))
        else:
            print("pip-audit found no known vulnerabilities.")
        return 0

    client = OSVClient(timeout=args.timeout)
    check_sla(findings, client)

    failing = [
        f for f in findings if f.severity in FAILING_SEVERITIES or f.severity == UNKNOWN_SEVERITY
    ]

    if args.json:
        print(
            json.dumps(
                {
                    "findings": [f.as_dict() for f in findings],
                    "failing": bool(failing),
                },
                indent=2,
            )
        )
    else:
        _print_report(findings)
        print(
            f"\n{len(findings)} finding(s) total; "
            f"{len(failing)} at or above the critical/high SLA threshold "
            f"(or unresolved)."
        )
        if failing:
            print("FAILED: critical/high severity (or unresolved-severity) vulnerabilities found.")
        else:
            print("PASSED: no critical/high severity vulnerabilities found.")

    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
