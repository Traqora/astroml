#!/usr/bin/env python3
"""Check for available system package updates in Dockerfiles.

Parses apt-get install commands in Dockerfiles and reports which
packages have newer versions available in the base distro repos.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


@dataclass
class SystemPackage:
    """A system package installed in a Dockerfile."""

    dockerfile: str
    line_number: int
    package: str
    version: str | None = None


@dataclass
class PackageUpdate:
    """An available update for a system package."""

    package: SystemPackage
    latest_version: str | None = None


@dataclass
class SystemPackageReport:
    """Report of system package status."""

    packages_checked: int = 0
    updates_available: list[PackageUpdate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


from dataclasses import dataclass, field


APT_INSTALL_PATTERN = re.compile(
    r"apt-get\s+install\s+-y\s+(.*)",
    re.IGNORECASE,
)

PACKAGE_CLEAN_PATTERN = re.compile(
    r"[a-zA-Z0-9][a-zA-Z0-9.+\-]*(?:=[0-9][0-9.+\-]*)?"
)


def parse_system_packages(dockerfile_path: str) -> list[SystemPackage]:
    """Extract apt-get install packages from a Dockerfile."""
    packages = []
    path = Path(dockerfile_path)
    if not path.exists():
        return packages

    content = path.read_text()

    for line_num, line in enumerate(content.splitlines(), start=1):
        match = APT_INSTALL_PATTERN.search(line)
        if match:
            pkg_str = match.group(1)
            # Handle line continuations by looking at previous content
            if pkg_str.endswith("\\"):
                pkg_str = pkg_str.rstrip("\\").strip()

            for pkg_match in PACKAGE_CLEAN_PATTERN.finditer(pkg_str):
                pkg = pkg_match.group(0)
                if "=" in pkg:
                    name, version = pkg.split("=", 1)
                    packages.append(SystemPackage(
                        dockerfile=dockerfile_path,
                        line_number=line_num,
                        package=name,
                        version=version,
                    ))
                else:
                    packages.append(SystemPackage(
                        dockerfile=dockerfile_path,
                        line_number=line_num,
                        package=pkg,
                    ))

    return packages


def check_package_updates(packages: list[SystemPackage]) -> list[PackageUpdate]:
    """Check which packages have updates available.

    Note: This is a simplified check. In a real environment, you'd
    query the package repository of the specific distro.
    """
    updates = []

    try:
        result = subprocess.run(
            ["apt-get", "update"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return updates

        for pkg in packages:
            try:
                result = subprocess.run(
                    ["apt-cache", "policy", pkg.package],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    # Parse apt-cache policy output
                    lines = result.stdout.splitlines()
                    for i, line in enumerate(lines):
                        if "Candidate:" in line:
                            candidate = line.split(":", 1)[1].strip()
                            if pkg.version and candidate != pkg.version:
                                updates.append(PackageUpdate(
                                    package=pkg,
                                    latest_version=candidate,
                                ))
                            break
            except subprocess.TimeoutExpired:
                continue

    except (subprocess.TimeoutExpired, FileNotFoundError):
        # apt-get not available (non-Debian system), skip
        pass

    return updates


def format_report(packages: list[SystemPackage], updates: list[PackageUpdate]) -> str:
    """Format the system package report as markdown."""
    lines = [
        "## Dockerfile System Package Report",
        "",
        f"Packages found: {len(packages)}",
        f"Updates available: {len(updates)}",
        "",
    ]

    if updates:
        lines.append("### Available Updates")
        lines.append("")
        lines.append("| Dockerfile | Package | Current | Latest |")
        lines.append("|------------|---------|---------|--------|")
        for update in updates:
            pkg = update.package
            current = pkg.version or "any"
            latest = update.latest_version or "unknown"
            lines.append(
                f"| {pkg.dockerfile}:{pkg.line_number} "
                f"| {pkg.package} | {current} | {latest} |"
            )
        lines.append("")

    if packages:
        lines.append("### All Packages")
        lines.append("")
        for pkg in packages:
            version_str = f" ({pkg.version})" if pkg.version else ""
            lines.append(f"- `{pkg.package}`{version_str} in {pkg.dockerfile}:{pkg.line_number}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Dockerfile system packages")
    parser.add_argument(
        "--dockerfiles",
        nargs="+",
        required=True,
        help="Dockerfile paths to check",
    )
    parser.add_argument(
        "--output",
        default="system_updates.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--report",
        default="system_report.md",
        help="Output markdown report file",
    )
    args = parser.parse_args()

    all_packages: list[SystemPackage] = []
    for dockerfile in args.dockerfiles:
        packages = parse_system_packages(dockerfile)
        all_packages.extend(packages)

    updates = check_package_updates(all_packages)

    # Write JSON output
    updates_data = [
        {
            "dockerfile": u.package.dockerfile,
            "line_number": u.package.line_number,
            "package": u.package.package,
            "current_version": u.package.version,
            "latest_version": u.latest_version,
        }
        for u in updates
    ]
    Path(args.output).write_text(json.dumps(updates_data, indent=2))

    # Write markdown report
    Path(args.report).write_text(format_report(all_packages, updates))

    print(f"Checked {len(all_packages)} packages, found {len(updates)} updates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
