#!/usr/bin/env python3
"""Check for Docker base image updates.

Compares the currently pinned base images in Dockerfiles against the
latest available versions on Docker Hub and GHCR.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ImageReference:
    """Represents a Docker image reference found in a Dockerfile."""

    dockerfile: str
    line_number: int
    stage: str
    image: str
    tag: str
    digest: str | None = None

    @property
    def full_ref(self) -> str:
        ref = f"{self.image}:{self.tag}"
        if self.digest:
            ref += f"@{self.digest}"
        return ref


@dataclass
class UpdateCandidate:
    """A candidate image update."""

    current: ImageReference
    latest_tag: str
    release_notes: str | None = None


@dataclass
class UpdateReport:
    """Report of available updates."""

    images_checked: int = 0
    updates_available: list[UpdateCandidate] = field(default_factory=list)
    up_to_date: list[ImageReference] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


DOCKERFILE_PATTERN = re.compile(
    r"^FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?",
    re.IGNORECASE | re.MULTILINE,
)

IMAGE_TAG_PATTERN = re.compile(
    r"^([a-zA-Z0-9._/-]+):([a-zA-Z0-9._-]+)(?:@sha256:[a-f0-9]+)?$"
)


def parse_dockerfile(path: str) -> list[ImageReference]:
    """Extract all FROM image references from a Dockerfile."""
    refs = []
    dockerfile = Path(path)
    if not dockerfile.exists():
        return refs

    content = dockerfile.read_text()
    current_stage = None

    for line_num, line in enumerate(content.splitlines(), start=1):
        match = DOCKERFILE_PATTERN.match(line.strip())
        if match:
            image = match.group(1)
            stage = match.group(2)

            if stage:
                current_stage = stage

            tag_match = IMAGE_TAG_PATTERN.match(image)
            if tag_match:
                refs.append(ImageReference(
                    dockerfile=path,
                    line_number=line_num,
                    stage=current_stage or "unknown",
                    image=tag_match.group(1),
                    tag=tag_match.group(2),
                ))

    return refs


def get_latest_tag(image: str, current_tag: str) -> str | None:
    """Get the latest tag for an image from Docker Hub or GHCR.

    For pinned patch versions (e.g., 3.11.9), checks for newer patches.
    For minor versions (e.g., 3.11), checks for latest patch.
    """
    try:
        result = subprocess.run(
            [
                "skopeo", "list-tags",
                f"docker://{image}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return None

        import json
        data = json.loads(result.stdout)
        tags = data.get("Tags", [])

        # Filter to semantic version tags matching the current pattern
        version_tags = [
            t for t in tags
            if re.match(r"^\d+\.\d+(\.\d+)?", t)
        ]

        if not version_tags:
            return None

        # Sort by version (simple semver comparison)
        def version_key(v: str) -> tuple:
            parts = v.split(".")
            return tuple(int(p) for p in parts if p.isdigit())

        version_tags.sort(key=version_key, reverse=True)

        current_key = version_key(current_tag)

        # Find the latest tag that's newer than current but same major/minor
        for tag in version_tags:
            tag_key = version_key(tag)
            if tag_key > current_key:
                # Only suggest same major.minor updates (safe updates)
                if len(current_key) >= 2 and len(tag_key) >= 2:
                    if tag_key[0] == current_key[0] and tag_key[1] == current_key[1]:
                        return tag
                    # For images like python:3.11.9, suggest latest patch
                    if len(current_key) == 3 and len(tag_key) == 3:
                        if tag_key[0] == current_key[0] and tag_key[1] == current_key[1]:
                            return tag

        return None

    except (subprocess.TimeoutExpired, Exception):
        return None


def check_updates(dockerfiles: list[str]) -> UpdateReport:
    """Check all Dockerfiles for available base image updates."""
    report = UpdateReport()
    all_refs: list[ImageReference] = []

    for dockerfile in dockerfiles:
        refs = parse_dockerfile(dockerfile)
        all_refs.extend(refs)

    report.images_checked = len(all_refs)

    for ref in all_refs:
        latest = get_latest_tag(ref.image, ref.tag)
        if latest and latest != ref.tag:
            report.updates_available.append(UpdateCandidate(
                current=ref,
                latest_tag=latest,
            ))
        else:
            report.up_to_date.append(ref)

    return report


def format_report(report: UpdateReport) -> str:
    """Format the update report as markdown."""
    lines = [
        "## Docker Base Image Update Report",
        "",
        f"Images checked: {report.images_checked}",
        f"Updates available: {len(report.updates_available)}",
        f"Up to date: {len(report.up_to_date)}",
        "",
    ]

    if report.updates_available:
        lines.append("### Available Updates")
        lines.append("")
        lines.append("| Dockerfile | Stage | Current | Latest |")
        lines.append("|------------|-------|---------|--------|")
        for update in report.updates_available:
            ref = update.current
            lines.append(
                f"| {ref.dockerfile}:{ref.line_number} | {ref.stage} "
                f"| `{ref.full_ref}` | `{ref.image}:{update.latest_tag}` |"
            )
        lines.append("")

    if report.errors:
        lines.append("### Errors")
        lines.append("")
        for error in report.errors:
            lines.append(f"- {error}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check for Docker base image updates")
    parser.add_argument(
        "--dockerfiles",
        nargs="+",
        required=True,
        help="Dockerfile paths to check",
    )
    parser.add_argument(
        "--output",
        default="image_updates.json",
        help="Output JSON file for updates",
    )
    parser.add_argument(
        "--report",
        default="update_report.md",
        help="Output markdown report file",
    )
    args = parser.parse_args()

    report = check_updates(args.dockerfiles)

    # Write JSON output
    updates_data = [
        {
            "dockerfile": u.current.dockerfile,
            "line_number": u.current.line_number,
            "stage": u.current.stage,
            "image": u.current.image,
            "current_tag": u.current.tag,
            "latest_tag": u.latest_tag,
            "current_ref": u.current.full_ref,
            "new_ref": f"{u.current.image}:{u.latest_tag}",
        }
        for u in report.updates_available
    ]

    Path(args.output).write_text(json.dumps(updates_data, indent=2))

    # Write markdown report
    Path(args.report).write_text(format_report(report))

    if report.updates_available:
        print(f"Found {len(report.updates_available)} update(s) available")
        return 0
    else:
        print("All base images are up to date")
        return 0


if __name__ == "__main__":
    sys.exit(main())
