#!/usr/bin/env python3
"""Apply Docker base image updates from a JSON update file.

Reads the output of check_base_images.py and modifies the Dockerfiles
in place to use the latest tags.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def apply_updates(updates: list[dict], dockerfile_path: str) -> bool:
    """Apply image updates to a Dockerfile.

    Args:
        updates: List of update dictionaries for this Dockerfile.
        dockerfile_path: Path to the Dockerfile to modify.

    Returns:
        True if any changes were made.
    """
    path = Path(dockerfile_path)
    if not path.exists():
        print(f"Warning: {dockerfile_path} not found, skipping")
        return False

    content = path.read_text()
    original = content

    for update in updates:
        line_num = update["line_number"]
        current_ref = update["current_ref"]
        new_ref = update["new_ref"]

        lines = content.splitlines()
        if line_num <= len(lines):
            line = lines[line_num - 1]
            # Replace the image reference in the FROM line
            new_line = line.replace(current_ref, new_ref)
            if new_line != line:
                lines[line_num - 1] = new_line
                content = "\n".join(lines)

    if content != original:
        path.write_text(content)
        return True
    return False


def git_commit_changes(dockerfiles: list[str], message: str) -> bool:
    """Commit the Dockerfile changes to git.

    Args:
        dockerfiles: List of modified Dockerfile paths.
        message: Commit message.

    Returns:
        True if commit was successful.
    """
    try:
        subprocess.run(["git", "add"] + dockerfiles, check=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Docker base image updates")
    parser.add_argument(
        "--updates",
        required=True,
        help="Path to JSON file with updates",
    )
    parser.add_argument(
        "--commit-message",
        default="deps: update Docker base images",
        help="Git commit message",
    )
    args = parser.parse_args()

    updates_file = Path(args.updates)
    if not updates_file.exists():
        print(f"Error: {args.updates} not found")
        return 1

    updates = json.loads(updates_file.read_text())
    if not updates:
        print("No updates to apply")
        return 0

    # Group updates by Dockerfile
    by_dockerfile: dict[str, list[dict]] = {}
    for update in updates:
        df = update["dockerfile"]
        by_dockerfile.setdefault(df, []).append(update)

    modified: list[str] = []
    for dockerfile, df_updates in by_dockerfile.items():
        if apply_updates(df_updates, dockerfile):
            modified.append(dockerfile)
            print(f"Updated {dockerfile}: {len(df_updates)} change(s)")

    if modified:
        print(f"\nModified {len(modified)} Dockerfile(s)")
        return 0
    else:
        print("No changes applied")
        return 0


if __name__ == "__main__":
    sys.exit(main())
