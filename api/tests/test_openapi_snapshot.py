"""
OpenAPI schema snapshot test (issue #951).

Covers: the generated OpenAPI schema's public surface (paths, methods,
tags) matches a committed snapshot, so an accidental route removal,
method change, or tag rename is caught in review instead of shipping
silently as a breaking change to API consumers.

Regenerating the snapshot after an intentional API change:
    python -m api.tests.test_openapi_snapshot --update
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SNAPSHOT_PATH = Path(__file__).parent / "openapi_snapshot.json"


def _extract_public_surface(schema: dict) -> dict:
    """Reduce the full OpenAPI schema to its stable public surface.

    Full-schema comparison would make this test churn on every docstring
    or example tweak; comparing paths/methods/tags instead means it only
    fails when the actual contract (what routes exist, what verbs they
    accept, how they're categorized) changes.
    """
    paths = {}
    for path, methods in schema.get("paths", {}).items():
        paths[path] = sorted(
            m.upper()
            for m in methods
            if m.lower()
            in {
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "options",
                "head",
            }
        )
    tags = sorted(t["name"] for t in schema.get("tags", []))
    return {"paths": paths, "tags": tags}


def _load_snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text())


def _write_snapshot(surface: dict) -> None:
    SNAPSHOT_PATH.write_text(json.dumps(surface, indent=2, sort_keys=True) + "\n")


@pytest.mark.xdist_group("api_openapi")
class TestOpenApiSnapshot:
    def test_snapshot_file_exists(self):
        assert SNAPSHOT_PATH.exists(), (
            "Snapshot missing. Generate it with: "
            "python -m api.tests.test_openapi_snapshot --update"
        )

    def test_public_surface_matches_snapshot(self, client):
        current = _extract_public_surface(client.get("/openapi.json").json())
        snapshot = _load_snapshot()

        current_paths = set(current["paths"])
        snapshot_paths = set(snapshot["paths"])

        removed = snapshot_paths - current_paths
        added = current_paths - snapshot_paths
        assert not removed, (
            f"Routes removed from the API without a snapshot update: {sorted(removed)}. "
            "If this is intentional, regenerate the snapshot: "
            "python -m api.tests.test_openapi_snapshot --update"
        )
        assert not added, (
            f"New routes not yet captured in the snapshot: {sorted(added)}. "
            "Regenerate the snapshot: python -m api.tests.test_openapi_snapshot --update"
        )

        for path in current_paths:
            assert current["paths"][path] == snapshot["paths"][path], (
                f"HTTP methods changed for {path}: "
                f"was {snapshot['paths'][path]}, now {current['paths'][path]}"
            )

        assert current["tags"] == snapshot["tags"], (
            f"API tags changed: was {snapshot['tags']}, now {current['tags']}. "
            "Regenerate the snapshot: python -m api.tests.test_openapi_snapshot --update"
        )

    def test_openapi_json_is_served(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")


def _regenerate_from_running_app() -> None:
    """Standalone entry point: build the app in-process and write the snapshot."""
    from api.app import app  # noqa: PLC0415

    surface = _extract_public_surface(app.openapi())
    _write_snapshot(surface)
    print(f"Wrote {SNAPSHOT_PATH} ({len(surface['paths'])} paths, {len(surface['tags'])} tags)")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _regenerate_from_running_app()
    else:
        print(__doc__)
