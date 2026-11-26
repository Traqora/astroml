"""Environment snapshot recorded with a run (issue #1217)."""

from __future__ import annotations

import json
from importlib import metadata

import pytest

from astroml.utils import environment
from astroml.utils.environment import TRACKED_PACKAGES, collect_environment, package_versions


class TestPackageVersions:
    def test_an_installed_package_reports_its_version(self) -> None:
        assert package_versions(["pytest"])["pytest"] == metadata.version("pytest")

    def test_a_package_that_is_not_installed_is_none(self) -> None:
        assert package_versions(["definitely-not-a-real-distribution"]) == {
            "definitely-not-a-real-distribution": None
        }

    def test_the_order_of_the_request_is_kept(self) -> None:
        names = ["numpy", "definitely-not-a-real-distribution", "pytest"]
        assert list(package_versions(names)) == names

    def test_it_reads_metadata_without_importing_the_package(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A broken or missing optional library must not stop the run from
        # recording its environment, so nothing is imported to ask for a version.
        monkeypatch.setattr(
            environment.metadata, "version", lambda name: "1.2.3" if name == "torch" else None
        )
        assert package_versions(["torch"]) == {"torch": "1.2.3"}

    def test_a_lookup_that_raises_package_not_found_becomes_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(name: str) -> str:
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(environment.metadata, "version", missing)
        assert package_versions(["torch", "numpy"]) == {"torch": None, "numpy": None}


class TestCollectEnvironment:
    def test_it_describes_the_interpreter_platform_and_packages(self) -> None:
        snapshot = collect_environment()

        assert set(snapshot) == {"python", "implementation", "platform", "packages"}
        assert snapshot["python"].count(".") >= 2
        assert list(snapshot["packages"]) == list(TRACKED_PACKAGES)

    def test_it_is_json_serialisable(self) -> None:
        snapshot = collect_environment()
        assert json.loads(json.dumps(snapshot)) == snapshot

    def test_the_package_list_can_be_chosen(self) -> None:
        snapshot = collect_environment(["numpy"])
        assert list(snapshot["packages"]) == ["numpy"]

    def test_the_tracked_set_covers_the_training_stack(self) -> None:
        assert {"astroml", "torch", "torch-geometric", "numpy"} <= set(TRACKED_PACKAGES)
