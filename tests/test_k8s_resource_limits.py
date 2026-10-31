"""Regression tests for k8s resource limits/requests audit (GitHub Issue #854).

Verifies that every container defined across k8s/*.yaml (Deployment,
CronJob, DaemonSet, StatefulSet) declares complete resources.limits and
resources.requests for both cpu and memory. A container missing a cpu or
memory limit can burst unbounded and starve its node; a container missing a
request makes the scheduler's bin-packing decisions unreliable.

The audit found fluentd (k8s/logging.yaml) missing resources.limits.cpu
(it had a memory limit and both requests, but no cpu limit) — fixed
alongside this test.
"""

from __future__ import annotations

import glob
import pathlib

import pytest
import yaml

_ROOT = pathlib.Path(__file__).parent.parent
_K8S_DIR = _ROOT / "k8s"

_CONTAINER_PARENT_KINDS = {"Deployment", "CronJob", "DaemonSet", "StatefulSet"}


def _iter_containers() -> list[tuple[str, str, str, dict]]:
    """Yields (file, kind, container_name, resources) for every container."""
    results: list[tuple[str, str, str, dict]] = []
    for fname in sorted(glob.glob(str(_K8S_DIR / "*.yaml"))):
        with open(fname) as f:
            docs = yaml.safe_load_all(f)
            for doc in docs:
                if not doc or doc.get("kind") not in _CONTAINER_PARENT_KINDS:
                    continue
                kind = doc["kind"]
                pod_spec = (
                    doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]
                    if kind == "CronJob"
                    else doc["spec"]["template"]["spec"]
                )
                for container in pod_spec.get("containers", []):
                    results.append(
                        (
                            pathlib.Path(fname).name,
                            f"{kind}/{doc['metadata']['name']}",
                            container["name"],
                            container.get("resources", {}),
                        )
                    )
    return results


def test_k8s_manifests_exist() -> None:
    if not _K8S_DIR.exists():
        pytest.skip("k8s/ directory not found")
    assert (
        len(_iter_containers()) > 0
    ), "expected at least one container across k8s/*.yaml"


@pytest.mark.parametrize(
    "field",
    ["limits.cpu", "limits.memory", "requests.cpu", "requests.memory"],
)
def test_every_container_has_complete_resource_bounds(field: str) -> None:
    section, key = field.split(".")
    missing = []
    for fname, workload, container_name, resources in _iter_containers():
        value = resources.get(section, {}).get(key)
        if value is None:
            missing.append(f"{fname}:{workload}:{container_name}")

    assert (
        not missing
    ), f"the following containers are missing resources.{field}: {missing}"
