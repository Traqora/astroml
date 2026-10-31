"""Regression tests for the feature-store PodDisruptionBudget (GitHub Issue #855).

Verifies that:
- k8s/pod-disruption-budget.yaml exists and parses as valid YAML.
- It defines a PodDisruptionBudget targeting feature-store with
  minAvailable >= 1, so voluntary disruptions (node drains, cluster
  upgrades) cannot evict every feature-store replica at once.
- It is scoped to the astroml namespace.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

_ROOT = pathlib.Path(__file__).parent.parent
_PDB_FILE = _ROOT / "k8s" / "pod-disruption-budget.yaml"


def _load_pdbs() -> list[dict]:
    if not _PDB_FILE.exists():
        pytest.skip("k8s/pod-disruption-budget.yaml not found")
    with _PDB_FILE.open() as f:
        return [doc for doc in yaml.safe_load_all(f) if doc]


def test_pdb_file_exists_and_parses() -> None:
    pdbs = _load_pdbs()
    assert (
        len(pdbs) > 0
    ), "pod-disruption-budget.yaml must define at least one PodDisruptionBudget"
    for pdb in pdbs:
        assert pdb["apiVersion"] == "policy/v1"
        assert pdb["kind"] == "PodDisruptionBudget"


def test_feature_store_pdb_targets_feature_store_with_min_available() -> None:
    pdbs = _load_pdbs()
    feature_store_pdbs = [
        p
        for p in pdbs
        if p["spec"]["selector"]["matchLabels"].get("app") == "feature-store"
    ]
    assert (
        len(feature_store_pdbs) == 1
    ), "expected exactly one PDB selecting app=feature-store"

    pdb = feature_store_pdbs[0]
    assert (
        pdb["spec"]["minAvailable"] >= 1
    ), "feature-store PDB must guarantee at least one available replica during disruptions"


def test_pdb_scoped_to_astroml_namespace() -> None:
    pdbs = _load_pdbs()
    for pdb in pdbs:
        assert pdb["metadata"]["namespace"] == "astroml"
