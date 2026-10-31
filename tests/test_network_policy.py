"""Regression tests for the astroml namespace NetworkPolicy (GitHub Issue #856).

Verifies that:
- k8s/network-policy.yaml exists and parses as valid YAML.
- A default-deny policy covers both Ingress and Egress for every pod in the
  namespace (an empty podSelector).
- Every policy targets the `astroml` namespace.
- DNS egress (UDP/TCP 53) is explicitly allowed, since default-deny egress
  would otherwise break DNS resolution for every pod.
- The workloads that actually need postgres/redis access (astroml-ingestion,
  astroml-training, feature-store) have an explicit allow for each.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

_ROOT = pathlib.Path(__file__).parent.parent
_POLICY_FILE = _ROOT / "k8s" / "network-policy.yaml"


def _load_policies() -> list[dict]:
    if not _POLICY_FILE.exists():
        pytest.skip("k8s/network-policy.yaml not found")
    with _POLICY_FILE.open() as f:
        return [doc for doc in yaml.safe_load_all(f) if doc]


def test_network_policy_file_exists_and_parses() -> None:
    policies = _load_policies()
    assert (
        len(policies) > 0
    ), "network-policy.yaml must define at least one NetworkPolicy"
    for policy in policies:
        assert policy["apiVersion"] == "networking.k8s.io/v1"
        assert policy["kind"] == "NetworkPolicy"


def test_default_deny_covers_ingress_and_egress() -> None:
    policies = _load_policies()
    default_deny = [p for p in policies if p["metadata"]["name"] == "default-deny-all"]
    assert len(default_deny) == 1, "expected exactly one default-deny-all policy"

    policy = default_deny[0]
    assert (
        policy["spec"]["podSelector"] == {}
    ), "default-deny-all must select every pod in the namespace (empty podSelector)"
    assert set(policy["spec"]["policyTypes"]) == {
        "Ingress",
        "Egress",
    }, "default-deny-all must cover both Ingress and Egress"


def test_every_policy_targets_astroml_namespace() -> None:
    policies = _load_policies()
    for policy in policies:
        assert (
            policy["metadata"]["namespace"] == "astroml"
        ), f"{policy['metadata']['name']} must be scoped to the astroml namespace"


def test_dns_egress_allowed() -> None:
    policies = _load_policies()
    dns_policies = [p for p in policies if p["metadata"]["name"] == "allow-dns-egress"]
    assert len(dns_policies) == 1, "expected an allow-dns-egress policy"

    egress_rules = dns_policies[0]["spec"]["egress"]
    ports = {
        (rule_port["protocol"], rule_port["port"])
        for rule in egress_rules
        for rule_port in rule.get("ports", [])
    }
    assert ("UDP", 53) in ports
    assert ("TCP", 53) in ports


@pytest.mark.parametrize(
    "workload", ["astroml-ingestion", "astroml-training", "feature-store"]
)
def test_workloads_can_reach_postgres_and_redis(workload: str) -> None:
    policies = _load_policies()

    postgres_policy = next(
        p for p in policies if p["metadata"]["name"] == "allow-workloads-to-postgres"
    )
    redis_policy = next(
        p for p in policies if p["metadata"]["name"] == "allow-workloads-to-redis"
    )

    for policy in (postgres_policy, redis_policy):
        allowed_apps = {
            value
            for rule in policy["spec"]["ingress"]
            for source in rule["from"]
            for value in source.get("podSelector", {})
            .get("matchExpressions", [{}])[0]
            .get("values", [])
        }
        assert (
            workload in allowed_apps
        ), f"{policy['metadata']['name']} does not allow traffic from {workload}"
