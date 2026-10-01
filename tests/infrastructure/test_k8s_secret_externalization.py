"""k8s secret externalization regression test (#942).

``k8s/astroml-deployment.yaml`` used to ship the database password inside a
**ConfigMap** (``DATABASE_URL`` with an embedded password). ConfigMaps are
not secret material: they are world-readable inside the namespace, often
synced by GitOps tooling that assumes non-sensitive data, and they end up
scattered in etcd snapshots and CI logs. The fix moves ``DATABASE_URL`` to
an ``Opaque`` Secret referenced via ``secretKeyRef`` from every workload in
the file, and leaves only the non-sensitive ``REDIS_URL`` on the ConfigMap.

This test parses the manifest (no cluster required) and pins the contract:

* the Secret exists, is ``Opaque``, and is the only place ``DATABASE_URL``
  appears as a value
* every workload (Deployment/CronJob) injects ``DATABASE_URL`` through
  ``secretKeyRef`` — never ``configMapKeyRef`` or a literal ``value:``
* no literal password-looking string remains in any ConfigMap ``data``
* non-sensitive endpoints (``REDIS_URL``) stay on the ConfigMap
* no credential URI appears in the Secret placeholder (rotation marker)
"""

import re
from pathlib import Path

import yaml

MANIFEST = Path(__file__).resolve().parents[2] / "k8s" / "astroml-deployment.yaml"


def _load_docs():
    return list(yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")))


def _workload_containers(doc):
    if doc["kind"] == "Deployment":
        pod_spec = doc["spec"]["template"]["spec"]
    elif doc["kind"] == "CronJob":
        pod_spec = doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    else:
        return []
    return pod_spec.get("containers", [])


def _env_entries(doc):
    entries = []
    for container in _workload_containers(doc):
        entries.extend(container.get("env", []))
    return entries


def test_secret_exists_and_is_opaque():
    docs = _load_docs()
    secrets = [d for d in docs if d["kind"] == "Secret"]
    assert secrets, "astroml-deployment.yaml must define a Secret for DATABASE_URL (#942)"
    db_secrets = [s for s in secrets if "DATABASE_URL" in (s.get("stringData") or {})]
    assert db_secrets, "Secret must carry the DATABASE_URL key"
    assert db_secrets[0]["type"] == "Opaque"


def test_database_url_never_uses_configmap_or_literal():
    docs = _load_docs()
    for doc in docs:
        if doc["kind"] not in ("Deployment", "CronJob"):
            continue
        for env in _env_entries(doc):
            if env.get("name") != "DATABASE_URL":
                continue
            source = env.get("valueFrom", {})
            assert "secretKeyRef" in source, (
                f"{doc['kind']}/{doc['metadata']['name']}: DATABASE_URL must come "
                "from secretKeyRef, not configMapKeyRef or a literal value (#942)"
            )
            assert "configMapKeyRef" not in source
            assert "value" not in env, "DATABASE_URL must not be inlined as a literal"


def test_configmaps_carry_no_database_url():
    docs = _load_docs()
    for doc in docs:
        if doc["kind"] != "ConfigMap":
            continue
        data = doc.get("data") or {}
        assert "DATABASE_URL" not in data, (
            f"ConfigMap/{doc['metadata']['name']} still carries DATABASE_URL; "
            "move it to the Secret (#942)"
        )


def test_secret_placeholder_requires_rotation():
    """The committed Secret must be a rotation marker, not a usable password."""
    docs = _load_docs()
    for doc in docs:
        if doc["kind"] != "Secret":
            continue
        value = (doc.get("stringData") or {}).get("DATABASE_URL", "")
        assert "REPLACE_WITH" in value or "change_me" in value, (
            "committed DATABASE_URL must stay a rotation placeholder; real "
            "credentials belong in the cluster, not in git (#942)"
        )


def test_non_sensitive_env_stays_on_configmap():
    """REDIS_URL has no credential and must keep working from the ConfigMap."""
    docs = _load_docs()
    configmaps = [d for d in docs if d["kind"] == "ConfigMap"]
    assert configmaps, "REDIS_URL ConfigMap must still exist"
    assert "REDIS_URL" in (configmaps[0].get("data") or {})
    for doc in docs:
        if doc["kind"] not in ("Deployment", "CronJob"):
            continue
        redis = [e for e in _env_entries(doc) if e.get("name") == "REDIS_URL"]
        assert redis, f"{doc['kind']}/{doc['metadata']['name']} lost REDIS_URL"
        assert "configMapKeyRef" in redis[0].get("valueFrom", {})


def test_no_password_shaped_string_in_manifest():
    """No committed *:<password>@ URI other than the rotation placeholder."""
    text = MANIFEST.read_text(encoding="utf-8")
    pattern = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^/\s\"']+:[^@\s\"']+@")
    for match in pattern.finditer(text):
        uri = match.group(0)
        assert (
            "REPLACE_WITH" in uri or "change_me" in uri
        ), f"non-placeholder credential URI in manifest: {uri!r} (#942)"
