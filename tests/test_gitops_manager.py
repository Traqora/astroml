"""Tests for GitOps manager."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock

import pytest

from astroml.deployment.gitops_manager import (
    DeploymentPhase,
    DriftReport,
    GitOpsManager,
    HealthStatus,
    SyncStatus,
)


class MockHttpClient:
    """Mock HTTP client for testing."""

    def __init__(self):
        self.get_responses = {}
        self.post_responses = {}
        self.last_get_url = None
        self.last_post_url = None
        self.last_post_body = None

    def get(self, url: str, headers: dict[str, str]) -> dict[str, Any]:
        self.last_get_url = url
        return self.get_responses.get(url, {})

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        self.last_post_url = url
        self.last_post_body = json
        return self.post_responses.get(url, {})


@pytest.fixture
def http_client():
    return MockHttpClient()


@pytest.fixture
def manager(http_client):
    return GitOpsManager(
        argocd_server="https://argocd.example.com",
        auth_token="test-token",
        application_name="test-app",
        http_client=http_client,
    )


def test_init_without_http_client():
    mgr = GitOpsManager(argocd_server="https://argocd.example.com", auth_token="test-token")
    with pytest.raises(RuntimeError, match="No HTTP client configured"):
        mgr.get_application_status()

    with pytest.raises(RuntimeError, match="No HTTP client configured"):
        mgr.trigger_sync()


def test_get_application_status(manager, http_client):
    expected_url = "https://argocd.example.com/api/v1/applications/test-app"
    http_client.get_responses[expected_url] = {"status": {"sync": {"status": "Synced"}}}

    status = manager.get_application_status()

    assert status == {"status": {"sync": {"status": "Synced"}}}
    assert http_client.last_get_url == expected_url


def test_trigger_sync(manager, http_client):
    expected_url = "https://argocd.example.com/api/v1/applications/test-app/sync"
    http_client.post_responses[expected_url] = {
        "status": {
            "sync": {"status": "Synced", "revision": "HEAD"},
            "health": {"status": "Healthy"},
            "resources": [{"kind": "Deployment"}],
        }
    }

    result = manager.trigger_sync(revision="HEAD", prune=True, dry_run=False)

    assert http_client.last_post_url == expected_url
    assert http_client.last_post_body == {"revision": "HEAD", "prune": True, "dryRun": False}

    assert result.application == "test-app"
    assert result.sync_status == SyncStatus.SYNCED
    assert result.health_status == HealthStatus.HEALTHY
    assert result.revision == "HEAD"
    assert result.resources_synced == 1


def test_trigger_sync_unknown_status(manager, http_client):
    expected_url = "https://argocd.example.com/api/v1/applications/test-app/sync"
    http_client.post_responses[expected_url] = {}

    result = manager.trigger_sync()

    assert result.sync_status == SyncStatus.UNKNOWN
    assert result.health_status == HealthStatus.UNKNOWN
    assert result.resources_synced == 0


def test_detect_drift_no_drift(manager, http_client):
    expected_url = "https://argocd.example.com/api/v1/applications/test-app"
    http_client.get_responses[expected_url] = {
        "status": {
            "sync": {"status": "Synced"},
            "resources": [{"status": "Synced", "kind": "Deployment"}],
        }
    }

    report = manager.detect_drift()

    assert report.application == "test-app"
    assert len(report.drifted_resources) == 0
    assert not report.remediated


def test_detect_drift_with_drift(manager, http_client):
    expected_url = "https://argocd.example.com/api/v1/applications/test-app"
    http_client.get_responses[expected_url] = {
        "status": {
            "sync": {"status": "OutOfSync"},
            "resources": [{"status": "OutOfSync", "kind": "Deployment", "name": "astroml-model"}],
        }
    }

    report = manager.detect_drift()

    assert report.application == "test-app"
    assert len(report.drifted_resources) == 1
    assert report.drifted_resources[0]["name"] == "astroml-model"


def test_remediate_drift(manager, http_client):
    expected_post_url = "https://argocd.example.com/api/v1/applications/test-app/sync"
    http_client.post_responses[expected_post_url] = {
        "status": {
            "sync": {"status": "Synced", "revision": "HEAD"},
            "health": {"status": "Healthy"},
        }
    }

    report = DriftReport(
        application="test-app", detected_at=datetime.now(timezone.utc), drifted_resources=[{}]
    )
    result = manager.remediate_drift(report)

    assert result.sync_status == SyncStatus.SYNCED
    assert report.remediated
    assert report.remediated_at is not None


def test_create_deployment(manager):
    record = manager.create_deployment(
        deployment_id="dep-1",
        model_version="1.0.0",
        image_tag="gitops-1.0.0",
        environment="staging",
        initiated_by="alice",
    )

    assert record.deployment_id == "dep-1"
    assert record.phase == DeploymentPhase.PENDING
    assert record.environment == "staging"


def test_approve_deployment(manager):
    manager.create_deployment(
        deployment_id="dep-1",
        model_version="1.0.0",
        image_tag="gitops-1.0.0",
        environment="production",
        initiated_by="alice",
    )

    record = manager.approve_deployment("dep-1", "bob")

    assert record.phase == DeploymentPhase.SYNCING
    assert record.approved_by == "bob"
    assert record.approved_at is not None


def test_approve_deployment_not_found(manager):
    with pytest.raises(ValueError, match="Deployment not found"):
        manager.approve_deployment("nonexistent", "bob")


def test_approve_deployment_not_pending(manager):
    manager.create_deployment(
        deployment_id="dep-1",
        model_version="1.0.0",
        image_tag="gitops-1.0.0",
        environment="production",
        initiated_by="alice",
    )
    manager.approve_deployment("dep-1", "bob")

    with pytest.raises(ValueError, match="is not pending"):
        manager.approve_deployment("dep-1", "charlie")


def test_complete_deployment_success(manager, http_client):
    manager.create_deployment(
        deployment_id="dep-1",
        model_version="1.0.0",
        image_tag="gitops-1.0.0",
        environment="staging",
        initiated_by="alice",
    )

    sync_result = manager.trigger_sync()  # returns UNKNOWN status by default mock
    sync_result.sync_status = SyncStatus.SYNCED

    record = manager.complete_deployment("dep-1", sync_result)

    assert record.phase == DeploymentPhase.SYNCED
    assert record.completed_at is not None


def test_complete_deployment_failure(manager, http_client):
    manager.create_deployment(
        deployment_id="dep-1",
        model_version="1.0.0",
        image_tag="gitops-1.0.0",
        environment="staging",
        initiated_by="alice",
    )

    sync_result = manager.trigger_sync()
    sync_result.sync_status = SyncStatus.OUT_OF_SYNC

    record = manager.complete_deployment("dep-1", sync_result)

    assert record.phase == DeploymentPhase.FAILED
    assert "OutOfSync" in record.error
    assert record.completed_at is not None


def test_rollback_deployment(manager):
    manager.create_deployment(
        deployment_id="dep-1",
        model_version="1.0.0",
        image_tag="gitops-1.0.0",
        environment="staging",
        initiated_by="alice",
    )

    record = manager.rollback_deployment("dep-1")

    assert record.phase == DeploymentPhase.ROLLED_BACK
    assert record.rolled_back
    assert record.completed_at is not None


def test_get_dashboard(manager, http_client):
    # Setup some state
    manager.create_deployment("dep-1", "1.0", "tag1", "staging", "alice")
    manager.approve_deployment("dep-1", "bob")
    manager.create_deployment("dep-2", "2.0", "tag2", "prod", "alice")

    # Add a drift report
    expected_url = "https://argocd.example.com/api/v1/applications/test-app"
    http_client.get_responses[expected_url] = {
        "status": {"sync": {"status": "OutOfSync"}, "resources": [{"status": "OutOfSync"}]}
    }
    manager.detect_drift()

    dashboard = manager.get_dashboard()

    assert dashboard["application"] == "test-app"
    assert dashboard["total_deployments"] == 2
    assert dashboard["deployments_by_phase"][DeploymentPhase.SYNCING.value] == 1
    assert dashboard["deployments_by_phase"][DeploymentPhase.PENDING.value] == 1
    assert dashboard["latest_deployment"]["deployment_id"] == "dep-2"
    assert dashboard["drift_summary"]["total_reports"] == 1
    assert dashboard["drift_summary"]["unresolved"] == 1


def test_get_dashboard_empty(manager):
    dashboard = manager.get_dashboard()

    assert dashboard["total_deployments"] == 0
    assert dashboard["deployments_by_phase"] == {}
    assert dashboard["latest_deployment"] == {}
    assert dashboard["drift_summary"]["total_reports"] == 0
