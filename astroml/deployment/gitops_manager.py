"""GitOps manager for declarative model deployments with ArgoCD.

Implements:
- Automated sync from Git to cluster via ArgoCD API
- Drift detection and remediation
- Deployment approval workflow
- Rollback via Git revert
- GitOps dashboard status reporting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    """ArgoCD application sync status."""

    SYNCED = "Synced"
    OUT_OF_SYNC = "OutOfSync"
    UNKNOWN = "Unknown"


class HealthStatus(str, Enum):
    """ArgoCD application health status."""

    HEALTHY = "Healthy"
    PROGRESSING = "Progressing"
    DEGRADED = "Degraded"
    SUSPENDED = "Suspended"
    MISSING = "Missing"
    UNKNOWN = "Unknown"


class DeploymentPhase(str, Enum):
    """Phase of a GitOps deployment."""

    PENDING = "pending"
    VALIDATING = "validating"
    SYNCING = "syncing"
    SYNCED = "synced"
    DRIFTED = "drifted"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class SyncResult:
    """Result of an ArgoCD sync operation."""

    application: str
    sync_status: SyncStatus
    health_status: HealthStatus
    revision: str
    synced_at: datetime
    resources_synced: int
    message: str = ""


@dataclass
class DriftReport:
    """Report of detected configuration drift."""

    application: str
    detected_at: datetime
    drifted_resources: list[dict[str, Any]] = field(default_factory=list)
    remediated: bool = False
    remediated_at: datetime | None = None


@dataclass
class DeploymentRecord:
    """Record of a single GitOps deployment."""

    deployment_id: str
    model_version: str
    image_tag: str
    environment: str
    phase: DeploymentPhase
    initiated_by: str
    initiated_at: datetime
    approved_by: str | None = None
    approved_at: datetime | None = None
    completed_at: datetime | None = None
    rolled_back: bool = False
    error: str | None = None


class GitOpsManager:
    """Manages GitOps-based model deployments via ArgoCD.

    Provides programmatic access to ArgoCD sync operations, drift detection,
    deployment approval workflow, and rollback via Git revert.

    Args:
        argocd_server: ArgoCD API server URL.
        auth_token: Bearer token for ArgoCD API authentication.
        application_name: Name of the ArgoCD Application resource.
        http_client: Optional HTTP client for dependency injection in tests.
    """

    def __init__(
        self,
        argocd_server: str,
        auth_token: str,
        application_name: str = "astroml-model-deployment",
        http_client: Any | None = None,
    ) -> None:
        self._server = argocd_server.rstrip("/")
        self._token = auth_token
        self._application_name = application_name
        self._http_client = http_client
        self._deployments: list[DeploymentRecord] = []
        self._drift_reports: list[DriftReport] = []

    # ------------------------------------------------------------------
    # ArgoCD API helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Return HTTP headers for ArgoCD API requests."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _api_url(self, path: str) -> str:
        """Build a full ArgoCD API URL."""
        return f"{self._server}/api/v1/{path.lstrip('/')}"

    def _get(self, path: str) -> dict[str, Any]:
        """Perform a GET request against the ArgoCD API.

        Args:
            path: API path relative to /api/v1/.

        Returns:
            Parsed JSON response body.

        Raises:
            RuntimeError: If the HTTP client is not configured.
        """
        if self._http_client is None:
            raise RuntimeError("No HTTP client configured. Inject an http_client for live calls.")
        return self._http_client.get(self._api_url(path), headers=self._headers())

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Perform a POST request against the ArgoCD API.

        Args:
            path: API path relative to /api/v1/.
            body: JSON-serialisable request body.

        Returns:
            Parsed JSON response body.

        Raises:
            RuntimeError: If the HTTP client is not configured.
        """
        if self._http_client is None:
            raise RuntimeError("No HTTP client configured. Inject an http_client for live calls.")
        return self._http_client.post(self._api_url(path), json=body, headers=self._headers())

    # ------------------------------------------------------------------
    # Sync operations
    # ------------------------------------------------------------------

    def get_application_status(self) -> dict[str, Any]:
        """Fetch the current status of the ArgoCD Application.

        Returns:
            Raw ArgoCD Application status dict.
        """
        response = self._get(f"applications/{self._application_name}")
        logger.info("Fetched status for application: %s", self._application_name)
        return response

    def trigger_sync(
        self,
        revision: str = "HEAD",
        prune: bool = True,
        dry_run: bool = False,
    ) -> SyncResult:
        """Trigger an ArgoCD sync for the managed application.

        Args:
            revision: Git revision to sync to. Defaults to HEAD.
            prune: Whether to prune resources not in Git. Defaults to True.
            dry_run: If True, performs a dry-run sync without making changes.

        Returns:
            SyncResult describing the outcome of the sync.
        """
        body: dict[str, Any] = {
            "revision": revision,
            "prune": prune,
            "dryRun": dry_run,
        }
        response = self._post(f"applications/{self._application_name}/sync", body)
        status = response.get("status", {})
        sync_status = SyncStatus(status.get("sync", {}).get("status", SyncStatus.UNKNOWN))
        health_status = HealthStatus(status.get("health", {}).get("status", HealthStatus.UNKNOWN))
        result = SyncResult(
            application=self._application_name,
            sync_status=sync_status,
            health_status=health_status,
            revision=status.get("sync", {}).get("revision", revision),
            synced_at=datetime.now(timezone.utc),
            resources_synced=len(status.get("resources", [])),
        )
        logger.info(
            "Sync triggered: application=%s revision=%s dry_run=%s result=%s",
            self._application_name,
            revision,
            dry_run,
            sync_status.value,
        )
        return result

    # ------------------------------------------------------------------
    # Drift detection and remediation
    # ------------------------------------------------------------------

    def detect_drift(self) -> DriftReport:
        """Detect configuration drift between Git and the live cluster state.

        Drift is defined as the ArgoCD Application being in OutOfSync status.

        Returns:
            DriftReport with details of any drifted resources.
        """
        status = self.get_application_status()
        sync_status = SyncStatus(
            status.get("status", {}).get("sync", {}).get("status", SyncStatus.UNKNOWN)
        )
        resources = status.get("status", {}).get("resources", [])
        drifted: list[dict[str, Any]] = [r for r in resources if r.get("status") == "OutOfSync"]
        report = DriftReport(
            application=self._application_name,
            detected_at=datetime.now(timezone.utc),
            drifted_resources=drifted,
        )
        self._drift_reports.append(report)
        if drifted:
            logger.warning(
                "Drift detected: %d resource(s) out of sync for application %s",
                len(drifted),
                self._application_name,
            )
        else:
            logger.info(
                "No drift detected for application %s (sync_status=%s)",
                self._application_name,
                sync_status.value,
            )
        return report

    def remediate_drift(self, report: DriftReport) -> SyncResult:
        """Remediate detected drift by triggering a self-heal sync.

        Args:
            report: A DriftReport produced by detect_drift().

        Returns:
            SyncResult from the remediation sync.
        """
        logger.info(
            "Remediating drift for application %s (%d resource(s))",
            self._application_name,
            len(report.drifted_resources),
        )
        result = self.trigger_sync(prune=True)
        report.remediated = True
        report.remediated_at = datetime.now(timezone.utc)
        return result

    # ------------------------------------------------------------------
    # Approval workflow
    # ------------------------------------------------------------------

    def create_deployment(
        self,
        deployment_id: str,
        model_version: str,
        image_tag: str,
        environment: str,
        initiated_by: str,
    ) -> DeploymentRecord:
        """Create a pending deployment record awaiting approval.

        Args:
            deployment_id: Unique identifier for this deployment.
            model_version: Semantic version of the model (e.g. "1.2.3").
            image_tag: Container image tag to deploy.
            environment: Target environment ("staging" or "production").
            initiated_by: Actor who initiated the deployment.

        Returns:
            DeploymentRecord in PENDING phase.
        """
        record = DeploymentRecord(
            deployment_id=deployment_id,
            model_version=model_version,
            image_tag=image_tag,
            environment=environment,
            phase=DeploymentPhase.PENDING,
            initiated_by=initiated_by,
            initiated_at=datetime.now(timezone.utc),
        )
        self._deployments.append(record)
        logger.info(
            "Deployment created: id=%s model=%s env=%s",
            deployment_id,
            model_version,
            environment,
        )
        return record

    def approve_deployment(self, deployment_id: str, approved_by: str) -> DeploymentRecord:
        """Approve a pending deployment, allowing it to proceed to sync.

        Args:
            deployment_id: ID of the deployment to approve.
            approved_by: Actor granting approval.

        Returns:
            Updated DeploymentRecord.

        Raises:
            ValueError: If no pending deployment with the given ID exists.
        """
        record = self._get_deployment(deployment_id)
        if record.phase != DeploymentPhase.PENDING:
            raise ValueError(
                f"Deployment {deployment_id} is not pending (phase={record.phase.value})"
            )
        record.approved_by = approved_by
        record.approved_at = datetime.now(timezone.utc)
        record.phase = DeploymentPhase.SYNCING
        logger.info("Deployment approved: id=%s approved_by=%s", deployment_id, approved_by)
        return record

    def complete_deployment(self, deployment_id: str, sync_result: SyncResult) -> DeploymentRecord:
        """Mark a deployment as completed based on a SyncResult.

        Args:
            deployment_id: ID of the deployment to complete.
            sync_result: Result returned by trigger_sync().

        Returns:
            Updated DeploymentRecord.

        Raises:
            ValueError: If no deployment with the given ID exists.
        """
        record = self._get_deployment(deployment_id)
        if sync_result.sync_status == SyncStatus.SYNCED:
            record.phase = DeploymentPhase.SYNCED
        else:
            record.phase = DeploymentPhase.FAILED
            record.error = f"Sync status: {sync_result.sync_status.value}"
        record.completed_at = datetime.now(timezone.utc)
        logger.info("Deployment completed: id=%s phase=%s", deployment_id, record.phase.value)
        return record

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback_deployment(self, deployment_id: str) -> DeploymentRecord:
        """Mark a deployment as rolled back.

        The actual rollback is performed externally via ``git revert`` in CI.
        This method updates the internal record to reflect the rollback.

        Args:
            deployment_id: ID of the deployment to roll back.

        Returns:
            Updated DeploymentRecord with phase ROLLED_BACK.

        Raises:
            ValueError: If no deployment with the given ID exists.
        """
        record = self._get_deployment(deployment_id)
        record.phase = DeploymentPhase.ROLLED_BACK
        record.rolled_back = True
        record.completed_at = datetime.now(timezone.utc)
        logger.info("Deployment rolled back: id=%s", deployment_id)
        return record

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_dashboard(self) -> dict[str, Any]:
        """Return a summary dashboard of all GitOps deployment activity.

        Returns:
            Dict containing deployment counts by phase, drift reports,
            and latest deployment details.
        """
        phase_counts: dict[str, int] = {}
        for record in self._deployments:
            phase_counts[record.phase.value] = phase_counts.get(record.phase.value, 0) + 1

        latest = self._deployments[-1] if self._deployments else None
        latest_info: dict[str, Any] = {}
        if latest:
            latest_info = {
                "deployment_id": latest.deployment_id,
                "model_version": latest.model_version,
                "image_tag": latest.image_tag,
                "environment": latest.environment,
                "phase": latest.phase.value,
                "initiated_by": latest.initiated_by,
                "initiated_at": latest.initiated_at.isoformat(),
                "approved_by": latest.approved_by,
                "completed_at": (latest.completed_at.isoformat() if latest.completed_at else None),
                "rolled_back": latest.rolled_back,
            }

        drift_summary = {
            "total_reports": len(self._drift_reports),
            "unresolved": sum(1 for r in self._drift_reports if not r.remediated),
        }

        return {
            "application": self._application_name,
            "total_deployments": len(self._deployments),
            "deployments_by_phase": phase_counts,
            "latest_deployment": latest_info,
            "drift_summary": drift_summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_deployment(self, deployment_id: str) -> DeploymentRecord:
        """Look up a deployment record by ID.

        Args:
            deployment_id: Unique deployment identifier.

        Returns:
            Matching DeploymentRecord.

        Raises:
            ValueError: If not found.
        """
        for record in self._deployments:
            if record.deployment_id == deployment_id:
                return record
        raise ValueError(f"Deployment not found: {deployment_id}")
