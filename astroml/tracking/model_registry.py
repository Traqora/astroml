"""Model registry for managing ML models and their versions.

Enhanced with:
- Semantic versioning (major.minor.patch)
- Model metadata (framework, task_type, description)
- Performance metrics tracking
- Rollback capability
- A/B testing support
- Deployment tracking
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from astroml.db.schema import Model, ModelVersion
from astroml.db.session import get_session

logger = logging.getLogger(__name__)

# Status definitions
VALID_STATUSES = {"training", "trained", "staged", "deployed", "archived", "failed", "rollback"}

VALID_STATUS_TRANSITIONS = {
    "training": ["trained", "failed"],
    "trained": ["staged", "archived"],
    "staged": ["deployed", "archived"],
    "deployed": ["archived", "rollback"],
    "archived": [],  # Terminal state
    "failed": ["training"],  # Can retry training
    "rollback": ["deployed", "archived"],  # Can rollback to previous version
}


class DeploymentEnvironment(str, Enum):
    """Deployment environment for model versions."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"


class ModelStage(str, Enum):
    """Lifecycle stage for a model version."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class InvalidStatusTransitionError(ValueError):
    """Raised when an invalid status transition is attempted."""

    pass


class SemanticVersion:
    """Semantic version parser and comparator."""

    def __init__(self, version: str):
        self.version = version
        self.major, self.minor, self.patch = self._parse(version)

    @staticmethod
    def _parse(version: str) -> tuple[int, int, int]:
        """Parse semantic version string into major, minor, patch."""
        parts = version.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid semantic version: {version}. Expected format: major.minor.patch"
            )

        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])
        except ValueError:
            raise ValueError(f"Invalid semantic version: {version}. All parts must be integers.")

        return major, minor, patch

    def __lt__(self, other: SemanticVersion) -> bool:
        """Compare if this version is less than another."""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return self.major == other.major and self.minor == other.minor and self.patch == other.patch

    def __repr__(self) -> str:
        return self.version


class ModelRegistry:
    """Core class for managing ML models and their versions in the database.

    Provides CRUD operations for Model and ModelVersion entities,
    with helper methods for common registry operations.
    """

    def __init__(self, session: Session | None = None):
        """Initialize the registry.

        Args:
            session: Optional SQLAlchemy session. If not provided, creates a new session.
        """
        self._session = session
        self._owns_session = session is None

    @property
    def session(self) -> Session:
        """Get the SQLAlchemy session, creating one if needed."""
        if self._session is None:
            self._session = get_session()
        return self._session

    def close(self) -> None:
        """Close the session if we own it."""
        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> ModelRegistry:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Model CRUD operations
    # ------------------------------------------------------------------

    def create_model(
        self,
        name: str,
        framework: str,
        task_type: str,
        description: str | None = None,
        is_active: bool = True,
    ) -> Model:
        """Create a new model.

        Args:
            name: Unique model name
            framework: ML framework (pytorch, tensorflow, sklearn, etc.)
            task_type: Task type (classification, regression, etc.)
            description: Optional model description
            is_active: Whether the model is active

        Returns:
            Created Model instance

        Raises:
            ValueError: If a model with the same name already exists
        """
        existing = self.get_model_by_name(name)
        if existing:
            raise ValueError(f"Model with name '{name}' already exists")

        model = Model(
            name=name,
            description=description,
            framework=framework,
            task_type=task_type,
            is_active=is_active,
        )
        self.session.add(model)
        self.session.commit()
        self.session.refresh(model)
        logger.info("Created model: %s (id=%d)", name, model.id)
        return model

    def get_model(self, model_id: int) -> Model | None:
        """Get a model by ID."""
        return self.session.get(Model, model_id)

    def get_model_by_name(self, name: str) -> Model | None:
        """Get a model by name."""
        stmt = select(Model).where(Model.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_models(
        self,
        framework: str | None = None,
        task_type: str | None = None,
        is_active: bool | None = None,
    ) -> list[Model]:
        """List models with optional filters."""
        stmt = select(Model)
        if framework:
            stmt = stmt.where(Model.framework == framework)
        if task_type:
            stmt = stmt.where(Model.task_type == task_type)
        if is_active is not None:
            stmt = stmt.where(Model.is_active == is_active)
        stmt = stmt.order_by(Model.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    def update_model(
        self,
        model_id: int,
        description: str | None = None,
        is_active: bool | None = None,
    ) -> Model | None:
        """Update a model."""
        model = self.get_model(model_id)
        if not model:
            return None

        if description is not None:
            model.description = description
        if is_active is not None:
            model.is_active = is_active

        self.session.commit()
        self.session.refresh(model)
        logger.info("Updated model: %s (id=%d)", model.name, model.id)
        return model

    def delete_model(self, model_id: int) -> bool:
        """Delete a model and all its versions."""
        model = self.get_model(model_id)
        if not model:
            return False

        self.session.delete(model)
        self.session.commit()
        logger.info("Deleted model: %s (id=%d)", model.name, model_id)
        return True

    # ------------------------------------------------------------------
    # ModelVersion CRUD operations
    # ------------------------------------------------------------------

    def _get_next_version(self, model_name: str) -> str:
        """Get the next semantic version for a model."""
        # Get all versions for this model
        stmt = select(ModelVersion).where(ModelVersion.model.has(name=model_name))
        versions = list(self.session.execute(stmt).scalars().all())

        if not versions:
            return "0.1.0"

        # Parse versions and find the latest
        latest = None
        for v in versions:
            try:
                semver = SemanticVersion(v.version)
                if latest is None or semver > latest:
                    latest = semver
            except ValueError:
                continue

        if latest is None:
            return "0.1.0"

        # Increment patch version
        return f"{latest.major}.{latest.minor}.{latest.patch + 1}"

    def create_model_version(
        self,
        model_id: int,
        artifact_path: str,
        hyperparameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        status: str = "training",
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
        auto_version: bool = True,
    ) -> ModelVersion:
        """Create a new model version.

        Args:
            model_id: Parent model ID
            artifact_path: Path to model artifacts
            hyperparameters: Optional hyperparameters dict
            metrics: Optional metrics dict
            status: Version status (training, trained, deployed, etc.)
            version: Optional version string. If not provided and auto_version=True, auto-generates.
            metadata: Optional additional metadata
            auto_version: Whether to auto-generate version if not provided

        Returns:
            Created ModelVersion instance

        Raises:
            ValueError: If model not found or version already exists
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model with id {model_id} not found")

        # Determine version
        if version:
            # Validate semantic version format
            SemanticVersion(version)
        elif auto_version:
            version = self._get_next_version(model.name)
        else:
            raise ValueError("Version must be provided when auto_version is False")

        # Check if version already exists
        existing = self.get_model_version(model_id, version)
        if existing:
            raise ValueError(f"Version '{version}' already exists for model {model_id}")

        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            artifact_path=artifact_path,
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            status=status,
            metadata=metadata or {},
        )
        self.session.add(model_version)
        self.session.commit()
        self.session.refresh(model_version)

        logger.info(
            "Created model version: %s (id=%d, model_id=%d)",
            version,
            model_version.id,
            model_id,
        )
        return model_version

    def get_model_version(self, model_id: int, version: str) -> ModelVersion | None:
        """Get a specific model version."""
        stmt = select(ModelVersion).where(
            ModelVersion.model_id == model_id,
            ModelVersion.version == version,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_model_version_by_id(self, version_id: int) -> ModelVersion | None:
        """Get a model version by ID."""
        return self.session.get(ModelVersion, version_id)

    def list_model_versions(
        self,
        model_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelVersion]:
        """List model versions with optional filters."""
        stmt = select(ModelVersion)
        if model_id:
            stmt = stmt.where(ModelVersion.model_id == model_id)
        if status:
            stmt = stmt.where(ModelVersion.status == status)
        stmt = stmt.order_by(ModelVersion.created_at.desc())
        stmt = stmt.offset(offset).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def update_model_version_status(
        self,
        version_id: int,
        new_status: str,
        validate_transition: bool = True,
        **kwargs,
    ) -> ModelVersion | None:
        """Update a model version status with optional validation."""
        version = self.get_model_version_by_id(version_id)
        if not version:
            return None

        if validate_transition:
            self._validate_status_transition(version.status, new_status)

        version.status = new_status
        for key, value in kwargs.items():
            if hasattr(version, key):
                setattr(version, key, value)

        # If status is "deployed", set deployed_at
        if new_status == "deployed" and not kwargs.get("deployed_at"):
            version.deployed_at = datetime.now(timezone.utc)

        self.session.commit()
        self.session.refresh(version)
        logger.info(
            "Updated model version status: %s -> %s (id=%d)",
            version.version,
            new_status,
            version_id,
        )
        return version

    def update_model_version_metrics(
        self,
        version_id: int,
        metrics: dict[str, Any],
    ) -> ModelVersion | None:
        """Update metrics for a model version."""
        version = self.get_model_version_by_id(version_id)
        if not version:
            return None

        # Reassign rather than mutate in place (issue #738).
        #
        # ``metrics`` is a plain JSON column, so SQLAlchemy compares it by
        # identity: an in-place ``dict.update`` leaves the attribute pointing
        # at the same object, the instance is never marked dirty, ``commit``
        # writes nothing, and the ``refresh`` below then reloads the old value
        # over the change. The update was discarded in silence — the caller
        # got a version object back and no error.
        version.metrics = {**(version.metrics or {}), **metrics}
        self.session.commit()
        self.session.refresh(version)
        logger.info("Updated metrics for model version: %s (id=%d)", version.version, version_id)
        return version

    def update_model_version_metadata(
        self,
        version_id: int,
        metadata: dict[str, Any],
    ) -> ModelVersion | None:
        """Update metadata for a model version."""
        version = self.get_model_version_by_id(version_id)
        if not version:
            return None

        version.metadata.update(metadata)
        self.session.commit()
        self.session.refresh(version)
        logger.info("Updated metadata for model version: %s (id=%d)", version.version, version_id)
        return version

    def delete_model_version(self, version_id: int) -> bool:
        """Delete a model version."""
        version = self.get_model_version_by_id(version_id)
        if not version:
            return False

        self.session.delete(version)
        self.session.commit()
        logger.info("Deleted model version: %s (id=%d)", version.version, version_id)
        return True

    # ------------------------------------------------------------------
    # Advanced operations
    # ------------------------------------------------------------------

    def get_latest_version(self, model_id: int) -> ModelVersion | None:
        """Get the latest version of a model by creation time."""
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id)
            .order_by(ModelVersion.created_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_latest_deployed_version(self, model_id: int) -> ModelVersion | None:
        """Get the latest deployed version of a model."""
        stmt = (
            select(ModelVersion)
            .where(
                ModelVersion.model_id == model_id,
                ModelVersion.status == "deployed",
            )
            .order_by(ModelVersion.deployed_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_version_by_semver(
        self, model_id: int, major: int, minor: int, patch: int
    ) -> ModelVersion | None:
        """Get a model version by semantic version components."""
        version_str = f"{major}.{minor}.{patch}"
        return self.get_model_version(model_id, version_str)

    # ── Serving activation / rollback (issue #718) ──────────────────────────
    #
    # Both transitions go through _switch_serving_version, which performs the
    # demote-and-promote as a single unit of work. The previous implementation
    # committed twice — outgoing version first, incoming second — so a failure
    # between them left the model with *no* deployed version at all. It also
    # recorded lineage by assigning to ``version.metadata``, which is
    # SQLAlchemy's reserved MetaData attribute rather than a column, so the
    # transition record was silently discarded.

    def _switch_serving_version(
        self,
        target: ModelVersion,
        transition: str,
        reason: str,
        actor: str | None = None,
    ) -> tuple[ModelVersion, ModelVersion | None]:
        """Atomically make ``target`` the deployed version for its model.

        Returns ``(target, previous)`` where ``previous`` is the version that was
        serving beforehand, or ``None`` if there was none.
        """
        previous = self.get_latest_deployed_version(target.model_id)
        if previous is not None and previous.id == target.id:
            raise ValueError(f"Version '{target.version}' is already deployed")

        now = datetime.now(timezone.utc)
        record: dict[str, Any] = {
            "transition": transition,
            "at": now.isoformat(),
            "reason": reason,
            "actor": actor,
            "from_version": previous.version if previous else None,
            "to_version": target.version,
        }

        # One transaction: either serving moves to the new version and both
        # lineage records land, or nothing changes at all.
        try:
            if previous is not None:
                previous.status = "archived"
                previous.lineage = _append_lineage(
                    previous.lineage, {**record, "role": "superseded"}
                )

            target.status = "deployed"
            target.deployed_at = now
            target.lineage = _append_lineage(target.lineage, {**record, "role": "activated"})

            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

        self.session.refresh(target)
        if previous is not None:
            self.session.refresh(previous)

        logger.info(
            "Serving switched to version %s for model %d (%s: %s)",
            target.version,
            target.model_id,
            transition,
            reason,
        )
        return target, previous

    def activate(
        self,
        model_id: int,
        version: str,
        reason: str = "Activation requested",
        actor: str | None = None,
    ) -> tuple[ModelVersion, ModelVersion | None]:
        """Make ``version`` the served version for ``model_id``.

        Args:
            model_id: Model ID.
            version: Version string to activate.
            reason: Why the switch is happening; recorded in lineage.
            actor: Who requested it; recorded in lineage.

        Returns:
            Tuple of (activated_version, previously_deployed_version_or_None).

        Raises:
            ValueError: If the version does not exist, or is already deployed.
        """
        target = self.get_model_version(model_id, version)
        if not target:
            raise ValueError(f"Version '{version}' not found for model {model_id}")

        return self._switch_serving_version(
            target, transition="activate", reason=reason, actor=actor
        )

    def rollback_to_version(
        self,
        model_id: int,
        target_version: str,
        reason: str = "Rollback requested",
        actor: str | None = None,
    ) -> tuple[ModelVersion, ModelVersion | None]:
        """
        Rollback serving to a previous version.

        Args:
            model_id: Model ID
            target_version: Version to rollback to
            reason: Reason for rollback
            actor: Who requested the rollback; recorded in lineage

        Returns:
            Tuple of (target_version, previously_deployed_version_or_None)

        Raises:
            ValueError: If target version not found or is already deployed
        """
        target = self.get_model_version(model_id, target_version)
        if not target:
            raise ValueError(f"Target version '{target_version}' not found")

        if target.status == "deployed":
            raise ValueError(f"Target version '{target_version}' is already deployed")

        return self._switch_serving_version(
            target, transition="rollback", reason=reason, actor=actor
        )

    def get_version_history(self, model_id: int, limit: int = 10) -> list[dict[str, Any]]:
        """Get version history with status transitions for a model."""
        versions = self.list_model_versions(model_id=model_id, limit=limit)

        history = []
        for version in versions:
            history.append(
                {
                    "id": version.id,
                    "version": version.version,
                    "status": version.status,
                    "metrics": version.metrics,
                    "created_at": version.created_at.isoformat(),
                    "deployed_at": version.deployed_at.isoformat() if version.deployed_at else None,
                    "metadata": version.metadata,
                }
            )

        return history

    def compare_versions(
        self,
        version_ids: list[int],
    ) -> dict[str, Any]:
        """Compare multiple model versions across metrics."""
        versions = []
        for vid in version_ids:
            v = self.get_model_version_by_id(vid)
            if v:
                versions.append(v)

        if len(versions) < 2:
            return {"error": "At least 2 versions required for comparison"}

        # Collect all metric keys
        all_metrics = set()
        for v in versions:
            if v.metrics:
                all_metrics.update(v.metrics.keys())

        comparison = {
            "versions": [{"id": v.id, "version": v.version, "status": v.status} for v in versions],
            "metrics": {},
            "summary": {},
        }

        for metric in sorted(all_metrics):
            values = {}
            for v in versions:
                values[v.version] = v.metrics.get(metric) if v.metrics else None

            # Find best version for this metric (higher is better)
            numeric_values = [(v, val) for v, val in values.items() if val is not None]
            if numeric_values:
                best = max(numeric_values, key=lambda x: x[1])
                worst = min(numeric_values, key=lambda x: x[1])
                comparison["summary"][metric] = {
                    "best_version": best[0],
                    "best_value": best[1],
                    "worst_version": worst[0],
                    "worst_value": worst[1],
                }

            comparison["metrics"][metric] = values

        return comparison

    # ------------------------------------------------------------------
    # A/B Testing support
    # ------------------------------------------------------------------

    def create_ab_test(
        self,
        model_id: int,
        control_version: str,
        treatment_version: str,
        traffic_split: float = 0.5,
        metrics: list[str] = None,
    ) -> dict[str, Any]:
        """
        Set up an A/B test between two model versions.

        Args:
            model_id: Model ID
            control_version: Version to use as control
            treatment_version: Version to use as treatment
            traffic_split: Traffic split for treatment (0-1)
            metrics: Metrics to track for comparison

        Returns:
            A/B test configuration

        Raises:
            ValueError: If versions not found or invalid traffic split
        """
        control = self.get_model_version(model_id, control_version)
        treatment = self.get_model_version(model_id, treatment_version)

        if not control:
            raise ValueError(f"Control version '{control_version}' not found")
        if not treatment:
            raise ValueError(f"Treatment version '{treatment_version}' not found")

        if not 0 < traffic_split < 1:
            raise ValueError(f"Traffic split must be between 0 and 1, got {traffic_split}")

        # Set status to staged for A/B testing
        control.status = "staged"
        treatment.status = "staged"

        # Store A/B test configuration
        ab_config = {
            "ab_test": {
                "control_version": control_version,
                "treatment_version": treatment_version,
                "traffic_split": traffic_split,
                "metrics": metrics or [],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "active",
            }
        }

        # Update both versions with A/B test metadata
        control.metadata = {**(control.metadata or {}), **ab_config}
        treatment.metadata = {**(treatment.metadata or {}), **ab_config}

        self.session.commit()

        logger.info(
            "A/B test created: control=%s, treatment=%s, traffic_split=%.2f",
            control_version,
            treatment_version,
            traffic_split,
        )

        return ab_config["ab_test"]

    def get_ab_test_results(
        self,
        model_id: int,
        ab_test_id: str = None,
    ) -> dict[str, Any]:
        """Get A/B test results for a model."""
        # Implementation would query metrics storage
        # This is a placeholder that returns mock data
        return {
            "message": "A/B test results retrieval - implementation pending",
            "versions": self.list_model_versions(model_id=model_id),
        }

    # ------------------------------------------------------------------
    # Deployment tracking
    # ------------------------------------------------------------------

    def track_deployment(
        self,
        version_id: int,
        environment: DeploymentEnvironment,
        deployed_by: str | None = None,
        notes: str | None = None,
    ) -> ModelVersion | None:
        """
        Track deployment of a model version.

        Args:
            version_id: ModelVersion ID
            environment: Deployment environment
            deployed_by: User who deployed
            notes: Deployment notes

        Returns:
            Updated ModelVersion instance or None if not found
        """
        version = self.get_model_version_by_id(version_id)
        if not version:
            return None

        deployment_record = {
            "environment": environment.value,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "deployed_by": deployed_by,
            "notes": notes,
        }

        # Update metadata with deployment info
        version.metadata = {
            **(version.metadata or {}),
            "deployments": [
                *(version.metadata.get("deployments", []) if version.metadata else []),
                deployment_record,
            ],
            "latest_deployment": deployment_record,
        }

        # If environment is production, mark version as deployed
        if environment == DeploymentEnvironment.PRODUCTION:
            version.status = "deployed"
            version.deployed_at = datetime.now(timezone.utc)

        self.session.commit()
        self.session.refresh(version)

        logger.info(
            "Deployed version %s to %s environment",
            version.version,
            environment.value,
        )

        return version

    def get_deployment_history(
        self,
        version_id: int,
    ) -> list[dict[str, Any]]:
        """Get deployment history for a model version."""
        version = self.get_model_version_by_id(version_id)
        if not version:
            return []

        return version.metadata.get("deployments", []) if version.metadata else []

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_status_transition(from_status: str, to_status: str) -> None:
        """Validate that a status transition is allowed."""
        if to_status not in VALID_STATUSES:
            raise InvalidStatusTransitionError(f"Invalid target status: '{to_status}'")

        if from_status == to_status:
            return

        allowed_transitions = VALID_STATUS_TRANSITIONS.get(from_status, [])
        if to_status not in allowed_transitions:
            raise InvalidStatusTransitionError(
                f"Cannot transition from '{from_status}' to '{to_status}'. "
                f"Allowed transitions from '{from_status}': {allowed_transitions}"
            )

    def is_valid_semantic_version(self, version: str) -> bool:
        """Check if a version string is a valid semantic version."""
        try:
            SemanticVersion(version)
            return True
        except ValueError:
            return False


def _append_lineage(existing: dict[str, Any] | None, entry: dict[str, Any]) -> dict[str, Any]:
    """Append a transition to a version's lineage record.

    A new dict is returned rather than mutating in place: SQLAlchemy only marks a
    JSON column dirty on assignment, so mutating the existing dict would not be
    persisted.
    """
    events = list((existing or {}).get("events", []))
    events.append(entry)
    return {**(existing or {}), "events": events, "latest": entry}
