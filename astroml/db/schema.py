"""ORM re-exports and the Hydra config schema for the database layer (#992).

This module has two jobs:

1. **Backward-compatible re-exports** of every ORM model from
   :mod:`astroml.db.models` (issue #571). New code should import directly from
   :mod:`astroml.db.models`.
2. **The ``db`` config-group schema and its validators.** ADR-004 commits the
   project to Hydra/OmegaConf with "strong Python type safety through OmegaConf
   structured configs", but nothing checked a composed config before the
   database layer consumed it. Because
   :class:`astroml.db.session.DatabaseConfig` is a pydantic model with the
   default ``extra="ignore"`` policy, a mistyped key in a composed config
   (``db.prot=5432``, say) is dropped without a word: the run proceeds with the
   default port and the mistake only surfaces as a connection failure against
   the wrong database. The validators here turn that class of silent drop into
   an explicit, located error before anything connects.

Typical use from a Hydra entry point::

    from astroml.db.schema import assert_valid_config

    db_config = assert_valid_config(cfg)  # raises SchemaValidationError

Dependencies:
- astroml.db.models: ORM model definitions
- astroml.db.session: ``DatabaseConfig``, the contract this schema mirrors
- omegaconf: structured config nodes, and the configs Hydra composes into
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError

from astroml.db.models import *  # noqa: F401, F403
from astroml.db.models import Base
from astroml.db.session import DatabaseConfig

logger = logging.getLogger(__name__)


def validate_router_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Validate router weights and return them normalised to sum to 1.0 (issue #987).

    Raises:
        ValueError: if ``weights`` is empty, a key is blank, a weight is
            not a finite non-negative number, or all weights are zero.
    """
    if not weights:
        raise ValueError("router weights must not be empty")
    for name, weight in weights.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid route name: {name!r}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"weight for {name!r} must be a number, got {weight!r}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {name!r} must be finite and >= 0, got {weight!r}")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("router weights must not all be zero")
    return {name: float(weight) / total for name, weight in weights.items()}

DB_CONFIG_GROUP = "db"
"""Hydra config-group/package name the database schema is composed under."""

Severity = Literal["error", "warning"]

_PORT_MIN = 1
_PORT_MAX = 65535
_POSITIVE_FIELDS = ("pool_size", "pool_timeout", "pool_recycle")
_NON_NEGATIVE_FIELDS = ("max_overflow",)
_TABLE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class SchemaValidationError(ValueError):
    """Raised when a database config or the ORM metadata fails validation.

    Subclasses :class:`ValueError` so the existing
    :func:`astroml.db.session.load_database_config` error handling keeps
    catching it.

    Attributes:
        issues: Every :class:`ConfigIssue` found, in report order.
    """

    def __init__(self, issues: Sequence[ConfigIssue]) -> None:
        self.issues = tuple(issues)
        detail = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(f"database schema validation failed:\n{detail}")


@dataclass(frozen=True)
class ConfigIssue:
    """A single validation finding, anchored to the config path that caused it.

    Attributes:
        location: Dotted path of the offending value, e.g. ``db.pool_size``.
            Empty for findings about the ORM metadata as a whole.
        message: Human-readable description of what is wrong.
        severity: ``"error"`` blocks startup, ``"warning"`` does not.
    """

    location: str
    message: str
    severity: Severity = "error"

    def __str__(self) -> str:
        """Render the issue as ``location: message [severity]``."""
        where = self.location or "<root>"
        return f"{where}: {self.message} [{self.severity}]"


@dataclass
class ValidationReport:
    """Aggregate verdict for one validation run.

    Attributes:
        name: Identifier of what was validated, e.g. ``"db_config"``.
        issues: All findings, errors and warnings alike.
    """

    name: str = "db_schema"
    issues: list[ConfigIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ConfigIssue]:
        """Findings that block startup."""
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ConfigIssue]:
        """Findings that are advisory only."""
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def valid(self) -> bool:
        """True when no error-severity finding was recorded."""
        return not self.errors

    def __str__(self) -> str:
        """Render the report as a one-line summary."""
        verdict = "valid" if self.valid else "invalid"
        return f"{self.name}: {verdict} ({len(self.errors)} errors, {len(self.warnings)} warnings)"


@dataclass
class DatabaseSchema:
    """OmegaConf structured config for the ``db`` config group (issue #992).

    Field names, types and defaults mirror
    :class:`astroml.db.session.DatabaseConfig` one-to-one, so a node merged
    against this schema can be handed straight to ``DatabaseConfig`` without key
    translation. Keeping the two in step is the point: a field added to the
    pydantic model and not here is reported as an unknown key, and a field added
    only here is reported by :func:`assert_valid_config` as unusable.
    """

    host: str = "localhost"
    port: int = 5432
    name: str = "astroml"
    user: str = "astroml"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 1800


def db_schema_config() -> DictConfig:
    """Return a fresh ``db`` config node carrying the schema defaults.

    Returns:
        A ``DictConfig`` built from :class:`DatabaseSchema`, suitable as a
        Hydra default or as the merge target for a user-supplied node.
    """
    return OmegaConf.structured(DatabaseSchema)


def config_group_node(cfg: Any, group: str = DB_CONFIG_GROUP) -> tuple[DictConfig, str]:
    """Locate the database sub-config inside a composed Hydra config.

    Args:
        cfg: A composed ``DictConfig``, a plain mapping, or a bare ``db`` node.
        group: Config-group key to descend into. When ``cfg`` has no such key
            the mapping itself is treated as the database node, which is what
            lets callers validate a standalone ``db`` block unchanged.

    Returns:
        Tuple of the database node and the dotted prefix that findings are
        reported against (empty for a bare node).

    Raises:
        SchemaValidationError: If ``cfg`` is not a mapping, or if the ``group``
            key holds something that is not a mapping.
    """
    node = _as_mapping(cfg)
    if group not in node:
        return node, ""
    child = node[group]
    if not isinstance(child, (DictConfig, Mapping)):
        raise SchemaValidationError(
            [ConfigIssue(group, f"must be a mapping, got {type(child).__name__}")]
        )
    return _as_mapping(child), f"{group}."


def validate_config(cfg: Any, group: str = DB_CONFIG_GROUP) -> ValidationReport:
    """Validate a composed database config against the ``db`` schema.

    Checks, in order: unknown keys (which pydantic would silently ignore),
    value types via an OmegaConf structured merge, numeric ranges, and finally
    the pydantic model itself so that a rule living only in
    :class:`~astroml.db.session.DatabaseConfig` is still enforced.

    Args:
        cfg: A composed ``DictConfig``, a plain mapping, or a bare ``db`` node.
        group: Config-group key to descend into; see :func:`config_group_node`.

    Returns:
        A :class:`ValidationReport`. A merely invalid config is reported, not
        raised; use :func:`assert_valid_config` for the raising variant.
    """
    _, issues = _check_config(cfg, group)
    report = ValidationReport(name="db_config", issues=issues)
    _log_report(report)
    return report


def validate_metadata() -> ValidationReport:
    """Validate that the ORM ``Base.metadata`` is internally consistent (#992).

    Checks every table registered on the declarative base for the invariants the
    database layer relies on: a snake_case name usable as a SQL identifier, a
    primary key, and at least one column. Properties are asserted rather than a
    hardcoded table list, so a model added later is covered the moment it is
    declared, with no edit here.

    Returns:
        A :class:`ValidationReport` over the metadata findings.
    """
    report = ValidationReport(name="db_metadata")
    tables = Base.metadata.tables
    if not tables:
        report.issues.append(
            ConfigIssue("", "no tables registered on Base.metadata", severity="warning")
        )
    for name in sorted(tables):
        table = tables[name]
        if not _TABLE_NAME_RE.match(name):
            report.issues.append(
                ConfigIssue(name, f"table name {name!r} is not a snake_case SQL identifier")
            )
        if not table.primary_key.columns:
            report.issues.append(ConfigIssue(name, "table has no primary key"))
        if not table.columns:
            report.issues.append(ConfigIssue(name, "table has no columns"))
    _log_report(report)
    return report


def validate_all(cfg: Any, group: str = DB_CONFIG_GROUP) -> ValidationReport:
    """Run every database-layer validation and merge the findings.

    Args:
        cfg: A composed ``DictConfig``, a plain mapping, or a bare ``db`` node.
        group: Config-group key to descend into; see :func:`config_group_node`.

    Returns:
        A :class:`ValidationReport` covering both the config and the ORM
        metadata.
    """
    report = ValidationReport(name="db_schema")
    report.issues.extend(validate_config(cfg, group).issues)
    report.issues.extend(validate_metadata().issues)
    _log_report(report)
    return report


def assert_valid_config(cfg: Any, group: str = DB_CONFIG_GROUP) -> DatabaseConfig:
    """Validate a composed database config and return the resulting model.

    Args:
        cfg: A composed ``DictConfig``, a plain mapping, or a bare ``db`` node.
        group: Config-group key to descend into; see :func:`config_group_node`.

    Returns:
        A validated :class:`~astroml.db.session.DatabaseConfig` carrying the
        schema default for every key the config did not set.

    Raises:
        SchemaValidationError: If any error-severity finding was recorded.
    """
    merged, issues = _check_config(cfg, group)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors or merged is None:
        raise SchemaValidationError(errors or [ConfigIssue("", "config could not be validated")])
    return DatabaseConfig(**_container(merged))


def _check_config(cfg: Any, group: str) -> tuple[DictConfig | None, list[ConfigIssue]]:
    """Run every config check and return the merged node plus the findings."""
    node, prefix = config_group_node(cfg, group)
    issues = _unknown_key_issues(node, prefix)
    merged = _merge_known_keys(node, prefix, issues)
    if merged is not None:
        issues.extend(_range_issues(merged, prefix))
        issues.extend(_pydantic_issues(merged, prefix))
    return merged, _dedupe_by_location(issues)


def _dedupe_by_location(issues: list[ConfigIssue]) -> list[ConfigIssue]:
    """Keep the first finding per location, preserving report order.

    ``port`` is bounded both here and in ``DatabaseConfig``; without this the
    same mistake would be reported twice at the same path, which reads as two
    problems rather than as one. The local check runs first and is the more
    specific of the two, so it is the one that survives.
    """
    seen: set[str] = set()
    unique: list[ConfigIssue] = []
    for issue in issues:
        if issue.location in seen:
            continue
        seen.add(issue.location)
        unique.append(issue)
    return unique


def _as_mapping(cfg: Any) -> DictConfig:
    """Return ``cfg`` as a ``DictConfig``, whatever container it arrived in.

    Raises:
        SchemaValidationError: If ``cfg`` is not a mapping.
    """
    if isinstance(cfg, DictConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return OmegaConf.create(dict(cfg))
    raise SchemaValidationError([ConfigIssue("", f"expected a mapping, got {type(cfg).__name__}")])


def _container(node: DictConfig) -> dict[str, Any]:
    """Resolve ``node`` to a plain ``dict`` for keyword construction."""
    resolved = OmegaConf.to_container(node, resolve=True)
    if not isinstance(resolved, dict):
        raise SchemaValidationError(
            [ConfigIssue("", f"expected a mapping, got {type(resolved).__name__}")]
        )
    return resolved


def _known_only(node: DictConfig) -> DictConfig:
    """Return ``node`` with every key the schema does not model removed."""
    allowed = _schema_keys()
    return OmegaConf.create({key: node[key] for key in node.keys() if key in allowed})


def _schema_keys() -> frozenset[str]:
    """Return the field names modelled by :class:`DatabaseSchema`."""
    return frozenset(DatabaseSchema.__dataclass_fields__)


def _unknown_key_issues(node: DictConfig, prefix: str) -> list[ConfigIssue]:
    """Report keys pydantic would silently ignore when building DatabaseConfig."""
    allowed = _schema_keys()
    return [
        ConfigIssue(
            f"{prefix}{key}",
            f"unknown config key {key!r}; expected one of {sorted(allowed)}",
        )
        for key in sorted(node.keys())
        if key not in allowed
    ]


def _merge_known_keys(node: DictConfig, prefix: str, sink: list[ConfigIssue]) -> DictConfig | None:
    """Merge the schema-modelled keys into the structured config.

    OmegaConf raises when the schema cannot hold a value (a string where an int
    belongs, a list where a scalar belongs). Unknown keys are filtered out first
    so that :func:`_unknown_key_issues` reports each of them once instead of
    the merge aborting on the first.

    Args:
        node: The database config node.
        prefix: Location prefix for any issue recorded.
        sink: List that collected issues are appended to.

    Returns:
        The merged node, or ``None`` when a type error made the merge
        impossible.
    """
    try:
        return OmegaConf.merge(db_schema_config(), _known_only(node))
    except OmegaConfBaseException as exc:
        message = str(exc).splitlines()
        sink.append(
            ConfigIssue(
                f"{prefix}{getattr(exc, 'key', None) or 'db'}", message[0] if message else ""
            )
        )
        logger.debug("database config merge failed for %s: %s", prefix, exc)
        return None


def _range_issues(merged: DictConfig, prefix: str) -> list[ConfigIssue]:
    """Report numeric ranges the database layer depends on.

    pydantic already bounds ``port``, but the pool fields are unbounded there,
    so a zero or negative pool size only fails inside SQLAlchemy at engine
    construction, far away from the config that caused it.
    """
    issues: list[ConfigIssue] = []
    port = merged.get("port")
    if isinstance(port, int) and not _PORT_MIN <= port <= _PORT_MAX:
        issues.append(
            ConfigIssue(f"{prefix}port", f"must be within {_PORT_MIN}..{_PORT_MAX}, got {port}")
        )
    for field_name in _POSITIVE_FIELDS:
        value = merged.get(field_name)
        if isinstance(value, int) and value <= 0:
            issues.append(
                ConfigIssue(f"{prefix}{field_name}", f"must be a positive integer, got {value}")
            )
    for field_name in _NON_NEGATIVE_FIELDS:
        value = merged.get(field_name)
        if isinstance(value, int) and value < 0:
            issues.append(
                ConfigIssue(f"{prefix}{field_name}", f"must not be negative, got {value}")
            )
    return issues


def _pydantic_issues(merged: DictConfig, prefix: str) -> list[ConfigIssue]:
    """Re-validate through ``DatabaseConfig`` so its rules are never skipped.

    ``DatabaseConfig`` owns the rules this module does not restate, such as
    rejecting a whitespace-only ``name``; deferring to it keeps a single source
    of truth for what a valid database config is.
    """
    try:
        DatabaseConfig(**_container(merged))
    except (ValidationError, SchemaValidationError) as exc:
        return [_pydantic_issue(prefix, exc)]
    return []


def _pydantic_issue(prefix: str, exc: Exception) -> ConfigIssue:
    """Flatten a pydantic (or structural) error into one located issue."""
    if not isinstance(exc, ValidationError) or not exc.errors():
        return ConfigIssue(prefix.rstrip("."), str(exc).splitlines()[0])
    first = exc.errors()[0]
    where = "".join(f"{prefix}{part}." for part in first.get("loc", ()))
    return ConfigIssue(where.rstrip("."), str(first.get("msg", "invalid value")))


def _log_report(report: ValidationReport) -> None:
    """Emit the report through the module logger, one line per finding.

    Structured logging per CONTRIBUTING: a bare ``print`` would bypass the
    handler configuration a running pipeline has already set up, and the
    requirement for this change is explicit that findings are logged rather
    than swallowed.
    """
    for issue in report.issues:
        logger.log(
            logging.ERROR if issue.severity == "error" else logging.WARNING,
            "database schema validation: %s",
            issue,
        )
    logger.info("database schema validation complete: %s", report)
