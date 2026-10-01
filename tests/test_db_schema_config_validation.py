"""Regression tests for the Hydra config validation in ``astroml.db.schema`` (#992).

The gap these pin: ADR-004 adopts OmegaConf structured configs for type safety,
but the database layer never checked a composed config. Because
:class:`astroml.db.session.DatabaseConfig` is a pydantic model with the default
``extra="ignore"`` policy, a key Hydra cannot see (an inlined typo, a YAML file
loaded outside a config group) is dropped without a word and the run quietly
uses the default. The tests below cover the three places that can go wrong:

- the schema and the pydantic model drifting apart,
- a valid config group surviving composition and CLI overrides,
- every class of invalid config being *reported* rather than swallowed.
"""

from __future__ import annotations

import pathlib
from typing import Any, Iterator

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from astroml.db.models import Base
from astroml.db.schema import (
    DB_CONFIG_GROUP,
    ConfigIssue,
    DatabaseSchema,
    SchemaValidationError,
    assert_valid_config,
    config_group_node,
    db_schema_config,
    validate_all,
    validate_config,
    validate_metadata,
)
from astroml.db.session import DatabaseConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIPPED_DATABASE_YAML = REPO_ROOT / "config" / "database.yaml"

# The keys a valid `db` group may set. Kept literal so that a field renamed on
# the pydantic model has to be acknowledged here, rather than both sides
# drifting together under a test that compares the schema to itself.
EXPECTED_DB_KEYS = {
    "host",
    "port",
    "name",
    "user",
    "password",
    "pool_size",
    "max_overflow",
    "pool_timeout",
    "pool_recycle",
}


@pytest.fixture
def hydra_config_dir(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield a throwaway Hydra config dir with a valid ``db`` group.

    A real config group is what Hydra places in struct mode, which rejects an
    unknown key during composition. The invalid-config tests therefore use
    non-struct nodes instead; this fixture covers the other half of the story.
    """
    conf_dir = tmp_path / "conf"
    (conf_dir / "db").mkdir(parents=True)
    # Written as literal text, not yaml.safe_dump: that quotes the entry as
    # `- 'db: default'`, which Hydra reads as a config *name* rather than a
    # group/option pair, and then it looks for a file that cannot exist.
    (conf_dir / "config.yaml").write_text(
        "defaults:\n  - db: default\n  - _self_\n\nexperiment:\n  name: probe\n",
        encoding="utf-8",
    )
    (conf_dir / "db" / "default.yaml").write_text(
        yaml.safe_dump({key: _yaml_safe(value) for key, value in _defaults().items()}),
        encoding="utf-8",
    )
    yield conf_dir
    GlobalHydra.instance().clear()


def _defaults() -> dict[str, Any]:
    """Return the schema defaults as a plain dict."""
    return dict(OmegaConf.to_container(db_schema_config()))


def _yaml_safe(value: Any) -> Any:
    """Return ``value`` in a form ``yaml.safe_dump`` can serialise."""
    return "" if value is None else value


def _locations(issues: list[ConfigIssue]) -> set[str]:
    return {issue.location for issue in issues}


class TestSchemaMatchesTheDatabaseConfigContract:
    """``DatabaseSchema`` and ``DatabaseConfig`` must describe the same fields."""

    def test_schema_models_exactly_the_expected_keys(self):
        assert set(DatabaseSchema.__dataclass_fields__) == EXPECTED_DB_KEYS

    def test_every_database_config_field_is_modelled(self):
        # The whole value of the schema is that a key accepted by pydantic is
        # also a key this module knows about; otherwise a new pool setting could
        # be configured and still escape validation.
        assert set(DatabaseConfig.model_fields) == set(DatabaseSchema.__dataclass_fields__)

    def test_defaults_agree_with_the_pydantic_defaults(self):
        schema_defaults = _defaults()
        for name, expected in schema_defaults.items():
            assert getattr(DatabaseConfig(), name) == expected, name

    def test_db_schema_config_is_a_fresh_node_each_call(self):
        first = db_schema_config()
        first.port = 6543
        assert db_schema_config().port == 5432


class TestOrmMetadata:
    """The invariants ``validate_metadata`` claims are invariants."""

    def test_the_shipped_metadata_is_valid(self):
        report = validate_metadata()

        assert report.valid, f"metadata findings: {report.issues}"
        assert report.issues == []

    def test_every_table_is_named_and_keyed(self):
        tables = Base.metadata.tables
        assert tables, "expected the declarative base to carry tables"

        for name, table in tables.items():
            assert name == name.lower(), name
            assert table.primary_key.columns, f"{name} has no primary key"
            assert table.columns, f"{name} has no columns"

    def test_reporting_is_not_vacuous(self):
        # A validator that reports nothing for an empty registry would be
        # indistinguishable from one that always passes, so a warning is
        # required to tell the two apart.
        metadata = Base.metadata
        original = metadata.tables
        try:
            metadata.tables = {}
            report = validate_metadata()
        finally:
            metadata.tables = original

        assert report.valid is True
        assert [issue.severity for issue in report.issues] == ["warning"]


class TestValidConfigsPass:
    def test_the_schema_defaults_are_a_valid_config(self):
        report = validate_config(db_schema_config())

        assert report.valid, report.issues

    def test_assert_returns_a_populated_database_config(self):
        config = assert_valid_config({"db": {"host": "db.internal", "port": 5433}})

        assert isinstance(config, DatabaseConfig)
        assert config.host == "db.internal"
        assert config.port == 5433
        # Unset keys fall back to the schema default, not to a surprise.
        assert config.pool_size == DatabaseSchema().pool_size

    def test_a_bare_db_node_needs_no_group_key(self):
        node, prefix = config_group_node({"port": 6000})

        assert prefix == ""
        assert node["port"] == 6000
        assert validate_config({"port": 6000}).valid

    def test_the_group_key_is_honoured(self):
        cfg = OmegaConf.create({"database": {"port": 6000}, "db": {"port": 7000}})

        assert validate_config(cfg).valid
        assert assert_valid_config(cfg).port == 7000
        assert assert_valid_config(cfg, group="database").port == 6000

    def test_interpolations_are_resolved_before_validation(self):
        cfg = OmegaConf.create({"the_port": 7777, "db": {"port": "${the_port}"}})

        assert assert_valid_config(cfg).port == 7777

    def test_a_composed_hydra_config_validates(self, hydra_config_dir: pathlib.Path):
        with initialize_config_dir(config_dir=str(hydra_config_dir), version_base=None):
            cfg = compose(config_name="config")

        report = validate_config(cfg)

        assert report.valid, report.issues
        assert assert_valid_config(cfg).name == "astroml"

    def test_cli_overrides_reach_the_validated_config(self, hydra_config_dir: pathlib.Path):
        with initialize_config_dir(config_dir=str(hydra_config_dir), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["db.port=7000", "db.pool_size=3", "db.host=db.pooled"],
            )

        config = assert_valid_config(cfg)

        assert (config.port, config.pool_size, config.host) == (7000, 3, "db.pooled")

    def test_a_composed_override_out_of_range_is_reported(self, hydra_config_dir: pathlib.Path):
        # Hydra's struct mode rejects unknown keys but not out-of-range values,
        # so an override such as this composes cleanly and has to be caught
        # here instead.
        with initialize_config_dir(config_dir=str(hydra_config_dir), version_base=None):
            cfg = compose(config_name="config", overrides=["db.pool_size=0"])

        report = validate_config(cfg)

        assert report.valid is False
        assert "db.pool_size" in _locations(report.errors)


class TestInvalidConfigsAreReported:
    """Every case here is a config pydantic would accept and silently misapply."""

    def test_an_unknown_key_is_reported(self):
        # The bug this pins: `prot` alongside a valid `port` is dropped by
        # pydantic's extra="ignore", and the run connects to the default port.
        cfg = OmegaConf.create({"db": {"port": 5432, "prot": 5432}})

        report = validate_config(cfg)

        assert report.valid is False
        assert "db.prot" in _locations(report.errors)

    def test_a_bare_node_reports_unknown_keys_without_a_prefix(self):
        assert "prot" in _locations(validate_config({"prot": 1}).errors)

    def test_a_string_where_an_int_belongs_is_reported(self):
        report = validate_config({"db": {"port": "not-a-number"}})

        assert report.valid is False
        assert "db.port" in _locations(report.errors)

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"port": 0}, "db.port"),
            ({"port": 70000}, "db.port"),
            ({"pool_size": 0}, "db.pool_size"),
            ({"pool_timeout": -1}, "db.pool_timeout"),
            ({"pool_recycle": 0}, "db.pool_recycle"),
            ({"max_overflow": -1}, "db.max_overflow"),
        ],
    )
    def test_out_of_range_numbers_are_reported(self, overrides: dict, expected: str):
        report = validate_config({"db": overrides})

        assert report.valid is False
        assert expected in _locations(report.errors)

    def test_zero_max_overflow_is_allowed(self):
        # Disabling overflow is a legitimate choice, so it must not be treated
        # as the same mistake as a non-positive pool size.
        assert validate_config({"db": {"max_overflow": 0}}).valid

    @pytest.mark.parametrize("field_name", ["name", "user", "host"])
    def test_blank_strings_are_reported(self, field_name: str):
        # These three rules live only in DatabaseConfig; the validator defers
        # to it so the rule stays stated in one place.
        report = validate_config({"db": {field_name: "   "}})

        assert report.valid is False
        assert f"db.{field_name}" in _locations(report.errors)

    def test_an_empty_password_is_allowed(self):
        assert validate_config({"db": {"password": ""}}).valid

    def test_a_list_where_a_scalar_belongs_is_reported(self):
        report = validate_config({"db": {"name": ["astroml"]}})

        assert report.valid is False
        assert "db.name" in _locations(report.errors)

    def test_each_location_is_reported_once(self):
        # `port` is bounded here and in DatabaseConfig; two messages for one
        # mistake reads as two mistakes.
        report = validate_config({"db": {"port": 70000}})

        assert _locations(report.errors) == {"db.port"}

    def test_severity_of_a_finding_is_error(self):
        report = validate_config({"db": {"prot": 1}})

        assert report.warnings == []
        assert all(issue.severity == "error" for issue in report.errors)

    def test_the_issue_renders_its_location(self):
        issue = ConfigIssue("db.port", "must be within 1..65535, got 70000")

        assert "db.port" in str(issue)
        assert "65535" in str(issue)
        assert "error" in str(issue)

    def test_findings_are_logged_rather_than_swallowed(self, caplog):
        with caplog.at_level("ERROR", logger="astroml.db.schema"):
            report = validate_config({"db": {"prot": 1}})

        assert report.valid is False
        assert "db.prot" in caplog.text


class TestRaisingVariant:
    def test_assert_raises_on_an_unknown_key(self):
        with pytest.raises(SchemaValidationError) as excinfo:
            assert_valid_config({"db": {"prot": 5432}})

        assert "db.prot" in str(excinfo.value)

    def test_the_error_carries_every_issue(self):
        with pytest.raises(SchemaValidationError) as excinfo:
            assert_valid_config({"db": {"prot": 1, "port": 70000}})

        assert _locations(excinfo.value.issues) == {"db.prot", "db.port"}

    def test_the_error_is_a_value_error(self):
        # load_database_config's callers already handle ValueError, so the new
        # failure mode has to stay inside that contract.
        assert issubclass(SchemaValidationError, ValueError)

        with pytest.raises(ValueError):
            assert_valid_config({"db": {"prot": 1}})

    @pytest.mark.parametrize("cfg", [["a-list"], "a-string", 7, None])
    def test_a_non_mapping_config_is_rejected(self, cfg: Any):
        with pytest.raises(SchemaValidationError):
            assert_valid_config(cfg)

    def test_a_group_that_is_not_a_mapping_is_rejected(self):
        with pytest.raises(SchemaValidationError) as excinfo:
            assert_valid_config(OmegaConf.create({"db": 5}))

        assert DB_CONFIG_GROUP in _locations(excinfo.value.issues)


class TestValidateAll:
    def test_a_valid_config_yields_no_findings(self):
        report = validate_all({"db": {"host": "db.internal"}})

        assert report.valid, report.issues
        assert report.issues == []

    def test_metadata_and_config_findings_are_merged(self):
        report = validate_all({"db": {"prot": 1}})

        assert report.valid is False
        assert "db.prot" in _locations(report.errors)


class TestShippedDatabaseYaml:
    """What the shipped ``config/database.yaml`` means to this schema."""

    def test_the_shipped_file_exists(self):
        assert SHIPPED_DATABASE_YAML.is_file()

    def test_its_connection_keys_are_valid(self):
        loaded = yaml.safe_load(SHIPPED_DATABASE_YAML.read_text(encoding="utf-8"))

        report = validate_config(loaded, group="database")

        # host/port/name/user/password are what the loader actually reads, and
        # they must keep validating cleanly.
        assert _locations(report.errors) <= {"database.pool", "database.health_check"}

    def test_the_nested_pool_block_is_reported_as_unmodelled(self):
        # Recorded here rather than fixed here: nothing in the database layer
        # reads `database.pool`, so every value in it is inert today and the
        # schema says so instead of pretending otherwise. Flattening the file
        # is a separate change with its own migration story.
        loaded = yaml.safe_load(SHIPPED_DATABASE_YAML.read_text(encoding="utf-8"))

        report = validate_config(loaded, group="database")

        assert _locations(report.errors) == {"database.pool", "database.health_check"}
        assert all("unknown config key" in issue.message for issue in report.errors)

    def test_the_flat_template_in_the_loader_docstring_is_valid(self):
        # session._database_yaml_template is what an operator is told to copy
        # when their config is rejected, so it must satisfy the schema itself.
        from astroml.db.session import _database_yaml_template

        parsed = yaml.safe_load(_database_yaml_template())

        assert validate_config(
            parsed, group="database"
        ).valid, "the schema-by-example printed in database config errors is itself invalid"


class TestContainerRoundTrip:
    @pytest.mark.parametrize(
        "config",
        [
            db_schema_config(),
            OmegaConf.create({"db": _defaults()}),
        ],
        ids=["defaults", "composed"],
    )
    def test_the_defaults_survive_the_round_trip(self, config: DictConfig):
        merged = assert_valid_config(config)

        assert merged == DatabaseConfig(**_defaults())
        assert merged.to_url().startswith("postgresql://")

    def test_a_partial_config_keeps_its_own_values(self):
        merged = assert_valid_config({"db": {"host": "h", "port": 6543, "max_overflow": 0}})

        assert (merged.host, merged.port, merged.max_overflow) == ("h", 6543, 0)
        assert merged.pool_size == DatabaseConfig().pool_size
