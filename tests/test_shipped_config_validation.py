"""Every shipped config file parses and passes validation (issue #719).

CI should catch a typo or an enum drift in a committed YAML file before it
reaches a startup path. These tests walk every ``*.yaml``/``*.yml`` under
``config/`` and ``configs/``, assert each one parses, and run the startup
dry-run validators over the configs that have schemas.

The sweep is deliberately data-driven: a config added later is covered
automatically, without anyone remembering to add a test for it.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIRS = ("config", "configs")


def _shipped_configs() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for directory in CONFIG_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        files.extend(sorted(root.rglob("*.yaml")))
        files.extend(sorted(root.rglob("*.yml")))
    return files


SHIPPED_CONFIGS = _shipped_configs()


def _ids(paths: list[pathlib.Path]) -> list[str]:
    return [str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in paths]


def test_the_repository_ships_config_files():
    # A sweep that silently matches nothing would pass forever while testing
    # nothing, so assert the corpus is non-empty before relying on it.
    assert SHIPPED_CONFIGS, "expected YAML config files under config/ and configs/"


@pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=_ids(SHIPPED_CONFIGS))
class TestEveryShippedConfigParses:
    def test_is_valid_yaml(self, config_path: pathlib.Path):
        with config_path.open(encoding="utf-8") as handle:
            try:
                yaml.safe_load(handle)
            except yaml.YAMLError as exc:
                pytest.fail(f"{config_path} is not valid YAML: {exc}")

    def test_is_not_empty(self, config_path: pathlib.Path):
        with config_path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        assert loaded is not None, f"{config_path} parses to nothing"

    def test_top_level_is_a_mapping(self, config_path: pathlib.Path):
        with config_path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        # Every consumer in the codebase indexes configs by key; a bare list or
        # scalar at the top level would fail at startup rather than at load.
        assert isinstance(loaded, dict), f"{config_path} must be a mapping at the top level"

    def test_has_no_duplicate_keys(self, config_path: pathlib.Path):
        """A duplicated key silently wins over its earlier twin in YAML."""
        # Composed against the node tree rather than loaded: constructing the
        # document would already have collapsed the duplicate away.
        with config_path.open(encoding="utf-8") as handle:
            root = yaml.compose(handle, Loader=yaml.SafeLoader)

        duplicates: list[str] = []

        def _visit(node, path: str = "") -> None:
            if isinstance(node, yaml.MappingNode):
                seen: set[str] = set()
                for key_node, value_node in node.value:
                    key = str(getattr(key_node, "value", key_node))
                    where = f"{path}.{key}" if path else key
                    if key in seen:
                        duplicates.append(where)
                    seen.add(key)
                    _visit(value_node, where)
            elif isinstance(node, yaml.SequenceNode):
                for index, item in enumerate(node.value):
                    _visit(item, f"{path}[{index}]")

        _visit(root)
        assert not duplicates, f"{config_path} has duplicate keys: {duplicates}"

    def test_uses_no_tab_indentation(self, config_path: pathlib.Path):
        """Tabs are illegal as YAML indentation and produce confusing errors."""
        for lineno, line in enumerate(
            config_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.split("#", 1)[0]
            indent = stripped[: len(stripped) - len(stripped.lstrip())]
            assert "\t" not in indent, f"{config_path}:{lineno} indents with a tab"


class TestDatabaseConfigSchema:
    """config/database.yaml must satisfy the loader used at startup."""

    def test_ships_a_database_config(self):
        assert (REPO_ROOT / "config" / "database.yaml").is_file()

    def test_passes_the_startup_dry_run(self):
        from astroml.config_dry_run import validate_database_config

        result = validate_database_config(REPO_ROOT / "config" / "database.yaml")

        assert result.valid, f"shipped database.yaml failed validation: {result.errors}"
        assert result.errors == []

    def test_dry_run_reports_the_parsed_values(self):
        from astroml.config_dry_run import validate_database_config

        result = validate_database_config(REPO_ROOT / "config" / "database.yaml")

        # Guards against a config that "validates" because everything fell back
        # to defaults after a mis-spelled section name.
        assert result.details["name"] == "astroml"
        assert result.details["port"] == 5432
        assert result.details["url"].startswith("postgresql")

    def test_a_missing_file_is_reported_not_raised(self, tmp_path):
        from astroml.config_dry_run import validate_database_config

        result = validate_database_config(tmp_path / "nope.yaml")

        assert result.valid is False
        assert result.errors

    def test_a_malformed_file_is_reported_not_raised(self, tmp_path):
        from astroml.config_dry_run import validate_database_config

        bad = tmp_path / "database.yaml"
        bad.write_text("database:\n  port: not-a-number\n", encoding="utf-8")

        result = validate_database_config(bad)

        assert result.valid is False
        assert result.errors

    def test_an_empty_file_is_reported_not_raised(self, tmp_path):
        from astroml.config_dry_run import validate_database_config

        empty = tmp_path / "database.yaml"
        empty.write_text("", encoding="utf-8")

        result = validate_database_config(empty)

        # Whatever the verdict, it must be a verdict — not an exception escaping
        # into the startup path.
        assert isinstance(result.valid, bool)


class TestEnumDrift:
    """Values constrained to a known set must still be in that set."""

    def test_logging_levels_are_valid(self):
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"}
        offenders: list[str] = []

        for path in SHIPPED_CONFIGS:
            with path.open(encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            for key, value in _walk(loaded):
                if key in {"level", "log_level", "logging_level"} and isinstance(value, str):
                    if value.upper() not in valid:
                        offenders.append(f"{path.name}: {key}={value}")

        assert not offenders, f"invalid logging levels: {offenders}"

    def test_boolean_fields_are_real_booleans(self):
        """Catches ``enabled: "true"`` — a string that is always truthy."""
        offenders: list[str] = []

        for path in SHIPPED_CONFIGS:
            with path.open(encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            for key, value in _walk(loaded):
                if key.startswith(("enable", "is_", "use_")) or key.endswith("_enabled"):
                    if isinstance(value, str) and value.lower() in {"true", "false", "yes", "no"}:
                        offenders.append(f"{path.name}: {key}={value!r}")

        assert not offenders, f"boolean-looking values quoted as strings: {offenders}"

    def test_no_placeholder_secrets_are_committed(self):
        """A committed real-looking secret is worse than an empty default."""
        suspicious = ("BEGIN RSA PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY")
        offenders: list[str] = []

        for path in SHIPPED_CONFIGS:
            text = path.read_text(encoding="utf-8")
            for marker in suspicious:
                if marker in text:
                    offenders.append(f"{path.name}: contains {marker}")

        assert not offenders, offenders


def _walk(node, prefix: str = ""):
    """Yield (key, value) for every scalar leaf in a nested mapping/sequence."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                yield from _walk(value, prefix=str(key))
            else:
                yield str(key), value
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, prefix=prefix)
