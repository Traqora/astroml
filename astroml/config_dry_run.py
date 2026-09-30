from astroml.utils.exceptions import AstroMLError
"""Startup configuration dry-run validation.

Validates all configuration sources without starting any services.
Exits with non-zero status on the first validation failure.

Example:
    python -m astroml.cli config --dry-run
    python -m astroml.cli --config ./custom/db.yaml config --dry-run
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any

from astroml.db.session import DatabaseConfig, load_database_config

logger = logging.getLogger("astroml.config.dry_run")


@dataclass
class ValidationResult:
    """Result of a configuration validation check."""

    name: str
    # Defaults to False so a validator can construct its result first and set
    # the verdict once its checks pass. Every validator in this module does
    # exactly that — without a default they all raised TypeError on
    # construction, which made the whole dry-run path unrunnable (#719).
    valid: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def validate_database_config(config_path: pathlib.Path | None = None) -> ValidationResult:
    """Validate database configuration without connecting.

    Checks YAML schema, field types, and URL format without
    actually opening a database connection.
    """
    result = ValidationResult(name="database_config")
    try:
        db_config = load_database_config(config_path)
        result.details = {
            "host": db_config.host,
            "port": db_config.port,
            "name": db_config.name,
            "user": db_config.user,
            "pool_size": db_config.pool_size,
            "url": db_config.to_url(),
        }

        if db_config.password == "":
            result.warnings.append("Database password is empty")
        if db_config.host in ("localhost", "127.0.0.1"):
            result.warnings.append(
                f"Database host is '{db_config.host}' - ensure this is intentional"
            )

        result.valid = True
    except FileNotFoundError as e:
        result.errors.append(str(e))
        result.valid = False
    except ValueError as e:
        result.errors.append(str(e))
        result.valid = False
    except AstroMLError as e:
        result.errors.append(f"Unexpected error: {e}")
        result.valid = False

    return result


def validate_environment_variables() -> ValidationResult:
    """Validate required environment variables."""
    result = ValidationResult(name="environment_variables")

    required_vars: dict[str, str] = {
        "ASTROML_DATABASE_URL": "Database connection URL",
    }

    optional_vars: dict[str, str] = {
        "ASTROML_ENV": "Runtime environment",
        "ASTROML_LOG_LEVEL": "Logging level",
        "ASTROML_HORIZON_URL": "Stellar Horizon server URL",
        "ASTROML_REDIS_URL": "Redis connection URL",
        "ASTROML_PERSIST_CHUNK_SIZE": "Ingestion batch chunk size",
    }

    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value:
            result.details[var] = value
        else:
            result.warnings.append(
                f"{var} not set ({description}) - will use defaults"
            )

    for var, description in optional_vars.items():
        value = os.environ.get(var)
        if value:
            result.details[var] = value

    result.valid = True
    return result


def validate_imports() -> ValidationResult:
    """Validate that all required Python packages are importable."""
    result = ValidationResult(name="python_imports")

    required_modules = [
        ("sqlalchemy", "SQLAlchemy ORM"),
        ("alembic", "Database migrations"),
        ("pydantic", "Configuration validation"),
        ("prometheus_client", "Metrics collection"),
    ]

    optional_modules = [
        ("stellar_sdk", "Stellar blockchain SDK"),
        ("aiohttp", "Async HTTP client"),
        ("celery", "Distributed task queue"),
        ("redis", "Redis client"),
        ("torch", "PyTorch ML framework"),
    ]

    for module_name, description in required_modules:
        try:
            __import__(module_name)
            result.details[module_name] = "available"
        except ImportError as e:
            result.errors.append(
                f"Required module '{module_name}' ({description}) not found: {e}"
            )
            result.valid = False

    for module_name, description in optional_modules:
        try:
            __import__(module_name)
            result.details[module_name] = "available"
        except ImportError:
            result.warnings.append(
                f"Optional module '{module_name}' ({description}) not found"
            )

    if not result.errors:
        result.valid = True

    return result


def validate_file_paths() -> ValidationResult:
    """Validate that required directories and files exist or can be created."""
    result = ValidationResult(name="file_paths")

    required_paths = [
        pathlib.Path("config"),
        pathlib.Path("migrations"),
    ]

    writable_paths = [
        pathlib.Path("data"),
        pathlib.Path("logs"),
        pathlib.Path(".astroml_state"),
    ]

    for path in required_paths:
        if path.exists():
            result.details[str(path)] = "exists"
        else:
            result.errors.append(f"Required path '{path}' does not exist")
            result.valid = False

    for path in writable_paths:
        if path.exists():
            if os.access(path, os.W_OK):
                result.details[str(path)] = "exists, writable"
            else:
                result.errors.append(f"Path '{path}' exists but is not writable")
                result.valid = False
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                result.details[str(path)] = "created"
            except OSError as e:
                result.errors.append(f"Cannot create path '{path}': {e}")
                result.valid = False

    if not result.errors:
        result.valid = True

    return result


def run_dry_run(
    config_path: pathlib.Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run all configuration validations without starting services.

    Args:
        config_path: Optional path to database YAML config.
        verbose: Whether to print detailed output.

    Returns:
        Dictionary with validation results:
        - valid: bool indicating overall success
        - results: list of ValidationResult objects
        - exit_code: recommended exit code (0 or 1)
    """
    checks = [
        ("Database Config", lambda: validate_database_config(config_path)),
        ("Environment Variables", validate_environment_variables),
        ("Python Imports", validate_imports),
        ("File Paths", validate_file_paths),
    ]

    results: list[ValidationResult] = []
    all_valid = True

    for name, check_fn in checks:
        result = check_fn()
        results.append(result)

        if not result.valid:
            all_valid = False

        if verbose:
            status = "PASS" if result.valid else "FAIL"
            print(f"[{status}] {name}")
            if result.details:
                for key, value in result.details.items():
                    print(f"  {key}: {value}")
            for error in result.errors:
                print(f"  ERROR: {error}")
            for warning in result.warnings:
                print(f"  WARN: {warning}")

    return {
        "valid": all_valid,
        "results": results,
        "exit_code": 0 if all_valid else 1,
    }
