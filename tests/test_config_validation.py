"""Tests for configuration layer validation (issue #575)."""
from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from main import DatabaseSettings, GraphSettings, Settings, TrainingSettings, get_settings, validate_config


class TestDatabaseSettings:
    def test_valid_postgres_url(self):
        settings = DatabaseSettings(url="postgresql://user:pass@localhost:5432/astroml")
        assert "postgresql" in settings.url

    def test_valid_sqlite_url(self):
        settings = DatabaseSettings(url="sqlite:///./test.db")
        assert "sqlite" in settings.url

    def test_invalid_url_scheme(self):
        with pytest.raises(ValidationError, match="Invalid database URL scheme"):
            DatabaseSettings(url="mysql://user:pass@localhost/db")

    def test_postgres_url_missing_db_name(self):
        with pytest.raises(ValidationError, match="must include a database name"):
            DatabaseSettings(url="postgresql://user:pass@localhost:5432/")

    def test_pool_size_positive(self):
        settings = DatabaseSettings(pool_size=5)
        assert settings.pool_size == 5

    def test_pool_size_zero_raises(self):
        with pytest.raises(ValidationError):
            DatabaseSettings(pool_size=0)

    def test_default_pool_size(self):
        settings = DatabaseSettings()
        assert settings.pool_size == 10


class TestTrainingSettings:
    def test_valid_learning_rate(self):
        settings = TrainingSettings(learning_rate=0.01, epochs=100)
        assert settings.learning_rate == 0.01

    def test_learning_rate_zero_raises(self):
        with pytest.raises(ValidationError):
            TrainingSettings(learning_rate=0.0, epochs=100)

    def test_learning_rate_negative_raises(self):
        with pytest.raises(ValidationError):
            TrainingSettings(learning_rate=-0.1, epochs=100)

    def test_valid_epochs(self):
        settings = TrainingSettings(epochs=100)
        assert settings.epochs == 100

    def test_epochs_zero_raises(self):
        with pytest.raises(ValidationError):
            TrainingSettings(epochs=0)

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValidationError):
            TrainingSettings(batch_size=0)


class TestGraphSettings:
    def test_valid_window_duration(self):
        settings = GraphSettings(window_duration_minutes=60)
        assert settings.window_duration_minutes == 60

    def test_window_duration_property(self):
        settings = GraphSettings(window_duration_minutes=30)
        assert isinstance(settings.window_duration, timedelta)
        assert settings.window_duration == timedelta(minutes=30)

    def test_window_duration_zero_raises(self):
        with pytest.raises(ValidationError):
            GraphSettings(window_duration_minutes=0)


class TestSettings:
    def test_valid_port(self):
        settings = Settings(port=8080)
        assert settings.port == 8080

    def test_port_too_low_raises(self):
        with pytest.raises(ValidationError, match="not in valid range"):
            Settings(port=800)

    def test_port_too_high_raises(self):
        with pytest.raises(ValidationError, match="not in valid range"):
            Settings(port=99999)

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValidationError, match="Invalid log level"):
            Settings(log_level="TRACE")

    def test_valid_log_level(self):
        settings = Settings(log_level="DEBUG")
        assert settings.log_level == "DEBUG"

    def test_log_level_case_insensitive(self):
        settings = Settings(log_level="info")
        assert settings.log_level == "INFO"

    def test_default_values(self):
        settings = Settings()
        assert settings.app_name == "AstroML Dashboard API"
        assert settings.port == 8000


class TestGetSettings:
    def test_returns_settings_instance(self):
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_settings_are_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


class TestValidateConfig:
    def test_valid_config_returns_success(self):
        result = validate_config()
        assert result["valid"] is True
        assert "settings" in result

    def test_config_validation_direct_error(self):
        with pytest.raises(Exception):
            DatabaseSettings(url="mysql://invalid:invalid@localhost/db")

    def test_config_validation_catches_port_error(self):
        with pytest.raises(Exception):
            Settings(port=80)
