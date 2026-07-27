"""Application configuration using Pydantic BaseSettings.

Validation rules:
- Database URL must be a valid PostgreSQL/SQLite connection string
- Training hyperparameters must be positive (learning_rate > 0, epochs > 0)
- Graph window duration must be a positive timedelta
- Port must be in valid range (1024-65535)
- Log level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL
"""
from datetime import timedelta
from typing import Optional, Any
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    url: str = Field(default="sqlite:///./astroml.db", env="DATABASE_URL", description="Database connection URL (PostgreSQL or SQLite)")
    pool_size: int = Field(default=10, ge=1, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, ge=0, env="DATABASE_MAX_OVERFLOW")

    @field_validator("url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        parsed = urlparse(v)
        scheme = parsed.scheme or ""
        if not any(supported in scheme for supported in ("postgresql", "sqlite")):
            raise ValueError(
                f"Invalid database URL scheme '{scheme}'. "
                f"Must be a PostgreSQL (postgresql://) or SQLite (sqlite:///) URL."
            )
        if "postgresql" in scheme:
            if not parsed.hostname:
                raise ValueError("PostgreSQL URL must include a hostname.")
            if not parsed.path or parsed.path.strip("/") == "":
                raise ValueError("PostgreSQL URL must include a database name.")
        return v


class TrainingSettings(BaseSettings):
    learning_rate: float = Field(default=0.01, gt=0, env="TRAINING_LEARNING_RATE", description="Learning rate (must be > 0)")
    epochs: int = Field(default=200, gt=0, env="TRAINING_EPOCHS", description="Number of training epochs (must be > 0)")
    batch_size: Optional[int] = Field(default=None, ge=1, env="TRAINING_BATCH_SIZE")
    weight_decay: float = Field(default=5e-4, ge=0, env="TRAINING_WEIGHT_DECAY")


class GraphSettings(BaseSettings):
    window_duration_minutes: int = Field(default=60, gt=0, env="GRAPH_WINDOW_DURATION_MINUTES", description="Graph window duration in minutes (must be > 0)")

    @property
    def window_duration(self) -> timedelta:
        return timedelta(minutes=self.window_duration_minutes)


class Settings(BaseSettings):
    app_name: str = Field(default="AstroML Dashboard API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    api_key_name: str = Field(default="X-API-Key", env="API_KEY_NAME")
    allowed_origins: list[str] = Field(default=["http://localhost:5173", "http://localhost:3000"], env="ALLOWED_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: list[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: list[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, ge=1, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1024 <= v <= 65535:
            raise ValueError(f"Port {v} is not in valid range (1024-65535)")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError(f"Secret key must be at least 16 characters long (got {len(v)} characters)")
        return v

    @model_validator(mode="after")
    def validate_experiment_consistency(self) -> "Settings":
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def validate_config() -> dict[str, Any]:
    errors: list[str] = []
    settings_dict: dict[str, Any] = {}
    try:
        settings = get_settings()
        settings_dict = settings.model_dump()
    except Exception as e:
        errors.append(str(e))
        return {"valid": False, "errors": errors, "settings": settings_dict}
    return {"valid": len(errors) == 0, "errors": errors, "settings": settings_dict}
