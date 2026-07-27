"""CLI configuration loader for LLM operations."""

import os
import pathlib
from typing import Any


DEFAULT_CONFIG_PATH = pathlib.Path.home() / ".config" / "astroml" / "llm.yaml"

DEFAULTS: dict[str, Any] = {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1024,
    "api_key": "",
}


def load_cli_config(config_path: str | None = None) -> dict[str, Any]:
    """Load CLI configuration from YAML file, falling back to defaults and env vars."""
    config = dict(DEFAULTS)

    cfg_path = pathlib.Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path) as f:
                loaded = yaml.safe_load(f) or {}
            config.update(loaded)
        except Exception:
            pass

    env_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if env_key:
        config["api_key"] = env_key

    return config
