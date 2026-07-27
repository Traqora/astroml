import yaml
import os
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from .exceptions import ConfigurationError

class LLMConfig(BaseModel):
    """
    LLM Configuration System for managing model parameters and provider settings.
    
    Parameters:
    - model_name: The name of the LLM model to use.
    - temperature: Controls randomness. Lower is more deterministic, higher is more random.
    - max_tokens: Maximum number of tokens to generate in the response.
    - top_p: Nucleus sampling probability.
    - provider_params: Additional provider-specific parameters (e.g., streaming, stop sequences).
    """
    model_name: str = Field(default="gpt-4", description="The LLM model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for sampling")
    max_tokens: int = Field(default=1024, ge=1, description="Max tokens for response")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    provider_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Provider-specific params")

    @classmethod
    def load_from_yaml(cls, file_path: str) -> "LLMConfig":
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


class RateLimitSettings(BaseModel):
    requests_per_minute: int = Field(default=0, ge=0)
    tokens_per_minute: int = Field(default=0, ge=0)


class CostBudgetSettings(BaseModel):
    daily_limit: float = Field(default=0.0, ge=0.0)
    monthly_limit: float = Field(default=0.0, ge=0.0)


class ProviderSettings(BaseModel):
    provider_name: str
    model_name: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    rate_limits: RateLimitSettings = Field(default_factory=RateLimitSettings)
    cost_budget: CostBudgetSettings = Field(default_factory=CostBudgetSettings)
    provider_params: Dict[str, Any] = Field(default_factory=dict)


class GlobalLLMSettings(BaseModel):
    default_provider: str = Field(default="openai")
    fallback_chain: List[str] = Field(default_factory=list)
    providers: Dict[str, ProviderSettings] = Field(default_factory=dict)


def load_all_configs(config_dir: str = None) -> GlobalLLMSettings:
    """Load configuration files and validate settings, failing fast if invalid."""
    if not config_dir:
        # Check standard path
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = os.path.join(base_dir, "configs", "llm")
    
    config_file = os.path.join(config_dir, "config.yaml")
    if not os.path.exists(config_file):
        # Fallback default configuration
        return GlobalLLMSettings(
            default_provider="openai",
            fallback_chain=["openai", "anthropic", "local"],
            providers={
                "openai": ProviderSettings(provider_name="openai", model_name="gpt-4"),
                "anthropic": ProviderSettings(provider_name="anthropic", model_name="claude-3-opus-20240229"),
                "local": ProviderSettings(provider_name="local", model_name="meta-llama/Llama-2-7b-chat-hf")
            }
        )

    with open(config_file, 'r') as f:
        global_data = yaml.safe_load(f) or {}

    providers_dir = os.path.join(config_dir, "providers")
    providers = {}
    if os.path.exists(providers_dir):
        for fname in sorted(os.listdir(providers_dir)):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                p_path = os.path.join(providers_dir, fname)
                with open(p_path, 'r') as f:
                    p_data = yaml.safe_load(f)
                if p_data:
                    p_name = p_data.get("provider_name") or os.path.splitext(fname)[0]
                    try:
                        providers[p_name] = ProviderSettings(**p_data)
                    except Exception as e:
                        raise ConfigurationError(f"Validation failed for provider config '{fname}': {e}") from e

    # Fail fast if default provider or fallback chain references unconfigured providers
    default_p = global_data.get("default_provider", "openai")
    if default_p not in providers:
        raise ConfigurationError(f"Default provider '{default_p}' is not configured in providers directory.")

    fallback_chain = global_data.get("fallback_chain", [])
    for fb in fallback_chain:
        if fb not in providers:
            raise ConfigurationError(f"Fallback provider '{fb}' is not configured in providers directory.")

    global_data["providers"] = providers
    try:
        return GlobalLLMSettings(**global_data)
    except Exception as e:
        raise ConfigurationError(f"Validation failed for global LLM config: {e}") from e

# Load and validate configs on module import to fail fast on startup
try:
    llm_settings = load_all_configs()
except Exception as e:
    # Re-raise so importing anything from here fails fast
    raise e
