"""Factory for LLM Providers with automatic fallback chains."""
import os
import logging
from typing import Dict, Type, List
from .base import LLMProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .huggingface import HuggingFaceProvider
from .local import LocalProvider

logger = logging.getLogger(__name__)

_PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "huggingface": HuggingFaceProvider,
    "local": LocalProvider,
}

def get_llm_provider(provider_name: str = None, **kwargs) -> LLMProvider:
    """Get the configured LLM provider, with fallback providers attached."""
    from ..config import llm_settings
    from ..secrets import get_api_key

    provider_name = provider_name or os.getenv("LLM_PROVIDER") or llm_settings.default_provider
    provider_name = provider_name.lower().strip()
    
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
        
    # Extract API key
    api_key = kwargs.pop("api_key", None)
    if not api_key:
        api_key = get_api_key(provider_name)

    primary_prov = _PROVIDERS[provider_name](api_key=api_key, **kwargs)

    # Initialize secondaries from fallback chain config
    secondaries: List[LLMProvider] = []
    fallback_chain = llm_settings.fallback_chain
    for fallback_name in fallback_chain:
        if fallback_name != provider_name and fallback_name in _PROVIDERS:
            try:
                f_key = get_api_key(fallback_name)
                secondaries.append(_PROVIDERS[fallback_name](api_key=f_key))
            except Exception as e:
                logger.warning(f"Failed to initialize fallback provider {fallback_name}: {e}")

    primary_prov.fallback_providers = secondaries
    return primary_prov
