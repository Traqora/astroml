"""Custom exceptions for the LLM Service layer."""

class LLMError(Exception):
    """Base exception for all LLM errors."""
    pass

class ConfigurationError(LLMError):
    """Raised when there is an issue with configuration validation on startup."""
    pass

class RateLimitExceededError(LLMError):
    """Raised when a request would exceed the rate limits (requests/min or tokens/min)."""
    pass

class CostBudgetExceededError(LLMError):
    """Raised when the daily or monthly cost budget is exceeded."""
    pass

class ProviderAPIError(LLMError):
    """Raised when a provider API request fails (e.g. status code 500, 429)."""
    pass
