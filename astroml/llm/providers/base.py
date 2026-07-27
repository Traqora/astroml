"""Base LLM Provider interface and unified response structures."""
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterator
from pydantic import BaseModel, Field
from ..exceptions import ProviderAPIError

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Unified response format wrapper."""
    text: str
    prompt_tokens: int = Field(default=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens used")
    cost: float = Field(default=0.0, description="Estimated USD cost of the call")
    latency: float = Field(default=0.0, description="Call latency in seconds")
    provider: str = Field(default="", description="Name of the LLM provider")
    model: str = Field(default="", description="Name of the model used")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.fallback_providers: List['LLMProvider'] = []

    @abstractmethod
    def _generate_raw(self, prompt: str, tools: list[dict] | None = None, **kwargs: Any) -> str:
        """Internal method to generate response from the specific provider."""
        pass

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the LLM, keeping backwards compatibility."""
        response = self.generate_detailed(prompt, **kwargs)
        return response.text

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_executor: Any = None,
        max_tool_calls: int = 10,
        **kwargs: Any,
    ) -> str:
        """Generate a response with tool-calling loop.

        Sends prompt + tool definitions to the LLM, executes any tool calls
        the LLM requests, feeds results back, and repeats until the LLM
        returns a final text response.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        current_tools = tools

        for _ in range(max_tool_calls):
            raw = self._generate_raw(json.dumps(messages), tools=current_tools, **kwargs)

            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                return raw

            tool_calls = result.get("tool_calls")
            content = result.get("content", "")

            if not tool_calls:
                return content or raw

            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            for tc in tool_calls:
                if tool_executor is None:
                    tool_result = {"error": "No tool executor available"}
                else:
                    try:
                        import asyncio
                        tool_result = asyncio.run(
                            tool_executor.execute(tc["function"]["name"], tc["function"]["arguments"])
                        )
                    except Exception as e:
                        tool_result = {"error": str(e)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": json.dumps(tool_result),
                })

            current_tools = None

        return json.dumps(messages[-1]) if messages else "Max tool calls reached"

    def generate_detailed(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate response with unified response format, cost tracking, rate limits, budgets, retries."""
        from ..rate_limiter import get_rate_limiter, get_budget_manager
        from ..secrets import get_api_key
        from ..config import llm_settings

        provider_name = self.__class__.__name__.lower().replace("provider", "")
        model_name = kwargs.get("model", self.model)
        
        # Resolve config settings
        p_settings = llm_settings.providers.get(provider_name)
        req_limit = p_settings.rate_limits.requests_per_minute if p_settings else 0
        token_limit = p_settings.rate_limits.tokens_per_minute if p_settings else 0
        daily_limit = p_settings.cost_budget.daily_limit if p_settings else 0.0
        monthly_limit = p_settings.cost_budget.monthly_limit if p_settings else 0.0

        rate_limiter = get_rate_limiter(provider_name, req_limit, token_limit)
        budget_manager = get_budget_manager(provider_name, daily_limit, monthly_limit)

        # Enforce cost budget check before sending
        budget_manager.check_budget()

        # Enforce rate limit check before sending (estimate ~1 token per 4 chars for prompt)
        estimated_tokens = max(1, len(prompt) // 4) + kwargs.get("max_tokens", 1024)
        rate_limiter.check_and_record(estimated_tokens)

        # Retry logic with exponential backoff for transient errors (429, 500, 502, 503, 504)
        max_retries = 3
        delay = 1.0
        backoff_factor = 2.0
        
        start_time = time.time()
        text = ""
        
        for attempt in range(max_retries + 1):
            try:
                # Decrypt keys on-the-fly (audit logged)
                _ = get_api_key(provider_name)
                
                text = self._generate_raw(prompt, **kwargs)
                break
            except Exception as e:
                err_str = str(e)
                # Check for transient errors
                is_transient = any(code in err_str for code in ["429", "500", "502", "503", "504"]) or "rate limit" in err_str.lower()
                if not is_transient or attempt == max_retries:
                    # Check fallback before raising error
                    if self.fallback_providers:
                        logger.warning(f"Primary provider {provider_name} failed. Trying fallbacks...")
                        for fb in self.fallback_providers:
                            try:
                                return fb.generate_detailed(prompt, **kwargs)
                            except Exception as fb_err:
                                logger.warning(f"Fallback provider failed: {fb_err}")
                    
                    # API keys are never logged or exposed in error messages
                    logger.error(f"LLM Provider API error. Secrets are never exposed. Detail: {err_str[:100]}")
                    raise ProviderAPIError(f"LLM API Error: {err_str[:100]}") from None
                
                # Exponential backoff
                sleep_time = delay + (time.time() % 0.1) # small jitter
                logger.warning(f"Transient error on {provider_name} (attempt {attempt+1}/{max_retries}). Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                delay *= backoff_factor

        latency = time.time() - start_time
        
        # Token usage and cost calculation
        usage = self.get_token_usage()
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        
        # Update rate limiter with actual tokens used
        rate_limiter.update_actual_tokens(total_tokens)
        
        # Cost calculation
        cost = self._calculate_cost(prompt_tokens, completion_tokens, model_name)
        
        # Update budget spend
        budget_manager.record_spend(cost)

        # Structured logging (latency, tokens, cost)
        logger.info(
            f"LLM Call: provider={provider_name}, model={model_name}, "
            f"latency={latency:.2f}s, tokens={total_tokens}, cost=${cost:.5f}"
        )

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency=latency,
            provider=provider_name,
            model=model_name
        )

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        provider_name = self.__class__.__name__.lower().replace("provider", "")
        if provider_name == "openai":
            return (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000.0
        elif provider_name == "anthropic":
            return (prompt_tokens * 0.015 + completion_tokens * 0.075) / 1000.0
        return 0.0

    @abstractmethod
    def get_token_usage(self) -> Dict[str, int]:
        """Return the token usage for the last generation."""
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the LLM response."""
        pass

    @abstractmethod
    def embed(self, text: str, **kwargs: Any) -> List[float]:
        """Generate text embedding."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        pass

    def health_check(self) -> bool:
        """Verify the provider is reachable and healthy."""
        try:
            self.generate("health_check_ping", max_tokens=5)
            return True
        except Exception:
            return False
