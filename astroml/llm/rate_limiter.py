"""Rate limiter and cost budget manager for LLM requests."""
import time
import logging
from typing import Dict, List, Tuple, Optional
from .exceptions import RateLimitExceededError, CostBudgetExceededError

logger = logging.getLogger(__name__)

class ProviderRateLimiter:
    """Tracks rate limits for requests per minute and tokens per minute."""
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_history: List[float] = []
        self.token_history: List[Tuple[float, int]] = []

    def _clean_old_history(self, now: float) -> None:
        cutoff = now - 60.0
        self.request_history = [t for t in self.request_history if t > cutoff]
        self.token_history = [(t, tokens) for t, tokens in self.token_history if t > cutoff]

    def check_and_record(self, estimated_tokens: int) -> None:
        """Enforce rate limits before sending requests."""
        now = time.time()
        self._clean_old_history(now)

        if self.requests_per_minute > 0 and len(self.request_history) >= self.requests_per_minute:
            raise RateLimitExceededError(
                f"Rate limit exceeded: Max {self.requests_per_minute} requests/min. "
                f"Active requests in last 60s: {len(self.request_history)}"
            )

        current_tokens = sum(tokens for _, tokens in self.token_history)
        if self.tokens_per_minute > 0 and (current_tokens + estimated_tokens) > self.tokens_per_minute:
            raise RateLimitExceededError(
                f"Rate limit exceeded: Max {self.tokens_per_minute} tokens/min. "
                f"Current active tokens + estimated prompt tokens: {current_tokens} + {estimated_tokens} = {current_tokens + estimated_tokens}"
            )

        self.request_history.append(now)
        self.token_history.append((now, estimated_tokens))

    def update_actual_tokens(self, actual_tokens: int) -> None:
        """Update the temporary token estimate with the actual response usage."""
        if self.token_history:
            t, _ = self.token_history[-1]
            self.token_history[-1] = (t, actual_tokens)


class CostBudgetManager:
    """Tracks daily and monthly cost budgets, triggering alerts at 80% and 100%."""
    def __init__(self, daily_limit: float, monthly_limit: float):
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.daily_spend = 0.0
        self.monthly_spend = 0.0
        self.last_reset_day = time.strftime("%Y-%m-%d")
        self.last_reset_month = time.strftime("%Y-%m")
        self.alerted_80_daily = False
        self.alerted_100_daily = False
        self.alerted_80_monthly = False
        self.alerted_100_monthly = False

    def _reset_if_needed(self) -> None:
        current_day = time.strftime("%Y-%m-%d")
        current_month = time.strftime("%Y-%m")
        
        if current_day != self.last_reset_day:
            self.daily_spend = 0.0
            self.last_reset_day = current_day
            self.alerted_80_daily = False
            self.alerted_100_daily = False
            
        if current_month != self.last_reset_month:
            self.monthly_spend = 0.0
            self.last_reset_month = current_month
            self.alerted_80_monthly = False
            self.alerted_100_monthly = False

    def check_budget(self) -> None:
        """Enforces limits before sending requests."""
        self._reset_if_needed()
        
        if self.daily_limit > 0 and self.daily_spend >= self.daily_limit:
            raise CostBudgetExceededError(
                f"Cost budget exceeded: Daily spend limit of ${self.daily_limit:.2f} reached."
            )
            
        if self.monthly_limit > 0 and self.monthly_spend >= self.monthly_limit:
            raise CostBudgetExceededError(
                f"Cost budget exceeded: Monthly spend limit of ${self.monthly_limit:.2f} reached."
            )

    def record_spend(self, cost: float) -> None:
        """Adds spend and triggers alerts at 80% and 100%."""
        self._reset_if_needed()
        self.daily_spend += cost
        self.monthly_spend += cost

        # Check daily alerts
        if self.daily_limit > 0:
            pct = self.daily_spend / self.daily_limit
            if pct >= 1.0 and not self.alerted_100_daily:
                self.alerted_100_daily = True
                logger.warning(f"ALERT: 100% daily cost budget reached! Spend: ${self.daily_spend:.2f}/${self.daily_limit:.2f}")
            elif pct >= 0.8 and not self.alerted_80_daily:
                self.alerted_80_daily = True
                logger.warning(f"ALERT: 80% daily cost budget reached! Spend: ${self.daily_spend:.2f}/${self.daily_limit:.2f}")

        # Check monthly alerts
        if self.monthly_limit > 0:
            pct = self.monthly_spend / self.monthly_limit
            if pct >= 1.0 and not self.alerted_100_monthly:
                self.alerted_100_monthly = True
                logger.warning(f"ALERT: 100% monthly cost budget reached! Spend: ${self.monthly_spend:.2f}/${self.monthly_limit:.2f}")
            elif pct >= 0.8 and not self.alerted_80_monthly:
                self.alerted_80_monthly = True
                logger.warning(f"ALERT: 80% monthly cost budget reached! Spend: ${self.monthly_spend:.2f}/${self.monthly_limit:.2f}")


# Global registries
_RATE_LIMITERS: Dict[str, ProviderRateLimiter] = {}
_BUDGET_MANAGERS: Dict[str, CostBudgetManager] = {}

def get_rate_limiter(provider: str, req_limit: int = 0, token_limit: int = 0) -> ProviderRateLimiter:
    provider = provider.lower().strip()
    if provider not in _RATE_LIMITERS:
        _RATE_LIMITERS[provider] = ProviderRateLimiter(req_limit, token_limit)
    return _RATE_LIMITERS[provider]

def get_budget_manager(scope: str = "global", daily_limit: float = 0.0, monthly_limit: float = 0.0) -> CostBudgetManager:
    scope = scope.lower().strip()
    if scope not in _BUDGET_MANAGERS:
        _BUDGET_MANAGERS[scope] = CostBudgetManager(daily_limit, monthly_limit)
    return _BUDGET_MANAGERS[scope]
