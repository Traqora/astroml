import time
import logging
import random
from typing import Callable, Any, List, Type, Tuple
from functools import wraps

logger = logging.getLogger(__name__)

class ErrorTaxonomy:
    class ProviderError(Exception): pass
    class RateLimitError(ProviderError): pass
    class AuthError(ProviderError): pass
    class ServerError(ProviderError): pass
    class ClientError(ProviderError): pass
    class TimeoutError(ProviderError): pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPENED after {self.failures} failures.")

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
                return True
            return False
            
        if self.state == "HALF-OPEN":
            return True
            
        return False

def with_exponential_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0, exceptions: Tuple[Type[Exception]] = (Exception,)):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if retries == max_retries:
                        raise e
                    
                    delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1), max_delay)
                    logger.warning(f"Operation failed with {e}. Retrying in {delay:.2f}s (Attempt {retries + 1}/{max_retries})")
                    time.sleep(delay)
                    retries += 1
        return wrapper
    return decorator

class FallbackProviderChain:
    def __init__(self, providers: List[Callable]):
        self.providers = providers
        self.circuit_breakers = {id(provider): CircuitBreaker(failure_threshold=5) for provider in providers}

    def execute(self, *args, **kwargs) -> Any:
        for provider in self.providers:
            cb = self.circuit_breakers[id(provider)]
            
            if not cb.can_execute():
                logger.info(f"Skipping provider {provider.__name__} due to OPEN circuit breaker.")
                continue

            try:
                # Wrap with exponential backoff for transient errors
                @with_exponential_backoff(max_retries=2, exceptions=(ErrorTaxonomy.RateLimitError, ErrorTaxonomy.TimeoutError))
                def run_provider():
                    return provider(*args, **kwargs)
                
                result = run_provider()
                cb.record_success()
                return result
            
            except Exception as e:
                cb.record_failure()
                logger.error(f"Provider {provider.__name__} failed with {e}. Falling back...")
                
        raise ErrorTaxonomy.ProviderError("All fallback providers failed.")
