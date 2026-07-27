"""Enhanced Stellar SDK-based streaming service with robust error handling.

Uses the Stellar SDK for improved reliability and provides comprehensive
rate limiting and connection drop handling for effects and operations streaming.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional, Dict, Any

from stellar_sdk import Server
from stellar_sdk.exceptions import (
    BadRequestError,
    ConnectionError,
    NotFoundError,
    BaseHorizonError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from astroml.db.session import get_session
from astroml.db.schema import Effect, Operation
from astroml.ingestion.parsers import parse_effect, parse_operation
from astroml.ingestion.metrics import (
    STREAM_RECORDS_PROCESSED,
    STREAM_ERRORS,
    STREAM_CONNECTION_HEALTH,
    STREAM_RATE_LIMIT_BACKOFF,
    STREAM_PROCESSING_LATENCY,
    STREAM_CURSOR,
    STREAM_LAG_SECONDS
)

try:
    from api.routers.streaming import update_stream_state
    _stream_state_available = True
except ImportError:
    _stream_state_available = False

logger = logging.getLogger("astroml.ingestion.enhanced_stream")


@dataclass
class EnhancedStreamConfig:
    """Configuration for enhanced streaming service."""
    
    horizon_url: str = "https://horizon-testnet.stellar.org"
    stream_type: str = "effects"  # "effects" or "operations"
    cursor: Optional[str] = None
    max_retries: int = 5
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    rate_limit_backoff: float = 5.0
    connection_timeout: float = 30.0
    stream_timeout: float = 60.0
    health_check_interval: float = 30.0
    batch_size: int = 100
    batch_timeout: float = 5.0


class RateLimitTracker:
    """Tracks rate limit status and implements adaptive throttling."""
    
    def __init__(self, backoff_factor: float = 1.5):
        self.backoff_factor = backoff_factor
        self.last_rate_limit_time: Optional[float] = None
        self.current_backoff: float = 1.0
        self.request_count: int = 0
        self.window_start: float = time.time()
        self.window_size: float = 60.0  # 1-minute window
        
    def record_request(self) -> None:
        """Record a request timestamp."""
        self.request_count += 1
        now = time.time()
        if now - self.window_start > self.window_size:
            self.request_count = 1
            self.window_start = now
    
    def handle_rate_limit(self) -> float:
        """Calculate backoff time after rate limit hit."""
        self.last_rate_limit_time = time.time()
        self.current_backoff = min(self.current_backoff * self.backoff_factor, 300.0)
        return self.current_backoff
    
    def get_request_rate(self) -> float:
        """Get current requests per second."""
        elapsed = time.time() - self.window_start
        return self.request_count / max(elapsed, 1.0)
    
    def should_throttle(self) -> bool:
        """Determine if we should throttle based on recent rate limit."""
        if self.last_rate_limit_time is None:
            return False
        
        time_since_limit = time.time() - self.last_rate_limit_time
        return time_since_limit < self.current_backoff


class ConnectionHealthMonitor:
    """Monitors connection health and detects drops."""
    
    def __init__(self, check_interval: float = 30.0, max_consecutive_failures: int = 3):
        self.check_interval = check_interval
        self.last_successful_request: Optional[float] = None
        self.last_health_check: Optional[float] = None
        self.is_healthy: bool = True
        self.consecutive_failures: int = 0
        self.max_consecutive_failures: int = max_consecutive_failures
        
    def record_success(self) -> None:
        """Record a successful request."""
        self.last_successful_request = time.time()
        self.consecutive_failures = 0
        self.is_healthy = True
        
    def record_failure(self) -> None:
        """Record a failed request."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.is_healthy = False
            
    def should_check_health(self) -> bool:
        """Determine if health check is needed."""
        if self.last_health_check is None:
            return True
        
        return time.time() - self.last_health_check >= self.check_interval
    
    def is_connection_stale(self) -> bool:
        """Check if connection appears stale based on inactivity."""
        if self.last_successful_request is None:
            return False
        
        return time.time() - self.last_successful_request > self.check_interval * 2


class EnhancedStellarStream:
    """Enhanced streaming service with Stellar SDK and robust error handling."""
    
    def __init__(self, config: EnhancedStreamConfig) -> None:
        self.config = config
        self.server = Server(horizon_url=config.horizon_url)
        self.rate_tracker = RateLimitTracker()
        self.health_monitor = ConnectionHealthMonitor(config.health_check_interval)
        self._running: bool = False
        self._cursor: Optional[str] = config.cursor
        self._processed_count: int = 0
        self._last_record_created_at: Optional[float] = None
        self._stream_id = f"{config.stream_type}-{config.horizon_url.replace('https://', '').replace('http://', '').replace('/', '_')}"
        
    def _update_stream_state(self) -> None:
        """Update the stream state in the API router."""
        if not _stream_state_available:
            return
            
        # Calculate lag
        lag_seconds = None
        if self._last_record_created_at:
            lag_seconds = time.time() - self._last_record_created_at
            
        # Determine status
        if self.health_monitor.is_healthy and self._running:
            status = "active"
        elif self.health_monitor.consecutive_failures > 0:
            status = "error"
        else:
            status = "inactive"
            
        state = {
            "stream_type": self.config.stream_type,
            "horizon_url": self.config.horizon_url,
            "is_healthy": self.health_monitor.is_healthy,
            "status": status,
            "cursor": self._cursor,
            "processed_count": self._processed_count,
            "consecutive_failures": self.health_monitor.consecutive_failures,
            "current_backoff": self.rate_tracker.current_backoff,
            "lag_seconds": lag_seconds
        }
        
        update_stream_state(self._stream_id, state)
        
    async def __aenter__(self) -> "EnhancedStellarStream":
        """Async context manager entry."""
        self._running = True
        STREAM_CONNECTION_HEALTH.labels(
            stream_type=self.config.stream_type,
            horizon_url=self.config.horizon_url
        ).set(1)
        self._update_stream_state()
        logger.info(
            "EnhancedStellarStream initialized | horizon=%s stream=%s cursor=%s",
            self.config.horizon_url,
            self.config.stream_type,
            self._cursor or "now"
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self._running = False
        STREAM_CONNECTION_HEALTH.labels(
            stream_type=self.config.stream_type,
            horizon_url=self.config.horizon_url
        ).set(0)
        self._update_stream_state()
        logger.info(
            "EnhancedStellarStream shutdown | processed=%d final_cursor=%s",
            self._processed_count,
            self._cursor
        )
        
    @retry(
        retry=retry_if_exception_type((ConnectionError, BaseHorizonError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def _check_server_health(self) -> bool:
        """Check if Horizon server is responsive."""
        try:
            root = await asyncio.to_thread(self.server.root)
            self.health_monitor.record_success()
            STREAM_CONNECTION_HEALTH.labels(
                stream_type=self.config.stream_type,
                horizon_url=self.config.horizon_url
            ).set(1)
            logger.debug("Server health check passed | horizon_version=%s", root.get("horizon_version"))
            return True
        except Exception as e:
            self.health_monitor.record_failure()
            STREAM_ERRORS.labels(
                stream_type=self.config.stream_type,
                horizon_url=self.config.horizon_url,
                error_type="health_check_failure"
            ).inc()
            STREAM_CONNECTION_HEALTH.labels(
                stream_type=self.config.stream_type,
                horizon_url=self.config.horizon_url
            ).set(0)
            logger.warning("Server health check failed: %s", e)
            return False
    
    async def _handle_rate_limit(self, error: BaseHorizonError) -> None:
        """Handle rate limit errors with adaptive backoff."""
        backoff = self.rate_tracker.handle_rate_limit()
        STREAM_RATE_LIMIT_BACKOFF.labels(
            stream_type=self.config.stream_type,
            horizon_url=self.config.horizon_url
        ).set(backoff)
        STREAM_ERRORS.labels(
            stream_type=self.config.stream_type,
            horizon_url=self.config.horizon_url,
            error_type="rate_limit"
        ).inc()
        
        # In stellar-sdk, BaseHorizonError has a response attribute
        reset_time = None
        if hasattr(error, 'response') and error.response:
            reset_time = error.response.headers.get('Retry-After')
            if reset_time:
                try:
                    wait_time = float(reset_time)
                    logger.warning(
                        "Rate limit hit | waiting=%.1fs adaptive_backoff=%.1fs",
                        wait_time,
                        backoff
                    )
                    await asyncio.sleep(wait_time)
                    return
                except ValueError:
                    pass
        
        await asyncio.sleep(backoff)
    
    async def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection errors with health monitoring."""
        self.health_monitor.record_failure()
        STREAM_ERRORS.labels(
            stream_type=self.config.stream_type,
            horizon_url=self.config.horizon_url,
            error_type="connection_error"
        ).inc()
        self._update_stream_state()
        
        if self.health_monitor.consecutive_failures >= self.health_monitor.max_consecutive_failures:
            logger.error(
                "Multiple consecutive failures (%d), attempting server health check",
                self.health_monitor.consecutive_failures
            )
            await self._check_server_health()
        
        # Exponential backoff for connection issues
        delay = min(
            self.config.base_retry_delay * (2 ** self.health_monitor.consecutive_failures),
            self.config.max_retry_delay
        )
        
        logger.warning(
            "Connection error (attempt %d): %s. Reconnecting in %.1fs...",
            self.health_monitor.consecutive_failures,
            error,
            delay
        )
        await asyncio.sleep(delay)
    
    async def _stream_effects(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream effects using Stellar SDK with robust error handling."""
        try:
            effects_builder = self.server.effects()
            
            if self._cursor:
                effects_builder = effects_builder.cursor(self._cursor)
            else:
                effects_builder = effects_builder.order("asc")
            
            async for effect in self._stream_with_retry(effects_builder):
                yield effect
                
        except Exception as e:
            logger.exception("Error in effects stream: %s", e)
            raise
    
    async def _stream_operations(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream operations using Stellar SDK with robust error handling."""
        try:
            operations_builder = self.server.operations()
            
            if self._cursor:
                operations_builder = operations_builder.cursor(self._cursor)
            else:
                operations_builder = operations_builder.order("asc")
            
            async for operation in self._stream_with_retry(operations_builder):
                yield operation
                
        except Exception as e:
            logger.exception("Error in operations stream: %s", e)
            raise
    
    async def _stream_with_retry(self, builder) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream with comprehensive retry logic and error handling."""
        retry_count = 0
        
        while self._running and retry_count < self.config.max_retries:
            try:
                # Check rate limiting before making request
                if self.rate_tracker.should_throttle():
                    wait_time = self.rate_tracker.current_backoff
                    logger.info("Rate limiting active | waiting=%.1fs", wait_time)
                    await asyncio.sleep(wait_time)
                
                # Check connection health if needed
                if self.health_monitor.should_check_health():
                    await self._check_server_health()
                
                # Make the request
                self.rate_tracker.record_request()
                
                start_time = time.time()
                if self.config.stream_type == "effects":
                    records = await asyncio.to_thread(
                        lambda: builder.limit(self.config.batch_size).call()
                    )
                else:  # operations
                    records = await asyncio.to_thread(
                        lambda: builder.limit(self.config.batch_size).call()
                    )
                
                # Reset retry count on success
                retry_count = 0
                self.health_monitor.record_success()
                STREAM_CONNECTION_HEALTH.labels(
                    stream_type=self.config.stream_type,
                    horizon_url=self.config.horizon_url
                ).set(1)
                
                # Process records
                batch_records = records.get("_embedded", {}).get("records", [])
                if not batch_records:
                    logger.debug("No records in response, waiting...")
                    await asyncio.sleep(1.0)
                    continue
                
                for record in batch_records:
                    if not self._running:
                        break
                    
                    yield record
                    self._cursor = record.get("paging_token")
                    self._processed_count += 1
                    
                    # Update metrics per record
                    STREAM_RECORDS_PROCESSED.labels(
                        stream_type=self.config.stream_type,
                        horizon_url=self.config.horizon_url
                    ).inc()
                
                # Update batch metrics
                duration = time.time() - start_time
                STREAM_PROCESSING_LATENCY.labels(
                    stream_type=self.config.stream_type,
                    horizon_url=self.config.horizon_url
                ).observe(duration)
                
                # Update cursor metric (try to parse numeric part if possible)
                if self._cursor:
                    try:
                        # Paging tokens are often numeric strings or have numeric parts
                        cursor_val = float(self._cursor.split('-')[0])
                        STREAM_CURSOR.labels(
                            stream_type=self.config.stream_type,
                            horizon_url=self.config.horizon_url
                        ).set(cursor_val)
                    except (ValueError, IndexError):
                        pass
                
                # Update cursor for next batch
                if records.get("_links", {}).get("next", {}).get("href"):
                    cursor = records["_links"]["next"]["href"].split("cursor=")[-1]
                    if cursor != self._cursor:
                        self._cursor = cursor
                        builder = builder.cursor(cursor)
                
            except BaseHorizonError as e:
                if getattr(e, "status", None) == 429:
                    await self._handle_rate_limit(e)
                    continue
                
                retry_count += 1
                await self._handle_connection_error(e)
                continue
                
            except ConnectionError as e:
                retry_count += 1
                await self._handle_connection_error(e)
                continue
                
            except BadRequestError as e:
                logger.error("Bad request error: %s", e)
                raise
                
            except NotFoundError as e:
                logger.warning("Resource not found: %s", e)
                await asyncio.sleep(5.0)
                continue
                
            except Exception as e:
                retry_count += 1
                logger.exception("Unexpected error (attempt %d): %s", retry_count, e)
                
                if retry_count >= self.config.max_retries:
                    logger.error("Max retries exceeded, stopping")
                    raise
                
                await asyncio.sleep(self.config.base_retry_delay * retry_count)
        
        if not self._running:
            logger.info("Stream stopped by user")
        else:
            logger.error("Stream stopped after max retries")
    
    async def run(self) -> None:
        """Main streaming loop."""
        logger.info("Starting enhanced stream | type=%s", self.config.stream_type)
        
        try:
            if self.config.stream_type == "effects":
                async for effect in self._stream_effects():
                    await self._process_effect(effect)
            else:
                async for operation in self._stream_operations():
                    await self._process_operation(operation)
                    
        except Exception as e:
            logger.exception("Fatal error in stream: %s", e)
            raise
        finally:
            logger.info("Stream completed | processed=%d final_cursor=%s", 
                       self._processed_count, self._cursor)
    
    async def _process_effect(self, effect_data: Dict[str, Any]) -> None:
        """Process and persist a single effect."""
        try:
            effect = parse_effect(effect_data)
            logger.debug("Processing effect %d (type=%s)", effect.id, effect.type)
            
            await asyncio.to_thread(self._persist_effect, effect)
            
        except Exception as e:
            logger.exception("Failed to process effect: %s", e)
    
    async def _process_operation(self, operation_data: Dict[str, Any]) -> None:
        """Process and persist a single operation."""
        try:
            operation = parse_operation(operation_data)
            logger.debug("Processing operation %d (type=%s)", operation.id, operation.type)
            
            await asyncio.to_thread(self._persist_operation, operation)
            
        except Exception as e:
            logger.exception("Failed to process operation: %s", e)
    
    @staticmethod
    def _persist_effect(effect: Effect) -> None:
        """Persist effect to database."""
        session = get_session()
        try:
            session.merge(effect)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @staticmethod
    def _persist_operation(operation: Operation) -> None:
        """Persist operation to database."""
        session = get_session()
        try:
            session.merge(operation)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @property
    def cursor(self) -> Optional[str]:
        """Current cursor position."""
        return self._cursor
    
    @property
    def processed_count(self) -> int:
        """Number of records processed."""
        return self._processed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "cursor": self._cursor,
            "processed_count": self._processed_count,
            "request_rate": self.rate_tracker.get_request_rate(),
            "is_healthy": self.health_monitor.is_healthy,
            "consecutive_failures": self.health_monitor.consecutive_failures,
            "current_backoff": self.rate_tracker.current_backoff,
        }
