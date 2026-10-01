from astroml.utils.exceptions import AstroMLError
"""Enhanced Stellar SDK-based streaming service with robust error handling.

Uses the Stellar SDK for improved reliability and provides comprehensive
rate limiting and connection drop handling for effects and operations streaming.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from stellar_sdk import Server
from stellar_sdk.exceptions import (
    BadRequestError,
    BaseHorizonError,
    ConnectionError,
    NotFoundError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from astroml.db.schema import Effect, Operation
from astroml.db.session import get_session
from astroml.ingestion.batch import BatchBuffer, chunked_merge
from astroml.ingestion.metrics import (
    BATCH_BUFFER_SIZE,
    BATCH_FLUSH_DURATION,
    BATCH_FLUSH_TOTAL,
    STREAM_CONNECTION_HEALTH,
    STREAM_CURSOR,
    STREAM_ERRORS,
    STREAM_PROCESSING_LATENCY,
    STREAM_RATE_LIMIT_BACKOFF,
    STREAM_RECORDS_PROCESSED,
)
from astroml.ingestion.parsers import parse_effect, parse_operation

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
    cursor: str | None = None
    max_retries: int = 5
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    rate_limit_backoff: float = 5.0
    connection_timeout: float = 30.0
    stream_timeout: float = 60.0
    health_check_interval: float = 30.0
    batch_size: int = 100
    batch_timeout: float = 5.0
    persist_chunk_size: int = 50


class RateLimitTracker:
    """Tracks rate limit status and implements adaptive throttling."""

    def __init__(self, backoff_factor: float = 1.5) -> None:
        """Start with no requests recorded and the minimum backoff applied.

        Args:
            backoff_factor: Multiplier applied to the wait each time Horizon
                rate-limits us, up to the 300 second ceiling in
                :meth:`handle_rate_limit`.

        Side effects: takes the current wall clock as the start of the first
            measurement window, so a tracker built long before its first
            request opens that window early.
        """
        self.backoff_factor = backoff_factor
        self.last_rate_limit_time: float | None = None
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

    def __init__(self, check_interval: float = 30.0, max_consecutive_failures: int = 3) -> None:
        """Start assumed healthy, with no requests and no health check yet.

        Args:
            check_interval: Minimum seconds between health probes, as read by
                :meth:`should_check_health`.
            max_consecutive_failures: Failures in a row before the connection
                is declared unhealthy.

        Side effects: none — the clock is deliberately not read here. A
            monitor that has never seen a request reports ``is_healthy``
            ``True`` and ``should_check_health`` ``True``, which is the
            conservative pair: probe early, do not assume death.
        """
        self.check_interval = check_interval
        self.last_successful_request: float | None = None
        self.last_health_check: float | None = None
        self.is_healthy: bool = True
        self.consecutive_failures: int = 0
        self.max_consecutive_failures: int = max_consecutive_failures

    def record_success(self) -> None:
        """Record a successful request."""
        self.last_successful_request = time.time()
        self.consecutive_failures = 0
        self.is_healthy = True

    def record_failure(self) -> None:
        """Count one failed request, declaring the connection unhealthy at the limit.

        Side effects: increments the consecutive-failure count and, once it
            reaches ``max_consecutive_failures``, clears ``is_healthy``. A
            single success afterwards resets both (see
            :meth:`record_success`), so a flapping connection stays healthy
            while a genuinely dead one stays flagged.
        """
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
        self._cursor: str | None = config.cursor
        self._processed_count: int = 0
        self._last_record_created_at: float | None = None
        self._stream_id = f"{config.stream_type}-{config.horizon_url.replace('https://', '').replace('http://', '').replace('/', '_')}"
        self._batch_buffer: BatchBuffer | None = None

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
            "lag_seconds": lag_seconds,
        }

        update_stream_state(self._stream_id, state)

    async def __aenter__(self) -> EnhancedStellarStream:
        """Async context manager entry."""
        self._running = True
        session = get_session()
        self._batch_buffer = BatchBuffer(
            session,
            chunk_size=self.config.persist_chunk_size,
            flush_on_exit=True,
        )
        STREAM_CONNECTION_HEALTH.labels(
            stream_type=self.config.stream_type, horizon_url=self.config.horizon_url
        ).set(1)
        self._update_stream_state()
        logger.info(
            "EnhancedStellarStream initialized | horizon=%s stream=%s cursor=%s chunk_size=%d",
            self.config.horizon_url,
            self.config.stream_type,
            self._cursor or "now",
            self.config.persist_chunk_size,
        )
        return self

    async def __aexit__(self, exc_type, _exc_val, _exc_tb) -> None:
        """Async context manager exit."""
        self._running = False
        if self._batch_buffer is not None:
            try:
                flushed = self._batch_buffer.total_flushed
                self._batch_buffer.close()
                logger.info("Batch buffer closed | total_flushed=%d flush_count=%d",
                    flushed, self._batch_buffer.flush_count)
            except AstroMLError:
                logger.exception("Error closing batch buffer")
            finally:
                self._batch_buffer = None
        STREAM_CONNECTION_HEALTH.labels(
            stream_type=self.config.stream_type, horizon_url=self.config.horizon_url
        ).set(0)
        self._update_stream_state()
        logger.info(
            "EnhancedStellarStream shutdown | processed=%d final_cursor=%s",
            self._processed_count,
            self._cursor,
        )

    @retry(
        retry=retry_if_exception_type((ConnectionError, BaseHorizonError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _check_server_health(self) -> bool:
        """Check if Horizon server is responsive."""
        try:
            root = await asyncio.to_thread(self.server.root)
            self.health_monitor.record_success()
            STREAM_CONNECTION_HEALTH.labels(
                stream_type=self.config.stream_type, horizon_url=self.config.horizon_url
            ).set(1)
            logger.debug(
                "Server health check passed | horizon_version=%s", root.get("horizon_version")
            )
            return True
        except AstroMLError as e:
            self.health_monitor.record_failure()
            STREAM_ERRORS.labels(
                stream_type=self.config.stream_type,
                horizon_url=self.config.horizon_url,
                error_type="health_check_failure",
            ).inc()
            STREAM_CONNECTION_HEALTH.labels(
                stream_type=self.config.stream_type, horizon_url=self.config.horizon_url
            ).set(0)
            logger.warning("Server health check failed: %s", e)
            return False

    async def _handle_rate_limit(self, error: BaseHorizonError) -> None:
        """Handle rate limit errors with adaptive backoff."""
        backoff = self.rate_tracker.handle_rate_limit()
        STREAM_RATE_LIMIT_BACKOFF.labels(
            stream_type=self.config.stream_type, horizon_url=self.config.horizon_url
        ).set(backoff)
        STREAM_ERRORS.labels(
            stream_type=self.config.stream_type,
            horizon_url=self.config.horizon_url,
            error_type="rate_limit",
        ).inc()

        # In stellar-sdk, BaseHorizonError has a response attribute
        reset_time = None
        if hasattr(error, "response") and error.response:
            reset_time = error.response.headers.get("Retry-After")
            if reset_time:
                try:
                    wait_time = float(reset_time)
                    logger.warning(
                        "Rate limit hit | waiting=%.1fs adaptive_backoff=%.1fs", wait_time, backoff
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
            error_type="connection_error",
        ).inc()
        self._update_stream_state()

        if self.health_monitor.consecutive_failures >= self.health_monitor.max_consecutive_failures:
            logger.error(
                "Multiple consecutive failures (%d), attempting server health check",
                self.health_monitor.consecutive_failures,
            )
            await self._check_server_health()

        # Exponential backoff for connection issues
        delay = min(
            self.config.base_retry_delay * (2**self.health_monitor.consecutive_failures),
            self.config.max_retry_delay,
        )

        logger.warning(
            "Connection error (attempt %d): %s. Reconnecting in %.1fs...",
            self.health_monitor.consecutive_failures,
            error,
            delay,
        )
        await asyncio.sleep(delay)

    async def _stream_effects(self) -> AsyncGenerator[dict[str, Any], None]:
        """Stream effects using Stellar SDK with robust error handling."""
        try:
            effects_builder = self.server.effects()

            if self._cursor:
                effects_builder = effects_builder.cursor(self._cursor)
            else:
                effects_builder = effects_builder.order("asc")

            async for effect in self._stream_with_retry(effects_builder):
                yield effect

        except AstroMLError as e:
            logger.exception("Error in effects stream: %s", e)
            raise

    async def _stream_operations(self) -> AsyncGenerator[dict[str, Any], None]:
        """Stream operations using Stellar SDK with robust error handling."""
        try:
            operations_builder = self.server.operations()

            if self._cursor:
                operations_builder = operations_builder.cursor(self._cursor)
            else:
                operations_builder = operations_builder.order("asc")

            async for operation in self._stream_with_retry(operations_builder):
                yield operation

        except AstroMLError as e:
            logger.exception("Error in operations stream: %s", e)
            raise

    async def _stream_with_retry(self, builder) -> AsyncGenerator[dict[str, Any], None]:
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
                    stream_type=self.config.stream_type, horizon_url=self.config.horizon_url
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
                        stream_type=self.config.stream_type, horizon_url=self.config.horizon_url
                    ).inc()

                # Update batch metrics
                duration = time.time() - start_time
                STREAM_PROCESSING_LATENCY.labels(
                    stream_type=self.config.stream_type, horizon_url=self.config.horizon_url
                ).observe(duration)

                # Update cursor metric (try to parse numeric part if possible)
                if self._cursor:
                    try:
                        # Paging tokens are often numeric strings or have numeric parts
                        cursor_val = float(self._cursor.split("-")[0])
                        STREAM_CURSOR.labels(
                            stream_type=self.config.stream_type, horizon_url=self.config.horizon_url
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

            except AstroMLError as e:
                retry_count += 1
                logger.exception("Unexpected error (attempt %d): %s", retry_count, e)

                if retry_count >= self.config.max_retries:
                    logger.error("Max retries exceeded, stopping")
                    raise

                delay = min(
                    self.config.base_retry_delay * (2 ** (retry_count - 1)),
                    self.config.max_retry_delay
                )
                await asyncio.sleep(delay)

        if not self._running:
            logger.info("Stream stopped by user")
        else:
            logger.error("Stream stopped after max retries")

    async def run(self) -> None:
        """Consume the configured stream until it ends or the block exits.

        Dispatches on ``config.stream_type``: ``"effects"`` streams effects,
        anything else streams operations. Each record is handed to the
        batching persister, so durability follows
        ``config.persist_chunk_size`` rather than one write per record.

        Raises:
            Exception: Anything the underlying stream raises that the retry loop
                in :meth:`_stream_with_retry` could not absorb — a bad request,
                or ``max_retries`` exhausted. It is logged, reported in the
                closing line, and re-raised rather than swallowed, so a caller
                cannot mistake a dead stream for a finished one.

        Note:
            Must be awaited from inside ``async with`` — that is what sets
            ``_running`` and opens the batch buffer. Called bare, the inner
            ``while self._running`` loop exits at once and this returns having
            processed nothing, with no error to say so.
        """
        logger.info("Starting enhanced stream | type=%s", self.config.stream_type)

        try:
            if self.config.stream_type == "effects":
                async for effect in self._stream_effects():
                    await self._process_effect(effect)
            else:
                async for operation in self._stream_operations():
                    await self._process_operation(operation)

        except AstroMLError as e:
            logger.exception("Fatal error in stream: %s", e)
            raise
        finally:
            logger.info(
                "Stream completed | processed=%d final_cursor=%s",
                self._processed_count,
                self._cursor,
            )

    async def _process_effect(self, effect_data: dict[str, Any]) -> None:
        """Process and buffer a single effect for batch persistence."""
        try:
            effect = parse_effect(effect_data)
            logger.debug("Processing effect %d (type=%s)", effect.id, effect.type)

            if self._batch_buffer is not None:
                self._batch_buffer.add(effect)
            else:
                self._persist_effect(effect)

        except AstroMLError as e:
            logger.exception("Failed to process effect: %s", e)

    async def _process_operation(self, operation_data: dict[str, Any]) -> None:
        """Process and buffer a single operation for batch persistence."""
        try:
            operation = parse_operation(operation_data)
            logger.debug("Processing operation %d (type=%s)", operation.id, operation.type)

            if self._batch_buffer is not None:
                self._batch_buffer.add(operation)
            else:
                self._persist_operation(operation)

        except AstroMLError as e:
            logger.exception("Failed to process operation: %s", e)

    @staticmethod
    def _persist_effect(effect: Effect) -> None:
        """Persist effect to database."""
        session = get_session()
        try:
            session.merge(effect)
            session.commit()
        except AstroMLError:
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
        except AstroMLError:
            session.rollback()
            raise
        finally:
            session.close()

    @property
    def cursor(self) -> str | None:
        """Horizon paging token to resume from, or ``None`` for "not yet set".

        Advanced as records are consumed, so after a restart a new stream
        constructed with this value as its ``cursor`` continues where the last
        one stopped. ``None`` means no record has been seen under this
        instance — treat it as "start where the config says", not as "resume
        from the beginning of time".
        """
        return self._cursor

    @property
    def processed_count(self) -> int:
        """Number of records processed."""
        return self._processed_count

    def get_stats(self) -> dict[str, Any]:
        """Get streaming statistics."""
        return {
            "cursor": self._cursor,
            "processed_count": self._processed_count,
            "request_rate": self.rate_tracker.get_request_rate(),
            "is_healthy": self.health_monitor.is_healthy,
            "consecutive_failures": self.health_monitor.consecutive_failures,
            "current_backoff": self.rate_tracker.current_backoff,
        }
