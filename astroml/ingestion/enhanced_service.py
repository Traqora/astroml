"""Enhanced ingestion service orchestrator for robust Stellar data streaming.

Provides high-level service management for multiple concurrent streams
with comprehensive monitoring and recovery capabilities.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Set

from prometheus_client import start_http_server

from astroml.ingestion.enhanced_stream import EnhancedStellarStream, EnhancedStreamConfig
from astroml.ingestion.state import StreamStateManager

logger = logging.getLogger("astroml.ingestion.enhanced_service")


class StreamService:
    """High-level service for managing multiple Stellar data streams."""
    
    def __init__(self, configs: List[EnhancedStreamConfig]) -> None:
        self.configs = configs
        self.streams: Dict[str, EnhancedStellarStream] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.state_manager = StreamStateManager()
        self._running: bool = False
        self._shutdown_event = asyncio.Event()
        
    async def start(self) -> None:
        """Start all configured streams."""
        self._running = True
        self._install_signal_handlers()
        
        logger.info("Starting enhanced stream service with %d streams", len(self.configs))
        
        # Initialize streams
        for config in self.configs:
            stream_id = f"{config.stream_type}_{config.horizon_url}"
            stream = EnhancedStellarStream(config)
            self.streams[stream_id] = stream
            
            # Load saved state if available
            saved_cursor = self.state_manager.get_cursor(stream_id)
            if saved_cursor and not config.cursor:
                stream._cursor = saved_cursor
                logger.info("Loaded saved cursor for %s: %s", stream_id, saved_cursor)
        
        # Start stream tasks
        for stream_id, stream in self.streams.items():
            task = asyncio.create_task(self._run_stream_with_monitoring(stream_id, stream))
            self.tasks[stream_id] = task
            logger.info("Started stream task: %s", stream_id)
        
        # Wait for shutdown
        await self._shutdown_event.wait()
        await self._shutdown()
    
    async def _run_stream_with_monitoring(self, stream_id: str, stream: EnhancedStellarStream) -> None:
        """Run a stream with continuous monitoring and recovery."""
        retry_count = 0
        max_retries = stream.config.max_retries
        
        while self._running and retry_count < max_retries:
            try:
                logger.info("Starting stream %s (attempt %d)", stream_id, retry_count + 1)
                
                async with stream:
                    await stream.run()
                
                # Stream completed successfully
                logger.info("Stream %s completed successfully", stream_id)
                break
                
            except asyncio.CancelledError:
                logger.info("Stream %s cancelled", stream_id)
                break
                
            except Exception as e:
                retry_count += 1
                logger.error(
                    "Stream %s failed (attempt %d): %s",
                    stream_id, retry_count, e
                )
                
                if retry_count >= max_retries:
                    logger.error("Max retries exceeded for stream %s", stream_id)
                    break
                
                # Save current state before retry
                if stream.cursor:
                    self.state_manager.save_cursor(stream_id, stream.cursor)
                
                # Exponential backoff before retry
                delay = min(stream.config.base_retry_delay * (2 ** retry_count), 
                          stream.config.max_retry_delay)
                logger.info("Retrying stream %s in %.1fs...", stream_id, delay)
                await asyncio.sleep(delay)
        
        # Save final state
        if stream.cursor:
            self.state_manager.save_cursor(stream_id, stream.cursor)
    
    async def _shutdown(self) -> None:
        """Graceful shutdown of all streams."""
        logger.info("Shutting down enhanced stream service...")
        
        self._running = False
        
        # Cancel all tasks
        for stream_id, task in self.tasks.items():
            if not task.done():
                logger.info("Cancelling stream task: %s", stream_id)
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        
        logger.info("Enhanced stream service shutdown complete")
    
    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal, sig)
    
    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signals."""
        logger.info("Received signal %s, initiating shutdown...", sig.name)
        self.stop()

    def stop(self) -> None:
        """Signal the service to stop."""
        self._shutdown_event.set()

    def get_service_stats(self) -> Dict[str, any]:
        """Get comprehensive service statistics."""
        stats = {
            "service_running": self._running,
            "active_streams": len([s for s in self.streams.values() if s._running]),
            "total_streams": len(self.streams),
            "stream_stats": {}
        }
        
        for stream_id, stream in self.streams.items():
            stats["stream_stats"][stream_id] = stream.get_stats()
        
        return stats


class MultiHorizonService:
    """Service for managing streams across multiple Horizon instances."""
    
    def __init__(self, prometheus_port: int = 8000) -> None:
        self.services: Dict[str, StreamService] = {}
        self._running: bool = False
        self._shutdown_event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []
        self.prometheus_port = prometheus_port
    
    def add_horizon_service(self, horizon_url: str, stream_types: List[str]) -> None:
        """Add a service for a specific Horizon instance."""
        configs = []
        for stream_type in stream_types:
            config = EnhancedStreamConfig(
                horizon_url=horizon_url,
                stream_type=stream_type
            )
            configs.append(config)
        
        service = StreamService(configs)
        service_id = horizon_url.replace("https://", "").replace("http://", "").replace("/", "_")
        self.services[service_id] = service
        
        logger.info("Added Horizon service: %s (streams: %s)", service_id, stream_types)
    
    async def start_all(self) -> None:
        """Start all Horizon services concurrently."""
        self._running = True
        self._install_signal_handlers()
        
        # Start Prometheus metrics server
        try:
            start_http_server(self.prometheus_port)
            logger.info("Prometheus metrics server started on port %d", self.prometheus_port)
        except Exception as e:
            logger.error("Failed to start Prometheus metrics server: %s", e)
        
        logger.info("Starting %d Horizon services", len(self.services))
        
        for service_id, service in self.services.items():
            task = asyncio.create_task(self._run_service_with_monitoring(service_id, service))
            self.tasks.append(task)
        
        # Wait for shutdown event
        await self._shutdown_event.wait()
        await self.stop_all()
        
        # Wait for all service tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Multi-horizon service shutdown complete")
    
    async def _run_service_with_monitoring(self, service_id: str, service: StreamService) -> None:
        """Run a service with monitoring."""
        try:
            await service.start()
        except asyncio.CancelledError:
            logger.info("Service %s cancelled", service_id)
        except Exception as e:
            logger.error("Service %s failed: %s", service_id, e)
        finally:
            if self._running and not self._shutdown_event.is_set():
                # If a service fails unexpectedly while we are still "running", 
                # we might want to shut down everything or just log it.
                # For now, let's just log it.
                logger.warning("Service %s stopped unexpectedly", service_id)
    
    async def stop_all(self) -> None:
        """Stop all services."""
        if not self._running:
            return
            
        logger.info("Stopping all Horizon services...")
        self._running = False
        
        for service in self.services.values():
            service.stop()
            
        self._shutdown_event.set()

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal, sig)
    
    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signals."""
        logger.info("Received signal %s, initiating multi-service shutdown...", sig.name)
        self._shutdown_event.set()

    def get_all_stats(self) -> Dict[str, any]:
        """Get aggregated statistics from all services."""
        all_stats = {
            "timestamp": datetime.now().isoformat(),
            "running": self._running,
            "services_count": len(self.services),
            "services": {}
        }
        
        for service_id, service in self.services.items():
            all_stats["services"][service_id] = service.get_service_stats()
            
        return all_stats


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """Configure structured logging.

    Delegates to :func:`astroml.utils.logging.configure_logging` so log
    level (``ASTROML_LOG_LEVEL``) and format (``ASTROML_LOG_FORMAT=
    text|json``) are consistent across every astroml entry point. See
    issue #195.
    """
    from astroml.utils.logging import configure_logging

    configure_logging()


async def run_single_stream(config: EnhancedStreamConfig) -> None:
    """Run a single enhanced stream."""
    async with EnhancedStellarStream(config) as stream:
        await stream.run()
        
        # Print final stats
        stats = stream.get_stats()
        logger.info(
            "Stream completed | processed=%d cursor=%s",
            stats["processed_count"],
            stats["cursor"]
        )


async def run_multi_stream_service(horizon_urls: List[str], stream_types: List[str]) -> None:
    """Run multi-Horizon streaming service."""
    service = MultiHorizonService()
    
    for horizon_url in horizon_urls:
        service.add_horizon_service(horizon_url, stream_types)
    
    try:
        await service.start_all()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await service.stop_all()


if __name__ == "__main__":
    _configure_logging()
    
    # Example usage - this would be expanded with proper CLI argument parsing
    config = EnhancedStreamConfig(
        horizon_url="https://horizon-testnet.stellar.org",
        stream_type="effects",
        cursor=None
    )
    
    asyncio.run(run_single_stream(config))
