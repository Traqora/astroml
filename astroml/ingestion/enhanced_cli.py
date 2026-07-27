"""Enhanced CLI for the robust Stellar ingestion service.

Provides command-line interface for running enhanced streams with
comprehensive configuration options and monitoring capabilities.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from astroml.ingestion.enhanced_service import (
    EnhancedStreamConfig,
    MultiHorizonService,
    run_multi_stream_service,
    run_single_stream,
)
from astroml.ingestion.state import StreamStateManager

logger = logging.getLogger("astroml.ingestion.enhanced_cli")


def _configure_logging(level: str = "INFO") -> None:
    """Configure structured logging.

    Delegates to :func:`astroml.utils.logging.configure_logging` so log
    level (``ASTROML_LOG_LEVEL``) and format (``ASTROML_LOG_FORMAT=
    text|json``) are consistent across every astroml entry point. See
    issue #195.
    """
    from astroml.utils.logging import configure_logging

    configure_logging(level=level)


def _parse_enhanced_args() -> argparse.Namespace:
    """Parse command-line arguments for enhanced streaming."""
    parser = argparse.ArgumentParser(
        description="Enhanced Stellar blockchain data ingestion with robust error handling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream effects from testnet
  python -m astroml.ingestion.enhanced_cli --stream-type effects --horizon testnet
  
  # Stream operations from mainnet with cursor
  python -m astroml.ingestion.enhanced_cli --stream-type operations --horizon mainnet --cursor 12345678
  
  # Multi-Horizon streaming
  python -m astroml.ingestion.enhanced_cli --multi --horizons testnet,mainnet --streams effects,operations
  
  # Custom rate limiting
  python -m astroml.ingestion.enhanced_cli --stream-type effects --max-retries 10 --rate-limit-backoff 10
        """
    )
    
    # Basic configuration
    parser.add_argument(
        "--stream-type",
        choices=["effects", "operations"],
        default="effects",
        help="Type of data to stream (default: effects)",
    )
    
    parser.add_argument(
        "--horizon",
        choices=["testnet", "mainnet"],
        default="testnet",
        help="Horizon network to use (default: testnet)",
    )
    
    parser.add_argument(
        "--horizon-url",
        type=str,
        help="Custom Horizon URL (overrides --horizon)",
    )
    
    parser.add_argument(
        "--cursor",
        type=str,
        help="Starting cursor/paging_token. Use 'now' for live-only.",
    )
    
    # Multi-Horizon options
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Enable multi-Horizon streaming mode",
    )
    
    parser.add_argument(
        "--horizons",
        type=str,
        help="Comma-separated list of horizons (testnet,mainnet,custom URLs)",
    )
    
    parser.add_argument(
        "--streams",
        type=str,
        default="effects,operations",
        help="Comma-separated list of stream types for multi-mode (default: effects,operations)",
    )
    
    # Retry and error handling
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts (default: 5)",
    )
    
    parser.add_argument(
        "--base-retry-delay",
        type=float,
        default=1.0,
        help="Base retry delay in seconds (default: 1.0)",
    )
    
    parser.add_argument(
        "--max-retry-delay",
        type=float,
        default=60.0,
        help="Maximum retry delay in seconds (default: 60.0)",
    )
    
    parser.add_argument(
        "--rate-limit-backoff",
        type=float,
        default=5.0,
        help="Rate limit backoff multiplier in seconds (default: 5.0)",
    )
    
    # Performance tuning
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for API requests (default: 100)",
    )
    
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=5.0,
        help="Batch timeout in seconds (default: 5.0)",
    )
    
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=30.0,
        help="Connection timeout in seconds (default: 30.0)",
    )
    
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=60.0,
        help="Stream timeout in seconds (default: 60.0)",
    )
    
    # Health monitoring
    parser.add_argument(
        "--health-check-interval",
        type=float,
        default=30.0,
        help="Health check interval in seconds (default: 30.0)",
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    
    # State management
    parser.add_argument(
        "--state-dir",
        type=str,
        default="state",
        help="Directory for storing stream state (default: state)",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved state",
    )
    
    return parser.parse_args()


def _get_horizon_url(horizon: str, custom_url: str = None) -> str:
    """Get Horizon URL from horizon name or custom URL."""
    if custom_url:
        return custom_url
    
    horizon_urls = {
        "testnet": "https://horizon-testnet.stellar.org",
        "mainnet": "https://horizon.stellar.org",
    }
    
    if horizon not in horizon_urls:
        raise ValueError(f"Unknown horizon: {horizon}")
    
    return horizon_urls[horizon]


def _create_enhanced_config(args: argparse.Namespace) -> EnhancedStreamConfig:
    """Create EnhancedStreamConfig from CLI arguments."""
    horizon_url = _get_horizon_url(args.horizon, args.horizon_url)
    
    # Load cursor from state if resuming
    cursor = args.cursor
    if args.resume and not cursor:
        state_manager = StreamStateManager()
        stream_id = f"{args.stream_type}_{horizon_url.replace('https://', '').replace('http://', '').replace('/', '_')}"
        cursor = state_manager.get_cursor(stream_id)
        if cursor:
            logger.info("Resumed from saved cursor: %s", cursor)
    
    return EnhancedStreamConfig(
        horizon_url=horizon_url,
        stream_type=args.stream_type,
        cursor=cursor,
        max_retries=args.max_retries,
        base_retry_delay=args.base_retry_delay,
        max_retry_delay=args.max_retry_delay,
        rate_limit_backoff=args.rate_limit_backoff,
        connection_timeout=args.connection_timeout,
        stream_timeout=args.stream_timeout,
        health_check_interval=args.health_check_interval,
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout,
    )


def _parse_horizon_list(horizons_str: str) -> list[str]:
    """Parse comma-separated horizon list."""
    horizons = []
    for horizon in horizons_str.split(","):
        horizon = horizon.strip()
        if horizon in ["testnet", "mainnet"]:
            horizons.append(_get_horizon_url(horizon))
        elif horizon.startswith("http"):
            horizons.append(horizon)
        else:
            raise ValueError(f"Invalid horizon: {horizon}")
    return horizons


def _parse_stream_list(streams_str: str) -> list[str]:
    """Parse comma-separated stream type list."""
    valid_streams = ["effects", "operations"]
    streams = []
    
    for stream in streams_str.split(","):
        stream = stream.strip()
        if stream not in valid_streams:
            raise ValueError(f"Invalid stream type: {stream}")
        streams.append(stream)
    
    return streams


async def _run_single_stream_enhanced(args: argparse.Namespace) -> None:
    """Run a single enhanced stream."""
    config = _create_enhanced_config(args)
    
    logger.info(
        "Starting enhanced stream | horizon=%s type=%s cursor=%s",
        config.horizon_url,
        config.stream_type,
        config.cursor or "now"
    )
    
    try:
        await run_single_stream(config)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Stream failed: %s", e)
        raise


async def _run_multi_stream_enhanced(args: argparse.Namespace) -> None:
    """Run multi-Horizon enhanced streaming."""
    horizon_urls = _parse_horizon_list(args.horizons)
    stream_types = _parse_stream_list(args.streams)
    
    logger.info(
        "Starting multi-Horizon service | horizons=%s streams=%s",
        horizon_urls,
        stream_types
    )
    
    try:
        await run_multi_stream_service(horizon_urls, stream_types)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Multi-Horizon service failed: %s", e)
        raise


async def _main() -> None:
    """Main CLI entry point."""
    args = _parse_enhanced_args()
    _configure_logging(args.log_level)
    
    # Ensure state directory exists
    state_dir = Path(args.state_dir)
    state_dir.mkdir(exist_ok=True)
    
    try:
        if args.multi:
            await _run_multi_stream_enhanced(args)
        else:
            await _run_single_stream_enhanced(args)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
