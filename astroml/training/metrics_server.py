"""Prometheus metrics server initialization and management.

This module provides utilities for starting and managing the Prometheus
metrics HTTP server for exporting training and ingestion metrics.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_metrics_server_running = False
_metrics_server_port = 8000


def get_metrics_port() -> int:
    """Get Prometheus metrics server port from config or environment.
    
    Returns:
        Port number for metrics server
    """
    global _metrics_server_port
    
    # Check environment variable first
    if 'PROMETHEUS_PORT' in os.environ:
        try:
            port = int(os.environ['PROMETHEUS_PORT'])
            _metrics_server_port = port
            return port
        except ValueError:
            logger.warning(
                "Invalid PROMETHEUS_PORT environment variable: %s",
                os.environ['PROMETHEUS_PORT']
            )
    
    return _metrics_server_port


def start_metrics_server(port: Optional[int] = None) -> bool:
    """Start Prometheus metrics HTTP server.
    
    Args:
        port: Optional port number. If not provided, uses environment variable
              or default (8000).
              
    Returns:
        True if server started successfully, False if already running or on error
    """
    global _metrics_server_running, _metrics_server_port
    
    if _metrics_server_running:
        logger.debug("Metrics server already running on port %d", _metrics_server_port)
        return False
    
    try:
        from prometheus_client import start_http_server
        
        if port is None:
            port = get_metrics_port()
        else:
            _metrics_server_port = port
        
        start_http_server(port)
        _metrics_server_running = True
        
        logger.info(
            "Prometheus metrics server started on port %d",
            port
        )
        logger.info(
            "Metrics endpoint: http://localhost:%d/metrics",
            port
        )
        
        return True
        
    except OSError as e:
        if e.errno == 48 or e.errno == 98:  # Port already in use
            logger.warning(
                "Port %d already in use, metrics server may already be running",
                port
            )
            _metrics_server_running = True
            return False
        else:
            logger.error("Failed to start metrics server: %s", e)
            return False
            
    except Exception as e:
        logger.error("Failed to start metrics server: %s", e)
        return False


def is_metrics_server_running() -> bool:
    """Check if metrics server is running.
    
    Returns:
        True if running, False otherwise
    """
    return _metrics_server_running


def set_metrics_port(port: int) -> None:
    """Set Prometheus metrics server port.
    
    Should be called before start_metrics_server().
    
    Args:
        port: Port number
    """
    global _metrics_server_port
    _metrics_server_port = port
