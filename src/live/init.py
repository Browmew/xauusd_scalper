"""
Live trading module for XAUUSD scalping system.

This module contains the core components for live trading execution:
- LiveFeedHandler: Asynchronous WebSocket data ingestion
- SmartOrderRouter: Multi-broker order execution interface
"""

from .feed_handler import LiveFeedHandler
from .order_router import SmartOrderRouter

__all__ = [
    "LiveFeedHandler",
    "SmartOrderRouter",
]