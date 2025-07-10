"""
Backtesting module for the XAUUSD scalping system.

This module provides comprehensive backtesting capabilities including
exchange simulation, performance reporting, and risk analysis.
"""

from .exchange_simulator import ExchangeSimulator, Order, OrderType, OrderStatus

__all__ = [
    "ExchangeSimulator",
    "Order", 
    "OrderType",
    "OrderStatus"
]