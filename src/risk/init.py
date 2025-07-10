"""
Risk management module for XAUUSD scalping strategy.

This module provides risk management capabilities including position sizing,
daily loss limits, session filtering, and news blackout functionality.
"""

from .manager import RiskManager

__all__ = ['RiskManager']