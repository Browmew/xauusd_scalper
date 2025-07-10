# src/data_ingestion/__init__.py
"""
Data Ingestion Module for XAUUSD High-Frequency Trading System.

This module provides efficient loading and alignment of tick-level and Level 2 
order book data from compressed CSV files.
"""

from .loader import DataLoader

__all__ = ['DataLoader']