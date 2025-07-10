"""
General-purpose helper functions for the XAUUSD scalping system.

This module provides core utilities including configuration loading, path management,
performance timing, and validation functions. All functions are designed for minimal
overhead to support low-latency trading requirements.

Big-O Considerations:
- Config loading: O(1) after initial load (cached)
- Path operations: O(1) 
- Validation functions: O(1) to O(n) depending on data size
- Timing utilities: O(1)
"""

import os
import sys
import time
import yaml
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar
from functools import wraps, lru_cache
import threading
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Global config cache
_config_cache: Optional[Dict[str, Any]] = None
_config_lock = threading.RLock()

@dataclass
class SystemInfo:
    """System information for optimization decisions."""
    os_name: str
    cpu_count: int
    total_memory_gb: float
    has_cuda: bool
    python_version: str
    platform: str


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Absolute path to project root
    """
    current_file = Path(__file__).resolve()
    # Navigate up from src/utils/helpers.py to project root
    return current_file.parent.parent.parent


def get_config_path() -> Path:
    """
    Get the path to the main configuration file.
    
    Returns:
        Path: Absolute path to config.yml
    """
    return get_project_root() / "configs" / "config.yml"


@lru_cache(maxsize=1)
def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load and cache the project configuration from YAML file.
    
    Thread-safe singleton pattern with caching for performance.
    
    Args:
        config_path: Optional path to config file. Defaults to standard location.
        
    Returns:
        Dict containing the full configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    global _config_cache
    
    if config_path is None:
        config_path = get_config_path()
    else:
        config_path = Path(config_path)
    
    with _config_lock:
        if _config_cache is not None:
            return _config_cache
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                _config_cache = yaml.safe_load(f)
                
            # Validate required sections
            required_sections = ['data', 'features', 'models', 'backtesting', 'live', 'risk', 'logging']
            missing_sections = [section for section in required_sections if section not in _config_cache]
            
            if missing_sections:
                raise ValueError(f"Missing required config sections: {missing_sections}")
                
            return _config_cache
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file {config_path}: {e}")


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Args:
        key_path: Dot-separated path to config value (e.g., 'models.lightgbm.num_leaves')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
        
    Example:
        >>> get_config_value('data.historical.tick_files')
        ['XAUUSD_2023_ticks.csv.gz']
    """
    config = load_config()
    keys = key_path.split('.')
    
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def validate_config() -> List[str]:
    """
    Validate the loaded configuration for completeness and correctness.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    config = load_config()
    
    # Validate data paths
    data_config = config.get('data', {})
    historical_path = get_project_root() / data_config.get('historical_data_path', 'data/historical')
    if not historical_path.exists():
        errors.append(f"Historical data path does not exist: {historical_path}")
    
    # Validate model parameters
    models_config = config.get('models', {})
    if 'lightgbm' in models_config:
        lgb_config = models_config['lightgbm']
        if lgb_config.get('num_leaves', 0) <= 0:
            errors.append("LightGBM num_leaves must be positive")
    
    # Validate risk parameters
    risk_config = config.get('risk', {})
    max_loss = risk_config.get('max_loss_usd', 0)
    if max_loss <= 0:
        errors.append("Risk max_loss_usd must be positive")
    
    return errors


def get_system_info() -> SystemInfo:
    """
    Get system information for optimization decisions.
    
    Returns:
        SystemInfo object with system details
    """
    # Check for CUDA availability
    has_cuda = False
    try:
        import cupy
        has_cuda = cupy.cuda.is_available()
    except ImportError:
        pass
    
    # Get memory info
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        total_memory_gb = 0.0
    
    return SystemInfo(
        os_name=platform.system(),
        cpu_count=os.cpu_count() or 1,
        total_memory_gb=total_memory_gb,
        has_cuda=has_cuda,
        python_version=platform.python_version(),
        platform=platform.platform()
    )


def timing_decorator(func: F) -> F:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
            
            # Import here to avoid circular imports
            from .logging import get_logger
            logger = get_logger(__name__)
            logger.debug(
                "Function execution completed",
                function=func.__name__,
                execution_time_ms=round(execution_time, 3)
            )
            return result
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            from .logging import get_logger
            logger = get_logger(__name__)
            logger.error(
                "Function execution failed",
                function=func.__name__,
                execution_time_ms=round(execution_time, 3),
                error=str(e)
            )
            raise
    return wrapper


class PerformanceTimer:
    """Context manager for measuring execution time with minimal overhead."""
    
    def __init__(self, name: str = "operation", log_result: bool = True):
        self.name = name
        self.log_result = log_result
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> 'PerformanceTimer':
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        if self.log_result and self.start_time is not None:
            execution_time = (self.end_time - self.start_time) * 1000
            from .logging import get_logger
            logger = get_logger(__name__)
            
            if exc_type is None:
                logger.debug(
                    "Timer completed",
                    operation=self.name,
                    execution_time_ms=round(execution_time, 3)
                )
            else:
                logger.error(
                    "Timer completed with error",
                    operation=self.name,
                    execution_time_ms=round(execution_time, 3),
                    error=str(exc_val)
                )
    
    @property
    def elapsed_ms(self) -> Optional[float]:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return None
        end_time = self.end_time or time.perf_counter()
        return (end_time - self.start_time) * 1000


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value  
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if abs(denominator) < 1e-10:  # Use small epsilon for floating point comparison
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def is_market_hours(timestamp: pd.Timestamp, timezone: str = 'UTC') -> bool:
    """
    Check if a timestamp falls within standard XAUUSD trading hours.
    
    XAUUSD trades nearly 24/5, but we exclude weekends and low-liquidity periods.
    
    Args:
        timestamp: Timestamp to check
        timezone: Timezone for the timestamp
        
    Returns:
        True if within trading hours
    """
    # Convert to UTC if needed
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize(timezone)
    timestamp_utc = timestamp.tz_convert('UTC')
    
    # Check if weekend (Saturday or Sunday in UTC)
    if timestamp_utc.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Exclude Friday 21:00 UTC to Sunday 21:00 UTC (weekend break)
    if timestamp_utc.weekday() == 4 and timestamp_utc.hour >= 21:  # Friday after 21:00
        return False
    if timestamp_utc.weekday() == 6 and timestamp_utc.hour < 21:   # Sunday before 21:00
        return False
    
    return True


def round_to_pip(value: float, pip_size: float = 0.01) -> float:
    """
    Round a price to the nearest pip for XAUUSD.
    
    Args:
        value: Price value to round
        pip_size: Size of one pip (default 0.01 for XAUUSD)
        
    Returns:
        Price rounded to nearest pip
    """
    return round(value / pip_size) * pip_size


def calculate_lot_size(account_balance: float, risk_pct: float, stop_loss_pips: float, 
                      pip_value: float = 1.0) -> float:
    """
    Calculate position size based on risk management rules.
    
    Args:
        account_balance: Current account balance in USD
        risk_pct: Risk percentage per trade (0.01 = 1%)
        stop_loss_pips: Stop loss distance in pips
        pip_value: Value per pip per lot (default 1.0 for XAUUSD mini lots)
        
    Returns:
        Lot size to risk the specified percentage
    """
    if stop_loss_pips <= 0 or pip_value <= 0:
        return 0.0
    
    risk_amount = account_balance * risk_pct
    lot_size = risk_amount / (stop_loss_pips * pip_value)
    
    # Round to reasonable lot sizes (0.01 increments)
    return round(lot_size, 2)


def format_currency(amount: float, currency: str = 'USD') -> str:
    """
    Format a currency amount for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == 'USD':
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"