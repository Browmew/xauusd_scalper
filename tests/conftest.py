"""
Pytest configuration and shared fixtures for the XAUUSD scalping stack test suite.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone


@pytest.fixture(scope='session')
def sample_data_df() -> pd.DataFrame:
    """
    Generate sample OHLCV and order book data for testing feature functions.
    
    Returns:
        pd.DataFrame: Test dataset with the exact schema expected by feature functions.
                     Contains 100 rows of realistic market data with proper column types.
    """
    # Create realistic time index with 100ms frequency
    periods = 100
    start_time = datetime(2024, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
    index = pd.date_range(
        start=start_time,
        periods=periods,
        freq='100ms',
        tz='UTC'
    )
    
    # Generate realistic XAUUSD price data around $2000
    np.random.seed(42)  # For reproducible tests
    base_price = 2000.0
    
    # Generate price walk with realistic volatility
    price_changes = np.random.normal(0, 0.5, periods)
    prices = base_price + np.cumsum(price_changes)
    
    # Create OHLC from price series with realistic spread and volatility
    opens = prices.copy()
    highs = opens + np.abs(np.random.normal(0, 0.3, periods))
    lows = opens - np.abs(np.random.normal(0, 0.3, periods))
    closes = opens + np.random.normal(0, 0.2, periods)
    
    # Ensure OHLC consistency (high >= max(open, close), low <= min(open, close))
    for i in range(periods):
        high_min = max(opens[i], closes[i])
        low_max = min(opens[i], closes[i])
        highs[i] = max(highs[i], high_min)
        lows[i] = min(lows[i], low_max)
    
    # Generate realistic volume data
    volumes = np.random.lognormal(mean=2.0, sigma=0.5, size=periods)
    
    # Create order book data (5 levels each side)
    order_book_data = {}
    
    # Generate bid side (below mid price)
    for level in range(1, 6):
        spread_offset = level * 0.1  # Increasing spread by level
        bid_prices = closes - spread_offset
        bid_volumes = np.random.exponential(scale=100.0, size=periods)
        
        order_book_data[f'bid_price_{level}'] = bid_prices.astype(np.float32)
        order_book_data[f'bid_volume_{level}'] = bid_volumes.astype(np.float32)
    
    # Generate ask side (above mid price)
    for level in range(1, 6):
        spread_offset = level * 0.1  # Increasing spread by level
        ask_prices = closes + spread_offset
        ask_volumes = np.random.exponential(scale=100.0, size=periods)
        
        order_book_data[f'ask_price_{level}'] = ask_prices.astype(np.float32)
        order_book_data[f'ask_volume_{level}'] = ask_volumes.astype(np.float32)
    
    # Construct the complete DataFrame
    data = {
        'open': opens.astype(np.float32),
        'high': highs.astype(np.float32),
        'low': lows.astype(np.float32),
        'close': closes.astype(np.float32),
        'volume': volumes.astype(np.float32),
        **order_book_data
    }
    
    df = pd.DataFrame(data, index=index)
    
    # Ensure no NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df


@pytest.fixture(scope='session')
def sample_data_df() -> pd.DataFrame:
    """
    Generate sample OHLCV and order book data for testing feature functions.
    """
    periods = 100
    start_time = datetime(2024, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
    index = pd.date_range(
        start=start_time,
        periods=periods,
        freq='100ms',
        tz='UTC'
    )
    
    np.random.seed(42)
    base_price = 2000.0
    price_changes = np.random.normal(0, 0.5, periods)
    prices = base_price + np.cumsum(price_changes)
    
    opens = prices.copy()
    highs = opens + np.abs(np.random.normal(0, 0.3, periods))
    lows = opens - np.abs(np.random.normal(0, 0.3, periods))
    closes = opens + np.random.normal(0, 0.2, periods)
    
    for i in range(periods):
        high_min = max(opens[i], closes[i])
        low_max = min(opens[i], closes[i])
        highs[i] = max(highs[i], high_min)
        lows[i] = min(lows[i], low_max)
    
    volumes = np.random.lognormal(mean=2.0, sigma=0.5, size=periods)
    
    # Create the main data structure with required columns
    data = {
        'timestamp': [ts.isoformat() for ts in index],
        'open': opens.astype(np.float32),
        'high': highs.astype(np.float32),
        'low': lows.astype(np.float32),
        'close': closes.astype(np.float32),
        'bid': (closes - 0.05).astype(np.float32),
        'ask': (closes + 0.05).astype(np.float32),
        'volume': volumes.astype(np.float32),
        'bid_volume': volumes.astype(np.float32)  # Add missing bid_volume
    }
    
    # Add order book data
    for level in range(1, 6):
        spread_offset = level * 0.1
        bid_prices = closes - spread_offset
        ask_prices = closes + spread_offset
        bid_volumes = np.random.exponential(scale=100.0, size=periods)
        ask_volumes = np.random.exponential(scale=100.0, size=periods)
        
        data[f'bid_price_{level}'] = bid_prices.astype(np.float32)
        data[f'bid_volume_{level}'] = bid_volumes.astype(np.float32)
        data[f'ask_price_{level}'] = ask_prices.astype(np.float32)
        data[f'ask_volume_{level}'] = ask_volumes.astype(np.float32)
    
    df = pd.DataFrame(data)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

@pytest.fixture(scope='session')
def sample_price_series(sample_data_df: pd.DataFrame) -> pd.Series:
    """
    Extract close prices as a Series for functions that only need price data.
    """
    return sample_data_df['close']


@pytest.fixture
def market_data_small() -> pd.DataFrame:
    """Generate a smaller dataset for performance-sensitive tests."""
    periods = 20
    index = pd.date_range(
        start='2024-01-01 09:30:00',
        periods=periods,
        freq='100ms',
        tz='UTC'
    )
    
    base_price = 2000.0
    price_increment = 0.1
    prices = [base_price + i * price_increment for i in range(periods)]
    
    data = {
        'timestamp': [ts.isoformat() for ts in index],
        'open': [p - 0.05 for p in prices],
        'high': [p + 0.1 for p in prices],
        'low': [p - 0.1 for p in prices],
        'close': prices,
        'bid': [p - 0.05 for p in prices],
        'ask': [p + 0.05 for p in prices],
        'volume': [100.0] * periods,
        'bid_volume': [50.0] * periods
    }
    
    # Add order book data
    for level in range(1, 6):
        data[f'bid_price_{level}'] = [p - 0.1 * level for p in prices]
        data[f'bid_volume_{level}'] = [50.0] * periods
        data[f'ask_price_{level}'] = [p + 0.1 * level for p in prices]
        data[f'ask_volume_{level}'] = [50.0] * periods
    
    # Convert to proper dtypes
    for col in data:
        if col != 'timestamp':
            data[col] = np.array(data[col], dtype=np.float32)
    
    return pd.DataFrame(data)