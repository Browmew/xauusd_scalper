# src/features/extended_features.py
"""
Extended feature calculation functions for XAUUSD scalping strategy.

This module contains a comprehensive catalog of technical indicators and features
for use in the trading pipeline. All functions are optimized with numba for
low-latency processing.
"""

import numpy as np
import pandas as pd
from numba import jit
from typing import Union, Tuple
import warnings

warnings.filterwarnings('ignore')


@jit(nopython=True)
def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Optimized rolling mean calculation."""
    result = np.full_like(values, np.nan)
    for i in range(window - 1, len(values)):
        result[i] = np.mean(values[i - window + 1:i + 1])
    return result


@jit(nopython=True)
def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Optimized rolling standard deviation calculation."""
    result = np.full_like(values, np.nan)
    for i in range(window - 1, len(values)):
        result[i] = np.std(values[i - window + 1:i + 1])
    return result


@jit(nopython=True)
def _ema_calculation(values: np.ndarray, alpha: float) -> np.ndarray:
    """Optimized exponential moving average calculation."""
    result = np.full_like(values, np.nan)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def simple_moving_average(df: pd.DataFrame, period: int = 20, 
                         price_col: str = 'close') -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        df: Input DataFrame with OHLCV data
        period: Moving average period
        price_col: Column to calculate SMA on
        
    Returns:
        pd.Series: SMA values
    """
    values = df[price_col].values
    sma_values = _rolling_mean(values, period)
    return pd.Series(sma_values, index=df.index, name=f'sma_{period}')


def exponential_moving_average(df: pd.DataFrame, period: int = 20,
                              price_col: str = 'close') -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        df: Input DataFrame with OHLCV data
        period: EMA period
        price_col: Column to calculate EMA on
        
    Returns:
        pd.Series: EMA values
    """
    alpha = 2.0 / (period + 1.0)
    values = df[price_col].values
    ema_values = _ema_calculation(values, alpha)
    return pd.Series(ema_values, index=df.index, name=f'ema_{period}')


def weighted_moving_average(df: pd.DataFrame, period: int = 20,
                           price_col: str = 'close') -> pd.Series:
    """
    Calculate Weighted Moving Average (WMA).
    
    Args:
        df: Input DataFrame with OHLCV data
        period: WMA period
        price_col: Column to calculate WMA on
        
    Returns:
        pd.Series: WMA values
    """
    weights = np.arange(1, period + 1)
    wma_values = df[price_col].rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return pd.Series(wma_values, index=df.index, name=f'wma_{period}')


def relative_strength_index(df: pd.DataFrame, period: int = 14,
                           price_col: str = 'close') -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df: Input DataFrame with OHLCV data
        period: RSI period
        price_col: Column to calculate RSI on
        
    Returns:
        pd.Series: RSI values (0-100)
    """
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi, index=df.index, name=f'rsi_{period}')


def macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
         signal_period: int = 9, price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: Input DataFrame with OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
        price_col: Column to calculate MACD on
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = exponential_moving_average(df, fast_period, price_col)
    ema_slow = exponential_moving_average(df, slow_period, price_col)
    
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(
        pd.DataFrame({'close': macd_line}), signal_period)
    histogram = macd_line - signal_line
    
    macd_line.name = 'macd'
    signal_line.name = 'macd_signal'
    histogram.name = 'macd_histogram'
    
    return macd_line, signal_line, histogram


def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14,
                         d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Args:
        df: Input DataFrame with OHLCV data
        k_period: %K period
        d_period: %D period (SMA of %K)
        
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    
    k_percent = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    k_percent.name = f'stoch_k_{k_period}'
    d_percent.name = f'stoch_d_{d_period}'
    
    return k_percent, d_percent


def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                   price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: Input DataFrame with OHLCV data
        period: Moving average period
        std_dev: Standard deviation multiplier
        price_col: Column to calculate bands on
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    middle_band = simple_moving_average(df, period, price_col)
    std = df[price_col].rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    upper_band.name = f'bb_upper_{period}'
    middle_band.name = f'bb_middle_{period}'
    lower_band.name = f'bb_lower_{period}'
    
    return upper_band, middle_band, lower_band


def average_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: Input DataFrame with OHLCV data
        period: ATR period
        
    Returns:
        pd.Series: ATR values
    """
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = pd.Series(true_range).rolling(window=period).mean()
    
    return pd.Series(atr, index=df.index, name=f'atr_{period}')


def williams_percent_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Williams %R oscillator.
    
    Args:
        df: Input DataFrame with OHLCV data
        period: Lookback period
        
    Returns:
        pd.Series: Williams %R values (-100 to 0)
    """
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    
    williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    
    return pd.Series(williams_r, index=df.index, name=f'williams_r_{period}')


def commodity_channel_index(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).
    
    Args:
        df: Input DataFrame with OHLCV data
        period: CCI period
        
    Returns:
        pd.Series: CCI values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    return pd.Series(cci, index=df.index, name=f'cci_{period}')


def rate_of_change(df: pd.DataFrame, period: int = 12,
                  price_col: str = 'close') -> pd.Series:
    """
    Calculate Rate of Change (ROC).
    
    Args:
        df: Input DataFrame with OHLCV data
        period: ROC period
        price_col: Column to calculate ROC on
        
    Returns:
        pd.Series: ROC values
    """
    roc = ((df[price_col] - df[price_col].shift(period)) / 
           df[price_col].shift(period)) * 100
    
    return pd.Series(roc, index=df.index, name=f'roc_{period}')


def momentum(df: pd.DataFrame, period: int = 10,
            price_col: str = 'close') -> pd.Series:
    """
    Calculate Momentum indicator.
    
    Args:
        df: Input DataFrame with OHLCV data
        period: Momentum period
        price_col: Column to calculate momentum on
        
    Returns:
        pd.Series: Momentum values
    """
    mom = df[price_col] - df[price_col].shift(period)
    
    return pd.Series(mom, index=df.index, name=f'momentum_{period}')


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        df: Input DataFrame with OHLCV data
        
    Returns:
        pd.Series: OBV values
    """
    price_change = df['close'].diff()
    volume_direction = np.where(price_change > 0, df['volume'],
                               np.where(price_change < 0, -df['volume'], 0))
    obv = pd.Series(volume_direction).cumsum()
    
    return pd.Series(obv, index=df.index, name='obv')


def volume_weighted_average_price(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        df: Input DataFrame with OHLCV data
        
    Returns:
        pd.Series: VWAP values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    return pd.Series(vwap, index=df.index, name='vwap')


def price_volume_trend(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Price Volume Trend (PVT).
    
    Args:
        df: Input DataFrame with OHLCV data
        
    Returns:
        pd.Series: PVT values
    """
    price_change = df['close'].pct_change()
    pvt = (price_change * df['volume']).cumsum()
    
    return pd.Series(pvt, index=df.index, name='pvt')


def donchian_channel(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Donchian Channel.
    
    Args:
        df: Input DataFrame with OHLCV data
        period: Channel period
        
    Returns:
        Tuple of (Upper channel, Middle channel, Lower channel)
    """
    upper_channel = df['high'].rolling(window=period).max()
    lower_channel = df['low'].rolling(window=period).min()
    middle_channel = (upper_channel + lower_channel) / 2
    
    upper_channel.name = f'donchian_upper_{period}'
    middle_channel.name = f'donchian_middle_{period}'
    lower_channel.name = f'donchian_lower_{period}'
    
    return upper_channel, middle_channel, lower_channel


def keltner_channel(df: pd.DataFrame, period: int = 20,
                   multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channel.
    
    Args:
        df: Input DataFrame with OHLCV data
        period: EMA and ATR period
        multiplier: ATR multiplier
        
    Returns:
        Tuple of (Upper channel, Middle channel, Lower channel)
    """
    middle_channel = exponential_moving_average(df, period)
    atr = average_true_range(df, period)
    
    upper_channel = middle_channel + (multiplier * atr)
    lower_channel = middle_channel - (multiplier * atr)
    
    upper_channel.name = f'keltner_upper_{period}'
    middle_channel.name = f'keltner_middle_{period}'
    lower_channel.name = f'keltner_lower_{period}'
    
    return upper_channel, middle_channel, lower_channel


def aroon(df: pd.DataFrame, period: int = 25) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Aroon Up and Aroon Down indicators.
    
    Args:
        df: Input DataFrame with OHLCV data
        period: Aroon period
        
    Returns:
        Tuple of (Aroon Up, Aroon Down)
    """
    aroon_up = []
    aroon_down = []
    
    for i in range(len(df)):
        if i < period - 1:
            aroon_up.append(np.nan)
            aroon_down.append(np.nan)
        else:
            window = df.iloc[i - period + 1:i + 1]
            high_idx = window['high'].idxmax()
            low_idx = window['low'].idxmin()
            
            periods_since_high = i - df.index.get_loc(high_idx)
            periods_since_low = i - df.index.get_loc(low_idx)
            
            aroon_up.append(((period - periods_since_high) / period) * 100)
            aroon_down.append(((period - periods_since_low) / period) * 100)
    
    aroon_up_series = pd.Series(aroon_up, index=df.index, name=f'aroon_up_{period}')
    aroon_down_series = pd.Series(aroon_down, index=df.index, name=f'aroon_down_{period}')
    
    return aroon_up_series, aroon_down_series


def day_of_week(df: pd.DataFrame) -> pd.Series:
    """
    Extract day of week feature.
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        pd.Series: Day of week (0=Monday, 6=Sunday)
    """
    return pd.Series(df.index.dayofweek, index=df.index, name='day_of_week')


def hour_of_day(df: pd.DataFrame) -> pd.Series:
    """
    Extract hour of day feature.
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        pd.Series: Hour of day (0-23)
    """
    return pd.Series(df.index.hour, index=df.index, name='hour_of_day')


def is_london_session(df: pd.DataFrame) -> pd.Series:
    """
    Identify London trading session (8:00-16:00 GMT).
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        pd.Series: Boolean series for London session
    """
    hour = df.index.hour
    london_session = (hour >= 8) & (hour < 16)
    return pd.Series(london_session, index=df.index, name='is_london_session')


def is_new_york_session(df: pd.DataFrame) -> pd.Series:
    """
    Identify New York trading session (13:00-21:00 GMT).
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        pd.Series: Boolean series for New York session
    """
    hour = df.index.hour
    ny_session = (hour >= 13) & (hour < 21)
    return pd.Series(ny_session, index=df.index, name='is_ny_session')


def is_overlap_session(df: pd.DataFrame) -> pd.Series:
    """
    Identify London-New York overlap session (13:00-16:00 GMT).
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        pd.Series: Boolean series for overlap session
    """
    hour = df.index.hour
    overlap_session = (hour >= 13) & (hour < 16)
    return pd.Series(overlap_session, index=df.index, name='is_overlap_session')


def volatility_ratio(df: pd.DataFrame, short_period: int = 10,
                    long_period: int = 30) -> pd.Series:
    """
    Calculate volatility ratio.
    
    Args:
        df: Input DataFrame with OHLCV data
        short_period: Short-term volatility period
        long_period: Long-term volatility period
        
    Returns:
        pd.Series: Volatility ratio
    """
    short_vol = df['close'].pct_change().rolling(window=short_period).std()
    long_vol = df['close'].pct_change().rolling(window=long_period).std()
    vol_ratio = short_vol / long_vol
    
    return pd.Series(vol_ratio, index=df.index, name=f'vol_ratio_{short_period}_{long_period}')


def price_distance_from_sma(df: pd.DataFrame, period: int = 20,
                           price_col: str = 'close') -> pd.Series:
    """
    Calculate price distance from SMA as percentage.
    
    Args:
        df: Input DataFrame with OHLCV data
        period: SMA period
        price_col: Price column
        
    Returns:
        pd.Series: Price distance from SMA (%)
    """
    sma = simple_moving_average(df, period, price_col)
    distance = ((df[price_col] - sma) / sma) * 100
    
    return pd.Series(distance, index=df.index, name=f'price_dist_sma_{period}')


def bollinger_bandwidth(df: pd.DataFrame, period: int = 20,
                       std_dev: float = 2.0) -> pd.Series:
    """
    Calculate Bollinger Band Bandwidth.
    
    Args:
        df: Input DataFrame with OHLCV data
        period: Bollinger band period
        std_dev: Standard deviation multiplier
        
    Returns:
        pd.Series: Bollinger bandwidth
    """
    upper, middle, lower = bollinger_bands(df, period, std_dev)
    bandwidth = ((upper - lower) / middle) * 100
    
    return pd.Series(bandwidth, index=df.index, name=f'bb_bandwidth_{period}')


def bollinger_percent_b(df: pd.DataFrame, period: int = 20,
                       std_dev: float = 2.0, price_col: str = 'close') -> pd.Series:
    """
    Calculate Bollinger %B.
    
    Args:
        df: Input DataFrame with OHLCV data
        period: Bollinger band period
        std_dev: Standard deviation multiplier
        price_col: Price column
        
    Returns:
        pd.Series: Bollinger %B
    """
    upper, middle, lower = bollinger_bands(df, period, std_dev, price_col)
    percent_b = (df[price_col] - lower) / (upper - lower)
    
    return pd.Series(percent_b, index=df.index, name=f'bb_percent_b_{period}')


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index (MFI).
    
    Args:
        df: Input DataFrame with OHLCV data
        period: MFI period
        
    Returns:
        pd.Series: MFI values (0-100)
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_flow_sum = positive_flow.rolling(window=period).sum()
    negative_flow_sum = negative_flow.rolling(window=period).sum()
    
    money_ratio = positive_flow_sum / negative_flow_sum
    mfi = 100 - (100 / (1 + money_ratio))
    
    return pd.Series(mfi, index=df.index, name=f'mfi_{period}')


def chaikin_oscillator(df: pd.DataFrame, fast_period: int = 3,
                      slow_period: int = 10) -> pd.Series:
    """
    Calculate Chaikin Oscillator.
    
    Args:
        df: Input DataFrame with OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        
    Returns:
        pd.Series: Chaikin Oscillator values
    """
    adl = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    adl = adl.fillna(0) * df['volume']
    adl = adl.cumsum()
    
    adl_df = pd.DataFrame({'close': adl})
    fast_ema = exponential_moving_average(adl_df, fast_period)
    slow_ema = exponential_moving_average(adl_df, slow_period)
    
    chaikin_osc = fast_ema - slow_ema
    
    return pd.Series(chaikin_osc, index=df.index, name=f'chaikin_osc_{fast_period}_{slow_period}')