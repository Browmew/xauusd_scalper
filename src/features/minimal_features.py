# src/features/minimal_features.py
"""
Minimal feature set for XAUUSD scalping strategy.
Implements core features with high-performance vectorized calculations.
"""

import numpy as np
import pandas as pd
import numba
from typing import Union, Tuple
from numba import jit


def hma_5(prices: pd.Series) -> pd.Series:
    """
    Calculate Hull Moving Average with period 5.
    
    HMA(n) = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    
    Args:
        prices: Series of price values (typically close prices)
        
    Returns:
        Series with HMA-5 values
    """
    def wma(data: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average calculation."""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    period = 5
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # Calculate WMAs
    wma_half = wma(prices, half_period)
    wma_full = wma(prices, period)
    
    # HMA intermediate calculation
    hma_intermediate = 2 * wma_half - wma_full
    
    # Final HMA
    hma = wma(hma_intermediate, sqrt_period)
    
    return hma


@jit(nopython=True)
def _gkyz_volatility_numba(
    opens: np.ndarray, 
    highs: np.ndarray, 
    lows: np.ndarray, 
    closes: np.ndarray,
    window: int
) -> np.ndarray:
    """
    Numba-optimized GKYZ volatility calculation.
    """
    n = len(opens)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        gk_sum = 0.0
        rs_sum = 0.0
        
        for j in range(i - window + 1, i + 1):
            if j > 0:  # Need previous close
                # Garman-Klass component
                log_hl = np.log(highs[j] / lows[j])
                log_co = np.log(closes[j] / opens[j])
                gk = 0.5 * log_hl * log_hl - (2 * np.log(2) - 1) * log_co * log_co
                
                # Rogers-Satchell component  
                log_ho = np.log(highs[j] / opens[j])
                log_hc = np.log(highs[j] / closes[j])
                log_lo = np.log(lows[j] / opens[j])
                log_lc = np.log(lows[j] / closes[j])
                rs = log_ho * log_hc + log_lo * log_lc
                
                gk_sum += gk
                rs_sum += rs
        
        # Yang-Zhang adjustment (simplified)
        result[i] = np.sqrt((gk_sum + rs_sum) / window * 252)  # Annualized
    
    return result


def gkyz_volatility(ohlc_data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Garman-Klass-Yang-Zhang volatility estimator.
    
    A superior volatility estimator that uses OHLC data to provide
    more accurate volatility estimates than simple returns-based methods.
    
    Args:
        ohlc_data: DataFrame with 'open', 'high', 'low', 'close' columns
        window: Rolling window period for volatility calculation
        
    Returns:
        Series with GKYZ volatility values (annualized)
    """
    opens = ohlc_data['open'].values
    highs = ohlc_data['high'].values
    lows = ohlc_data['low'].values
    closes = ohlc_data['close'].values
    
    volatility = _gkyz_volatility_numba(opens, highs, lows, closes, window)
    
    return pd.Series(volatility, index=ohlc_data.index, name='gkyz_volatility')


def order_book_imbalance_5l(data: pd.DataFrame) -> pd.Series:
    """
    Calculate 5-level Order Book Imbalance (OBI).
    
    Measures the imbalance between bid and ask volumes across the top 5 levels
    of the order book. Positive values indicate bid pressure, negative values
    indicate ask pressure.
    
    Args:
        data: DataFrame with bid_volume_1-5 and ask_volume_1-5 columns
        
    Returns:
        Series with OBI values ranging from -1 to 1
    """
    # Sum volumes across 5 levels
    bid_volumes = data[[f'bid_volume_{i}' for i in range(1, 6)]].sum(axis=1)
    ask_volumes = data[[f'ask_volume_{i}' for i in range(1, 6)]].sum(axis=1)
    
    # Calculate imbalance
    total_volume = bid_volumes + ask_volumes
    imbalance = (bid_volumes - ask_volumes) / total_volume
    
    # Handle division by zero
    imbalance = imbalance.fillna(0)
    
    return imbalance.rename('obi_5l')


@jit(nopython=True)
def _vpvr_breakout_numba(
    prices: np.ndarray,
    volumes: np.ndarray,
    lookback: int,
    min_volume_threshold: float
) -> np.ndarray:
    """
    Numba-optimized VPVR breakout detection.
    """
    n = len(prices)
    result = np.zeros(n)
    
    for i in range(lookback, n):
        # Get price range for lookback period
        start_idx = i - lookback
        price_slice = prices[start_idx:i]
        volume_slice = volumes[start_idx:i]
        
        # Create price-volume profile
        min_price = np.min(price_slice)
        max_price = np.max(price_slice)
        
        if max_price > min_price:
            # Discretize prices into bins
            n_bins = min(20, lookback // 5)  # Adaptive number of bins
            price_step = (max_price - min_price) / n_bins
            
            # Find highest volume price level
            max_volume = 0.0
            high_volume_price = min_price
            
            for bin_idx in range(n_bins):
                bin_start = min_price + bin_idx * price_step
                bin_end = bin_start + price_step
                
                # Sum volume in this price bin
                bin_volume = 0.0
                for j in range(len(price_slice)):
                    if bin_start <= price_slice[j] < bin_end:
                        bin_volume += volume_slice[j]
                
                if bin_volume > max_volume:
                    max_volume = bin_volume
                    high_volume_price = bin_start + price_step / 2
            
            # Check for breakout
            current_price = prices[i]
            if max_volume > min_volume_threshold:
                # Breakout above high volume area
                if current_price > high_volume_price * 1.001:  # 10 pip threshold
                    result[i] = 1.0
                # Breakdown below high volume area
                elif current_price < high_volume_price * 0.999:
                    result[i] = -1.0
    
    return result


def vpvr_breakout(data: pd.DataFrame, lookback: int = 100, volume_threshold_multiplier: float = 2.0) -> pd.Series:
    """
    Detect Volume Profile Visible Range (VPVR) breakouts.
    
    Identifies when price breaks through areas of high volume concentration,
    which often act as support/resistance levels.
    
    Args:
        data: DataFrame with 'close' and 'volume' columns
        lookback: Period to analyze for volume profile
        volume_threshold_multiplier: Minimum volume threshold as multiple of average
        
    Returns:
        Series with breakout signals (1: bullish breakout, -1: bearish breakdown, 0: no signal)
    """
    prices = data['close'].values
    volumes = data['volume'].values
    
    # Calculate dynamic volume threshold
    avg_volume = np.mean(volumes[~np.isnan(volumes)])
    min_volume_threshold = avg_volume * volume_threshold_multiplier
    
    breakouts = _vpvr_breakout_numba(prices, volumes, lookback, min_volume_threshold)
    
    return pd.Series(breakouts, index=data.index, name='vpvr_breakout')


@jit(nopython=True)
def _liquidity_sweep_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int,
    sweep_threshold: float
) -> np.ndarray:
    """
    Numba-optimized liquidity sweep detection.
    """
    n = len(highs)
    result = np.zeros(n)
    
    for i in range(lookback + 1, n):
        # Find recent high and low
        start_idx = i - lookback
        recent_high = np.max(highs[start_idx:i])
        recent_low = np.min(lows[start_idx:i])
        
        current_high = highs[i]
        current_low = lows[i]
        current_close = closes[i]
        prev_close = closes[i-1]
        
        # Bullish liquidity sweep (sweep below recent low then rally)
        if (current_low < recent_low and 
            current_close > prev_close and
            current_close > current_low + (current_high - current_low) * sweep_threshold):
            result[i] = 1.0
            
        # Bearish liquidity sweep (sweep above recent high then decline)
        elif (current_high > recent_high and 
              current_close < prev_close and
              current_close < current_high - (current_high - current_low) * sweep_threshold):
            result[i] = -1.0
    
    return result


def liquidity_sweep_detection(data: pd.DataFrame, lookback: int = 20, sweep_threshold: float = 0.6) -> pd.Series:
    """
    Detect liquidity sweep patterns.
    
    Identifies when price briefly moves beyond recent highs/lows to trigger
    stop losses, then quickly reverses direction. These are often signs of
    institutional activity.
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        lookback: Period to look back for recent highs/lows
        sweep_threshold: Minimum reversal fraction of the range
        
    Returns:
        Series with sweep signals (1: bullish sweep, -1: bearish sweep, 0: no signal)
    """
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    
    sweeps = _liquidity_sweep_numba(highs, lows, closes, lookback, sweep_threshold)
    
    return pd.Series(sweeps, index=data.index, name='liquidity_sweep')


@jit(nopython=True)
def _atr_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """
    Numba-optimized ATR calculation.
    """
    n = len(highs)
    atr_values = np.full(n, np.nan)
    
    if n < 2:
        return atr_values
    
    # Calculate True Range
    tr_values = np.zeros(n)
    for i in range(1, n):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr_values[i] = max(tr1, tr2, tr3)
    
    # Calculate ATR using Wilder's smoothing
    if n > period:
        # First ATR is simple average
        atr_values[period] = np.mean(tr_values[1:period+1])
        
        # Subsequent ATRs use Wilder's smoothing
        for i in range(period + 1, n):
            atr_values[i] = (atr_values[i-1] * (period - 1) + tr_values[i]) / period
    
    return atr_values


def atr_14(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Average True Range with 14-period.
    
    ATR measures market volatility by calculating the average of true ranges
    over a specified period. True Range is the maximum of:
    - Current High - Current Low
    - |Current High - Previous Close|
    - |Current Low - Previous Close|
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        
    Returns:
        Series with ATR-14 values
    """
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    
    atr_values = _atr_numba(highs, lows, closes, 14)
    
    return pd.Series(atr_values, index=data.index, name='atr_14')

# ──────────────────────────────────────────────────────────────
# ✂  paste at the *end* of src/features/minimal_features.py  ✂
# ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

__all__ = [
    # previously-existing exports …,
    "calculate_price_features",
    "calculate_spread_features",
    "calculate_volume_features",
    "calculate_momentum_features",
]

# ------------------------------------------------------------------
# Basic “minimal” feature set used by FeaturePipeline
# ------------------------------------------------------------------
def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core price-based features for the minimal pipeline.

    Returns a DataFrame with:
      • price_return_1, _5, _10         – pct-change of close
      • price_volatility_10, _30        – rolling σ of 1-bar returns
    """
    closes = df["close"]
    one_bar_ret = closes.pct_change()
    out = pd.DataFrame(
        {
            "price_return_1":  one_bar_ret,
            "price_return_5":  closes.pct_change(5),
            "price_return_10": closes.pct_change(10),
            "price_volatility_10": one_bar_ret.rolling(10).std(ddof=0),
            "price_volatility_30": one_bar_ret.rolling(30).std(ddof=0),
        },
        index=df.index,
    ).fillna(0.0)
    return out


def calculate_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Absolute and relative bid/ask spread.  If bid/ask columns are missing,
    returns zeros so downstream code never breaks.
    """
    if {"bid", "ask"}.issubset(df.columns):
        spread = df["ask"] - df["bid"]
        rel_spread = spread / df["close"].replace(0, np.nan)
    else:
        spread = rel_spread = pd.Series(0.0, index=df.index)

    return pd.DataFrame(
        {
            "spread_abs": spread,
            "spread_rel": rel_spread.replace([np.inf, -np.inf], 0).fillna(0),
        },
        index=df.index,
    )


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple volume stats.

      • volume_mean_10, volume_std_10  – rolling mean / std
      • volume_ratio_5                 – current vol ÷ 5-bar mean
    """
    vol = df["volume"]
    mean10 = vol.rolling(10).mean()
    out = pd.DataFrame(
        {
            "volume_mean_10": mean10,
            "volume_std_10": vol.rolling(10).std(ddof=0),
            "volume_ratio_5": vol / vol.rolling(5).mean().replace(0, np.nan),
        },
        index=df.index,
    ).fillna(0.0)
    return out


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple price-difference momentum features (not percentage).

      • momentum_1, _5, _10  – close diff over N bars
    """
    closes = df["close"]
    out = pd.DataFrame(
        {
            "momentum_1": closes.diff(1),
            "momentum_5": closes.diff(5),
            "momentum_10": closes.diff(10),
        },
        index=df.index,
    ).fillna(0.0)
    return out
# ──────────────────────────────────────────────────────────────
