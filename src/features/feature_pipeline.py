"""
Feature pipeline for XAUUSD scalping system.
Handles both minimal and extended feature engineering with vectorized operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
import logging
from numba import jit

from .minimal_features import (
    calculate_price_features,
    calculate_spread_features, 
    calculate_volume_features,
    calculate_momentum_features
)

# Import all extended features
from .extended_features import (
    simple_moving_average,
    exponential_moving_average,
    relative_strength_index,
    bollinger_bands,
    macd,
    stochastic_oscillator,
    williams_r,
    commodity_channel_index,
    average_true_range,
    parabolic_sar,
    ichimoku_cloud,
    fibonacci_retracements,
    pivot_points,
    support_resistance_levels,
    volume_weighted_average_price,
    money_flow_index,
    accumulation_distribution_line,
    on_balance_volume,
    chaikin_oscillator,
    volume_rate_of_change,
    price_volume_trend,
    ease_of_movement,
    negative_volume_index,
    positive_volume_index,
    volume_zone_oscillator,
    klinger_oscillator,
    twiggs_money_flow,
    elder_ray_index,
    chande_momentum_oscillator,
    detrended_price_oscillator,
    ultimate_oscillator,
    mass_index,
    choppiness_index,
    aroon_oscillator,
    balance_of_power,
    typical_price,
    weighted_close,
    median_price,
    true_range,
    directional_movement_index,
    trix,
    vortex_indicator,
    know_sure_thing,
    schaff_trend_cycle,
    elder_force_index,
    emv_oscillator,
    money_flow_oscillator,
    price_oscillator,
    absolute_price_oscillator,
    percentage_price_oscillator,
    linear_regression_slope,
    linear_regression_intercept,
    standard_error,
    r_squared,
    time_series_forecast,
    correlation_coefficient,
    beta_coefficient,
    historical_volatility,
    chaikin_volatility,
    standard_deviation,
    variance,
    covariance,
    mean_deviation,
    median_absolute_deviation
)


class FeaturePipeline:
    """
    Advanced feature engineering pipeline for high-frequency trading.
    
    Supports both minimal and extended feature sets with configurable parameters.
    All computations are vectorized and optimized for low-latency operations.
    """
    
    def __init__(self, features_config: Dict[str, Any]):
        """
        Initialize the feature pipeline.
        
        Args:
            features_config: Configuration dictionary containing feature parameters
        """
        self.features_config = features_config
        self.logger = logging.getLogger(__name__)
        
        # Cache frequently used config values
        self.use_extended_features = features_config.get('use_extended_features', False)
        self.feature_params = features_config.get('features', {})
        
        self.logger.info(f"FeaturePipeline initialized with extended_features={self.use_extended_features}")
    
    def transform(self, tick_data: pd.DataFrame, l2_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transform raw market data into engineered features.
        
        Args:
            tick_data: Tick-by-tick price and volume data
            l2_data: Level 2 order book data (optional)
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.debug(f"Transforming {len(tick_data)} tick data points")
        
        # Start with minimal features
        features_df = self._calculate_minimal_features(tick_data, l2_data)
        
        # Add extended features if enabled
        if self.use_extended_features:
            extended_features = self._calculate_extended_features(tick_data)
            features_df = pd.concat([features_df, extended_features], axis=1)
            
        # Remove any NaN values that may have been introduced
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        self.logger.debug(f"Generated {features_df.shape[1]} features")
        return features_df
    
    # This corrected version directly calls the functions that DO exist
    # in minimal_features.py, achieving the same goal as your original code.

    def _calculate_minimal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the core minimal feature set."""
        self.logger.debug("Calculating minimal features")

        # This corrected version directly calls the functions that DO exist
        # in the minimal_features.py file.

        features = {
            'hma_5': hma_5(data['close']),
            'gkyz_volatility': gkyz_volatility(data, window=14),
            'obi_5l': order_book_imbalance_5l(data),
            'vpvr_breakout': vpvr_breakout(data),
            'liquidity_sweep': liquidity_sweep_detection(data),
            'atr_14': atr_14(data)
        }

        # Combine all individual feature Series into a single DataFrame
        return pd.DataFrame(features)
    
    def _calculate_extended_features(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the extended feature set using all available indicators.
        
        Args:
            tick_data: Input tick data
            
        Returns:
            DataFrame with extended features
        """
        self.logger.debug("Calculating extended features")
        
        prices = tick_data['price'].values
        volumes = tick_data['volume'].values
        high_prices = tick_data.get('high', prices).values
        low_prices = tick_data.get('low', prices).values
        close_prices = prices
        
        # Get feature parameters from config
        sma_periods = self.feature_params.get('sma_periods', [5, 10, 20, 50])
        ema_periods = self.feature_params.get('ema_periods', [5, 10, 20, 50])
        rsi_period = self.feature_params.get('rsi_period', 14)
        bb_period = self.feature_params.get('bollinger_period', 20)
        bb_std = self.feature_params.get('bollinger_std', 2.0)
        macd_fast = self.feature_params.get('macd_fast', 12)
        macd_slow = self.feature_params.get('macd_slow', 26)
        macd_signal = self.feature_params.get('macd_signal', 9)
        stoch_k = self.feature_params.get('stoch_k', 14)
        stoch_d = self.feature_params.get('stoch_d', 3)
        williams_period = self.feature_params.get('williams_period', 14)
        cci_period = self.feature_params.get('cci_period', 20)
        atr_period = self.feature_params.get('atr_period', 14)
        
        extended_features = {}
        
        # Moving Averages
        for period in sma_periods:
            sma_values = simple_moving_average(close_prices, period)
            extended_features[f'sma_{period}'] = sma_values
            extended_features[f'price_sma_{period}_ratio'] = close_prices / np.where(sma_values != 0, sma_values, 1)
            
        for period in ema_periods:
            ema_values = exponential_moving_average(close_prices, period)
            extended_features[f'ema_{period}'] = ema_values
            extended_features[f'price_ema_{period}_ratio'] = close_prices / np.where(ema_values != 0, ema_values, 1)
        
        # Momentum Indicators
        rsi_values = relative_strength_index(close_prices, rsi_period)
        extended_features['rsi'] = rsi_values
        extended_features['rsi_oversold'] = (rsi_values < 30).astype(float)
        extended_features['rsi_overbought'] = (rsi_values > 70).astype(float)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = bollinger_bands(close_prices, bb_period, bb_std)
        extended_features['bb_upper'] = bb_upper
        extended_features['bb_middle'] = bb_middle
        extended_features['bb_lower'] = bb_lower
        extended_features['bb_position'] = (close_prices - bb_lower) / np.where((bb_upper - bb_lower) != 0, (bb_upper - bb_lower), 1)
        extended_features['bb_squeeze'] = ((bb_upper - bb_lower) / bb_middle).fillna(0)
        
        # MACD
        macd_line, macd_signal_line, macd_histogram = macd(close_prices, macd_fast, macd_slow, macd_signal)
        extended_features['macd'] = macd_line
        extended_features['macd_signal'] = macd_signal_line
        extended_features['macd_histogram'] = macd_histogram
        extended_features['macd_bullish'] = (macd_line > macd_signal_line).astype(float)
        
        # Stochastic Oscillator
        stoch_k_values, stoch_d_values = stochastic_oscillator(high_prices, low_prices, close_prices, stoch_k, stoch_d)
        extended_features['stoch_k'] = stoch_k_values
        extended_features['stoch_d'] = stoch_d_values
        extended_features['stoch_oversold'] = (stoch_d_values < 20).astype(float)
        extended_features['stoch_overbought'] = (stoch_d_values > 80).astype(float)
        
        # Williams %R
        williams_values = williams_r(high_prices, low_prices, close_prices, williams_period)
        extended_features['williams_r'] = williams_values
        
        # Commodity Channel Index
        cci_values = commodity_channel_index(high_prices, low_prices, close_prices, cci_period)
        extended_features['cci'] = cci_values
        extended_features['cci_extreme'] = (np.abs(cci_values) > 100).astype(float)
        
        # Average True Range
        atr_values = average_true_range(high_prices, low_prices, close_prices, atr_period)
        extended_features['atr'] = atr_values
        extended_features['atr_normalized'] = atr_values / close_prices
        
        # Volume Indicators
        if len(volumes) > 0:
            vwap_values = volume_weighted_average_price(close_prices, volumes)
            extended_features['vwap'] = vwap_values
            extended_features['price_vwap_ratio'] = close_prices / np.where(vwap_values != 0, vwap_values, 1)
            
            mfi_values = money_flow_index(high_prices, low_prices, close_prices, volumes, 14)
            extended_features['mfi'] = mfi_values
            
            ad_line = accumulation_distribution_line(high_prices, low_prices, close_prices, volumes)
            extended_features['ad_line'] = ad_line
            
            obv_values = on_balance_volume(close_prices, volumes)
            extended_features['obv'] = obv_values
            
            # Volume rate of change
            vol_roc = volume_rate_of_change(volumes, 10)
            extended_features['volume_roc'] = vol_roc
        
        # Price-based indicators
        typical_price_values = typical_price(high_prices, low_prices, close_prices)
        extended_features['typical_price'] = typical_price_values
        
        weighted_close_values = weighted_close(high_prices, low_prices, close_prices)
        extended_features['weighted_close'] = weighted_close_values
        
        median_price_values = median_price(high_prices, low_prices)
        extended_features['median_price'] = median_price_values
        
        # Volatility indicators
        historical_vol = historical_volatility(close_prices, 20)
        extended_features['historical_volatility'] = historical_vol
        
        # Advanced momentum indicators
        cmo_values = chande_momentum_oscillator(close_prices, 14)
        extended_features['cmo'] = cmo_values
        
        dpo_values = detrended_price_oscillator(close_prices, 20)
        extended_features['dpo'] = dpo_values
        
        ultimate_osc = ultimate_oscillator(high_prices, low_prices, close_prices, 7, 14, 28)
        extended_features['ultimate_oscillator'] = ultimate_osc
        
        # Trend indicators
        dmi_plus, dmi_minus, adx = directional_movement_index(high_prices, low_prices, close_prices, 14)
        extended_features['dmi_plus'] = dmi_plus
        extended_features['dmi_minus'] = dmi_minus
        extended_features['adx'] = adx
        extended_features['trend_strength'] = (adx > 25).astype(float)
        
        trix_values = trix(close_prices, 14)
        extended_features['trix'] = trix_values
        
        # Support/Resistance levels
        support_levels, resistance_levels = support_resistance_levels(high_prices, low_prices, close_prices, 20)
        extended_features['nearest_support'] = support_levels
        extended_features['nearest_resistance'] = resistance_levels
        extended_features['support_distance'] = (close_prices - support_levels) / close_prices
        extended_features['resistance_distance'] = (resistance_levels - close_prices) / close_prices
        
        # Statistical features
        linear_slope = linear_regression_slope(close_prices, 20)
        extended_features['linear_slope'] = linear_slope
        extended_features['trend_direction'] = np.sign(linear_slope)
        
        r_squared_values = r_squared(close_prices, 20)
        extended_features['r_squared'] = r_squared_values
        extended_features['trend_reliability'] = (r_squared_values > 0.7).astype(float)
        
        # Create DataFrame from extended features
        result_df = pd.DataFrame(extended_features, index=tick_data.index)
        
        self.logger.debug(f"Generated {len(extended_features)} extended features")
        return result_df
    
    def get_feature_names(self) -> list:
        """
        Get list of all feature names that will be generated.
        
        Returns:
            List of feature names
        """
        minimal_features = [
            'price_return_1', 'price_return_5', 'price_return_10',
            'price_volatility_10', 'price_volatility_30',
            'volume_mean_10', 'volume_std_10', 'volume_ratio_5',
            'momentum_1', 'momentum_5', 'momentum_10'
        ]
        
        if not self.use_extended_features:
            return minimal_features
        
        # Extended feature names (would be dynamically generated based on config)
        extended_features = []
        
        # Add SMA features
        sma_periods = self.feature_params.get('sma_periods', [5, 10, 20, 50])
        for period in sma_periods:
            extended_features.extend([f'sma_{period}', f'price_sma_{period}_ratio'])
        
        # Add EMA features
        ema_periods = self.feature_params.get('ema_periods', [5, 10, 20, 50])
        for period in ema_periods:
            extended_features.extend([f'ema_{period}', f'price_ema_{period}_ratio'])
        
        # Add other extended features
        extended_features.extend([
            'rsi', 'rsi_oversold', 'rsi_overbought',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_squeeze',
            'macd', 'macd_signal', 'macd_histogram', 'macd_bullish',
            'stoch_k', 'stoch_d', 'stoch_oversold', 'stoch_overbought',
            'williams_r', 'cci', 'cci_extreme', 'atr', 'atr_normalized',
            'vwap', 'price_vwap_ratio', 'mfi', 'ad_line', 'obv', 'volume_roc',
            'typical_price', 'weighted_close', 'median_price',
            'historical_volatility', 'cmo', 'dpo', 'ultimate_oscillator',
            'dmi_plus', 'dmi_minus', 'adx', 'trend_strength', 'trix',
            'nearest_support', 'nearest_resistance', 'support_distance', 'resistance_distance',
            'linear_slope', 'trend_direction', 'r_squared', 'trend_reliability'
        ])
        
        return minimal_features + extended_features