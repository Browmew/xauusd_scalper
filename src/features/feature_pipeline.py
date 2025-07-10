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
    hma_5,
    gkyz_volatility,
    order_book_imbalance_5l,
    vpvr_breakout,
    liquidity_sweep_detection,
    atr_14,
    calculate_price_features,
    calculate_spread_features, 
    calculate_volume_features,
    calculate_momentum_features
)

# Import only the functions that actually exist in extended_features.py
from .extended_features import (
    simple_moving_average,
    exponential_moving_average,
    weighted_moving_average,
    relative_strength_index,
    macd,
    stochastic_oscillator,
    bollinger_bands,
    average_true_range,
    williams_percent_r,
    commodity_channel_index,
    rate_of_change,
    momentum,
    on_balance_volume,
    volume_weighted_average_price,
    price_volume_trend,
    donchian_channel,
    keltner_channel,
    aroon,
    day_of_week,
    hour_of_day,
    is_london_session,
    is_new_york_session,
    is_overlap_session,
    volatility_ratio,
    price_distance_from_sma,
    bollinger_bandwidth,
    bollinger_percent_b,
    money_flow_index,
    chaikin_oscillator
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
    
    def _calculate_minimal_features(self, data: pd.DataFrame, l2_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate the core minimal feature set."""
        self.logger.debug("Calculating minimal features")

        # Use the functions that DO exist in minimal_features.py
        features = {}
        
        # Price features
        if 'close' in data.columns:
            features['hma_5'] = hma_5(data['close'])
            
        # OHLC volatility
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            features['gkyz_volatility'] = gkyz_volatility(data, window=14)
            features['atr_14'] = atr_14(data)

        # Order book features (if L2 data available)
        if l2_data is not None or all(f'bid_volume_{i}' in data.columns for i in range(1, 6)):
            features['obi_5l'] = order_book_imbalance_5l(data if l2_data is None else l2_data)

        # Volume profile features
        if 'volume' in data.columns and 'close' in data.columns:
            features['vpvr_breakout'] = vpvr_breakout(data)

        # Liquidity sweep detection
        if all(col in data.columns for col in ['high', 'low', 'close']):
            features['liquidity_sweep'] = liquidity_sweep_detection(data)

        # Use the basic feature calculation functions
        price_features = calculate_price_features(data)
        spread_features = calculate_spread_features(data)
        volume_features = calculate_volume_features(data)
        momentum_features = calculate_momentum_features(data)
        
        # Combine all features
        all_features = {**features}
        
        # Add the basic features
        for df in [price_features, spread_features, volume_features, momentum_features]:
            for col in df.columns:
                all_features[col] = df[col]

        # Combine all individual feature Series into a single DataFrame
        return pd.DataFrame(all_features, index=data.index)
    
    def _calculate_extended_features(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the extended feature set using available indicators.
        
        Args:
            tick_data: Input tick data
            
        Returns:
            DataFrame with extended features
        """
        self.logger.debug("Calculating extended features")
        
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
            sma_values = simple_moving_average(tick_data, period)
            extended_features[f'sma_{period}'] = sma_values
            extended_features[f'price_sma_{period}_ratio'] = tick_data['close'] / sma_values.replace(0, np.nan)
            
        for period in ema_periods:
            ema_values = exponential_moving_average(tick_data, period)
            extended_features[f'ema_{period}'] = ema_values
            extended_features[f'price_ema_{period}_ratio'] = tick_data['close'] / ema_values.replace(0, np.nan)
        
        # Momentum Indicators
        rsi_values = relative_strength_index(tick_data, rsi_period)
        extended_features['rsi'] = rsi_values
        extended_features['rsi_oversold'] = (rsi_values < 30).astype(float)
        extended_features['rsi_overbought'] = (rsi_values > 70).astype(float)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = bollinger_bands(tick_data, bb_period, bb_std)
        extended_features['bb_upper'] = bb_upper
        extended_features['bb_middle'] = bb_middle
        extended_features['bb_lower'] = bb_lower
        extended_features['bb_position'] = (tick_data['close'] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        extended_features['bb_squeeze'] = ((bb_upper - bb_lower) / bb_middle).fillna(0)
        
        # MACD
        macd_line, macd_signal_line, macd_histogram = macd(tick_data, macd_fast, macd_slow, macd_signal)
        extended_features['macd'] = macd_line
        extended_features['macd_signal'] = macd_signal_line
        extended_features['macd_histogram'] = macd_histogram
        extended_features['macd_bullish'] = (macd_line > macd_signal_line).astype(float)
        
        # Stochastic Oscillator
        stoch_k_values, stoch_d_values = stochastic_oscillator(tick_data, stoch_k, stoch_d)
        extended_features['stoch_k'] = stoch_k_values
        extended_features['stoch_d'] = stoch_d_values
        extended_features['stoch_oversold'] = (stoch_d_values < 20).astype(float)
        extended_features['stoch_overbought'] = (stoch_d_values > 80).astype(float)
        
        # Williams %R
        williams_values = williams_percent_r(tick_data, williams_period)
        extended_features['williams_r'] = williams_values
        
        # Commodity Channel Index
        cci_values = commodity_channel_index(tick_data, cci_period)
        extended_features['cci'] = cci_values
        extended_features['cci_extreme'] = (np.abs(cci_values) > 100).astype(float)
        
        # Average True Range
        atr_values = average_true_range(tick_data, atr_period)
        extended_features['atr'] = atr_values
        extended_features['atr_normalized'] = atr_values / tick_data['close']
        
        # Volume Indicators (if volume data available)
        if 'volume' in tick_data.columns:
            vwap_values = volume_weighted_average_price(tick_data)
            extended_features['vwap'] = vwap_values
            extended_features['price_vwap_ratio'] = tick_data['close'] / vwap_values.replace(0, np.nan)
            
            mfi_values = money_flow_index(tick_data, 14)
            extended_features['mfi'] = mfi_values
            
            obv_values = on_balance_volume(tick_data)
            extended_features['obv'] = obv_values
            
            # Price Volume Trend
            pvt_values = price_volume_trend(tick_data)
            extended_features['pvt'] = pvt_values
        
        # Additional technical indicators
        roc_values = rate_of_change(tick_data, 12)
        extended_features['roc'] = roc_values
        
        momentum_values = momentum(tick_data, 10)
        extended_features['momentum'] = momentum_values
        
        # Donchian Channel
        dc_upper, dc_middle, dc_lower = donchian_channel(tick_data, 20)
        extended_features['donchian_upper'] = dc_upper
        extended_features['donchian_middle'] = dc_middle
        extended_features['donchian_lower'] = dc_lower
        
        # Keltner Channel
        kc_upper, kc_middle, kc_lower = keltner_channel(tick_data, 20)
        extended_features['keltner_upper'] = kc_upper
        extended_features['keltner_middle'] = kc_middle
        extended_features['keltner_lower'] = kc_lower
        
        # Aroon indicators
        aroon_up, aroon_down = aroon(tick_data, 25)
        extended_features['aroon_up'] = aroon_up
        extended_features['aroon_down'] = aroon_down
        
        # Time-based features
        extended_features['day_of_week'] = day_of_week(tick_data)
        extended_features['hour_of_day'] = hour_of_day(tick_data)
        extended_features['is_london_session'] = is_london_session(tick_data)
        extended_features['is_ny_session'] = is_new_york_session(tick_data)
        extended_features['is_overlap_session'] = is_overlap_session(tick_data)
        
        # Statistical features
        volatility_ratio_values = volatility_ratio(tick_data, 10, 30)
        extended_features['volatility_ratio'] = volatility_ratio_values
        
        price_dist_sma = price_distance_from_sma(tick_data, 20)
        extended_features['price_distance_sma_20'] = price_dist_sma
        
        bb_bandwidth = bollinger_bandwidth(tick_data, 20, 2.0)
        extended_features['bb_bandwidth'] = bb_bandwidth
        
        bb_percent_b = bollinger_percent_b(tick_data, 20, 2.0)
        extended_features['bb_percent_b'] = bb_percent_b
        
        # Chaikin Oscillator
        chaikin_osc = chaikin_oscillator(tick_data, 3, 10)
        extended_features['chaikin_oscillator'] = chaikin_osc
        
        # Create DataFrame from extended features
        result_df = pd.DataFrame(extended_features, index=tick_data.index)
        
        # Fill any NaN values
        result_df = result_df.fillna(method='ffill').fillna(0)
        
        self.logger.debug(f"Generated {len(extended_features)} extended features")
        return result_df
    
    def get_feature_names(self) -> list:
        """
        Get list of all feature names that will be generated.
        
        Returns:
            List of feature names
        """
        minimal_features = [
            'hma_5', 'gkyz_volatility', 'atr_14', 'obi_5l', 
            'vpvr_breakout', 'liquidity_sweep',
            'price_return_1', 'price_return_5', 'price_return_10',
            'price_volatility_10', 'price_volatility_30',
            'spread_abs', 'spread_rel',
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
            'vwap', 'price_vwap_ratio', 'mfi', 'obv', 'pvt',
            'roc', 'momentum', 
            'donchian_upper', 'donchian_middle', 'donchian_lower',
            'keltner_upper', 'keltner_middle', 'keltner_lower',
            'aroon_up', 'aroon_down',
            'day_of_week', 'hour_of_day', 'is_london_session', 'is_ny_session', 'is_overlap_session',
            'volatility_ratio', 'price_distance_sma_20', 'bb_bandwidth', 'bb_percent_b',
            'chaikin_oscillator'
        ])
        
        return minimal_features + extended_features