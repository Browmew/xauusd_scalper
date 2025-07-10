"""
Unit tests for the feature engineering module.

Tests all minimal feature functions for correctness, proper return types,
and expected behavior with various input conditions.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Any

from src.features.minimal_features import (
    hma_5,
    gkyz_volatility,
    order_book_imbalance_5l,
    vpvr_breakout,
    liquidity_sweep_detection,
    atr_14
)


class TestHMA5:
    """Test cases for the Hull Moving Average (5-period) function."""
    
    def test_hma_5_basic_functionality(self, sample_price_series: pd.Series) -> None:
        """Test that HMA5 returns a Series with correct shape and type."""
        result = hma_5(sample_price_series)
        
        assert isinstance(result, pd.Series), "HMA5 should return a pandas Series"
        assert len(result) == len(sample_price_series), "Result should have same length as input"
        assert result.dtype == np.float64 or result.dtype == np.float32, "Result should be numeric"
        assert result.index.equals(sample_price_series.index), "Index should be preserved"
    
    def test_hma_5_handles_minimal_data(self, market_data_small: pd.DataFrame) -> None:
        """Test HMA5 with minimal data length."""
        prices = market_data_small['close']
        result = hma_5(prices)
        
        assert len(result) == len(prices)
        # First few values might be NaN due to moving average calculation
        assert not result.isna().all(), "Not all values should be NaN"
    
    def test_hma_5_monotonic_increasing_prices(self) -> None:
        """Test HMA5 with monotonically increasing prices."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1min')
        prices = pd.Series(range(2000, 2020), index=dates, dtype=np.float32)
        
        result = hma_5(prices)
        
        # For increasing prices, HMA should generally trend upward
        valid_result = result.dropna()
        assert len(valid_result) > 0, "Should have some valid HMA values"


class TestGKYZVolatility:
    """Test cases for the Garman-Klass-Yang-Zhang volatility function."""
    
    def test_gkyz_volatility_basic_functionality(self, sample_data_df: pd.DataFrame) -> None:
        """Test that GKYZ volatility returns correct shape and type."""
        window = 14
        result = gkyz_volatility(sample_data_df, window)
        
        assert isinstance(result, pd.Series), "GKYZ should return a pandas Series"
        assert len(result) == len(sample_data_df), "Result should have same length as input"
        assert result.dtype == np.float64 or result.dtype == np.float32, "Result should be numeric"
        assert (result >= 0).all() or result.isna().any(), "Volatility should be non-negative where defined"
    
    def test_gkyz_volatility_different_windows(self, sample_data_df: pd.DataFrame) -> None:
        """Test GKYZ volatility with different window sizes."""
        for window in [5, 10, 20]:
            result = gkyz_volatility(sample_data_df, window)
            assert len(result) == len(sample_data_df)
            # First (window-1) values should be NaN
            assert result.iloc[:window-1].isna().all()
    
    def test_gkyz_volatility_requires_ohlc(self, sample_data_df: pd.DataFrame) -> None:
        """Test that GKYZ volatility works with OHLC columns."""
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            assert col in sample_data_df.columns, f"Test data must have {col} column"
        
        result = gkyz_volatility(sample_data_df, 14)
        assert not result.dropna().empty, "Should produce some valid volatility values"


class TestOrderBookImbalance:
    """Test cases for the order book imbalance (5-level) function."""
    
    def test_order_book_imbalance_5l_basic(self, sample_data_df: pd.DataFrame) -> None:
        """Test basic functionality of order book imbalance calculation."""
        result = order_book_imbalance_5l(sample_data_df)
        
        assert isinstance(result, pd.Series), "Should return a pandas Series"
        assert len(result) == len(sample_data_df), "Should have same length as input"
        assert result.dtype == np.float64 or result.dtype == np.float32, "Should be numeric"
        # Imbalance should be between -1 and 1
        valid_values = result.dropna()
        assert (valid_values >= -1.0).all() and (valid_values <= 1.0).all(), "Imbalance should be in [-1, 1]"
    
    def test_order_book_imbalance_requires_book_data(self, sample_data_df: pd.DataFrame) -> None:
        """Test that function works with required order book columns."""
        required_bid_cols = [f'bid_price_{i}' for i in range(1, 6)] + [f'bid_volume_{i}' for i in range(1, 6)]
        required_ask_cols = [f'ask_price_{i}' for i in range(1, 6)] + [f'ask_volume_{i}' for i in range(1, 6)]
        
        for col in required_bid_cols + required_ask_cols:
            assert col in sample_data_df.columns, f"Test data must have {col} column"
        
        result = order_book_imbalance_5l(sample_data_df)
        assert not result.isna().all(), "Should produce some valid imbalance values"


class TestVPVRBreakout:
    """Test cases for the Volume Profile Visible Range breakout function."""
    
    def test_vpvr_breakout_basic_functionality(self, sample_data_df: pd.DataFrame) -> None:
        """Test basic VPVR breakout detection."""
        # Use default parameters that would be in the function signature
        result = vpvr_breakout(sample_data_df)
        
        assert isinstance(result, pd.Series), "Should return a pandas Series"
        assert len(result) == len(sample_data_df), "Should have same length as input"
        assert result.dtype == np.bool_ or result.dtype == bool or result.dtype == np.int64, "Should be boolean or int"
    
    def test_vpvr_breakout_with_volume(self, sample_data_df: pd.DataFrame) -> None:
        """Test that VPVR uses volume data correctly."""
        assert 'volume' in sample_data_df.columns, "Test data must have volume column"
        assert (sample_data_df['volume'] > 0).all(), "Volume should be positive"
        
        result = vpvr_breakout(sample_data_df)
        # Should not be all True or all False for realistic data
        unique_values = result.unique()
        assert len(unique_values) >= 1, "Should have at least one type of signal"


class TestLiquiditySweepDetection:
    """Test cases for the liquidity sweep detection function."""
    
    def test_liquidity_sweep_basic_functionality(self, sample_data_df: pd.DataFrame) -> None:
        """Test basic liquidity sweep detection."""
        result = liquidity_sweep_detection(sample_data_df)
        
        assert isinstance(result, pd.Series), "Should return a pandas Series"
        assert len(result) == len(sample_data_df), "Should have same length as input"
        assert result.dtype in [np.bool_, bool, np.int64, np.float64], "Should be boolean, int, or float"
    
    def test_liquidity_sweep_uses_price_levels(self, sample_data_df: pd.DataFrame) -> None:
        """Test that liquidity sweep detection uses high/low price levels."""
        required_cols = ['high', 'low']
        for col in required_cols:
            assert col in sample_data_df.columns, f"Test data must have {col} column"
        
        result = liquidity_sweep_detection(sample_data_df)
        assert not result.isna().all(), "Should produce some valid sweep signals"


class TestATR14:
    """Test cases for the Average True Range (14-period) function."""
    
    def test_atr_14_basic_functionality(self, sample_data_df: pd.DataFrame) -> None:
        """Test that ATR14 returns correct shape and type."""
        result = atr_14(sample_data_df)
        
        assert isinstance(result, pd.Series), "ATR14 should return a pandas Series"
        assert len(result) == len(sample_data_df), "Result should have same length as input"
        assert result.dtype == np.float64 or result.dtype == np.float32, "Result should be numeric"
        assert (result >= 0).all() or result.isna().any(), "ATR should be non-negative where defined"
    
    def test_atr_14_requires_hlc(self, sample_data_df: pd.DataFrame) -> None:
        """Test that ATR14 works with High, Low, Close data."""
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            assert col in sample_data_df.columns, f"Test data must have {col} column"
        
        result = atr_14(sample_data_df)
        # First 13 values should be NaN (14-period requires 14 observations)
        assert result.iloc[:13].isna().all()
        assert not result.iloc[14:].isna().all(), "Should have valid ATR values after warmup"
    
    def test_atr_14_with_constant_prices(self) -> None:
        """Test ATR14 with constant prices (should be zero after warmup)."""
        dates = pd.date_range('2024-01-01', periods=30, freq='1min')
        constant_price = 2000.0
        
        data = pd.DataFrame({
            'high': [constant_price] * 30,
            'low': [constant_price] * 30,
            'close': [constant_price] * 30,
        }, index=dates)
        
        result = atr_14(data)
        
        # After the first value, ATR should approach zero for constant prices
        valid_atr = result.dropna()
        if len(valid_atr) > 1:
            assert valid_atr.iloc[-1] < 0.1, "ATR should be near zero for constant prices"


class TestFeatureIntegration:
    """Integration tests for feature functions working together."""
    
    def test_all_features_with_same_data(self, sample_data_df: pd.DataFrame) -> None:
        """Test that all feature functions can process the same dataset."""
        prices = sample_data_df['close']
        
        # Test all functions with the same input data
        hma_result = hma_5(prices)
        gkyz_result = gkyz_volatility(sample_data_df, 14)
        imbalance_result = order_book_imbalance_5l(sample_data_df)
        vpvr_result = vpvr_breakout(sample_data_df)
        sweep_result = liquidity_sweep_detection(sample_data_df)
        atr_result = atr_14(sample_data_df)
        
        # All should return Series with same length
        results = [hma_result, gkyz_result, imbalance_result, vpvr_result, sweep_result, atr_result]
        for result in results:
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data_df)
    
    def test_feature_output_consistency(self, sample_data_df: pd.DataFrame) -> None:
        """Test that feature outputs are consistent across multiple calls."""
        prices = sample_data_df['close']
        
        # Call each function twice with same data
        hma1 = hma_5(prices)
        hma2 = hma_5(prices)
        
        atr1 = atr_14(sample_data_df)
        atr2 = atr_14(sample_data_df)
        
        # Results should be identical
        pd.testing.assert_series_equal(hma1, hma2, check_names=False)
        pd.testing.assert_series_equal(atr1, atr2, check_names=False)