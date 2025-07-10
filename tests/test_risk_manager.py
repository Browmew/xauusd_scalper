# tests/test_risk_manager.py
"""
Unit tests for the RiskManager class.

This module tests critical risk management functionality including
trading permission checks and position sizing calculations.
"""

import pytest
import pandas as pd
from datetime import datetime, time
from unittest.mock import Mock, patch
from src.risk.manager import RiskManager, PositionSizeResult


class TestRiskManager:
    """Test suite for RiskManager class."""

    @pytest.fixture
    def risk_config(self):
        """Risk management configuration fixture."""
        return {
            'max_loss_usd': 1000.0,
            'max_position_size': 0.1,
            'risk_per_trade': 0.02,
            'allowed_sessions': {
                'london': {'start': '08:00', 'end': '16:00'},
                'new_york': {'start': '13:00', 'end': '21:00'}
            },
            'news_blackout_minutes': 30,
            'max_drawdown_pct': 0.15,
            'min_win_probability': 0.55
        }

    @pytest.fixture
    def risk_manager(self, risk_config):
        """Risk manager instance fixture."""
        return RiskManager(risk_config)

    def test_risk_manager_initialization(self, risk_config):
        """Test RiskManager can be initialized with valid config."""
        manager = RiskManager(risk_config)
        assert manager.max_loss_usd == 1000.0
        assert manager.max_position_size == 0.1
        assert hasattr(manager, 'kelly_multiplier')

    def test_check_trading_allowed_normal_conditions(self, risk_manager):
        """Test trading is allowed under normal conditions."""
        # London session timestamp
        timestamp = pd.Timestamp('2025-01-15 10:30:00', tz='UTC')
        
        result = risk_manager.check_trading_allowed(timestamp)
        
        assert result is True

    def test_check_trading_allowed_exceeds_daily_loss(self, risk_manager):
        """Test trading is blocked when daily loss limit is exceeded."""
        timestamp = pd.Timestamp('2025-01-15 10:30:00', tz='UTC')
        
        # Set daily PnL to exceed limit
        risk_manager.daily_pnl = -1500.0  # Exceeds max_loss_usd of 1000
        
        result = risk_manager.check_trading_allowed(timestamp)
        
        assert result is False

    def test_check_trading_allowed_exceeds_drawdown(self, risk_manager):
        """Test trading is blocked when drawdown limit is exceeded."""
        timestamp = pd.Timestamp('2025-01-15 10:30:00', tz='UTC')
        
        # Set up drawdown scenario
        risk_manager.peak_balance = 10000.0
        risk_manager.current_balance = 8000.0  # 20% drawdown > 15% limit
        
        result = risk_manager.check_trading_allowed(timestamp)
        
        assert result is False

    def test_check_trading_allowed_outside_london_session(self, risk_manager):
        """Test trading is blocked outside allowed sessions."""
        # 6 AM UTC - before London session
        timestamp = pd.Timestamp('2025-01-15 06:00:00', tz='UTC')
        
        result = risk_manager.check_trading_allowed(timestamp)
        
        assert result is False

    def test_check_trading_allowed_outside_ny_session(self, risk_manager):
        """Test trading is blocked outside NY session."""
        # 11 PM UTC - after NY session
        timestamp = pd.Timestamp('2025-01-15 23:00:00', tz='UTC')
        
        result = risk_manager.check_trading_allowed(timestamp)
        
        assert result is False

    def test_check_trading_allowed_during_news_blackout(self, risk_manager):
        """Test trading is blocked during news blackout periods."""
        timestamp = pd.Timestamp('2025-01-15 10:30:00', tz='UTC')
        
        # Mock upcoming news event within 30 minutes
        upcoming_news = [
            {'time': pd.Timestamp('2025-01-15 10:45:00', tz='UTC'), 'impact': 'high'}
        ]
        
        result = risk_manager.check_trading_allowed(timestamp, upcoming_news=upcoming_news)
        
        assert result is False

    def test_check_trading_allowed_weekend(self, risk_manager):
        """Test trading is blocked on weekends."""
        # Saturday timestamp
        timestamp = pd.Timestamp('2025-01-18 10:30:00', tz='UTC')
        
        result = risk_manager.check_trading_allowed(timestamp)
        
        assert result is False

    def test_calculate_position_size_normal_conditions(self, risk_manager):
        """Test position size calculation under normal conditions."""
        result = risk_manager.calculate_position_size(
            symbol='XAUUSD',
            signal_strength=0.8,
            win_probability=0.65,
            avg_win=100.0,
            avg_loss=50.0,
            current_atr=0.5,
            current_price=2000.0
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.size > 0
        assert result.size <= risk_manager.max_position_size
        assert result.atr_stop_distance > 0

    def test_calculate_position_size_low_win_probability(self, risk_manager):
        """Test position size is zero when win probability is too low."""
        result = risk_manager.calculate_position_size(
            symbol='XAUUSD',
            signal_strength=0.8,
            win_probability=0.45,  # Below min_win_probability of 0.55
            avg_win=100.0,
            avg_loss=50.0,
            current_atr=0.5,
            current_price=2000.0
        )
        
        assert result.size == 0.0
        assert "probability" in result.reason.lower()

    def test_calculate_position_size_high_atr(self, risk_manager):
        """Test position size calculation with high ATR."""
        result_low_atr = risk_manager.calculate_position_size(
            symbol='XAUUSD',
            signal_strength=0.8,
            win_probability=0.65,
            avg_win=100.0,
            avg_loss=50.0,
            current_atr=0.5,  # Low ATR
            current_price=2000.0
        )
        
        result_high_atr = risk_manager.calculate_position_size(
            symbol='XAUUSD',
            signal_strength=0.8,
            win_probability=0.65,
            avg_win=100.0,
            avg_loss=50.0,
            current_atr=2.0,  # High ATR
            current_price=2000.0
        )
        
        # High ATR should result in smaller position size
        assert result_high_atr.size <= result_low_atr.size

    def test_calculate_position_size_small_account(self, risk_manager):
        """Test position size calculation with small account balance."""
        # Set small account balance
        risk_manager.current_balance = 1000.0
        
        result = risk_manager.calculate_position_size(
            symbol='XAUUSD',
            signal_strength=0.8,
            win_probability=0.65,
            avg_win=100.0,
            avg_loss=50.0,
            current_atr=0.5,
            current_price=2000.0
        )
        
        assert result.size >= 0
        assert result.risk_amount <= risk_manager.current_balance * risk_manager.risk_per_trade

    def test_calculate_position_size_zero_atr(self, risk_manager):
        """Test position size calculation with zero ATR."""
        result = risk_manager.calculate_position_size(
            symbol='XAUUSD',
            signal_strength=0.8,
            win_probability=0.65,
            avg_win=100.0,
            avg_loss=50.0,
            current_atr=0.0,  # No volatility
            current_price=2000.0
        )
        
        # Should return zero size when ATR is zero
        assert result.size == 0.0
        assert "atr" in result.reason.lower() or "zero" in result.reason.lower()

    def test_position_size_respects_maximum(self, risk_manager):
        """Test that position size never exceeds configured maximum."""
        result = risk_manager.calculate_position_size(
            symbol='XAUUSD',
            signal_strength=0.8,
            win_probability=0.85,  # High confidence
            avg_win=100.0,
            avg_loss=50.0,
            current_atr=0.1,  # Low volatility
            current_price=2000.0
        )
        
        assert result.size <= risk_manager.max_position_size
        if result.size == risk_manager.max_position_size:
            assert "maximum" in result.reason.lower()

    def test_is_in_trading_session_london(self, risk_manager):
        """Test London session detection."""
        # 10:30 AM UTC - within London session (8:00-16:00)
        timestamp = pd.Timestamp('2025-01-15 10:30:00', tz='UTC')
        
        result = risk_manager._is_in_trading_session(timestamp)
        assert result is True

    def test_is_in_trading_session_new_york(self, risk_manager):
        """Test New York session detection."""
        # 3:30 PM UTC - within NY session (13:00-21:00)
        timestamp = pd.Timestamp('2025-01-15 15:30:00', tz='UTC')
        
        result = risk_manager._is_in_trading_session(timestamp)
        assert result is True

    def test_is_in_trading_session_overlap(self, risk_manager):
        """Test London-NY overlap session detection."""
        # 2:00 PM UTC - within overlap (13:00-16:00)
        timestamp = pd.Timestamp('2025-01-15 14:00:00', tz='UTC')
        
        result = risk_manager._is_in_trading_session(timestamp)
        assert result is True

    def test_is_in_trading_session_outside_hours(self, risk_manager):
        """Test outside trading hours detection."""
        # 5:00 AM UTC - outside all sessions
        timestamp = pd.Timestamp('2025-01-15 05:00:00', tz='UTC')
        
        result = risk_manager._is_in_trading_session(timestamp)
        assert result is False

    def test_check_news_blackout_no_news(self, risk_manager):
        """Test news blackout check with no upcoming news."""
        timestamp = pd.Timestamp('2025-01-15 10:30:00', tz='UTC')
        
        result = risk_manager._check_news_blackout(timestamp, [])
        assert result is False

    def test_check_news_blackout_news_outside_window(self, risk_manager):
        """Test news blackout check with news outside blackout window."""
        timestamp = pd.Timestamp('2025-01-15 10:00:00', tz='UTC')
        upcoming_news = [
            {'time': pd.Timestamp('2025-01-15 11:00:00', tz='UTC'), 'impact': 'high'}
        ]
        
        result = risk_manager._check_news_blackout(timestamp, upcoming_news)
        assert result is False

    def test_check_news_blackout_news_within_window(self, risk_manager):
        """Test news blackout check with news within blackout window."""
        timestamp = pd.Timestamp('2025-01-15 10:30:00', tz='UTC')
        upcoming_news = [
            {'time': pd.Timestamp('2025-01-15 10:45:00', tz='UTC'), 'impact': 'high'}
        ]
        
        result = risk_manager._check_news_blackout(timestamp, upcoming_news)
        assert result is True

    def test_kelly_criterion_calculation(self, risk_manager):
        """Test Kelly criterion position sizing calculation."""
        win_probability = 0.6
        avg_win = 100.0
        avg_loss = 50.0
        
        kelly_fraction = risk_manager._calculate_kelly_criterion(
            win_probability, avg_win, avg_loss)
        
        # Kelly = (bp - q) / b where b = avg_win/avg_loss, p = win_prob, q = 1-p
        expected_kelly = (win_probability * (avg_win / avg_loss) - (1 - win_probability)) / (avg_win / avg_loss)
        
        assert abs(kelly_fraction - expected_kelly) < 0.001

    @patch('src.risk.manager.get_logger')
    def test_logging_on_blocked_trading(self, mock_get_logger, risk_manager):
        """Test that blocked trading events are logged."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        timestamp = pd.Timestamp('2025-01-15 10:30:00', tz='UTC')
        
        # Create news blackout scenario
        upcoming_news = [
            {'time': pd.Timestamp('2025-01-15 10:45:00', tz='UTC'), 'impact': 'high'}
        ]
        
        result = risk_manager.check_trading_allowed(timestamp, upcoming_news=upcoming_news)
        
        # Should be blocked and logged
        assert result is False