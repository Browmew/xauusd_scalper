# tests/test_backtesting_engine.py
"""
Integration tests for the BacktestEngine class.

This module tests the orchestration logic of the backtesting engine
using mocked dependencies to verify proper integration.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.backtesting.engine import BacktestEngine
from src.backtesting.reporting import BacktestReport


class TestBacktestEngine:
    """Test suite for BacktestEngine class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration fixture."""
        return {
            'data': {
                'symbol': 'XAUUSD',
                'timeframe': '1m',
                'start_date': '2025-01-01',
                'end_date': '2025-01-31'
            },
            'features': {
                'use_extended_features': True,
                'feature_list': ['sma_20', 'rsi_14', 'atr_14']
            },
            'model': {
                'type': 'xgboost',
                'model_path': 'models/xauusd_model.pkl'
            },
            'risk': {
                'max_loss_usd': 1000.0,
                'max_position_size': 0.1,
                'risk_per_trade': 0.02
            },
            'backtesting': {
                'initial_balance': 10000.0,
                'commission': 0.0001,
                'slippage': 0.0001
            }
        }

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for BacktestEngine."""
        mocks = {
            'data_loader': Mock(),
            'feature_pipeline': Mock(),
            'model_predictor': Mock(),
            'risk_manager': Mock(),
            'exchange_simulator': Mock()
        }
        
        # Configure mock behaviors
        df = pd.DataFrame({
            'open': [1800.0] * 100,
            'high': [1805.0] * 100,
            'low': [1795.0] * 100,
            'close': [1802.0] * 100,
            'volume': [1000] * 100
        })
        df.index = pd.date_range('2025-01-01', periods=100, freq='1min', tz='UTC')
        mocks['data_loader'].load_and_align.return_value = df
        
        features_df = pd.DataFrame({
            'close': [1802.0] * 100,
            'sma_20': [1800.0] * 100,
            'rsi_14': [50.0] * 100,
            'atr_14': [2.0] * 100
        })
        features_df.index = pd.date_range('2025-01-01', periods=100, freq='1min', tz='UTC')
        mocks['feature_pipeline'].transform.return_value = features_df
        
        mocks['model_predictor'].predict.return_value = [0.6] * 100  # Win probabilities
        mocks['risk_manager'].check_trading_allowed.return_value = True
        mocks['risk_manager'].calculate_position_size.return_value = Mock(
            size=0.05, stop_loss_distance=1.0, risk_amount=100.0, reason="Normal conditions"
        )
        
        mocks['exchange_simulator'].execute_trade.return_value = Mock(
            success=True, fill_price=1802.0, commission=0.18, slippage=0.05
        )
        
        return mocks

    @patch('src.backtesting.engine.DataLoader')
    @patch('src.backtesting.engine.FeaturePipeline')
    @patch('src.backtesting.engine.ModelPredictor')
    @patch('src.backtesting.engine.RiskManager')
    @patch('src.backtesting.engine.ExchangeSimulator')
    def test_engine_initialization(self, mock_exchange, mock_risk, mock_model,
                                  mock_features, mock_data, mock_config):
        """Test BacktestEngine can be initialized without errors."""
        # Configure mocks to return Mock instances
        mock_data.return_value = Mock()
        mock_features.return_value = Mock()
        mock_model.return_value = Mock()
        mock_risk.return_value = Mock()
        mock_exchange.return_value = Mock()
        
        engine = BacktestEngine(mock_config)
        
        # Assert that all components were instantiated
        assert engine is not None
        mock_data.assert_called_once()
        mock_features.assert_called_once()
        mock_model.assert_called_once()
        mock_risk.assert_called_once()
        mock_exchange.assert_called_once()

    @patch('src.backtesting.engine.DataLoader')
    @patch('src.backtesting.engine.FeaturePipeline')
    @patch('src.backtesting.engine.ModelPredictor')
    @patch('src.backtesting.engine.RiskManager')
    @patch('src.backtesting.engine.ExchangeSimulator')
    def test_engine_components_configuration(self, mock_exchange, mock_risk, mock_model,
                                           mock_features, mock_data, mock_config):
        """Test that engine components are configured with correct parameters."""
        mock_data.return_value = Mock()
        mock_features.return_value = Mock()
        mock_model.return_value = Mock()
        mock_risk.return_value = Mock()
        mock_exchange.return_value = Mock()
        
        engine = BacktestEngine(mock_config)
        
        # Verify components were initialized with correct config sections
        mock_data.assert_called_once()
        mock_features.assert_called_with(mock_config['features'])
        mock_model.assert_called_once()
        mock_risk.assert_called_with(mock_config['risk'])
        mock_exchange.assert_called_once()

    @patch('src.backtesting.engine.DataLoader')
    @patch('src.backtesting.engine.FeaturePipeline')
    @patch('src.backtesting.engine.ModelPredictor')
    @patch('src.backtesting.engine.RiskManager')
    @patch('src.backtesting.engine.ExchangeSimulator')
    def test_engine_run_small_backtest(self, mock_exchange, mock_risk, mock_model,
                                    mock_features, mock_data, mock_config,
                                    sample_data_df):
        """Test engine can run a small backtest without crashing."""
        # Setup mocks with the injected behavior
        mock_data_instance = Mock()
        mock_features_instance = Mock()
        mock_model_instance = Mock()
        mock_risk_instance = Mock()
        mock_exchange_instance = Mock()
        
        mock_data.return_value = mock_data_instance
        mock_features.return_value = mock_features_instance
        mock_model.return_value = mock_model_instance
        mock_risk.return_value = mock_risk_instance
        mock_exchange.return_value = mock_exchange_instance
        
        # Use first 10 rows of sample data for small backtest
        small_data = sample_data_df.head(10).copy()
        
        # Configure mock behaviors
        mock_data_instance.load_and_align.return_value = small_data
        mock_features_instance.transform.return_value = small_data  # Features added
        mock_model_instance.predict_with_metadata.return_value = Mock(
            prediction=0.6, confidence=0.7, features_used=5, timestamp=None
        )
        
        from src.risk.manager import RiskCheckResult
        mock_risk_instance.check_trading_allowed.return_value = RiskCheckResult(
            allowed=True, reason="Normal conditions", current_drawdown=0.01, daily_pnl=0.0
        )
        
        from src.risk.manager import PositionSizeResult
        mock_risk_instance.calculate_position_size.return_value = PositionSizeResult(
            size=0.05, reason="Normal", max_allowed=0.1, kelly_fraction=0.2,
            atr_stop_distance=1.0, stop_loss_distance=1.0, risk_amount=100.0
        )
        
        mock_exchange_instance.get_recent_fills.return_value = []
        mock_exchange_instance.get_performance_stats.return_value = {}
        
        # Initialize engine and run backtest
        engine = BacktestEngine(mock_config)
        
        result = engine.run_backtest()
        
        # Assert backtest completed successfully
        assert result is not None
        
        # Verify that key methods were called
        mock_data_instance.load_and_align.assert_called()
        mock_features_instance.transform.assert_called()

    @patch('src.backtesting.engine.DataLoader')
    @patch('src.backtesting.engine.FeaturePipeline')
    @patch('src.backtesting.engine.ModelPredictor')
    @patch('src.backtesting.engine.RiskManager')
    @patch('src.backtesting.engine.ExchangeSimulator')
    def test_engine_orchestration_flow(self, mock_exchange, mock_risk, mock_model,
                                    mock_features, mock_data, mock_config,
                                    sample_data_df):
        """Test the complete orchestration flow of the engine."""
        # Setup mocks
        mock_data_instance = Mock()
        mock_features_instance = Mock()
        mock_model_instance = Mock()
        mock_risk_instance = Mock()
        mock_exchange_instance = Mock()
        
        mock_data.return_value = mock_data_instance
        mock_features.return_value = mock_features_instance
        mock_model.return_value = mock_model_instance
        mock_risk.return_value = mock_risk_instance
        mock_exchange.return_value = mock_exchange_instance
        
        # Configure pipeline data flow
        raw_data = sample_data_df.head(5).copy()
        featured_data = raw_data.copy()
        featured_data['sma_20'] = 1800.0
        featured_data['rsi_14'] = 50.0
        
        mock_data_instance.load_and_align.return_value = raw_data
        mock_features_instance.transform.return_value = featured_data
        
        # Mock predict_with_metadata to return proper objects
        prediction_results = [
            Mock(prediction=0.7, confidence=0.8, features_used=5, timestamp=None),
            Mock(prediction=0.6, confidence=0.7, features_used=5, timestamp=None),
            Mock(prediction=0.8, confidence=0.9, features_used=5, timestamp=None),
            Mock(prediction=0.5, confidence=0.6, features_used=5, timestamp=None),
            Mock(prediction=0.9, confidence=0.95, features_used=5, timestamp=None)
        ]
        mock_model_instance.predict_with_metadata.side_effect = prediction_results
        
        # Risk manager responses
        from src.risk.manager import RiskCheckResult, PositionSizeResult
        risk_results = [
            RiskCheckResult(allowed=True, reason="OK", current_drawdown=0.01, daily_pnl=0.0),
            RiskCheckResult(allowed=True, reason="OK", current_drawdown=0.01, daily_pnl=0.0),
            RiskCheckResult(allowed=False, reason="Blocked", current_drawdown=0.01, daily_pnl=0.0),
            RiskCheckResult(allowed=True, reason="OK", current_drawdown=0.01, daily_pnl=0.0),
            RiskCheckResult(allowed=True, reason="OK", current_drawdown=0.01, daily_pnl=0.0)
        ]
        mock_risk_instance.check_trading_allowed.side_effect = risk_results
        
        mock_risk_instance.calculate_position_size.return_value = PositionSizeResult(
            size=0.03, reason="Normal", max_allowed=0.1, kelly_fraction=0.15,
            atr_stop_distance=1.5, stop_loss_distance=1.5, risk_amount=75.0
        )
        
        # Exchange simulator responses
        mock_exchange_instance.get_recent_fills.return_value = []
        mock_exchange_instance.get_performance_stats.return_value = {}
        
        # Run backtest
        engine = BacktestEngine(mock_config)
        result = engine.run_backtest()
        
        # Verify orchestration sequence
        mock_data_instance.load_and_align.assert_called()
        mock_features_instance.transform.assert_called()
        
        # Risk manager should be called for trading opportunities
        assert mock_risk_instance.check_trading_allowed.call_count >= 1

    @patch('src.backtesting.engine.DataLoader')
    @patch('src.backtesting.engine.FeaturePipeline')
    @patch('src.backtesting.engine.ModelPredictor')
    @patch('src.backtesting.engine.RiskManager')
    @patch('src.backtesting.engine.ExchangeSimulator')
    def test_engine_handles_trading_blocked_scenario(self, mock_exchange, mock_risk,
                                                    mock_model, mock_features, mock_data,
                                                    mock_config, sample_data_df):
        """Test engine behavior when trading is blocked by risk manager."""
        # Setup mocks
        mock_data_instance = Mock()
        mock_features_instance = Mock()
        mock_model_instance = Mock()
        mock_risk_instance = Mock()
        mock_exchange_instance = Mock()
        
        mock_data.return_value = mock_data_instance
        mock_features.return_value = mock_features_instance
        mock_model.return_value = mock_model_instance
        mock_risk.return_value = mock_risk_instance
        mock_exchange.return_value = mock_exchange_instance
        
        small_data = sample_data_df.head(3).copy()
        
        mock_data_instance.load_and_align.return_value = small_data
        mock_features_instance.transform.return_value = small_data
        mock_model_instance.predict_with_metadata.return_value = Mock(
            prediction=0.8, confidence=0.9, features_used=5, timestamp=None
        )
        
        # Risk manager blocks all trading
        from src.risk.manager import RiskCheckResult
        mock_risk_instance.check_trading_allowed.return_value = RiskCheckResult(
            allowed=False, reason="Trading blocked", current_drawdown=0.1, daily_pnl=-500.0
        )
        mock_exchange_instance.get_recent_fills.return_value = []
        mock_exchange_instance.get_performance_stats.return_value = {}
        
        # Run backtest
        engine = BacktestEngine(mock_config)
        result = engine.run_backtest()
        
        # Verify behavior when trading is blocked
        assert result is not None
        mock_risk_instance.check_trading_allowed.assert_called()
        # Position sizing should not be called when trading is blocked
        mock_risk_instance.calculate_position_size.assert_not_called()

    @patch('src.backtesting.engine.DataLoader')
    @patch('src.backtesting.engine.FeaturePipeline')
    @patch('src.backtesting.engine.ModelPredictor')
    @patch('src.backtesting.engine.RiskManager')
    @patch('src.backtesting.engine.ExchangeSimulator')
    def test_engine_handles_zero_position_size(self, mock_exchange, mock_risk,
                                            mock_model, mock_features, mock_data,
                                            mock_config, sample_data_df):
        """Test engine behavior when risk manager returns zero position size."""
        # Setup mocks
        mock_data_instance = Mock()
        mock_features_instance = Mock()
        mock_model_instance = Mock()
        mock_risk_instance = Mock()
        mock_exchange_instance = Mock()
        
        mock_data.return_value = mock_data_instance
        mock_features.return_value = mock_features_instance
        mock_model.return_value = mock_model_instance
        mock_risk.return_value = mock_risk_instance
        mock_exchange.return_value = mock_exchange_instance
        
        small_data = sample_data_df.head(3).copy()
        
        mock_data_instance.load_and_align.return_value = small_data
        mock_features_instance.transform.return_value = small_data
        mock_model_instance.predict_with_metadata.return_value = Mock(
            prediction=0.8, confidence=0.9, features_used=5, timestamp=None
        )
        
        # Risk manager allows trading but returns zero position size
        from src.risk.manager import RiskCheckResult, PositionSizeResult
        mock_risk_instance.check_trading_allowed.return_value = RiskCheckResult(
            allowed=True, reason="OK", current_drawdown=0.01, daily_pnl=0.0
        )
        mock_risk_instance.calculate_position_size.return_value = PositionSizeResult(
            size=0.0, reason="Win probability too low", max_allowed=0.1, kelly_fraction=0.0,
            atr_stop_distance=0.0, stop_loss_distance=0.0, risk_amount=0.0
        )
        mock_exchange_instance.get_recent_fills.return_value = []
        mock_exchange_instance.get_performance_stats.return_value = {}
        
        # Run backtest
        engine = BacktestEngine(mock_config)
        result = engine.run_backtest()
        
        # Verify behavior with zero position size
        assert result is not None
        mock_risk_instance.calculate_position_size.assert_called()

    @patch('src.backtesting.engine.DataLoader')
    @patch('src.backtesting.engine.FeaturePipeline')
    @patch('src.backtesting.engine.ModelPredictor')
    @patch('src.backtesting.engine.RiskManager')
    @patch('src.backtesting.engine.ExchangeSimulator')
    def test_engine_error_handling_data_load_failure(self, mock_exchange, mock_risk, mock_model,
                                                    mock_features, mock_data, mock_config):
        """Test engine handles data loading failures gracefully."""
        # Setup mocks
        mock_data_instance = Mock()
        mock_features_instance = Mock()
        mock_model_instance = Mock()
        mock_risk_instance = Mock()
        mock_exchange_instance = Mock()
        
        mock_data.return_value = mock_data_instance
        mock_features.return_value = mock_features_instance
        mock_model.return_value = mock_model_instance
        mock_risk.return_value = mock_risk_instance
        mock_exchange.return_value = mock_exchange_instance
        
        # Data loader raises exception
        mock_data_instance.load_and_align.side_effect = Exception("Data source unavailable")
        
        # Run backtest and expect it to handle the error
        engine = BacktestEngine(mock_config)
        
        with pytest.raises(Exception) as exc_info:
            engine.run_backtest()
        
        assert "Data source unavailable" in str(exc_info.value)

    @patch('src.backtesting.engine.get_logger')
    @patch('src.backtesting.engine.DataLoader')
    @patch('src.backtesting.engine.FeaturePipeline')
    @patch('src.backtesting.engine.ModelPredictor')
    @patch('src.backtesting.engine.RiskManager')
    @patch('src.backtesting.engine.ExchangeSimulator')
    def test_engine_logging_behavior(self, mock_exchange, mock_risk, mock_model,
                                mock_features, mock_data, mock_logger, mock_config,
                                sample_data_df):
        """Test that engine logs important events during backtest."""
        # Setup mocks
        mock_data_instance = Mock()
        mock_features_instance = Mock()
        mock_model_instance = Mock()
        mock_risk_instance = Mock()
        mock_exchange_instance = Mock()
        
        mock_data.return_value = mock_data_instance
        mock_features.return_value = mock_features_instance
        mock_model.return_value = mock_model_instance
        mock_risk.return_value = mock_risk_instance
        mock_exchange.return_value = mock_exchange_instance
        
        small_data = sample_data_df.head(2).copy()
        
        mock_data_instance.load_and_align.return_value = small_data
        mock_features_instance.transform.return_value = small_data
        mock_model_instance.predict_with_metadata.return_value = Mock(
            prediction=0.7, confidence=0.8, features_used=5, timestamp=None
        )
        
        from src.risk.manager import RiskCheckResult, PositionSizeResult
        mock_risk_instance.check_trading_allowed.return_value = RiskCheckResult(
            allowed=True, reason="OK", current_drawdown=0.01, daily_pnl=0.0
        )
        mock_risk_instance.calculate_position_size.return_value = PositionSizeResult(
            size=0.05, reason="Normal", max_allowed=0.1, kelly_fraction=0.2,
            atr_stop_distance=1.0, stop_loss_distance=1.0, risk_amount=100.0
        )
        mock_exchange_instance.get_recent_fills.return_value = []
        mock_exchange_instance.get_performance_stats.return_value = {}
        
        # Run backtest
        engine = BacktestEngine(mock_config)
        result = engine.run_backtest()
        
        # Verify that logging occurred (mock_logger is the get_logger function)
        mock_logger.assert_called()

    def test_engine_backtest_report_structure(self, mock_config, sample_data_df):
        """Test that engine returns properly structured BacktestReport."""
        with patch('src.backtesting.engine.DataLoader') as mock_data, \
            patch('src.backtesting.engine.FeaturePipeline') as mock_features, \
            patch('src.backtesting.engine.ModelPredictor') as mock_model, \
            patch('src.backtesting.engine.RiskManager') as mock_risk, \
            patch('src.backtesting.engine.ExchangeSimulator') as mock_exchange:
            
            # Setup mocks
            mock_data_instance = Mock()
            mock_features_instance = Mock()
            mock_model_instance = Mock()
            mock_risk_instance = Mock()
            mock_exchange_instance = Mock()
            
            mock_data.return_value = mock_data_instance
            mock_features.return_value = mock_features_instance
            mock_model.return_value = mock_model_instance
            mock_risk.return_value = mock_risk_instance
            mock_exchange.return_value = mock_exchange_instance
            
            small_data = sample_data_df.head(1).copy()
            
            mock_data_instance.load_and_align.return_value = small_data
            mock_features_instance.transform.return_value = small_data
            mock_model_instance.predict_with_metadata.return_value = Mock(
                prediction=0.7, confidence=0.8, features_used=5, timestamp=None
            )
            
            from src.risk.manager import RiskCheckResult
            mock_risk_instance.check_trading_allowed.return_value = RiskCheckResult(
                allowed=False, reason="Test blocked", current_drawdown=0.01, daily_pnl=0.0
            )
            mock_exchange_instance.get_recent_fills.return_value = []
            mock_exchange_instance.get_performance_stats.return_value = {}
            mock_risk_instance.get_risk_metrics.return_value = {}
            
            # Run backtest
            engine = BacktestEngine(mock_config)
            result = engine.run_backtest()
            
            # Verify report structure
            from src.backtesting.reporting import BacktestReport
            assert isinstance(result, BacktestReport)
            assert hasattr(result, 'total_return')
            assert hasattr(result, 'sharpe_ratio')
            assert hasattr(result, 'max_drawdown')
            assert hasattr(result, 'total_trades')
            assert hasattr(result, 'win_rate')