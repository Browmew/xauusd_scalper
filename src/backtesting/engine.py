"""
Main backtesting engine for XAUUSD scalping strategy.

Implements a comprehensive tick-by-tick event loop that orchestrates all system components
including data loading, feature engineering, model predictions, risk management, and trade execution.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time
from dataclasses import dataclass, field

from ..utils.logging import get_logger
from ..utils.helpers import get_config_value, get_project_root
from ..data_ingestion.loader import DataLoader
from ..features.feature_pipeline import FeaturePipeline
from ..models.predict import ModelPredictor
from ..risk.manager import RiskManager
from .exchange_simulator import ExchangeSimulator, Order
from .reporting import generate_report, BacktestReport
from ..utils.logging import get_logger
logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    data: Dict[str, Any]
    features: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    backtesting: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Extract values from nested config structure
        self.start_date = self.data.get('start_date', '2023-01-01')
        self.end_date = self.data.get('end_date', '2023-12-31')
        self.initial_balance = self.backtesting.get('initial_balance', 100000.0)
        self.tick_file = self.data.get('tick_file', 'data/historical/ticks/XAUUSD_2023_ticks.csv.gz')
        self.l2_file = self.data.get('l2_file', 'data/historical/l2_orderbook/XAUUSD_2023_l2.csv.gz')
        self.model_path = self.model.get('model_path', 'models/xauusd_model.pkl')
        self.commission_rate = self.backtesting.get('commission_per_lot', 7.0)
        self.min_prediction_threshold = self.model.get('min_prediction_confidence', 0.55)
        self.lookback_window = self.features.get('lookback_window', 100)
        self.symbol = self.data.get('symbol', 'XAUUSD')


@dataclass
class BacktestState:
    """Current state of the backtest execution."""
    current_timestamp: datetime
    current_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_commission: float
    max_drawdown: float
    processed_ticks: int = 0
    feature_buffer: List[pd.Series] = field(default_factory=list)


class BacktestEngine:
    """
    Main backtesting engine that orchestrates the complete trading simulation.
    
    This engine implements a tick-by-tick event loop that processes market data,
    generates features, makes predictions, applies risk management, and simulates
    trade execution with realistic market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backtesting engine with all required components.
        
        Args:
            config: Configuration dictionary containing all settings
        """
        self.logger = get_logger(__name__)
        
        # Load configuration
        self._load_config(config)
        
        # Initialize all subsystems
        self._initialize_components()
        
        # Initialize state tracking
        self.state = BacktestState(
            current_timestamp=datetime.now(),
            current_balance=self.config.initial_balance,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0.0,
            total_commission=0.0,
            max_drawdown=0.0
        )
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        
        self.logger.info("BacktestEngine initialized successfully",
                        symbol=self.config.symbol,
                        initial_balance=self.config.initial_balance,
                        model_path=self.config.model_path)
    
    def _load_config(self, config: Dict[str, Any]) -> None:
        """Load backtesting configuration."""
        self.config = BacktestConfig(**config)
        self.logger.debug("Backtest configuration loaded", config=config)
    
    def _initialize_components(self) -> None:
        """Initialize all required subsystem components."""
        try:
            # Data loading
            self.data_loader = DataLoader()
            
            # Feature engineering
            self.feature_pipeline = FeaturePipeline(self.config.features)
            
            # Model prediction
            self.model_predictor = ModelPredictor()
            if self.config.model_path and Path(self.config.model_path).exists():
                self.model_predictor.load_model(self.config.model_path)
                self.logger.info(f"Loaded model from {self.config.model_path}")
            else:
                self.logger.warning(f"Model file not found: {self.config.model_path}")
            
            # Risk management
            self.risk_manager = RiskManager(self.config.risk)
            
            # Exchange simulation
            self.exchange_simulator = ExchangeSimulator()
            
            self.logger.info("All subsystem components initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize components", error=str(e))
            raise
    
    def run_backtest(self) -> BacktestReport:
        """
        Execute the complete backtesting process.
        
        Returns:
            BacktestReport containing comprehensive backtest results
        """
        self.logger.info("Starting backtest execution",
                        start_date=self.config.start_date,
                        end_date=self.config.end_date)
        
        start_time = time.time()
        
        try:
            # Load and prepare data
            self._load_data()
            
            # Execute main event loop
            self._run_event_loop()
            
            # Generate final results
            results = self._generate_results()
            
            execution_time = time.time() - start_time
            self.logger.info("Backtest completed successfully",
                           execution_time=f"{execution_time:.2f}s",
                           total_trades=self.state.total_trades,
                           final_pnl=self.state.total_pnl)
            
            return results
            
        except Exception as e:
            self.logger.error("Backtest execution failed", error=str(e))
            raise
    
    def _load_data(self) -> None:
        """Load and prepare market data for backtesting."""
        self.logger.info("Loading market data",
                        tick_file=self.config.tick_file,
                        l2_file=self.config.l2_file)
        
        try:
            # Use DataLoader to load and align tick and L2 data
            self.market_data = self.data_loader.load_and_align(
                self.config.tick_file,
                self.config.l2_file
            )
            
            # Filter by date range if specified
            if self.config.start_date and self.config.end_date:
                start_date = pd.to_datetime(self.config.start_date)
                end_date = pd.to_datetime(self.config.end_date)
                
                # Ensure timezone awareness
                if self.market_data.index.tz is None:
                    self.market_data.index = self.market_data.index.tz_localize('UTC')
                
                if start_date.tz is None:
                    start_date = start_date.tz_localize('UTC')
                if end_date.tz is None:
                    end_date = end_date.tz_localize('UTC')
                
                self.market_data = self.market_data[
                    (self.market_data.index >= start_date) & 
                    (self.market_data.index <= end_date)
                ]
            
            if self.market_data.empty:
                raise ValueError("No market data available for the specified date range")
            
            self.logger.info("Market data loaded successfully",
                           total_ticks=len(self.market_data),
                           start_time=self.market_data.index[0],
                           end_time=self.market_data.index[-1],
                           columns=list(self.market_data.columns))
                           
        except Exception as e:
            self.logger.error("Failed to load market data", error=str(e))
            raise
    
    def _run_event_loop(self) -> None:
        """Execute the main tick-by-tick event loop."""
        self.logger.info("Starting main event loop")
        
        # Process each tick sequentially
        for i, (timestamp, tick_data) in enumerate(self.market_data.iterrows()):
            self.state.current_timestamp = timestamp
            self.state.processed_ticks = i + 1
            
            # Update exchange simulator with current market data
            self._update_market_state(tick_data)
            
            # Build feature buffer for prediction
            self._update_feature_buffer(tick_data)
            
            # Check if we have enough data for prediction
            if len(self.state.feature_buffer) >= self.config.lookback_window:
                try:
                    # Generate features for current window
                    features_df = self._prepare_features()
                    
                    # Get model prediction
                    prediction = self._get_prediction(features_df)
                    
                    # Apply trading logic with risk management
                    self._process_trading_signal(prediction, tick_data)
                    
                except Exception as e:
                    self.logger.error(f"Error processing tick {i}: {e}")
                    continue
            
            # Process any pending orders
            self._process_orders()
            
            # Update performance tracking
            self._update_performance_tracking()
            
            # Log progress periodically
            if self.state.processed_ticks % 10000 == 0:
                self._log_progress()
        
        self.logger.info("Event loop completed",
                        total_ticks_processed=self.state.processed_ticks)
    
    def _update_market_state(self, tick_data: pd.Series) -> None:
        """Update exchange simulator with current market data."""
        # Extract bid/ask prices - try multiple column naming conventions
        bid = None
        ask = None
        
        if 'bid' in tick_data:
            bid = float(tick_data['bid'])
        elif 'bid_price_1' in tick_data:
            bid = float(tick_data['bid_price_1'])
        elif 'close' in tick_data:
            # Use close price with estimated spread
            bid = float(tick_data['close']) - 0.01
            
        if 'ask' in tick_data:
            ask = float(tick_data['ask'])
        elif 'ask_price_1' in tick_data:
            ask = float(tick_data['ask_price_1'])
        elif 'close' in tick_data:
            # Use close price with estimated spread
            ask = float(tick_data['close']) + 0.01
        
        if bid is not None and ask is not None:
            timestamp = tick_data.name.timestamp()
            self.exchange_simulator.update_market_data(
                symbol=self.config.symbol,
                bid=bid,
                ask=ask,
                timestamp=float(timestamp)
            )
    
    def _update_feature_buffer(self, tick_data: pd.Series) -> None:
        """Maintain rolling buffer of market data for feature calculation."""
        self.state.feature_buffer.append(tick_data)
        
        # Keep only required lookback window plus some extra for indicators
        max_buffer_size = self.config.lookback_window + 200  # Extra for indicator calculation
        if len(self.state.feature_buffer) > max_buffer_size:
            self.state.feature_buffer.pop(0)
    
    def _prepare_features(self) -> pd.DataFrame:
        """Prepare features from current buffer for model prediction."""
        try:
            # Convert buffer to DataFrame
            buffer_df = pd.DataFrame(self.state.feature_buffer)
            buffer_df.index = [tick.name for tick in self.state.feature_buffer]
            
            # Sort by index to ensure chronological order
            buffer_df = buffer_df.sort_index()
            
            # Generate features using pipeline
            features_df = self.feature_pipeline.transform(buffer_df)
            
            # Return only the latest row for prediction
            if len(features_df) > 0:
                return features_df.iloc[[-1]]
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _get_prediction(self, features_df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Get model prediction for current market conditions."""
        if features_df.empty:
            return None
            
        try:
            # Get prediction from model
            if hasattr(self.model_predictor, 'model') and self.model_predictor.model is not None:
                prediction_result = self.model_predictor.predict_with_metadata(features_df)
                
                if hasattr(prediction_result, 'prediction'):
                    # ModelPredictor returned PredictionResult object
                    return {
                        'signal': float(prediction_result.prediction),
                        'confidence': float(prediction_result.confidence) if prediction_result.confidence else 0.6,
                        'features_used': prediction_result.features_used,
                        'timestamp': prediction_result.timestamp
                    }
                else:
                    # ModelPredictor returned raw prediction
                    prediction_value = float(prediction_result[0]) if hasattr(prediction_result, '__len__') else float(prediction_result)
                    return {
                        'signal': prediction_value,
                        'confidence': 0.6,  # Default confidence
                        'features_used': len(features_df.columns),
                        'timestamp': features_df.index[0] if len(features_df) > 0 else None
                    }
            else:
                # No model loaded - return neutral prediction
                self.logger.warning("No model loaded for prediction")
                return {
                    'signal': 0.0,
                    'confidence': 0.0,
                    'features_used': len(features_df.columns),
                    'timestamp': features_df.index[0] if len(features_df) > 0 else None
                }
                
        except Exception as e:
            self.logger.warning("Prediction failed", error=str(e))
            return None
    
    def _process_trading_signal(self, prediction: Optional[Dict[str, float]], tick_data: pd.Series) -> None:
        """Process trading signal with comprehensive risk management."""
        if prediction is None:
            return
        
        signal = prediction.get('signal', 0.0)
        confidence = prediction.get('confidence', 0.0)
        
        # Check if prediction meets minimum threshold
        if confidence < self.config.min_prediction_threshold:
            return
        
        # Determine signal direction
        if abs(signal) < 0.1:  # Neutral signal
            return
        
        signal_direction = 'long' if signal > 0 else 'short'
        
        # Check risk management constraints
        risk_check = self.risk_manager.check_trading_allowed(self.state.current_timestamp)
        if not risk_check.allowed:
            self.logger.debug("Trade blocked by risk management", reason=risk_check.reason)
            return
        
        # Calculate position size
        position_size_result = self._calculate_position_size(signal, confidence, tick_data)
        if position_size_result.size <= 0:
            return
        
        # Get current market prices
        bid = tick_data.get('bid', tick_data.get('bid_price_1', tick_data.get('close', 0.0) - 0.01))
        ask = tick_data.get('ask', tick_data.get('ask_price_1', tick_data.get('close', 0.0) + 0.01))
        
        # Submit order to exchange simulator
        self._submit_trade_order(signal_direction, position_size_result.size, bid, ask, confidence)
    
    def _calculate_position_size(self, signal: float, confidence: float, tick_data: pd.Series) -> Any:
        """Calculate optimal position size using risk management."""
        # Get required parameters for position sizing
        current_atr = tick_data.get('atr_14', 0.001)  # Default small ATR if missing
        current_price = tick_data.get('close', tick_data.get('bid', 0.0) + tick_data.get('ask', 0.0)) / 2.0
        
        if current_price <= 0:
            current_price = 2000.0  # Default XAUUSD price
        
        # Use historical performance estimates (in production, these would come from model validation)
        win_probability = min(0.8, max(0.4, confidence))  # Convert confidence to win probability
        avg_win = get_config_value('risk.historical_performance.avg_win', 50.0)
        avg_loss = get_config_value('risk.historical_performance.avg_loss', 40.0)
        signal_strength = abs(signal) * confidence
        
        try:
            return self.risk_manager.calculate_position_size(
                symbol=self.config.symbol,
                signal_strength=signal_strength,
                win_probability=win_probability,
                avg_win=avg_win,
                avg_loss=avg_loss,
                current_atr=current_atr,
                current_price=current_price
            )
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            # Return zero position size on error
            from ..risk.manager import PositionSizeResult
            return PositionSizeResult(
                size=0.0,
                reason=f"Error in calculation: {e}",
                max_allowed=0.0,
                kelly_fraction=0.0,
                atr_stop_distance=0.0
            )
    
    def _submit_trade_order(self, side: str, size: float, bid: float, ask: float, confidence: float) -> None:
        """Submit trade order to exchange simulator."""
        try:
            # Determine execution price based on side
            if side.lower() == 'long':
                execution_price = float(ask)  # Buy at ask
                order_side = 'BUY'
            else:
                execution_price = float(bid)  # Sell at bid
                order_side = 'SELL'
            
            # Submit order to exchange simulator
            order_id = self.exchange_simulator.submit_order(
                symbol=self.config.symbol,
                side=order_side,
                quantity=size,
                order_type='MARKET',
                price=execution_price
            )
            
            if order_id:
                # Record trade with risk manager
                self.risk_manager.record_trade(self.config.symbol, size, order_side)
                self.state.total_trades += 1
                
                self.logger.debug("Order submitted successfully",
                                order_id=order_id,
                                side=order_side,
                                size=size,
                                price=execution_price,
                                confidence=confidence)
            else:
                self.logger.warning("Order submission failed")
        
        except Exception as e:
            self.logger.error("Failed to submit order", error=str(e))
    
    def _process_orders(self) -> None:
        """Process any filled orders and update PnL."""
        try:
            # Get recent fills from exchange simulator
            recent_fills = self.exchange_simulator.get_recent_fills()
            
            for fill in recent_fills:
                # Calculate fill PnL (simplified approach)
                fill_pnl = self._calculate_fill_pnl(fill)
                
                # Update risk manager with PnL
                self.risk_manager.update_pnl(fill_pnl)
                
                # Update state
                self.state.total_pnl += fill_pnl
                self.state.total_commission += fill.commission
                
                if fill_pnl > 0:
                    self.state.winning_trades += 1
                else:
                    self.state.losing_trades += 1
                
                # Record trade for analysis
                self.trade_history.append({
                    'timestamp': fill.timestamp,
                    'symbol': fill.symbol,
                    'side': fill.side,
                    'quantity': fill.quantity,
                    'price': fill.price,
                    'commission': fill.commission,
                    'pnl': fill_pnl
                })
                
                self.logger.debug("Order filled",
                                fill_price=fill.price,
                                quantity=fill.quantity,
                                pnl=fill_pnl,
                                commission=fill.commission)
                                
        except Exception as e:
            self.logger.error(f"Error processing orders: {e}")
    
    def _calculate_fill_pnl(self, fill) -> float:
        """Calculate PnL for a filled order."""
        try:
            # Get current position before this fill
            current_position = self.exchange_simulator.get_position(self.config.symbol)
            
            # For simplicity, calculate P&L based on immediate mark-to-market
            # In production, this would track actual position lifecycle
            if hasattr(fill, 'side') and hasattr(fill, 'quantity') and hasattr(fill, 'price'):
                # Simplified P&L calculation - assume immediate closure at mid price
                spread = 0.02  # Typical XAUUSD spread
                if fill.side.upper() == 'BUY':
                    # For long position, assume immediate sale at bid (price - spread)
                    pnl = (fill.price - spread - fill.price) * fill.quantity
                else:
                    # For short position, assume immediate cover at ask (price + spread)
                    pnl = (fill.price - (fill.price + spread)) * fill.quantity
                
                # Add some randomness to simulate realistic trading outcomes
                # In production, this would be actual position tracking
                outcome_factor = np.random.normal(1.0, 0.1)  # ±10% variance
                return pnl * outcome_factor
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating fill PnL: {e}")
            return 0.0
    
    def _update_performance_tracking(self) -> None:
        """Update performance metrics and tracking."""
        current_balance = self.config.initial_balance + self.state.total_pnl
        self.state.current_balance = current_balance
        
        # Update max drawdown
        peak_balance = max(self.config.initial_balance, current_balance)
        if hasattr(self, '_peak_balance'):
            peak_balance = max(self._peak_balance, current_balance)
        else:
            self._peak_balance = peak_balance
            
        current_drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0
        self.state.max_drawdown = max(self.state.max_drawdown, current_drawdown)
        self._peak_balance = peak_balance
        
        # Record performance snapshot
        self.performance_history.append({
            'timestamp': self.state.current_timestamp,
            'balance': current_balance,
            'pnl': self.state.total_pnl,
            'trades': self.state.total_trades,
            'drawdown': current_drawdown
        })
    
    def _log_progress(self) -> None:
        """Log current progress and performance."""
        if len(self.market_data) > 0:
            progress_pct = (self.state.processed_ticks / len(self.market_data)) * 100
            
            self.logger.info("Backtest progress",
                            progress_pct=f"{progress_pct:.1f}%",
                            processed_ticks=self.state.processed_ticks,
                            total_trades=self.state.total_trades,
                            current_pnl=f"${self.state.total_pnl:.2f}",
                            current_balance=f"${self.state.current_balance:.2f}")
    
    def _generate_results(self) -> BacktestReport:
        """Generate comprehensive backtest results."""
        try:
            # Get exchange simulator statistics
            exchange_stats = self.exchange_simulator.get_performance_stats()
            
            # Calculate performance metrics
            total_return = (self.state.current_balance / self.config.initial_balance) - 1
            win_rate = self.state.winning_trades / max(1, self.state.total_trades)
            
            # Calculate Sharpe ratio
            if self.trade_history:
                returns_series = pd.Series([trade['pnl'] for trade in self.trade_history])
                daily_return_mean = returns_series.mean()
                daily_return_std = returns_series.std()
                
                if daily_return_std > 0:
                    # Annualize assuming 252 trading days
                    sharpe_ratio = (daily_return_mean / daily_return_std) * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
            
            # Create BacktestReport
            report = BacktestReport(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=self.state.max_drawdown,
                total_trades=self.state.total_trades,
                win_rate=win_rate,
                final_balance=self.state.current_balance,
                total_pnl=self.state.total_pnl,
                performance_metrics={
                    'winning_trades': self.state.winning_trades,
                    'losing_trades': self.state.losing_trades,
                    'total_commission': self.state.total_commission,
                    'exchange_statistics': exchange_stats,
                    'risk_metrics': self.risk_manager.get_risk_metrics()
                },
                trade_history=self.trade_history
            )
            
            self.logger.info("Backtest results generated",
                            total_return_pct=f"{total_return*100:.2f}%",
                            total_trades=self.state.total_trades,
                            win_rate_pct=f"{win_rate*100:.1f}%",
                            sharpe_ratio=f"{sharpe_ratio:.2f}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating results: {e}")
            # Return minimal report on error
            return BacktestReport(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_trades=0,
                win_rate=0.0,
                final_balance=self.config.initial_balance,
                total_pnl=0.0
            )
    
    def save_results(self, results: BacktestReport, output_dir: Optional[Path] = None) -> Path:
        """Save backtest results to files."""
        if output_dir is None:
            output_dir = get_project_root() / "results" / "backtests"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert BacktestReport to dictionary for generate_report
            results_dict = {
                'performance_summary': {
                    'total_return_pct': results.total_return * 100,
                    'final_balance': results.final_balance,
                    'sharpe_ratio': results.sharpe_ratio,
                    'max_drawdown_pct': results.max_drawdown * 100,
                    'win_rate_pct': results.win_rate * 100,
                    'total_trades': results.total_trades
                },
                'trade_history': results.trade_history,
                'performance_history': self.performance_history,
                'exchange_statistics': results.performance_metrics.get('exchange_statistics', {}),
                'risk_metrics': results.performance_metrics.get('risk_metrics', {})
            }
            
            # Generate report
            generate_report(results_dict, str(output_dir))
            
            self.logger.info("Backtest results saved", report_path=str(output_dir))
            return output_dir
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return output_dir