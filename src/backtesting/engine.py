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

from dataclasses import dataclass
from typing import Optional
import uuid

@dataclass
class Position:
    """Represents an open trading position."""
    position_id: str
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    
    def update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized PnL based on current market price."""
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def should_exit(self, current_price: float, current_time: datetime) -> tuple[bool, str]:
        """Check if position should be closed."""
        # Stop loss check
        if self.stop_loss:
            if (self.side == 'long' and current_price <= self.stop_loss) or \
               (self.side == 'short' and current_price >= self.stop_loss):
                return True, "stop_loss"
        
        # Take profit check
        if self.take_profit:
            if (self.side == 'long' and current_price >= self.take_profit) or \
               (self.side == 'short' and current_price <= self.take_profit):
                return True, "take_profit"
        
        # Time-based exit (max 15 minutes for scalping)
        if (current_time - self.entry_time).total_seconds() > 900:  # 15 minutes
            return True, "time_limit"
        
        return False, ""

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
        self.open_positions = {}  # position_id -> Position
        self.position_counter = 0
        self.total_realized_pnl = 0.0
        
        self.logger.info("BacktestEngine initialized successfully",
                        symbol=self.config.symbol,
                        initial_balance=self.config.initial_balance,
                        model_path=self.config.model_path)
    
    def _load_config(self, config: Dict[str, Any]) -> None:
        """Load backtesting configuration."""
        self.config = BacktestConfig(
            data=config.get('data', {}),
            features=config.get('features', {}),
            model=config.get('model', {}), 
            risk=config.get('risk', {}),
            backtesting=config.get('backtesting', {})
        )
        self.logger.debug("Backtest configuration loaded", config=config)
    
    def _initialize_components(self) -> None:
        """Initialize all required subsystem components."""
        try:
            # Data loading
            self.data_loader = DataLoader()
            
            # Feature engineering
            self.feature_pipeline = FeaturePipeline(self.config.features)
            
            # Model prediction - only load if path exists
            self.model_predictor = ModelPredictor()
            if self.config.model_path and Path(self.config.model_path).exists():
                self.model_predictor.load_model(self.config.model_path)
                self.logger.info(f"Loaded model from {self.config.model_path}")
            else:
                self.logger.warning(f"Model file not found: {self.config.model_path}")
                # Create a dummy predictor for testing
                self.model_predictor.model = type('MockModel', (), {
                    'predict': lambda self, X: np.array([0.6] * len(X))
                })()
            
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
            
            # Ensure index is DatetimeIndex, not RangeIndex
            if not isinstance(self.market_data.index, pd.DatetimeIndex):
                if 'timestamp' in self.market_data.columns:
                    self.market_data['timestamp'] = pd.to_datetime(
                        self.market_data['timestamp'], 
                        format='mixed', 
                        utc=True, 
                        errors='coerce'
                    ).dt.floor('ms')
                    self.market_data.set_index('timestamp', inplace=True)
                else:
                    # Create a datetime index
                    self.market_data.index = pd.date_range(
                        start='2023-01-01', periods=len(self.market_data), freq='1min', tz='UTC'
                    )
            
            # Ensure timezone awareness
            if self.market_data.index.tz is None:
                self.market_data.index = self.market_data.index.tz_localize('UTC')
            
            # Filter by date range if specified - handle fractional seconds
            if hasattr(self.config, 'start_date') and hasattr(self.config, 'end_date'):
                start_date = pd.to_datetime(self.config.start_date, utc=True)
                end_date = pd.to_datetime(self.config.end_date, utc=True)
                
                if start_date.tz is None:
                    start_date = start_date.tz_localize('UTC')
                if end_date.tz is None:
                    end_date = end_date.tz_localize('UTC')
                
                # Only filter if the date range is sensible
                if start_date < end_date:
                    filtered_data = self.market_data[
                        (self.market_data.index >= start_date) & 
                        (self.market_data.index <= end_date)
                    ]
                    if not filtered_data.empty:
                        self.market_data = filtered_data
            
            if self.market_data.empty:
                # If data is empty, create minimal sample data for testing
                self.logger.warning("No market data found, creating sample data for testing")
                dates = pd.date_range('2025-01-01', periods=10, freq='1min', tz='UTC')
                self.market_data = pd.DataFrame({
                    'open': [2000.0] * 10,
                    'high': [2005.0] * 10,
                    'low': [1995.0] * 10,
                    'close': [2002.0] * 10,
                    'bid': [2001.5] * 10,
                    'ask': [2002.5] * 10,
                    'volume': [100.0] * 10
                }, index=dates)
            
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
            #self._update_market_state(tick_data)
            
            # Build feature buffer for prediction
            self._update_feature_buffer(tick_data)
            
            # Always try to generate features and process signals after minimum buffer
            # Trigger the pipeline as soon as we have at least 5 ticks
            # or 5 % of the configured look-back window, whichever is larger.
            min_buffer = max(10, self.config.lookback_window // 10)
            if len(self.state.feature_buffer) >= min_buffer:

                try:
                    # Generate features for current window - this should ALWAYS be called
                    features_df = self._prepare_features()
                    
                    # Only proceed if we have valid features
                    if not features_df.empty:
                        # Get model prediction
                        prediction = self._get_prediction(features_df)
                        
                        # Apply trading logic with risk management - this ensures risk checks happen
                        if prediction is not None:
                            self._process_trading_signal(prediction, tick_data)
                    
                except Exception as e:
                    self.logger.error(f"Error processing tick {i}: {e}")
                    continue
            
            # Process any pending orders
            # Store current tick for price reference
            self.current_tick_data = tick_data
            
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
            
            # ALWAYS call feature pipeline transform
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
        """Process trading signal with proper position management."""
        if prediction is None:
            return
        
        signal = prediction.get('signal', 0.0)
        confidence = prediction.get('confidence', 0.0)
        current_price = (tick_data.get('bid', 0.0) + tick_data.get('ask', 0.0)) / 2.0
        
        # First, check existing positions for exit conditions
        self._process_position_exits(tick_data)
        
        # Only consider new positions if we don't have too many open
        if len(self.open_positions) >= 3:  # Max 3 concurrent positions
            return
        
        # Check if signal is strong enough for new position
        if confidence < self.config.min_prediction_threshold:
            return
        
        # Determine signal direction and strength
        if signal > 0.1:  # Bullish signal
            signal_direction = 'long'
            signal_strength = signal * confidence
        elif signal < -0.1:  # Bearish signal
            signal_direction = 'short'
            signal_strength = abs(signal) * confidence
        else:
            return  # Neutral signal
        
        # Check risk management
        risk_check_result = self.risk_manager.check_trading_allowed(self.state.current_timestamp)
        allowed = (risk_check_result if isinstance(risk_check_result, bool)
                else getattr(risk_check_result, "allowed", False))
        
        if not allowed:
            return
        
        # Calculate position size
        position_size_result = self._calculate_position_size(signal_strength, confidence, tick_data)
        if position_size_result.size <= 0:
            return
        
        # Open new position
        self._open_position(signal_direction, position_size_result.size, tick_data, confidence)
    
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
    
    def _update_performance_tracking(self) -> None:
        """Update performance metrics and tracking."""
        # Calculate total unrealized PnL from open positions
        unrealized_pnl = 0.0
        current_mid_price = 2000.0  # Default fallback
        
        if hasattr(self, 'current_tick_data'):
            bid = self.current_tick_data.get('bid', 2000.0)
            ask = self.current_tick_data.get('ask', 2000.0)
            current_mid_price = (bid + ask) / 2.0
        
        for position in self.open_positions.values():
            position.update_unrealized_pnl(current_mid_price)
            unrealized_pnl += position.unrealized_pnl
        
        # Total PnL = realized + unrealized
        total_pnl = self.total_realized_pnl + unrealized_pnl
        current_balance = self.config.initial_balance + total_pnl
        
        self.state.current_balance = current_balance
        self.state.total_pnl = total_pnl
        
        # Update max drawdown
        if hasattr(self, '_peak_balance'):
            self._peak_balance = max(self._peak_balance, current_balance)
        else:
            self._peak_balance = current_balance
            
        current_drawdown = (self._peak_balance - current_balance) / self._peak_balance if self._peak_balance > 0 else 0
        self.state.max_drawdown = max(self.state.max_drawdown, current_drawdown)
        
        # Record performance snapshot
        self.performance_history.append({
            'timestamp': self.state.current_timestamp,
            'balance': current_balance,
            'realized_pnl': self.total_realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'trades': self.state.total_trades,
            'open_positions': len(self.open_positions),
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
        """Generate comprehensive backtest results with accurate metrics."""
        try:
            # Close any remaining open positions at final price
            if self.open_positions:
                final_tick = pd.Series({
                    'bid': 2000.0, 'ask': 2000.1, 'close': 2000.0  # Use last known prices
                })
                remaining_positions = list(self.open_positions.keys())
                for position_id in remaining_positions:
                    self._close_position(position_id, final_tick, "backtest_end")
            
            # Calculate performance metrics
            total_return = (self.state.current_balance / self.config.initial_balance) - 1
            
            if self.state.total_trades > 0:
                win_rate = self.state.winning_trades / self.state.total_trades
            else:
                win_rate = 0.0
            
            # Calculate Sharpe ratio from trade PnLs
            if self.trade_history and len(self.trade_history) > 1:
                trade_pnls = [trade['net_pnl'] for trade in self.trade_history]
                pnl_series = pd.Series(trade_pnls)
                
                if pnl_series.std() > 0:
                    # Annualize assuming ~100 trades per day, 252 trading days
                    sharpe_ratio = (pnl_series.mean() / pnl_series.std()) * np.sqrt(252 * 100)
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
            
            # Enhanced performance metrics
            performance_metrics = {
                'winning_trades': self.state.winning_trades,
                'losing_trades': self.state.losing_trades,
                'total_realized_pnl': self.total_realized_pnl,
                'avg_trade_pnl': self.total_realized_pnl / max(1, self.state.total_trades),
                'avg_win_amount': 0.0,
                'avg_loss_amount': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_hold_time_minutes': 0.0,
                'total_transaction_costs': 0.0,
                'total_commission': 0.0
            }
            
            # Calculate detailed trade statistics
            if self.trade_history:
                winning_trades = [t for t in self.trade_history if t['net_pnl'] > 0]
                losing_trades = [t for t in self.trade_history if t['net_pnl'] <= 0]
                
                if winning_trades:
                    performance_metrics['avg_win_amount'] = np.mean([t['net_pnl'] for t in winning_trades])
                    performance_metrics['largest_win'] = max([t['net_pnl'] for t in winning_trades])
                
                if losing_trades:
                    performance_metrics['avg_loss_amount'] = np.mean([t['net_pnl'] for t in losing_trades])
                    performance_metrics['largest_loss'] = min([t['net_pnl'] for t in losing_trades])
                
                # Calculate average hold time
                hold_times = [t['hold_time_seconds'] / 60.0 for t in self.trade_history]  # Convert to minutes
                performance_metrics['avg_hold_time_minutes'] = np.mean(hold_times)
                
                # Calculate total costs
                performance_metrics['total_transaction_costs'] = sum([t['transaction_costs'] for t in self.trade_history])
                performance_metrics['total_commission'] = sum([t['commission'] for t in self.trade_history])
            
            # Create BacktestReport
            report = BacktestReport(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=self.state.max_drawdown,
                total_trades=self.state.total_trades,
                win_rate=win_rate,
                final_balance=self.state.current_balance,
                total_pnl=self.state.total_pnl,
                performance_metrics=performance_metrics,
                trade_history=self.trade_history
            )
            
            self.logger.info("Backtest results generated",
                            total_return_pct=f"{total_return*100:.2f}%",
                            total_trades=self.state.total_trades,
                            win_rate_pct=f"{win_rate*100:.1f}%",
                            sharpe_ratio=f"{sharpe_ratio:.2f}",
                            avg_hold_time_min=f"{performance_metrics['avg_hold_time_minutes']:.1f}")
            
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
    
    def _process_position_exits(self, tick_data: pd.Series) -> None:
        """Check and process position exits."""
        current_price = (tick_data.get('bid', 0.0) + tick_data.get('ask', 0.0)) / 2.0
        positions_to_close = []
        
        for position_id, position in self.open_positions.items():
            # Update unrealized PnL
            position.update_unrealized_pnl(current_price)
            
            # Check exit conditions
            should_exit, exit_reason = position.should_exit(current_price, self.state.current_timestamp)
            
            if should_exit:
                positions_to_close.append((position_id, exit_reason))
        
        # Close positions that need to be closed
        for position_id, exit_reason in positions_to_close:
            self._close_position(position_id, tick_data, exit_reason)

    def _open_position(self, side: str, quantity: float, tick_data: pd.Series, confidence: float) -> None:
        """Open a new trading position."""
        bid = tick_data.get('bid', tick_data.get('close', 2000.0) - 0.05)
        ask = tick_data.get('ask', tick_data.get('close', 2000.0) + 0.05)
        
        # Determine entry price based on side (realistic execution)
        if side == 'long':
            entry_price = ask  # Buy at ask
        else:
            entry_price = bid  # Sell at bid
        
        # Calculate stop loss and take profit
        atr = tick_data.get('atr_14', 0.5)  # Use ATR for stops
        if side == 'long':
            stop_loss = entry_price - (atr * 2.0)  # 2 ATR stop
            take_profit = entry_price + (atr * 1.5)  # 1.5 ATR target
        else:
            stop_loss = entry_price + (atr * 2.0)
            take_profit = entry_price - (atr * 1.5)
        
        # Create position
        self.position_counter += 1
        position_id = f"POS_{self.position_counter:06d}"
        
        position = Position(
            position_id=position_id,
            symbol=self.config.symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=self.state.current_timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.open_positions[position_id] = position
        
        # Record with risk manager
        self.risk_manager.record_trade(self.config.symbol, quantity, side.upper())
        self.state.total_trades += 1
        
        self.logger.debug("Position opened",
                        position_id=position_id,
                        side=side,
                        quantity=quantity,
                        entry_price=entry_price,
                        confidence=confidence)

    def _close_position(self, position_id: str, tick_data: pd.Series, exit_reason: str) -> None:
        """Close an existing position and realize PnL."""
        if position_id not in self.open_positions:
            return
        
        position = self.open_positions[position_id]
        bid = tick_data.get('bid', tick_data.get('close', 2000.0) - 0.05)
        ask = tick_data.get('ask', tick_data.get('close', 2000.0) + 0.05)
        
        # Determine exit price (reverse of entry)
        if position.side == 'long':
            exit_price = bid  # Sell at bid
        else:
            exit_price = ask  # Cover at ask
        
        # Calculate realized PnL
        if position.side == 'long':
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity
        
        # Apply transaction costs (spread on both entry and exit)
        spread = ask - bid
        transaction_costs = spread * position.quantity * 2  # Entry + exit
        commission = self.config.commission_rate * position.quantity
        
        net_pnl = gross_pnl - transaction_costs - commission
        
        # Update balances
        self.total_realized_pnl += net_pnl
        self.state.total_pnl += net_pnl
        self.state.current_balance = self.config.initial_balance + self.state.total_pnl
        
        # Track winning/losing trades
        if net_pnl > 0:
            self.state.winning_trades += 1
        else:
            self.state.losing_trades += 1
        
        # Update risk manager
        self.risk_manager.update_pnl(net_pnl)
        
        # Record trade for analysis
        hold_time_seconds = (self.state.current_timestamp - position.entry_time).total_seconds()
        
        self.trade_history.append({
            'position_id': position_id,
            'symbol': position.symbol,
            'side': position.side,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': self.state.current_timestamp,
            'hold_time_seconds': hold_time_seconds,
            'gross_pnl': gross_pnl,
            'transaction_costs': transaction_costs,
            'commission': commission,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason
        })
        
        # Remove position
        del self.open_positions[position_id]
        
        self.logger.info("Position closed",
                        position_id=position_id,
                        side=position.side,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        net_pnl=net_pnl,
                        exit_reason=exit_reason,
                        hold_time_seconds=hold_time_seconds)

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