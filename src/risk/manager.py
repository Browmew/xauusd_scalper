"""
Risk Manager for XAUUSD scalping strategy.

Implements comprehensive risk management including:
- Dynamic position sizing using Kelly criterion and ATR-based stops
- Daily loss limits and drawdown protection
- Session filtering for optimal trading hours
- News blackout periods to avoid high-impact events
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

from ..utils.logging import get_logger
from ..utils.helpers import get_config_value, get_project_root


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    size: float
    reason: str
    max_allowed: float
    kelly_fraction: float
    atr_stop_distance: float


@dataclass
class RiskCheckResult:
    """Result of risk validation check."""
    allowed: bool
    reason: str
    current_drawdown: float
    daily_pnl: float


class RiskManager:
    """
    Comprehensive risk management system for intraday XAUUSD scalping.
    
    This class enforces trading limits, calculates position sizes using Kelly criterion,
    and manages exposure during news events and outside trading sessions.
    """

    def check_trading_allowed(self, timestamp: datetime = None, daily_pnl: float = None, 
                            current_drawdown: float = None, upcoming_news: List = None) -> bool:
        """Check if trading is allowed at given timestamp."""
        if timestamp is None:
            timestamp = datetime.now()
        if daily_pnl is None:
            daily_pnl = self.daily_pnl
        if current_drawdown is None:
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        if upcoming_news is None:
            upcoming_news = []
        
        # Reset daily state if new day
        self.reset_daily_state(timestamp)
        
        # Check daily loss limit
        if abs(daily_pnl) >= self.max_daily_loss_usd:
            return False
        
        # Check maximum drawdown
        if current_drawdown >= self.max_drawdown_pct:
            return False
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        # Check session hours
        if not self._is_in_trading_session(timestamp):
            return False
        
        # Check news blackout
        if self._check_news_blackout(timestamp, upcoming_news):
            return False
        
        return True

    def calculate_position_size(self, account_balance: float = None, atr: float = None, 
                            win_probability: float = None) -> PositionSizeResult:
        """Calculate position size based on risk parameters."""
        if account_balance is None:
            account_balance = self.current_balance
        if atr is None:
            atr = 1.0  # Default ATR
        if win_probability is None:
            win_probability = 0.6  # Default win probability
            return PositionSizeResult(
                size=0.0,
                reason="Win probability below minimum threshold",
                max_allowed=0.0,
                kelly_fraction=0.0,
                atr_stop_distance=0.0
            )
        
        if atr <= 0:
            return PositionSizeResult(
                size=0.0,
                reason="ATR too low or zero",
                max_allowed=0.0,
                kelly_fraction=0.0,
                atr_stop_distance=0.0
            )
        
        # Calculate Kelly fraction (simplified)
        avg_win = 100.0  # Default values
        avg_loss = 50.0
        kelly_fraction = self._calculate_kelly_criterion(win_probability, avg_win, avg_loss)
        
        # Calculate position size
        risk_amount = account_balance * self.kelly_multiplier * kelly_fraction
        stop_distance = atr * 2.0  # ATR multiplier
        position_size = risk_amount / stop_distance
        
        # Apply limits
        max_allowed = min(self.max_position_size, account_balance * 0.1)
        final_size = min(position_size, max_allowed)
        
        return PositionSizeResult(
            size=max(0.0, final_size),
            reason="Kelly-based sizing",
            max_allowed=max_allowed,
            kelly_fraction=kelly_fraction,
            atr_stop_distance=stop_distance
        )

    def _is_in_trading_session(self, timestamp: datetime) -> bool:
        """Check if timestamp is within allowed trading sessions."""
        if not self.allowed_sessions:
            return True
        
        hour = timestamp.hour
        
        # Check London session (8-16 UTC)
        if 'london' in self.allowed_sessions:
            if 8 <= hour < 16:
                return True
        
        # Check NY session (13-21 UTC)
        if 'new_york' in self.allowed_sessions:
            if 13 <= hour < 21:
                return True
        
        # Check weekends
        if timestamp.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        return False

    def _check_news_blackout(self, timestamp: datetime, upcoming_news: List) -> bool:
        """Check if timestamp is within news blackout period."""
        if not upcoming_news or self.news_blackout_minutes <= 0:
            return False
        
        for news in upcoming_news:
            news_time = news.get('time')
            if isinstance(news_time, str):
                news_time = pd.to_datetime(news_time)
            
            time_diff = abs((news_time - timestamp).total_seconds() / 60)
            if time_diff <= self.news_blackout_minutes:
                return True
        
        return False

    def _calculate_kelly_criterion(self, win_probability: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion fraction."""
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        return max(0.0, kelly)
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the risk manager with configuration and state tracking."""
        self.logger = get_logger(__name__)
        
        # Load configuration
        self._load_config(config)
        
        # Add missing attributes that tests expect
        self.max_loss_usd = self.max_daily_loss_usd  # Alias for tests
        
        # State tracking
        self.daily_pnl = 0.0
        self.peak_balance = self.initial_balance
        self.current_balance = self.initial_balance
        self.positions = {}  # symbol -> current position size
        self.daily_trades = 0
        self.last_reset_date = None
        
        # Load news events if available
        self._load_news_events()
        
        self.logger.info("RiskManager initialized", 
                        initial_balance=self.initial_balance,
                        max_daily_loss=self.max_daily_loss_usd,
                        max_position_size=self.max_position_size)
    
    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load risk management configuration from config file or provided config."""
        if config:
            # Use passed config
            self.max_daily_loss_usd = config.get('max_loss_usd', 1000.0)
            self.max_daily_trades = config.get('max_trades', 100)
            self.max_drawdown_pct = config.get('max_drawdown_pct', 0.15)
            
            # Position sizing
            self.kelly_multiplier = config.get('kelly_multiplier', 0.25)
            self.max_position_size = config.get('max_position_size', 0.1)
            self.min_position_size = config.get('min_position_size', 0.01)
            self.atr_stop_multiplier = config.get('atr_stop_multiplier', 2.0)
            
            # Session filtering
            self.allowed_sessions = config.get('allowed_sessions', {})
            self.timezone = config.get('timezone', 'UTC')
            
            # News blackout
            self.news_blackout_minutes = config.get('news_blackout_minutes', 30)
            self.high_impact_only = config.get('high_impact_only', True)
            
            # Initial balance
            self.initial_balance = config.get('initial_balance', 100000.0)
        else:
            # Use existing get_config_value calls
            self.max_daily_loss_usd = get_config_value('risk.daily_limits.max_loss_usd')
            self.max_daily_trades = get_config_value('risk.daily_limits.max_trades')
            self.max_drawdown_pct = get_config_value('risk.daily_limits.max_drawdown_pct')
            
            # Position sizing
            self.kelly_multiplier = get_config_value('risk.position_sizing.kelly_multiplier')
            self.max_position_size = get_config_value('risk.position_sizing.max_size_usd')
            self.min_position_size = get_config_value('risk.position_sizing.min_size_usd')
            self.atr_stop_multiplier = get_config_value('risk.position_sizing.atr_stop_multiplier')
            
            # Session filtering
            self.allowed_sessions = get_config_value('risk.session_filter.allowed_hours')
            self.timezone = get_config_value('risk.session_filter.timezone')
            
            # News blackout
            self.news_blackout_minutes = get_config_value('risk.news_blackout.minutes_before_after')
            self.high_impact_only = get_config_value('risk.news_blackout.high_impact_only')
            
            # Initial balance from backtesting config
            self.initial_balance = get_config_value('backtesting.initial_balance')
        
        self.logger.debug("Risk configuration loaded successfully")
    
    def _load_news_events(self) -> None:
        """Load news events from CSV file for blackout periods."""
        try:
            news_file = get_project_root() / "data" / "live" / "news_events.csv"
            if news_file.exists():
                self.news_events = pd.read_csv(news_file)
                self.news_events['datetime'] = pd.to_datetime(self.news_events['datetime'])
                
                if self.high_impact_only:
                    self.news_events = self.news_events[
                        self.news_events['impact'].str.upper() == 'HIGH'
                    ]
                
                self.logger.info("News events loaded", 
                               total_events=len(self.news_events))
            else:
                self.news_events = pd.DataFrame()
                self.logger.warning("News events file not found, proceeding without news filter")
                
        except Exception as e:
            self.logger.error("Failed to load news events", error=str(e))
            self.news_events = pd.DataFrame()
    
    def reset_daily_state(self, current_date: datetime) -> None:
        """Reset daily tracking variables at start of new trading day."""
        if self.last_reset_date != current_date.date():
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date.date()
            self.logger.info("Daily state reset", date=current_date.date())
    
    def update_pnl(self, pnl_change: float) -> None:
        """Update current PnL and balance tracking."""
        self.daily_pnl += pnl_change
        self.current_balance += pnl_change
        
        # Update peak balance for drawdown calculation
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
    
    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        current_atr: float,
        current_price: float
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using Kelly criterion with ATR-based stops.
        
        Args:
            symbol: Trading symbol
            signal_strength: Model confidence (0-1)
            win_probability: Expected win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
            current_atr: Current ATR value for stop distance
            current_price: Current market price
            
        Returns:
            PositionSizeResult with calculated size and reasoning
        """
        # Calculate Kelly fraction
        if avg_loss <= 0 or win_probability <= 0:
            return PositionSizeResult(0.0, "Invalid win/loss parameters", 0.0, 0.0, 0.0)
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        
        # Apply Kelly multiplier and signal strength
        adjusted_kelly = kelly_fraction * self.kelly_multiplier * signal_strength
        
        # Calculate ATR-based stop distance
        atr_stop_distance = current_atr * self.atr_stop_multiplier
        stop_loss_pct = atr_stop_distance / current_price
        
        # Calculate position size based on risk per trade
        risk_amount = self.current_balance * abs(adjusted_kelly) * stop_loss_pct
        position_value = min(risk_amount / stop_loss_pct, self.max_position_size)
        position_value = max(position_value, self.min_position_size)
        
        # Apply maximum position limit
        max_allowed = min(self.max_position_size, self.current_balance * 0.1)  # Max 10% of balance
        final_size = min(position_value, max_allowed)
        
        reason = "Kelly-based sizing with ATR stops"
        if final_size == max_allowed:
            reason = "Limited by maximum position size"
        elif adjusted_kelly <= 0:
            final_size = 0.0
            reason = "Negative Kelly fraction - no position"
        
        return PositionSizeResult(
            size=final_size,
            reason=reason,
            max_allowed=max_allowed,
            kelly_fraction=adjusted_kelly,
            atr_stop_distance=atr_stop_distance
        )
    
    def check_trading_allowed(self, timestamp: datetime, symbol: str = "XAUUSD") -> RiskCheckResult:
        """
        Comprehensive check if trading is allowed at given timestamp.
        
        Args:
            timestamp: Current timestamp to check
            symbol: Trading symbol
            
        Returns:
            RiskCheckResult indicating if trading is allowed and why
        """
        # Reset daily state if new day
        self.reset_daily_state(timestamp)
        
        # Calculate current drawdown
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.max_daily_loss_usd:
            return RiskCheckResult(
                allowed=False,
                reason=f"Daily loss limit exceeded: ${abs(self.daily_pnl):.2f}",
                current_drawdown=current_drawdown,
                daily_pnl=self.daily_pnl
            )
        
        # Check maximum drawdown
        if current_drawdown >= self.max_drawdown_pct:
            return RiskCheckResult(
                allowed=False,
                reason=f"Maximum drawdown exceeded: {current_drawdown*100:.2f}%",
                current_drawdown=current_drawdown,
                daily_pnl=self.daily_pnl
            )
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return RiskCheckResult(
                allowed=False,
                reason=f"Daily trade limit exceeded: {self.daily_trades}",
                current_drawdown=current_drawdown,
                daily_pnl=self.daily_pnl
            )
        
        # Check session hours
        if not self._is_trading_session_active(timestamp):
            return RiskCheckResult(
                allowed=False,
                reason="Outside allowed trading session",
                current_drawdown=current_drawdown,
                daily_pnl=self.daily_pnl
            )
        
        # Check news blackout
        if self._is_news_blackout_period(timestamp):
            return RiskCheckResult(
                allowed=False,
                reason="News blackout period active",
                current_drawdown=current_drawdown,
                daily_pnl=self.daily_pnl
            )
        
        return RiskCheckResult(
            allowed=True,
            reason="All risk checks passed",
            current_drawdown=current_drawdown,
            daily_pnl=self.daily_pnl
        )
    
    def _is_trading_session_active(self, timestamp: datetime) -> bool:
        """Check if current time is within allowed trading sessions."""
        if not self.allowed_sessions:
            return True  # No restrictions if not configured
        
        current_time = timestamp.time()
        
        for session in self.allowed_sessions:
            start_time = time.fromisoformat(session['start'])
            end_time = time.fromisoformat(session['end'])
            
            # Handle sessions that cross midnight
            if start_time <= end_time:
                if start_time <= current_time <= end_time:
                    return True
            else:
                if current_time >= start_time or current_time <= end_time:
                    return True
        
        return False
    
    def _is_news_blackout_period(self, timestamp: datetime) -> bool:
        """Check if current time is within news blackout period."""
        if self.news_events.empty or self.news_blackout_minutes <= 0:
            return False
        
        blackout_delta = pd.Timedelta(minutes=self.news_blackout_minutes)
        
        for _, event in self.news_events.iterrows():
            event_time = event['datetime']
            start_blackout = event_time - blackout_delta
            end_blackout = event_time + blackout_delta
            
            if start_blackout <= timestamp <= end_blackout:
                return True
        
        return False
    
    def record_trade(self, symbol: str, size: float, side: str) -> None:
        """Record a new trade for daily limit tracking."""
        self.daily_trades += 1
        
        # Update position tracking
        current_pos = self.positions.get(symbol, 0.0)
        if side.upper() == 'BUY':
            self.positions[symbol] = current_pos + size
        else:
            self.positions[symbol] = current_pos - size
        
        self.logger.debug("Trade recorded", 
                         symbol=symbol, 
                         size=size, 
                         side=side,
                         daily_trades=self.daily_trades,
                         new_position=self.positions[symbol])
    
    def get_current_position(self, symbol: str) -> float:
        """Get current position size for symbol."""
        return self.positions.get(symbol, 0.0)
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics for monitoring."""
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        return {
            'current_balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'current_drawdown_pct': current_drawdown * 100,
            'daily_loss_limit_used_pct': (abs(self.daily_pnl) / self.max_daily_loss_usd) * 100,
            'trade_limit_used_pct': (self.daily_trades / self.max_daily_trades) * 100,
            'positions': self.positions.copy()
        }