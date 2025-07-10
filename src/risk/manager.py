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
    
    def __init__(self):
        """Initialize the risk manager with configuration and state tracking."""
        self.logger = get_logger(__name__)
        
        # Load configuration
        self._load_config()
        
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
    
    def _load_config(self) -> None:
        """Load risk management configuration from config file."""
        # Daily limits
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