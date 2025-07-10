"""
Exchange simulator for realistic backtesting of trading strategies.

This module provides a comprehensive simulation of exchange behavior including
latency, slippage, commissions, and order management for XAUUSD trading.
"""

import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Deque
import pandas as pd
import numpy as np

from ..utils.helpers import get_config_value
from ..utils.logging import get_logger


class OrderType(Enum):
    """Order types supported by the exchange simulator."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status types."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """
    Represents a trading order in the exchange simulator.
    
    Attributes:
        order_id: Unique identifier for the order
        symbol: Trading symbol (e.g., 'XAUUSD')
        side: 'BUY' or 'SELL'
        order_type: Type of order (MARKET, LIMIT, etc.)
        quantity: Amount to trade in lots
        price: Order price (None for market orders)
        stop_price: Stop price for stop orders
        timestamp: When the order was created
        status: Current order status
        filled_quantity: Amount already filled
        average_fill_price: Average price of filled portions
        commission: Total commission charged
        slippage: Total slippage experienced
        latency_ms: Simulated network latency
    """
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    latency_ms: float = 0.0


@dataclass
class Fill:
    """Represents a trade execution."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: float
    commission: float
    slippage: float


class ExchangeSimulator:
    """
    Realistic exchange simulator for backtesting trading strategies.
    
    This simulator provides:
    - Latency simulation (5-50ms delays)
    - Commission calculation ($7/lot configurable)
    - Slippage modeling based on historical distributions
    - FIFO order queue management
    - Realistic spread handling
    
    All parameters are configurable via the backtesting section of config.yml.
    """
    
    def __init__(self):
        """Initialize the exchange simulator with configuration parameters."""
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.latency_range_ms = get_config_value('backtesting.latency_range_ms', [5, 50])
        self.commission_per_lot = get_config_value('backtesting.costs.commission_per_lot', 7.0)
        self.slippage_config = get_config_value('backtesting.costs.slippage', {
            'mean': 0.0001,
            'std': 0.0002,
            'max': 0.001
        })
        self.spread_config = get_config_value('backtesting.costs.spread', {
            'base': 0.0003,
            'volatility_multiplier': 1.5
        })
        
        # Internal state
        self.order_queue: Deque[Order] = deque()
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.order_counter: int = 0
        self.current_time: float = 0.0
        self.current_prices: Dict[str, Dict[str, float]] = {}  # {symbol: {bid, ask, mid}}
        
        # Performance tracking
        self.total_volume: float = 0.0
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0
        self.fill_history: List[Fill] = []
        
        self.logger.info("ExchangeSimulator initialized with realistic trading costs")
    
    def update_market_data(self, symbol: str, bid: float, ask: float, timestamp: float) -> None:
        """
        Update current market prices for a symbol.
        
        Args:
            symbol: Trading symbol
            bid: Current bid price
            ask: Current ask price  
            timestamp: Market data timestamp
        """
        self.current_time = timestamp
        self.current_prices[symbol] = {
            'bid': bid,
            'ask': ask,
            'mid': (bid + ask) / 2,
            'spread': ask - bid
        }
        
        # Process any pending orders that can now be filled
        self._process_pending_orders()
    
    def submit_order(self, symbol: str, side: str, quantity: float, 
                    order_type: OrderType = OrderType.MARKET, 
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> str:
        """
        Submit a new order to the exchange.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Amount to trade in lots
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            Order ID string
            
        Raises:
            ValueError: If order parameters are invalid
        """
        # Validate inputs
        if side not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid side: {side}")
        
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            raise ValueError(f"Price required for {order_type} orders")
        
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError(f"Stop price required for {order_type} orders")
        
        # Generate order ID
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:06d}"
        
        # Simulate network latency
        latency_ms = random.uniform(*self.latency_range_ms)
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            timestamp=self.current_time,
            latency_ms=latency_ms
        )
        
        # Add to queue for processing (simulating network delay)
        self.order_queue.append(order)
        self.logger.debug(f"Order {order_id} queued: {side} {quantity} {symbol} @ {price}")
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was cancelled, False if not found or already filled
        """
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            self.logger.debug(f"Order {order_id} cancelled")
            return True
        
        return False
    
    def _process_pending_orders(self) -> None:
        """Process orders in the queue, applying latency delays."""
        current_time_ms = self.current_time * 1000
        
        # Process queued orders (check if latency period has passed)
        while self.order_queue:
            order = self.order_queue[0]
            order_time_ms = order.timestamp * 1000
            
            if current_time_ms >= order_time_ms + order.latency_ms:
                # Latency period has passed, process the order
                order = self.order_queue.popleft()
                self._execute_order(order)
            else:
                break
        
        # Check existing pending orders for fill opportunities
        orders_to_process = list(self.pending_orders.values())
        for order in orders_to_process:
            if order.symbol in self.current_prices:
                self._try_fill_order(order)
    
    def _execute_order(self, order: Order) -> None:
        """
        Execute an order based on current market conditions.
        
        Args:
            order: Order to execute
        """
        if order.symbol not in self.current_prices:
            order.status = OrderStatus.REJECTED
            self.logger.warning(f"Order {order.order_id} rejected: No market data for {order.symbol}")
            return
        
        if order.order_type == OrderType.MARKET:
            # Market orders execute immediately at current price
            self._fill_market_order(order)
        else:
            # Limit/stop orders go to pending queue
            self.pending_orders[order.order_id] = order
            self.logger.debug(f"Order {order.order_id} added to pending orders")
    
    def _fill_market_order(self, order: Order) -> None:
        """
        Fill a market order immediately.
        
        Args:
            order: Market order to fill
        """
        prices = self.current_prices[order.symbol]
        
        # Determine execution price based on side
        if order.side == 'BUY':
            base_price = prices['ask']
        else:
            base_price = prices['bid']
        
        # Apply slippage
        slippage = self._calculate_slippage(order.quantity, prices['spread'])
        if order.side == 'BUY':
            execution_price = base_price + slippage
        else:
            execution_price = base_price - slippage
        
        # Calculate commission
        commission = self._calculate_commission(order.quantity)
        
        # Fill the order
        self._complete_fill(order, execution_price, order.quantity, commission, slippage)
    
    def _try_fill_order(self, order: Order) -> None:
        """
        Check if a pending order can be filled.
        
        Args:
            order: Pending order to check
        """
        prices = self.current_prices[order.symbol]
        can_fill = False
        execution_price = 0.0
        
        if order.order_type == OrderType.LIMIT:
            if order.side == 'BUY' and prices['ask'] <= order.price:
                can_fill = True
                execution_price = order.price
            elif order.side == 'SELL' and prices['bid'] >= order.price:
                can_fill = True
                execution_price = order.price
        
        elif order.order_type == OrderType.STOP:
            if order.side == 'BUY' and prices['ask'] >= order.stop_price:
                can_fill = True
                execution_price = prices['ask']
            elif order.side == 'SELL' and prices['bid'] <= order.stop_price:
                can_fill = True
                execution_price = prices['bid']
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # First check if stop is triggered
            stop_triggered = False
            if order.side == 'BUY' and prices['mid'] >= order.stop_price:
                stop_triggered = True
            elif order.side == 'SELL' and prices['mid'] <= order.stop_price:
                stop_triggered = True
            
            # If stop triggered, check if limit can be filled
            if stop_triggered:
                if order.side == 'BUY' and prices['ask'] <= order.price:
                    can_fill = True
                    execution_price = order.price
                elif order.side == 'SELL' and prices['bid'] >= order.price:
                    can_fill = True
                    execution_price = order.price
        
        if can_fill:
            # Apply slippage for stop orders (not for limit orders at exact price)
            slippage = 0.0
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                slippage = self._calculate_slippage(order.quantity, prices['spread'])
                if order.side == 'BUY':
                    execution_price += slippage
                else:
                    execution_price -= slippage
            
            commission = self._calculate_commission(order.quantity)
            self._complete_fill(order, execution_price, order.quantity, commission, slippage)
    
    def _complete_fill(self, order: Order, price: float, quantity: float, 
                      commission: float, slippage: float) -> None:
        """
        Complete a fill for an order.
        
        Args:
            order: Order being filled
            price: Execution price
            quantity: Quantity filled
            commission: Commission charged
            slippage: Slippage experienced
        """
        order.filled_quantity += quantity
        order.average_fill_price = price  # Simplified for single fills
        order.commission += commission
        order.slippage += abs(slippage)
        order.status = OrderStatus.FILLED
        
        # Remove from pending if fully filled
        if order.order_id in self.pending_orders:
            del self.pending_orders[order.order_id]
        
        # Add to filled orders
        self.filled_orders[order.order_id] = order
        
        # Create fill record
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            timestamp=self.current_time,
            commission=commission,
            slippage=abs(slippage)
        )
        self.fill_history.append(fill)
        
        # Update performance metrics
        self.total_volume += quantity
        self.total_commission += commission
        self.total_slippage += abs(slippage)
        
        self.logger.info(f"Order {order.order_id} filled: {quantity} @ {price:.5f}, "
                        f"commission: ${commission:.2f}, slippage: {slippage:.5f}")
    
    def _calculate_slippage(self, quantity: float, current_spread: float) -> float:
        """
        Calculate slippage based on order size and market conditions.
        
        Args:
            quantity: Order size in lots
            current_spread: Current bid-ask spread
            
        Returns:
            Slippage amount in price units
        """
        # Base slippage from configuration
        base_slippage = np.random.normal(
            self.slippage_config['mean'],
            self.slippage_config['std']
        )
        
        # Scale by order size (larger orders get more slippage)
        size_multiplier = 1 + (quantity - 1) * 0.1  # 10% more slippage per lot above 1
        
        # Scale by spread (wider spreads mean more slippage)
        spread_multiplier = max(1.0, current_spread / self.spread_config['base'])
        
        slippage = base_slippage * size_multiplier * spread_multiplier
        
        # Cap at maximum slippage
        slippage = np.clip(slippage, 0, self.slippage_config['max'])
        
        return slippage
    
    def _calculate_commission(self, quantity: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            quantity: Trade size in lots
            
        Returns:
            Commission amount in dollars
        """
        return quantity * self.commission_per_lot
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get the status of an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            OrderStatus or None if order not found
        """
        if order_id in self.pending_orders:
            return self.pending_orders[order_id].status
        elif order_id in self.filled_orders:
            return self.filled_orders[order_id].status
        else:
            return None
    
    def get_position(self, symbol: str) -> Dict[str, float]:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with position information
        """
        position = 0.0
        total_cost = 0.0
        total_commission = 0.0
        
        for fill in self.fill_history:
            if fill.symbol == symbol:
                if fill.side == 'BUY':
                    position += fill.quantity
                    total_cost += fill.quantity * fill.price
                else:
                    position -= fill.quantity
                    total_cost -= fill.quantity * fill.price
                total_commission += fill.commission
        
        avg_price = total_cost / abs(position) if position != 0 else 0.0
        
        return {
            'quantity': position,
            'average_price': avg_price,
            'total_commission': total_commission,
            'total_cost': total_cost
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get overall performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'total_volume': self.total_volume,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_orders': len(self.filled_orders),
            'pending_orders': len(self.pending_orders),
            'average_commission_per_lot': self.total_commission / self.total_volume if self.total_volume > 0 else 0,
            'average_slippage': self.total_slippage / len(self.fill_history) if self.fill_history else 0
        }
    
    def reset(self) -> None:
        """Reset the simulator state."""
        self.order_queue.clear()
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.fill_history.clear()
        self.order_counter = 0
        self.current_time = 0.0
        self.current_prices.clear()
        self.total_volume = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        self.logger.info("ExchangeSimulator reset")

    def get_recent_fills(self) -> List[Fill]:
        """Get recent fills from this trading session."""
        return self.fill_history.copy()