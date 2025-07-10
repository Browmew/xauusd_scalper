"""
Asynchronous WebSocket feed handler for live market data ingestion.

This module provides the LiveFeedHandler class that connects to broker WebSocket
endpoints, subscribes to XAUUSD market data, and queues incoming tick data for
consumption by the live trading engine.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import websockets
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logging.info("Using uvloop for asyncio event loop.")
except ImportError:
    logging.warning("uvloop not available, falling back to default asyncio event loop.")
    pass # uvloop is optional

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    logging.warning("uvloop not available, falling back to default event loop")


@dataclass
class TickData:
    """Standardized tick data structure."""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    spread: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.spread is None:
            self.spread = self.ask - self.bid


class LiveFeedHandler:
    """
    Asynchronous WebSocket feed handler for live market data.
    
    This class manages WebSocket connections to broker feeds, handles
    subscription management, implements robust reconnection logic with
    exponential backoff, and queues incoming market data for consumption
    by the live trading engine.
    """
    
    def __init__(
        self,
        websocket_url: str,
        subscription_message: Dict[str, Any],
        data_queue: asyncio.Queue,
        symbol: str = "XAUUSD",
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        heartbeat_interval: float = 30.0,
        message_timeout: float = 5.0,
    ):
        """
        Initialize the live feed handler.
        
        Args:
            websocket_url: WebSocket endpoint URL
            subscription_message: JSON message to send for data subscription
            data_queue: asyncio.Queue to place incoming tick data
            symbol: Trading symbol (default: XAUUSD)
            max_reconnect_attempts: Maximum reconnection attempts before giving up
            base_reconnect_delay: Base delay for exponential backoff (seconds)
            max_reconnect_delay: Maximum delay between reconnection attempts
            heartbeat_interval: Interval for sending heartbeat/ping messages
            message_timeout: Timeout for individual message operations
        """
        self.websocket_url = websocket_url
        self.subscription_message = subscription_message
        self.data_queue = data_queue
        self.symbol = symbol
        self.max_reconnect_attempts = max_reconnect_attempts
        self.base_reconnect_delay = base_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.heartbeat_interval = heartbeat_interval
        self.message_timeout = message_timeout
        
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_running = False
        self.reconnect_count = 0
        self.last_message_time = 0.0
        self.connection_start_time = 0.0
        
        # Message parsing callback - can be customized for different brokers
        self.message_parser: Callable[[Dict[str, Any]], Optional[TickData]] = self._default_message_parser
        
        # Statistics
        self.messages_received = 0
        self.messages_queued = 0
        self.connection_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def set_message_parser(self, parser: Callable[[Dict[str, Any]], Optional[TickData]]) -> None:
        """Set custom message parser for different broker message formats."""
        self.message_parser = parser
    
    async def start(self) -> None:
        """Start the feed handler with automatic reconnection."""
        self.is_running = True
        self.logger.info(f"Starting WebSocket feed handler for {self.symbol}")
        
        while self.is_running and self.reconnect_count < self.max_reconnect_attempts:
            try:
                await self._connect_and_run()
            except Exception as e:
                self.logger.error(f"Feed handler error: {e}")
                if self.is_running:
                    await self._handle_reconnection()
    
    async def stop(self) -> None:
        """Stop the feed handler gracefully."""
        self.logger.info("Stopping WebSocket feed handler")
        self.is_running = False
        
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
    
    async def _connect_and_run(self) -> None:
        """Establish WebSocket connection and run the main message loop."""
        self.connection_start_time = time.time()
        
        # Connect with timeout and proper headers
        connect_kwargs = {
            "ping_interval": self.heartbeat_interval,
            "ping_timeout": self.message_timeout,
            "close_timeout": self.message_timeout,
            "max_size": 2**20,  # 1MB max message size
            "max_queue": 32,    # Limit message queue size
        }
        
        async with websockets.connect(self.websocket_url, **connect_kwargs) as websocket:
            self.websocket = websocket
            self.connection_count += 1
            self.reconnect_count = 0  # Reset on successful connection
            
            self.logger.info(f"Connected to {self.websocket_url} (connection #{self.connection_count})")
            
            # Send subscription message
            await self._send_subscription()
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            try:
                # Main message processing loop
                await self._message_loop()
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
    
    async def _send_subscription(self) -> None:
        """Send subscription message to the WebSocket."""
        try:
            subscription_json = json.dumps(self.subscription_message)
            await asyncio.wait_for(
                self.websocket.send(subscription_json),
                timeout=self.message_timeout
            )
            self.logger.info(f"Sent subscription: {subscription_json}")
        except asyncio.TimeoutError:
            self.logger.error("Timeout sending subscription message")
            raise
        except Exception as e:
            self.logger.error(f"Error sending subscription: {e}")
            raise
    
    async def _message_loop(self) -> None:
        """Main message processing loop."""
        while self.is_running:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.message_timeout
                )
                
                self.last_message_time = time.time()
                self.messages_received += 1
                
                # Parse and queue the message
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                # Check if we've been without messages for too long
                if time.time() - self.last_message_time > self.heartbeat_interval * 2:
                    self.logger.warning("No messages received, connection may be stale")
                    break
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed")
                break
            except Exception as e:
                self.logger.error(f"Error in message loop: {e}")
                break
    
    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            # Parse JSON message
            data = json.loads(message)
            
            # Convert to standardized TickData using the configured parser
            tick_data = self.message_parser(data)
            
            if tick_data:
                # Queue the tick data (non-blocking)
                try:
                    self.data_queue.put_nowait(tick_data)
                    self.messages_queued += 1
                except asyncio.QueueFull:
                    self.logger.warning("Data queue is full, dropping message")
                    
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    def _default_message_parser(self, data: Dict[str, Any]) -> Optional[TickData]:
        """
        Default message parser for generic broker format.
        
        Expected format:
        {
            "symbol": "XAUUSD",
            "timestamp": 1234567890.123,
            "bid": 1950.25,
            "ask": 1950.75,
            "bid_size": 1000000,
            "ask_size": 1000000
        }
        """
        try:
            # Skip non-quote messages
            if data.get("type") != "quote" and "bid" not in data:
                return None
            
            # Ensure this is for our symbol
            if data.get("symbol") != self.symbol:
                return None
            
            # Parse timestamp
            timestamp_val = data.get("timestamp", time.time())
            if isinstance(timestamp_val, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp_val)
            else:
                timestamp = datetime.now()
            
            # Extract bid/ask prices
            bid = float(data["bid"])
            ask = float(data["ask"])
            
            # Optional size information
            bid_size = data.get("bid_size")
            ask_size = data.get("ask_size")
            if bid_size is not None:
                bid_size = float(bid_size)
            if ask_size is not None:
                ask_size = float(ask_size)
            
            return TickData(
                timestamp=timestamp,
                symbol=self.symbol,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size
            )
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing message data: {e}")
            return None
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat/ping messages to keep connection alive."""
        while self.is_running and self.websocket and not self.websocket.closed:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self.websocket and not self.websocket.closed:
                    await self.websocket.ping()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                break
    
    async def _handle_reconnection(self) -> None:
        """Handle reconnection with exponential backoff."""
        if not self.is_running:
            return
        
        self.reconnect_count += 1
        
        if self.reconnect_count >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            self.is_running = False
            return
        
        # Calculate delay with exponential backoff
        delay = min(
            self.base_reconnect_delay * (2 ** (self.reconnect_count - 1)),
            self.max_reconnect_delay
        )
        
        self.logger.info(f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_count})")
        await asyncio.sleep(delay)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feed handler statistics."""
        current_time = time.time()
        uptime = current_time - self.connection_start_time if self.connection_start_time > 0 else 0
        
        return {
            "is_running": self.is_running,
            "connection_count": self.connection_count,
            "reconnect_count": self.reconnect_count,
            "messages_received": self.messages_received,
            "messages_queued": self.messages_queued,
            "uptime_seconds": uptime,
            "last_message_age": current_time - self.last_message_time if self.last_message_time > 0 else None,
            "messages_per_second": self.messages_received / uptime if uptime > 0 else 0,
        }