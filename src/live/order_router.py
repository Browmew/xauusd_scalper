"""
Smart order router for multi-broker trade execution.

This module provides the SmartOrderRouter class that abstracts broker-specific
APIs and provides a unified interface for order execution via both REST and
WebSocket protocols.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urlencode

import aiohttp
import websockets


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Standardized order representation.
    
    This dataclass ensures compatibility between backtesting and live trading
    by providing a consistent order structure.
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    client_order_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate client order ID if not provided."""
        if self.client_order_id is None:
            self.client_order_id = f"scalper_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


@dataclass
class OrderResponse:
    """Order execution response."""
    order_id: str
    client_order_id: str
    status: OrderStatus
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    commission: Optional[float] = None
    timestamp: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BrokerConfig:
    """Broker configuration container."""
    
    def __init__(
        self,
        name: str,
        api_key: str,
        api_secret: str,
        rest_base_url: str,
        websocket_url: Optional[str] = None,
        passphrase: Optional[str] = None,
        sandbox: bool = False,
    ):
        self.name = name
        self.api_key = api_key
        self.api_secret = api_secret
        self.rest_base_url = rest_base_url
        self.websocket_url = websocket_url
        self.passphrase = passphrase
        self.sandbox = sandbox


class SmartOrderRouter:
    """
    Multi-broker order execution router.
    
    This class provides a unified interface for order execution across multiple
    brokers, supporting both REST API and WebSocket protocols. It includes
    comprehensive error handling, retry logic, and broker-specific adaptations.
    """
    
    async def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an order via REST API."""
        if not self._validate_order(order):
            return {'error': 'INVALID_ORDER', 'message': 'Order validation failed'}
        
        payload = self._create_payload(order)
        headers = self._create_headers(payload)
        
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
                
            async with self.session.post(
                url=self.api_url,
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_data = await response.json()
                    return error_data
        except aiohttp.ClientError as e:
            # Propagate HTTP-layer failures so unit tests can assert on them
            raise                
        except Exception as e:
            return {'error': 'EXECUTION_FAILED', 'message': str(e)}

    def __init__(self, api_url: str, api_key: str, api_secret: str):
        self.api_url = api_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.default_broker = "PRIMARY_BROKER"
        self.brokers: Dict[str, Dict[str, Any]] = {
            self.default_broker: {
                "api_url": api_url,
                "api_key": api_key,
                "api_secret": api_secret,
            }
        }

        
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[OrderResponse] = []
        
        # Configuration
        self.request_timeout = 5.0
        self.max_retries = 3
        self.retry_delay = 1.0
        
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self) -> None:
        """Initialize HTTP session and WebSocket connections."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.request_timeout),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=10)
        )
        
        self.logger.info("SmartOrderRouter started")
    
    async def stop(self) -> None:
        """Close all connections gracefully."""
        # Close WebSocket connections
        for broker_name, websocket in self.websocket_connections.items():
            if not websocket.closed:
                await websocket.close()
                self.logger.info(f"Closed WebSocket connection to {broker_name}")
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        self.logger.info("SmartOrderRouter stopped")
    
    async def execute_order_full(
        self,
        order: Order,
        broker_name: Optional[str] = None,
        use_websocket: bool = False
    ) -> OrderResponse:
        """
        Execute an order using the specified broker.
        
        Args:
            order: Order to execute
            broker_name: Broker to use (defaults to default_broker)
            use_websocket: Whether to use WebSocket instead of REST API
            
        Returns:
            OrderResponse with execution details
        """
        broker_name = broker_name or self.default_broker
        broker_config = self.brokers[broker_name]
        
        self.logger.info(
            f"Executing {order.side.value} order for {order.quantity} {order.symbol} "
            f"via {broker_name} ({'WebSocket' if use_websocket else 'REST'})"
        )
        
        # Add to active orders
        self.active_orders[order.client_order_id] = order
        
        try:
            if use_websocket:
                response = await self._execute_via_websocket(order, broker_config)
            else:
                response = await self._execute_via_rest(order, broker_config)
            
            # Update order tracking
            if response.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                self.active_orders.pop(order.client_order_id, None)
            
            self.order_history.append(response)
            return response
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            error_response = OrderResponse(
                order_id="",
                client_order_id=order.client_order_id,
                status=OrderStatus.REJECTED,
                error_message=str(e)
            )
            self.order_history.append(error_response)
            return error_response
    
    async def _execute_via_rest(self, order: Order, broker_config: BrokerConfig) -> OrderResponse:
        """
        Execute order via REST API.
        
        This method demonstrates a generic REST API order execution pattern
        that can be adapted for specific brokers like OANDA, Interactive Brokers,
        MetaTrader, etc.
        """
        # Prepare order payload
        payload = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": str(order.quantity),
            "timeInForce": order.time_in_force,
            "newClientOrderId": order.client_order_id,
            "timestamp": int(time.time() * 1000),
        }
        
        # Add price fields based on order type
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price:
            payload["price"] = str(order.price)
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price:
            payload["stopPrice"] = str(order.stop_price)
        
        # Generate signature for authenticated request
        signature = self._generate_signature(payload, broker_config.api_secret)
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": broker_config.api_key,
            "X-Signature": signature,
            "User-Agent": "XAUUSD-Scalper/1.0",
        }
        
        # Add passphrase if required (e.g., for some crypto exchanges)
        if broker_config.passphrase:
            headers["X-Passphrase"] = broker_config.passphrase
        
        # Execute request with retry logic
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{broker_config.rest_base_url}/api/v1/order",
                    json=payload,
                    headers=headers
                ) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        return self._parse_rest_response(response_data, order.client_order_id)
                    else:
                        error_msg = response_data.get("msg", f"HTTP {response.status}")
                        self.logger.error(f"REST API error: {error_msg}")
                        
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        else:
                            raise Exception(f"REST API error after {self.max_retries} attempts: {error_msg}")
                            
            except asyncio.TimeoutError:
                self.logger.warning(f"REST API timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise Exception("REST API timeout after multiple attempts")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"REST API error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
    
    async def _execute_via_websocket(self, order: Order, broker_config: BrokerConfig) -> OrderResponse:
        """
        Execute order via WebSocket API.
        
        This method demonstrates WebSocket order execution for brokers that
        support real-time order submission via WebSocket connections.
        """
        if not broker_config.websocket_url:
            raise ValueError(f"WebSocket URL not configured for broker {broker_config.name}")
        
        # Get or create WebSocket connection
        websocket = await self._get_websocket_connection(broker_config)
        
        # Prepare order message
        order_message = {
            "id": str(uuid.uuid4()),
            "method": "order.place",
            "params": {
                "symbol": order.symbol,
                "side": order.side.value,
                "type": order.order_type.value,
                "quantity": str(order.quantity),
                "clientOrderId": order.client_order_id,
                "timestamp": int(time.time() * 1000),
            }
        }
        
        # Add price fields based on order type
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price:
            order_message["params"]["price"] = str(order.price)
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price:
            order_message["params"]["stopPrice"] = str(order.stop_price)
        
        # Sign the message
        message_str = json.dumps(order_message["params"], sort_keys=True)
        signature = hmac.new(
            broker_config.api_secret.encode(),
            message_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        order_message["signature"] = signature
        order_message["apiKey"] = broker_config.api_key
        
        # Send order and wait for response
        try:
            await websocket.send(json.dumps(order_message))
            
            # Wait for response with timeout
            response_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=self.request_timeout
            )
            
            response_data = json.loads(response_message)
            return self._parse_websocket_response(response_data, order.client_order_id)
            
        except asyncio.TimeoutError:
            raise Exception("WebSocket order execution timeout")
        except websockets.exceptions.ConnectionClosed:
            # Remove closed connection and retry
            self.websocket_connections.pop(broker_config.name, None)
            raise Exception("WebSocket connection closed during order execution")
    
    async def _get_websocket_connection(self, broker_config: BrokerConfig) -> websockets.WebSocketServerProtocol:
        """Get or create WebSocket connection for a broker."""
        if broker_config.name not in self.websocket_connections:
            try:
                websocket = await websockets.connect(
                    broker_config.websocket_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                )
                
                # Send authentication message
                auth_message = {
                    "method": "auth",
                    "params": {
                        "apiKey": broker_config.api_key,
                        "timestamp": int(time.time() * 1000),
                    }
                }
                
                # Sign authentication
                auth_str = json.dumps(auth_message["params"], sort_keys=True)
                auth_signature = hmac.new(
                    broker_config.api_secret.encode(),
                    auth_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                auth_message["signature"] = auth_signature
                
                await websocket.send(json.dumps(auth_message))
                
                # Wait for auth confirmation
                auth_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                auth_data = json.loads(auth_response)
                
                if auth_data.get("result") != "success":
                    raise Exception(f"WebSocket authentication failed: {auth_data}")
                
                self.websocket_connections[broker_config.name] = websocket
                self.logger.info(f"WebSocket connection established to {broker_config.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to establish WebSocket connection to {broker_config.name}: {e}")
                raise
        
        return self.websocket_connections[broker_config.name]
    
    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate HMAC signature for REST API requests."""
        # Create query string from payload
        query_string = urlencode(sorted(payload.items()))
        
        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _parse_rest_response(self, response_data: Dict[str, Any], client_order_id: str) -> OrderResponse:
        """Parse REST API response into standardized OrderResponse."""
        try:
            # Generic response parsing - adapt for specific broker formats
            order_id = response_data.get("orderId", response_data.get("id", ""))
            status_str = response_data.get("status", "unknown").lower()
            
            # Map broker status to our enum
            status_mapping = {
                "new": OrderStatus.PENDING,
                "pending": OrderStatus.PENDING,
                "filled": OrderStatus.FILLED,
                "partially_filled": OrderStatus.PARTIALLY_FILLED,
                "cancelled": OrderStatus.CANCELLED,
                "canceled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
            }
            
            status = status_mapping.get(status_str, OrderStatus.PENDING)
            
            return OrderResponse(
                order_id=str(order_id),
                client_order_id=client_order_id,
                status=status,
                filled_quantity=float(response_data.get("executedQty", 0)),
                average_fill_price=float(response_data["price"]) if response_data.get("price") else None,
                commission=float(response_data["commission"]) if response_data.get("commission") else None,
            )
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing REST response: {e}")
            return OrderResponse(
                order_id="",
                client_order_id=client_order_id,
                status=OrderStatus.REJECTED,
                error_message=f"Response parsing error: {e}"
            )
    
    def _parse_websocket_response(self, response_data: Dict[str, Any], client_order_id: str) -> OrderResponse:
        """Parse WebSocket response into standardized OrderResponse."""
        try:
            # Handle error responses
            if "error" in response_data:
                return OrderResponse(
                    order_id="",
                    client_order_id=client_order_id,
                    status=OrderStatus.REJECTED,
                    error_message=response_data["error"].get("message", "Unknown error")
                )
            
            # Parse successful response
            result = response_data.get("result", {})
            order_id = result.get("orderId", result.get("id", ""))
            status_str = result.get("status", "unknown").lower()
            
            # Map status
            status_mapping = {
                "accepted": OrderStatus.PENDING,
                "filled": OrderStatus.FILLED,
                "partial": OrderStatus.PARTIALLY_FILLED,
                "cancelled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
            }
            
            status = status_mapping.get(status_str, OrderStatus.PENDING)
            
            return OrderResponse(
                order_id=str(order_id),
                client_order_id=client_order_id,
                status=status,
                filled_quantity=float(result.get("filledQuantity", 0)),
                average_fill_price=float(result["avgPrice"]) if result.get("avgPrice") else None,
            )
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing WebSocket response: {e}")
            return OrderResponse(
                order_id="",
                client_order_id=client_order_id,
                status=OrderStatus.REJECTED,
                error_message=f"Response parsing error: {e}"
            )
    
    async def cancel_order(self, order_id: str, broker_name: Optional[str] = None) -> OrderResponse:
        """Cancel an active order."""
        broker_name = broker_name or self.default_broker
        broker_config = self.brokers[broker_name]
        
        cancel_payload = {
            "orderId": order_id,
            "timestamp": int(time.time() * 1000),
        }
        
        signature = self._generate_signature(cancel_payload, broker_config.api_secret)
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": broker_config.api_key,
            "X-Signature": signature,
        }
        
        try:
            async with self.session.delete(
                f"{broker_config.rest_base_url}/api/v1/order",
                json=cancel_payload,
                headers=headers
            ) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return OrderResponse(
                        order_id=order_id,
                        client_order_id=response_data.get("clientOrderId", ""),
                        status=OrderStatus.CANCELLED,
                    )
                else:
                    raise Exception(f"Cancel failed: {response_data}")
                    
        except Exception as e:
            self.logger.error(f"Order cancellation failed: {e}")
            return OrderResponse(
                order_id=order_id,
                client_order_id="",
                status=OrderStatus.REJECTED,
                error_message=str(e)
            )
        
    def _create_payload(self, order: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': order['quantity'],
            'order_type': order['order_type'],
            'timestamp': datetime.now().isoformat()
        }
        if order['order_type'] == 'LIMIT' and 'price' in order:
            payload['price'] = order['price']
        return payload


    def _sign_payload(self, payload: Dict[str, Any]) -> str:
        import hmac
        import hashlib
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(payload.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _create_headers(self, payload: Dict[str, Any]) -> Dict[str, str]:
        return {
            'X-API-KEY': self.api_key,
            'X-SIGNATURE': self._sign_payload(payload),
            'Content-Type': 'application/json'
        }

    def _validate_order(self, order: Dict[str, Any]) -> bool:
        required_fields = ['symbol', 'side', 'quantity', 'order_type']
        if not all(field in order for field in required_fields):
            return False

        if order['side'] not in ['BUY', 'SELL']:
            return False

        if order['quantity'] <= 0:
            return False

        allowed_types = ['MARKET', 'LIMIT']
        if order['order_type'] not in allowed_types:
            return False

        return True

    
    def get_active_orders(self) -> Dict[str, Order]:
        """Get all active orders."""
        return self.active_orders.copy()
    
    def get_order_history(self) -> List[OrderResponse]:
        """Get order execution history."""
        return self.order_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order router statistics."""
        total_orders = len(self.order_history)
        filled_orders = len([r for r in self.order_history if r.status == OrderStatus.FILLED])
        
        return {
            "active_orders": len(self.active_orders),
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "fill_rate": filled_orders / total_orders if total_orders > 0 else 0,
            "websocket_connections": len(self.websocket_connections),
            "brokers_configured": len(self.brokers),
        }