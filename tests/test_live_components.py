"""
Unit tests for live trading components.
Tests LiveFeedHandler and SmartOrderRouter with mocked network calls.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp
import websockets
from datetime import datetime

from src.live.feed_handler import LiveFeedHandler
from src.live.order_router import SmartOrderRouter


class TestLiveFeedHandler:
    """Test suite for LiveFeedHandler component."""
    
    @pytest.fixture
    def mock_data_queue(self):
        """Create mock data queue for testing."""
        queue = Mock()
        queue.put_nowait = Mock()  # Use sync Mock instead of AsyncMock
        return queue
    
    @pytest.fixture
    def feed_handler_config(self):
        """Configuration for LiveFeedHandler testing."""
        return {
            'url': 'wss://test-feed.example.com/ws',
            'subscription_message': json.dumps({
                'subscribe': 'XAUUSD',
                'type': 'ticker'
            })
        }
    
    @pytest.fixture
    def sample_ws_messages(self):
        """Sample websocket messages for testing."""
        return [
            json.dumps({
                'symbol': 'XAUUSD',
                'bid': 2000.50,
                'ask': 2000.55,
                'timestamp': '2024-01-01T09:00:00.000Z'
            }),
            json.dumps({
                'symbol': 'XAUUSD',
                'bid': 2000.52,
                'ask': 2000.57,
                'timestamp': '2024-01-01T09:00:01.000Z'
            }),
            json.dumps({
                'symbol': 'XAUUSD',
                'bid': 2000.48,
                'ask': 2000.53,
                'timestamp': '2024-01-01T09:00:02.000Z'
            })
        ]
    
    def test_feed_handler_initialization(self, mock_data_queue, feed_handler_config):
        """Test LiveFeedHandler initialization."""
        handler = LiveFeedHandler(
            url=feed_handler_config['url'],
            data_queue=mock_data_queue,
            subscription_message=feed_handler_config['subscription_message']
        )
        
        assert handler.url == feed_handler_config['url']
        assert handler.data_queue == mock_data_queue
        assert handler.subscription_message == feed_handler_config['subscription_message']
        assert not handler.is_running
    
    @pytest.mark.asyncio
    @patch('websockets.connect')
    async def test_feed_handler_successful_connection(self, mock_ws_connect, mock_data_queue, feed_handler_config, sample_ws_messages):
        """Test successful websocket connection and message handling."""
        # Setup mock websocket
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        
        # Configure recv() to return messages then raise StopAsyncIteration
        recv_calls = sample_ws_messages + [StopAsyncIteration()]
        mock_ws.recv = AsyncMock(side_effect=recv_calls)
        
        mock_ws_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws_connect.return_value.__aexit__ = AsyncMock(return_value=None)
        
        handler = LiveFeedHandler(
            url=feed_handler_config['url'],
            data_queue=mock_data_queue,
            subscription_message=feed_handler_config['subscription_message']
        )
        
        # Run handler - should exit when StopAsyncIteration is raised
        await handler.run()
        
        # Verify websocket connection was attempted
        mock_ws_connect.assert_called_once_with(feed_handler_config['url'])
        
        # Verify subscription message was sent
        mock_ws.send.assert_called_once_with(feed_handler_config['subscription_message'])
        
        # Verify messages were processed and queued
        assert mock_data_queue.put_nowait.call_count == len(sample_ws_messages)
    
    @pytest.mark.asyncio
    @patch('websockets.connect')
    async def test_feed_handler_connection_error(self, mock_ws_connect, mock_data_queue, feed_handler_config):
        """Test handling of websocket connection errors."""
        # Make websocket connection fail
        mock_ws_connect.side_effect = websockets.exceptions.ConnectionClosed(None, None)
        
        handler = LiveFeedHandler(
            url=feed_handler_config['url'],
            data_queue=mock_data_queue,
            subscription_message=feed_handler_config['subscription_message']
        )
        
        # Handler should handle connection errors gracefully
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await handler.run()
        
        # Verify connection was attempted
        mock_ws_connect.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('websockets.connect')
    async def test_feed_handler_malformed_message(self, mock_ws_connect, mock_data_queue, feed_handler_config):
        """Test handling of malformed JSON messages."""
        # Setup mock websocket with malformed message
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        
        # Configure recv() to return malformed message then raise StopAsyncIteration
        mock_ws.recv = AsyncMock(side_effect=['invalid json message', StopAsyncIteration()])
        
        mock_ws_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws_connect.return_value.__aexit__ = AsyncMock(return_value=None)
        
        handler = LiveFeedHandler(
            url=feed_handler_config['url'],
            data_queue=mock_data_queue,
            subscription_message=feed_handler_config['subscription_message']
        )
        
        # Run handler - should exit when StopAsyncIteration is raised
        await handler.run()
        
        # Handler should continue running despite malformed message
        # and not queue the invalid data
        mock_data_queue.put.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_feed_handler_stop(self, mock_data_queue, feed_handler_config):
        """Test graceful stopping of feed handler."""
        handler = LiveFeedHandler(
            url=feed_handler_config['url'],
            data_queue=mock_data_queue,
            subscription_message=feed_handler_config['subscription_message']
        )
        
        # Test stop method
        handler.stop()
        assert not handler.is_running
    
    def test_parse_message_valid_json(self, mock_data_queue, feed_handler_config):
        """Test message parsing with valid JSON."""
        handler = LiveFeedHandler(
            url=feed_handler_config['url'],
            data_queue=mock_data_queue,
            subscription_message=feed_handler_config['subscription_message']
        )
        
        message = json.dumps({
            'symbol': 'XAUUSD',
            'bid': 2000.50,
            'ask': 2000.55,
            'timestamp': '2024-01-01T09:00:00.000Z'
        })
        
        parsed = handler._parse_message(message)
        
        assert parsed['symbol'] == 'XAUUSD'
        assert parsed['bid'] == 2000.50
        assert parsed['ask'] == 2000.55
        assert parsed['timestamp'] == '2024-01-01T09:00:00.000Z'
    
    def test_parse_message_invalid_json(self, mock_data_queue, feed_handler_config):
        """Test message parsing with invalid JSON."""
        handler = LiveFeedHandler(
            url=feed_handler_config['url'],
            data_queue=mock_data_queue,
            subscription_message=feed_handler_config['subscription_message']
        )
        
        # Should return None for invalid JSON
        parsed = handler._parse_message('invalid json')
        assert parsed is None


class TestSmartOrderRouter:
    """Test suite for SmartOrderRouter component."""
    
    @pytest.fixture
    def router_config(self):
        """Configuration for SmartOrderRouter testing."""
        return {
            'api_url': 'https://api.test-exchange.com/v1/orders',
            'api_key': 'test_api_key_123',
            'api_secret': 'test_api_secret_456'
        }
    
    @pytest.fixture
    def sample_order(self):
        """Sample order for testing."""
        return {
            'symbol': 'XAUUSD',
            'side': 'BUY',
            'quantity': 0.1,
            'order_type': 'MARKET',
            'timestamp': datetime.now().isoformat()
        }
    
    def test_router_initialization(self, router_config):
        """Test SmartOrderRouter initialization."""
        router = SmartOrderRouter(
            api_url=router_config['api_url'],
            api_key=router_config['api_key'],
            api_secret=router_config['api_secret']
        )
        
        assert router.api_url == router_config['api_url']
        assert router.api_key == router_config['api_key']
        assert router.api_secret == router_config['api_secret']
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_execute_order_success(self, mock_post, router_config, sample_order):
        """Test successful order execution."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'order_id': '12345',
            'status': 'FILLED',
            'fill_price': 2000.50
        })
        mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        router = SmartOrderRouter(
            api_url=router_config['api_url'],
            api_key=router_config['api_key'],
            api_secret=router_config['api_secret']
        )
        
        result = await router.execute_order(sample_order)
        
        # Verify HTTP request was made
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Verify URL
        assert call_args[1]['url'] == router_config['api_url']
        
        # Verify headers contain API key and signature
        headers = call_args[1]['headers']
        assert 'X-API-KEY' in headers
        assert headers['X-API-KEY'] == router_config['api_key']
        assert 'X-SIGNATURE' in headers
        
        # Verify payload contains order data
        payload = call_args[1]['json']
        assert payload['symbol'] == sample_order['symbol']
        assert payload['side'] == sample_order['side']
        assert payload['quantity'] == sample_order['quantity']
        
        # Verify result
        assert result['order_id'] == '12345'
        assert result['status'] == 'FILLED'
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_execute_order_http_error(self, mock_post, router_config, sample_order):
        """Test handling of HTTP errors during order execution."""
        # Setup mock to raise HTTP error
        mock_post.side_effect = aiohttp.ClientError("Connection failed")
        
        router = SmartOrderRouter(
            api_url=router_config['api_url'],
            api_key=router_config['api_key'],
            api_secret=router_config['api_secret']
        )
        
        with pytest.raises(aiohttp.ClientError):
            await router.execute_order(sample_order)
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_execute_order_api_error_response(self, mock_post, router_config, sample_order):
        """Test handling of API error responses."""
        # Setup mock to return error response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={
            'error': 'INVALID_ORDER',
            'message': 'Insufficient balance'
        })
        mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        router = SmartOrderRouter(
            api_url=router_config['api_url'],
            api_key=router_config['api_key'],
            api_secret=router_config['api_secret']
        )
        
        result = await router.execute_order(sample_order)
        
        # Should return the error response
        assert result['error'] == 'INVALID_ORDER'
        assert result['message'] == 'Insufficient balance'
    
    def test_create_payload(self, router_config, sample_order):
        """Test payload creation from order."""
        router = SmartOrderRouter(
            api_url=router_config['api_url'],
            api_key=router_config['api_key'],
            api_secret=router_config['api_secret']
        )
        
        payload = router._create_payload(sample_order)
        
        assert payload['symbol'] == sample_order['symbol']
        assert payload['side'] == sample_order['side']
        assert payload['quantity'] == sample_order['quantity']
        assert payload['order_type'] == sample_order['order_type']
        assert 'timestamp' in payload
    
    def test_sign_payload(self, router_config):
        """Test payload signing for authentication."""
        router = SmartOrderRouter(
            api_url=router_config['api_url'],
            api_key=router_config['api_key'],
            api_secret=router_config['api_secret']
        )
        
        payload = {
            'symbol': 'XAUUSD',
            'side': 'BUY',
            'quantity': 0.1,
            'timestamp': '2024-01-01T09:00:00'
        }
        
        signature = router._sign_payload(payload)
        
        # Signature should be a non-empty string
        assert isinstance(signature, str)
        assert len(signature) > 0
        
        # Same payload should produce same signature
        signature2 = router._sign_payload(payload)
        assert signature == signature2
        
        # Different payload should produce different signature
        different_payload = payload.copy()
        different_payload['quantity'] = 0.2
        different_signature = router._sign_payload(different_payload)
        assert signature != different_signature
    
    @pytest.mark.asyncio
    async def test_validate_order_valid(self, router_config):
        """Test order validation with valid order."""
        router = SmartOrderRouter(
            api_url=router_config['api_url'],
            api_key=router_config['api_key'],
            api_secret=router_config['api_secret']
        )
        
        valid_order = {
            'symbol': 'XAUUSD',
            'side': 'BUY',
            'quantity': 0.1,
            'order_type': 'MARKET'
        }
        
        # Should not raise exception
        is_valid = router._validate_order(valid_order)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_validate_order_invalid(self, router_config):
        """Test order validation with invalid orders."""
        router = SmartOrderRouter(
            api_url=router_config['api_url'],
            api_key=router_config['api_key'],
            api_secret=router_config['api_secret']
        )
        
        # Missing required fields
        invalid_orders = [
            {},  # Empty order
            {'symbol': 'XAUUSD'},  # Missing side, quantity, order_type
            {'symbol': 'XAUUSD', 'side': 'BUY'},  # Missing quantity, order_type
            {'symbol': 'XAUUSD', 'side': 'INVALID_SIDE', 'quantity': 0.1, 'order_type': 'MARKET'},  # Invalid side
            {'symbol': 'XAUUSD', 'side': 'BUY', 'quantity': -0.1, 'order_type': 'MARKET'},  # Negative quantity
            {'symbol': 'XAUUSD', 'side': 'BUY', 'quantity': 0.1, 'order_type': 'INVALID_TYPE'},  # Invalid order type
        ]
        
        for invalid_order in invalid_orders:
            is_valid = router._validate_order(invalid_order)
            assert is_valid is False
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_execute_order_with_limit_price(self, mock_post, router_config):
        """Test order execution with limit price."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'order_id': '12345', 'status': 'PENDING'})
        mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        router = SmartOrderRouter(
            api_url=router_config['api_url'],
            api_key=router_config['api_key'],
            api_secret=router_config['api_secret']
        )
        
        limit_order = {
            'symbol': 'XAUUSD',
            'side': 'BUY',
            'quantity': 0.1,
            'order_type': 'LIMIT',
            'price': 2000.00
        }
        
        result = await router.execute_order(limit_order)
        
        # Verify limit price was included in payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['price'] == 2000.00
        assert payload['order_type'] == 'LIMIT'
        
        assert result['order_id'] == '12345'
        assert result['status'] == 'PENDING'
    
    def test_headers_creation(self, router_config):
        """Test creation of request headers with authentication."""
        router = SmartOrderRouter(
            api_url=router_config['api_url'],
            api_key=router_config['api_key'],
            api_secret=router_config['api_secret']
        )
        
        payload = {'symbol': 'XAUUSD', 'side': 'BUY'}
        headers = router._create_headers(payload)
        
        assert 'X-API-KEY' in headers
        assert headers['X-API-KEY'] == router_config['api_key']
        assert 'X-SIGNATURE' in headers
        assert 'Content-Type' in headers
        assert headers['Content-Type'] == 'application/json'