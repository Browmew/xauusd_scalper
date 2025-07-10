"""
Structured logging system for the XAUUSD scalping system using structlog.

This module provides high-performance, structured logging with different configurations
for development and production environments. Designed for minimal latency impact
in live trading scenarios.

Big-O Considerations:
- Log message creation: O(1)
- JSON serialization: O(k) where k is the number of fields
- Async logging queue: O(1) for enqueue
- File I/O: Asynchronous, non-blocking for performance
"""

import sys
import json
import asyncio
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, Union, TextIO
from datetime import datetime
import structlog
from structlog.types import FilteringBoundLogger
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Import project utilities
from .helpers import get_config_value, ensure_directory, get_project_root


class AsyncLogHandler(logging.Handler):
    """
    Asynchronous log handler for high-performance logging in live trading.
    
    Uses a background thread and queue to prevent I/O blocking in the main thread.
    """
    
    def __init__(self, base_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.base_handler = base_handler
        self.log_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LogWriter")
        self.shutdown_event = threading.Event()
        
        # Start the background logging thread
        self.executor.submit(self._log_worker)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Queue log record for async processing."""
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop log message if queue is full to prevent blocking
            pass
    
    def _log_worker(self) -> None:
        """Background worker to process log records."""
        while not self.shutdown_event.is_set():
            try:
                record = self.log_queue.get(timeout=1.0)
                if record is None:  # Shutdown signal
                    break
                self.base_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # Prevent logging errors from crashing the worker
                pass
    
    def close(self) -> None:
        """Shutdown the async handler gracefully."""
        self.shutdown_event.set()
        self.log_queue.put(None)  # Shutdown signal
        self.executor.shutdown(wait=True)
        self.base_handler.close()
        super().close()


def _add_timestamp(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add high-precision timestamp to log events."""
    event_dict['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    return event_dict


def _add_level(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure log level is in the event dictionary."""
    if 'level' not in event_dict:
        event_dict['level'] = 'info'
    return event_dict


def _add_logger_name(logger: FilteringBoundLogger, _, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add logger name to event dictionary."""
    if hasattr(logger, '_context') and 'logger' in logger._context:
        event_dict['logger'] = logger._context['logger']
    return event_dict


class PerformanceFilter(logging.Filter):
    """Filter to drop or modify log records based on performance criteria."""
    
    def __init__(self, max_length: int = 1000):
        super().__init__()
        self.max_length = max_length
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records for performance."""
        # Truncate extremely long messages
        if hasattr(record, 'getMessage'):
            msg = record.getMessage()
            if len(msg) > self.max_length:
                record.msg = msg[:self.max_length] + "... [TRUNCATED]"
        
        return True


def setup_logging() -> None:
    """
    Setup structured logging system based on configuration.
    
    Configures different logging outputs and formats for development vs production.
    """
    # Load logging configuration
    log_config = get_config_value('logging', {})
    log_level = log_config.get('level', 'INFO').upper()
    enable_async = log_config.get('async_logging', True)
    log_to_file = log_config.get('file_logging', True)
    log_to_console = log_config.get('console_logging', True)
    
    # Create log directory
    log_dir = ensure_directory(get_project_root() / "logs")
    
    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        _add_timestamp,
        _add_level,
        _add_logger_name,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    # Add development-friendly formatting for console
    if log_to_console:
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback)
        )
    
    # Configure the standard library logger
    stdlib_logger = logging.getLogger()
    stdlib_logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    for handler in stdlib_logger.handlers[:]:
        stdlib_logger.removeHandler(handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        
        # Use JSON format for production, readable format for development
        if log_config.get('json_format', False):
            console_formatter = logging.Formatter(
                '%(message)s'  # structlog will handle the formatting
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(PerformanceFilter())
        
        if enable_async:
            console_handler = AsyncLogHandler(console_handler)
        
        stdlib_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_file = log_dir / f"xauusd_scalper_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Use rotating file handler to manage log size
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_config.get('max_file_size_mb', 100) * 1024 * 1024,
            backupCount=log_config.get('backup_count', 5),
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level))
        
        # Always use JSON format for file logs
        file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s | %(pathname)s:%(lineno)d'
    )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(PerformanceFilter())
        
        if enable_async:
            file_handler = AsyncLogHandler(file_handler)
        
        stdlib_logger.addHandler(file_handler)
    
    # Error file handler (always enabled for critical issues)
    error_file = log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    error_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s | %(pathname)s:%(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    error_handler.setFormatter(error_formatter)
    
    if enable_async:
        error_handler = AsyncLogHandler(error_handler)
    
    stdlib_logger.addHandler(error_handler)
    
    # Configure structlog
    structlog.configure(
        processors=processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class TradingLogger:
    """
    Specialized logger for trading operations with performance optimizations.
    
    Provides methods for common trading events with pre-structured fields.
    """
    
    def __init__(self, component: str):
        self.logger = get_logger(f"trading.{component}")
        self.component = component
    
    def trade_signal(self, symbol: str, signal: str, confidence: float, 
                    features: Optional[Dict[str, float]] = None, **kwargs) -> None:
        """Log a trading signal generation event."""
        self.logger.info(
            "Trading signal generated",
            component=self.component,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            features=features or {},
            **kwargs
        )
    
    def order_placed(self, order_id: str, symbol: str, side: str, quantity: float,
                    price: Optional[float] = None, order_type: str = "market", **kwargs) -> None:
        """Log an order placement event."""
        self.logger.info(
            "Order placed",
            component=self.component,
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            **kwargs
        )
    
    def order_filled(self, order_id: str, symbol: str, side: str, quantity: float,
                    fill_price: float, latency_ms: Optional[float] = None, **kwargs) -> None:
        """Log an order fill event."""
        self.logger.info(
            "Order filled",
            component=self.component,
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            latency_ms=latency_ms,
            **kwargs
        )
    
    def position_update(self, symbol: str, position: float, unrealized_pnl: float,
                       realized_pnl: float, **kwargs) -> None:
        """Log a position update event."""
        self.logger.info(
            "Position updated",
            component=self.component,
            symbol=symbol,
            position=position,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            **kwargs
        )
    
    def risk_event(self, event_type: str, symbol: str, current_risk: float,
                  max_risk: float, action_taken: str, **kwargs) -> None:
        """Log a risk management event."""
        self.logger.warning(
            "Risk management event",
            component=self.component,
            event_type=event_type,
            symbol=symbol,
            current_risk=current_risk,
            max_risk=max_risk,
            action_taken=action_taken,
            **kwargs
        )
    
    def performance_metric(self, metric_name: str, value: float, timeframe: str,
                          benchmark: Optional[float] = None, **kwargs) -> None:
        """Log a performance metric."""
        self.logger.info(
            "Performance metric",
            component=self.component,
            metric_name=metric_name,
            value=value,
            timeframe=timeframe,
            benchmark=benchmark,
            **kwargs
        )
    
    def latency_measurement(self, operation: str, latency_ms: float,
                           target_ms: Optional[float] = None, **kwargs) -> None:
        """Log a latency measurement."""
        level = "warning" if target_ms and latency_ms > target_ms else "debug"
        getattr(self.logger, level)(
            "Latency measurement",
            component=self.component,
            operation=operation,
            latency_ms=latency_ms,
            target_ms=target_ms,
            **kwargs
        )


class MarketDataLogger:
    """Specialized logger for market data events with minimal overhead."""
    
    def __init__(self):
        self.logger = get_logger("market_data")
        self.tick_count = 0
        self.log_interval = get_config_value('logging.market_data_log_interval', 1000)
    
    def tick_received(self, symbol: str, bid: float, ask: float, timestamp: str,
                     latency_ms: Optional[float] = None) -> None:
        """Log tick data reception (throttled for performance)."""
        self.tick_count += 1
        
        # Only log every N ticks to avoid overwhelming logs
        if self.tick_count % self.log_interval == 0:
            self.logger.debug(
                "Tick data batch",
                symbol=symbol,
                latest_bid=bid,
                latest_ask=ask,
                latest_timestamp=timestamp,
                latency_ms=latency_ms,
                tick_count=self.tick_count
            )
    
    def orderbook_update(self, symbol: str, levels: int, top_bid: float, top_ask: float,
                        total_bid_volume: float, total_ask_volume: float) -> None:
        """Log order book update."""
        self.logger.debug(
            "Order book updated",
            symbol=symbol,
            levels=levels,
            top_bid=top_bid,
            top_ask=top_ask,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume
        )


def shutdown_logging() -> None:
    """Gracefully shutdown the logging system."""
    # Get all loggers and close async handlers
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())  # Add root logger
    
    for logger in loggers:
        for handler in logger.handlers:
            if isinstance(handler, AsyncLogHandler):
                handler.close()
    
    # Shutdown structlog
    logging.shutdown()


# Initialize logging when module is imported
try:
    setup_logging()
except Exception as e:
    # Fallback to basic logging if setup fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    logging.getLogger(__name__).error(f"Failed to setup structured logging: {e}")