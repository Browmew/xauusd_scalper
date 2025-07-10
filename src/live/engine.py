# src/live/engine.py

"""
Live Trading Engine for XAUUSD Scalping System.

This module contains the main LiveEngine class that orchestrates real-time
trading operations with low-latency asynchronous processing.
"""

import asyncio
import time
import logging
from collections import deque
from typing import Optional, Dict, Any, Deque
from pathlib import Path
import yaml

from .feed_handler import LiveFeedHandler
from .order_router import SmartOrderRouter
from ..features.feature_pipeline import FeaturePipeline
from ..models.predict import ModelPredictor
from ..backtesting.engine import RiskManager
from ..backtesting.exchange_simulator import Order
from ..utils.logging import setup_logging


logger = logging.getLogger(__name__)


class LiveEngine:
    """
    Main live trading engine with asynchronous event loop.
    
    This class orchestrates real-time data ingestion, feature engineering,
    model prediction, risk management, and order execution in a low-latency
    asynchronous environment.
    
    Key Features:
    - Sub-50ms processing loop target
    - Asynchronous I/O operations
    - Rolling data buffer management
    - Real-time latency monitoring
    - Dry-run mode support
    - Graceful shutdown handling
    """
    
    def __init__(self, config_path: str = "configs/config.yml", 
                 config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the live trading engine.
        
        Args:
            config_path: Path to configuration file
            config_overrides: Optional configuration overrides
        """
        self.config_path = config_path
        self.config_overrides = config_overrides or {}
        
        # Load configuration
        self._load_config()
        
        # Initialize performance tracking
        self.loop_count = 0
        self.total_loop_time = 0.0
        self.max_loop_time = 0.0
        self.min_loop_time = float('inf')
        
        # State management
        self.is_running = False
        self.should_stop = False
        
        # Rolling data buffer for feature engineering
        buffer_size = self.config.get('live_trading', {}).get('buffer_size', 1000)
        self.data_buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)
        
        # Initialize core components
        self._initialize_components()
        
        logger.info("LiveEngine initialized successfully")
    
    def _load_config(self) -> None:
        """Load and validate configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Apply overrides
            for key, value in self.config_overrides.items():
                self.config[key] = value
                
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _initialize_components(self) -> None:
        """Initialize all trading components."""
        try:
            # Initialize live data feed handler
            self.feed_handler = LiveFeedHandler(
                config=self.config.get('data_feed', {})
            )
            
            # Initialize order router
            self.order_router = SmartOrderRouter(
                config=self.config.get('order_routing', {})
            )
            
            # Initialize feature pipeline
            self.feature_pipeline = FeaturePipeline(
                config=self.config.get('features', {})
            )
            
            # Initialize model predictor
            model_path = self.config.get('live_trading', {}).get('model_path')
            if not model_path:
                raise ValueError("Model path not specified in configuration")
            
            self.model_predictor = ModelPredictor(
                model_path=model_path,
                config=self.config.get('model', {})
            )
            
            # Initialize risk manager
            self.risk_manager = RiskManager(
                config=self.config.get('risk_management', {})
            )
            
            logger.info("All trading components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def run(self, dry_run: bool = False) -> None:
        """
        Run the main live trading loop.
        
        Args:
            dry_run: If True, log orders instead of executing them
        """
        logger.info(f"Starting LiveEngine in {'dry-run' if dry_run else 'live'} mode")
        
        try:
            self.is_running = True
            self.should_stop = False
            
            # Get data queue from feed handler
            data_queue = self.feed_handler.get_data_queue()
            
            # Start feed handler as background task
            feed_task = asyncio.create_task(self.feed_handler.run())
            
            # Start main processing loop
            await self._main_loop(data_queue, dry_run)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down gracefully...")
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
            raise
        finally:
            self.is_running = False
            self.should_stop = True
            
            # Cancel feed handler task
            if 'feed_task' in locals():
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass
            
            # Print performance statistics
            self._print_performance_stats()
            
            logger.info("LiveEngine shutdown complete")
    
    async def _main_loop(self, data_queue: asyncio.Queue, dry_run: bool) -> None:
        """
        Main asynchronous processing loop.
        
        Args:
            data_queue: Queue containing live market data
            dry_run: Whether to execute orders or just log them
        """
        logger.info("Starting main processing loop")
        
        while not self.should_stop:
            loop_start_time = time.perf_counter()
            
            try:
                # Get new tick data (with timeout to allow graceful shutdown)
                try:
                    tick_data = await asyncio.wait_for(data_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue  # Check should_stop flag and continue
                
                # Update rolling buffer
                self.data_buffer.append(tick_data)
                
                # Skip processing if buffer not sufficiently filled
                min_buffer_size = self.config.get('live_trading', {}).get('min_buffer_size', 100)
                if len(self.data_buffer) < min_buffer_size:
                    continue
                
                # Feature engineering
                features_start = time.perf_counter()
                feature_data = self._prepare_feature_data()
                features = self.feature_pipeline.transform(feature_data)
                features_time = time.perf_counter() - features_start
                
                # Model prediction
                prediction_start = time.perf_counter()
                signal = self.model_predictor.predict(features)
                prediction_time = time.perf_counter() - prediction_start
                
                # Risk management and trading decision
                risk_start = time.perf_counter()
                trading_decision = await self._make_trading_decision(
                    signal, tick_data, dry_run
                )
                risk_time = time.perf_counter() - risk_start
                
                # Calculate and log loop performance
                loop_end_time = time.perf_counter()
                loop_duration = loop_end_time - loop_start_time
                
                self._update_performance_stats(loop_duration)
                
                # Log detailed timing for monitoring
                if self.loop_count % 100 == 0:  # Log every 100th iteration
                    logger.info(
                        f"Loop {self.loop_count}: Total={loop_duration*1000:.2f}ms "
                        f"(Features={features_time*1000:.2f}ms, "
                        f"Prediction={prediction_time*1000:.2f}ms, "
                        f"Risk={risk_time*1000:.2f}ms)"
                    )
                
                # Warning if loop exceeds target latency
                if loop_duration > 0.05:  # 50ms threshold
                    logger.warning(
                        f"Loop latency exceeded target: {loop_duration*1000:.2f}ms"
                    )
                
            except Exception as e:
                logger.error(f"Error in main loop iteration {self.loop_count}: {e}")
                # Continue processing to maintain system stability
                continue
            
            self.loop_count += 1
    
    def _prepare_feature_data(self) -> Dict[str, Any]:
        """
        Prepare data from buffer for feature engineering.
        
        Returns:
            Dictionary containing recent tick data for feature calculation
        """
        # Convert deque to list for feature pipeline
        recent_ticks = list(self.data_buffer)
        
        return {
            'ticks': recent_ticks,
            'timestamp': recent_ticks[-1]['timestamp'] if recent_ticks else None
        }
    
    async def _make_trading_decision(self, signal: Dict[str, Any], 
                                   current_tick: Dict[str, Any], 
                                   dry_run: bool) -> Optional[Dict[str, Any]]:
        """
        Make and execute trading decision based on signal and risk checks.
        
        Args:
            signal: Model prediction signal
            current_tick: Current market tick data
            dry_run: Whether to execute orders or just log them
            
        Returns:
            Trading decision details or None if no action taken
        """
        try:
            # Extract signal information
            predicted_direction = signal.get('direction')  # 'long', 'short', or 'hold'
            confidence = signal.get('confidence', 0.0)
            
            # Skip if signal is to hold or confidence too low
            min_confidence = self.config.get('live_trading', {}).get('min_confidence', 0.6)
            if predicted_direction == 'hold' or confidence < min_confidence:
                return None
            
            # Check if trading is allowed
            if not self.risk_manager.check_trading_allowed():
                logger.debug("Trading not allowed by risk manager")
                return None
            
            # Calculate position size
            current_price = current_tick.get('bid', 0.0)
            position_size = self.risk_manager.calculate_position_size(
                price=current_price,
                direction=predicted_direction
            )
            
            if position_size <= 0:
                logger.debug("Position size too small, skipping trade")
                return None
            
            # Create order
            order = Order(
                symbol='XAUUSD',
                side='buy' if predicted_direction == 'long' else 'sell',
                quantity=position_size,
                price=current_price,
                order_type='market',
                timestamp=current_tick.get('timestamp')
            )
            
            # Execute or log order
            if dry_run:
                logger.info(
                    f"DRY RUN - Would execute order: {order.side.upper()} "
                    f"{order.quantity} XAUUSD @ {order.price:.5f} "
                    f"(confidence: {confidence:.3f})"
                )
            else:
                logger.info(
                    f"Executing order: {order.side.upper()} {order.quantity} "
                    f"XAUUSD @ {order.price:.5f} (confidence: {confidence:.3f})"
                )
                
                # Execute order through router
                execution_result = await self.order_router.execute_order(order)
                
                if execution_result.get('status') == 'filled':
                    logger.info(f"Order executed successfully: {execution_result}")
                else:
                    logger.warning(f"Order execution failed: {execution_result}")
            
            return {
                'order': order,
                'signal': signal,
                'executed': not dry_run,
                'timestamp': current_tick.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Error in trading decision: {e}")
            return None
    
    def _update_performance_stats(self, loop_duration: float) -> None:
        """Update performance tracking statistics."""
        self.total_loop_time += loop_duration
        self.max_loop_time = max(self.max_loop_time, loop_duration)
        self.min_loop_time = min(self.min_loop_time, loop_duration)
    
    def _print_performance_stats(self) -> None:
        """Print performance statistics summary."""
        if self.loop_count == 0:
            return
        
        avg_loop_time = self.total_loop_time / self.loop_count
        
        logger.info("=== Performance Statistics ===")
        logger.info(f"Total loops processed: {self.loop_count}")
        logger.info(f"Average loop time: {avg_loop_time*1000:.2f}ms")
        logger.info(f"Min loop time: {self.min_loop_time*1000:.2f}ms")
        logger.info(f"Max loop time: {self.max_loop_time*1000:.2f}ms")
        
        # Calculate percentile of loops under target
        target_met_pct = (self.loop_count - sum(1 for _ in range(self.loop_count) 
                          if self.max_loop_time > 0.05)) / self.loop_count * 100
        logger.info(f"Loops under 50ms target: {target_met_pct:.1f}%")
    
    async def stop(self) -> None:
        """Gracefully stop the trading engine."""
        logger.info("Stopping LiveEngine...")
        self.should_stop = True
        
        # Wait for current loop to complete
        while self.is_running:
            await asyncio.sleep(0.1)
        
        logger.info("LiveEngine stopped")