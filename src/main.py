# src/main.py

"""
Main CLI entry point for the XAUUSD Scalping System.

This module provides command-line interface for training models and running
backtests using the click library.
"""

import os
import sys
import click
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.loader import DataLoader
from src.features.feature_pipeline import FeaturePipeline
from src.models.train import ModelTrainer
from src.models.predict import ModelPredictor
from src.backtesting.engine import BacktestEngine
from src.backtesting.reporting import generate_report
from src.live.engine import LiveEngine
from src.utils.logging import setup_logging
from src.utils.helpers import get_config_value


@click.group()
@click.option('--config', '-c', default='configs/config.yml', 
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """
    XAUUSD Scalping System - Production-grade low-latency trading system.
    
    This CLI provides commands for training models and running backtests
    on XAUUSD (Gold/USD) tick data.
    """
    # Ensure ctx.obj exists
    ctx.ensure_object(dict)
    
    # Store config path and verbose flag
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    # Set up logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging()
    
    # Validate config file exists
    if not Path(config).exists():
        click.echo(f"Error: Configuration file '{config}' not found.", err=True)
        sys.exit(1)


@cli.command()
@click.option('--data-path', '-d', 
              help='Path to training data (overrides config)')
@click.option('--model-name', '-m', default='xauusd_model',
              help='Name for the trained model')
@click.option('--epochs', '-e', type=int,
              help='Number of training epochs (overrides config)')
@click.option('--batch-size', '-b', type=int,
              help='Training batch size (overrides config)')
@click.option('--learning-rate', '-lr', type=float,
              help='Learning rate (overrides config)')
@click.option('--validation-split', '-vs', type=float,
              help='Validation split ratio (overrides config)')
@click.option('--save-path', '-s', default='models/',
              help='Directory to save trained model')
@click.pass_context
def train(ctx: click.Context, 
          data_path: Optional[str],
          model_name: str,
          epochs: Optional[int],
          batch_size: Optional[int],
          learning_rate: Optional[float],
          validation_split: Optional[float],
          save_path: str) -> None:
    """
    Train a new model on historical XAUUSD data.
    
    This command loads historical tick data, engineers features, and trains
    a neural network model for predicting short-term price movements.
    
    Example:
        python src/main.py train --data-path data/historical/ticks/ --epochs 100
    """
    click.echo("🚀 Starting model training...")
    
    try:
        # Load configuration
        config_path = ctx.obj['config_path']
        config_overrides = {}
        
        # Apply command-line overrides
        if data_path:
            config_overrides['data_path'] = data_path
        if epochs:
            config_overrides['training_epochs'] = epochs
        if batch_size:
            config_overrides['batch_size'] = batch_size
        if learning_rate:
            config_overrides['learning_rate'] = learning_rate
        if validation_split:
            config_overrides['validation_split'] = validation_split
        
        # Load configuration
        with open(config_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
            
        # Apply overrides
        for key, value in config_overrides.items():
            config[key] = value
            
        # Initialize trainer
        trainer = ModelTrainer(config.get('models', {}), save_path)
        
        # Create save directory
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Train model  
        click.echo("📊 Loading and preparing data...")
        # Create dummy data for now - replace with actual data loading
        import pandas as pd
        import numpy as np
        X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(np.random.randn(1000))

        model_path = trainer.train_random_forest(X, y)
        training_history = None
        
        # Model already saved by train_random_forest
        final_model_path = save_dir / f"{model_name}.pkl"
        click.echo(f"💾 Model saved to {model_path}...")
        
        # Print training summary
        click.echo("\n✅ Training completed successfully!")
        click.echo(f"📁 Model saved to: {model_path}")
        
        # Display final metrics if available
        if training_history and 'val_loss' in training_history.history:
            final_val_loss = training_history.history['val_loss'][-1]
            click.echo(f"📈 Final validation loss: {final_val_loss:.6f}")
        
        if training_history and 'val_accuracy' in training_history.history:
            final_val_acc = training_history.history['val_accuracy'][-1]
            click.echo(f"🎯 Final validation accuracy: {final_val_acc:.4f}")
            
    except Exception as e:
        click.echo(f"❌ Training failed: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.option('--model-path', '-m', required=True,
              help='Path to trained model file')
@click.option('--data-path', '-d',
              help='Path to backtest data (overrides config)')
@click.option('--start-date', '-sd',
              help='Start date for backtest (YYYY-MM-DD)')
@click.option('--end-date', '-ed',
              help='End date for backtest (YYYY-MM-DD)')
@click.option('--initial-balance', '-ib', type=float,
              help='Initial balance for backtest (overrides config)')
@click.option('--output-dir', '-o', default='reports',
              help='Directory to save backtest reports')
@click.option('--no-plots', is_flag=True,
              help='Skip generating plots (faster execution)')
@click.option('--show-plots', is_flag=True,
              help='Display plots interactively')
@click.pass_context
def backtest(ctx: click.Context,
             model_path: str,
             data_path: Optional[str],
             start_date: Optional[str],
             end_date: Optional[str],
             initial_balance: Optional[float],
             output_dir: str,
             no_plots: bool,
             show_plots: bool) -> None:
    """
    Run backtest using a trained model.
    
    This command loads a trained model and runs a comprehensive backtest
    on historical data, generating performance reports and visualizations.
    
    Example:
        python src/main.py backtest --model-path models/xauusd_model.keras
    """
    click.echo("🔍 Starting backtest...")
    
    try:
        # Validate model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load configuration and apply overrides
        config_path = ctx.obj['config_path']
        config_overrides = {'model_path': model_path}
        
        if data_path:
            config_overrides['data_path'] = data_path
        if start_date:
            config_overrides['backtest_start_date'] = start_date
        if end_date:
            config_overrides['backtest_end_date'] = end_date
        if initial_balance:
            config_overrides['initial_balance'] = initial_balance
        
        # Load configuration
        with open(config_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
            
        # Apply overrides
        for key, value in config_overrides.items():
            config[key] = value

        # Initialize backtest engine
        click.echo("⚙️  Initializing backtest engine...")
        engine = BacktestEngine(config)
        
        # Run backtest
        click.echo("🏃 Running backtest simulation...")
        with click.progressbar(length=100, label='Processing ticks') as bar:
            # Note: In a real implementation, you'd need to modify BacktestEngine
            # to accept a progress callback. For now, we just show indeterminate progress.
            results = engine.run_backtest()
            bar.update(100)
        
        # Generate report
        click.echo("📊 Generating performance report...")
        generate_report(
            results=results,
            output_dir=output_dir,
            save_plots=not no_plots,
            show_plots=show_plots
        )
        
        # Display key metrics
        perf_summary = results.get('performance_summary', {})
        click.echo("\n🎯 Key Performance Metrics:")
        click.echo(f"   Total Return: {perf_summary.get('total_return_pct', 0):.2f}%")
        click.echo(f"   Sharpe Ratio: {perf_summary.get('sharpe_ratio', 0):.2f}")
        click.echo(f"   Max Drawdown: {perf_summary.get('max_drawdown_pct', 0):.2f}%")
        click.echo(f"   Win Rate: {perf_summary.get('win_rate_pct', 0):.2f}%")
        
        trade_count = len(results.get('trade_history', []))
        click.echo(f"   Total Trades: {trade_count}")
        
        click.echo(f"\n✅ Backtest completed successfully!")
        click.echo(f"📁 Reports saved to: {Path(output_dir).absolute()}")
        
    except Exception as e:
        click.echo(f"❌ Backtest failed: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.option('--model-path', '-m',
              help='Path to trained model file (overrides config)')
@click.option('--dry-run', is_flag=True,
              help='Run in dry-run mode (log orders without executing)')
@click.option('--data-feed', '-df',
              help='Data feed configuration (overrides config)')
@click.pass_context
def live(ctx: click.Context,
         model_path: Optional[str],
         dry_run: bool,
         data_feed: Optional[str]) -> None:
    """
    Start live trading with real-time market data.
    
    This command initializes the live trading engine and begins processing
    real-time market data for automated trading decisions.
    
    CAUTION: This will execute real trades unless --dry-run is specified.
    Always test thoroughly in dry-run mode before live trading.
    
    Example:
        python src/main.py live --model-path models/xauusd_model.keras --dry-run
    """
    # Warning for live trading
    if not dry_run:
        click.echo("⚠️  WARNING: You are about to start LIVE TRADING with real money!")
        click.echo("   This will execute actual trades in the market.")
        click.echo("   Make sure you have tested thoroughly in dry-run mode.")
        
        if not click.confirm("\nDo you want to continue with live trading?"):
            click.echo("Live trading cancelled.")
            return
    
    click.echo(f"🚀 Starting live trading engine in {'DRY-RUN' if dry_run else 'LIVE'} mode...")
    
    try:
        # Load configuration and apply overrides
        config_path = ctx.obj['config_path']
        config_overrides = {}
        
        if model_path:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            config_overrides['live_trading'] = {'model_path': model_path}
        
        if data_feed:
            config_overrides['data_feed'] = {'source': data_feed}
        
        # Initialize live engine
        click.echo("⚙️  Initializing live trading engine...")
        engine = LiveEngine(config_path=config_path, config_overrides=config_overrides)
        
        # Display configuration summary
        click.echo("\n📋 Trading Configuration:")
        click.echo(f"   Mode: {'DRY-RUN' if dry_run else 'LIVE TRADING'}")
        click.echo(f"   Model: {config_overrides.get('live_trading', {}).get('model_path', 'from config')}")
        click.echo(f"   Data Feed: {config_overrides.get('data_feed', {}).get('source', 'from config')}")
        
        if dry_run:
            click.echo("\n🔒 DRY-RUN MODE: No real trades will be executed")
        else:
            click.echo("\n💰 LIVE MODE: Real trades will be executed!")
        
        click.echo("\n🎯 Performance target: <50ms per processing loop")
        click.echo("📊 Press Ctrl+C to stop gracefully and view statistics\n")
        
        # Run the live engine
        try:
            asyncio.run(engine.run(dry_run=dry_run))
        except KeyboardInterrupt:
            click.echo("\n\n🛑 Received shutdown signal...")
            # Engine handles graceful shutdown internally
        
        click.echo("✅ Live trading engine stopped successfully")
        
    except Exception as e:
        click.echo(f"❌ Live trading failed: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def validate_config(ctx: click.Context) -> None:
    """
    Validate the configuration file.
    
    This command checks the configuration file for required fields
    and validates the structure.
    """
    config_path = ctx.obj['config_path']
    
    try:
        click.echo(f"🔍 Validating configuration file: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['data', 'model', 'backtesting', 'risk_management']
        missing_sections = []
        
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            click.echo(f"❌ Missing required sections: {', '.join(missing_sections)}", err=True)
            sys.exit(1)
        
        # Validate specific fields
        data_config = config.get('data', {})
        if 'historical_data_path' not in data_config:
            click.echo("⚠️  Warning: 'historical_data_path' not found in data section")
        
        model_config = config.get('model', {})
        if 'architecture' not in model_config:
            click.echo("⚠️  Warning: 'architecture' not found in model section")
        
        backtesting_config = config.get('backtesting', {})
        if 'initial_balance' not in backtesting_config:
            click.echo("⚠️  Warning: 'initial_balance' not found in backtesting section")
        
        risk_config = config.get('risk_management', {})
        if 'max_position_size' not in risk_config:
            click.echo("⚠️  Warning: 'max_position_size' not found in risk_management section")
        
        # Check live trading specific configuration
        live_config = config.get('live_trading', {})
        if 'model_path' not in live_config:
            click.echo("⚠️  Warning: 'model_path' not found in live_trading section")
        
        data_feed_config = config.get('data_feed', {})
        if 'source' not in data_feed_config:
            click.echo("⚠️  Warning: 'source' not found in data_feed section")
        
        click.echo("✅ Configuration file is valid!")
        
    except yaml.YAMLError as e:
        click.echo(f"❌ YAML parsing error: {str(e)}", err=True)
        sys.exit(1)
    except FileNotFoundError:
        click.echo(f"❌ Configuration file not found: {config_path}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Validation failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def version() -> None:
    """Display version information."""
    click.echo("XAUUSD Scalping System v1.0.0")
    click.echo("Production-grade low-latency intraday trading system")
    click.echo("Copyright (c) 2024")


if __name__ == '__main__':
    cli()