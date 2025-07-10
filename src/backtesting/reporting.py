# src/backtesting/reporting.py

"""
Comprehensive reporting module for backtesting results.

This module generates visual reports and statistical summaries from backtest
results produced by the BacktestEngine.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    sns = None
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class BacktestReport:
    """Results from a backtest execution."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    final_balance: float
    total_pnl: float
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)


def generate_report(
    results: Dict[str, Any], 
    output_dir: str = "reports", 
    save_plots: bool = True,
    show_plots: bool = False
) -> None:
    """
    Generate comprehensive backtest report with plots and statistics.
    
    Args:
        results: Dictionary containing backtest results from BacktestEngine
        output_dir: Directory to save report files
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots interactively
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"BACKTESTING REPORT - {timestamp}")
    print(f"{'='*60}")
    
    # Print performance summary
    _print_performance_summary(results.get('performance_summary', {}))
    
    # Print exchange statistics
    if 'exchange_statistics' in results:
        _print_exchange_statistics(results['exchange_statistics'])
    
    # Generate plots if data is available
    if save_plots or show_plots:
        if 'performance_history' in results and results['performance_history']:
            _plot_equity_curve(
                results['performance_history'], 
                output_path, 
                timestamp, 
                save_plots, 
                show_plots
            )
            _plot_drawdown(
                results['performance_history'], 
                output_path, 
                timestamp, 
                save_plots, 
                show_plots
            )
        
        if 'trade_history' in results and results['trade_history']:
            _plot_pnl_distribution(
                results['trade_history'], 
                output_path, 
                timestamp, 
                save_plots, 
                show_plots
            )
            _plot_trade_timeline(
                results['trade_history'], 
                output_path, 
                timestamp, 
                save_plots, 
                show_plots
            )
    
    # Save summary to text file
    if save_plots:
        _save_summary_to_file(results, output_path, timestamp)
    
    print(f"\nReport generated successfully!")
    if save_plots:
        print(f"Files saved to: {output_path.absolute()}")


def generate_backtest_report(results: Dict[str, Any], output_dir: str = "reports") -> Path:
    """
    Alias for generate_report for backward compatibility.
    
    Args:
        results: Dictionary containing backtest results
        output_dir: Directory to save report files
        
    Returns:
        Path to the generated report directory
    """
    generate_report(results, output_dir, save_plots=True, show_plots=False)
    return Path(output_dir)


def _print_performance_summary(performance_summary: Dict[str, float]) -> None:
    """Print formatted performance summary statistics."""
    print(f"\n{'PERFORMANCE SUMMARY':^40}")
    print("-" * 40)
    
    metrics = [
        ("Total Return", performance_summary.get('total_return_pct', 0), "%"),
        ("Final Balance", performance_summary.get('final_balance', 0), "$"),
        ("Sharpe Ratio", performance_summary.get('sharpe_ratio', 0), ""),
        ("Sortino Ratio", performance_summary.get('sortino_ratio', 0), ""),
        ("Calmar Ratio", performance_summary.get('calmar_ratio', 0), ""),
        ("Max Drawdown", performance_summary.get('max_drawdown_pct', 0), "%"),
        ("Win Rate", performance_summary.get('win_rate_pct', 0), "%"),
        ("Profit Factor", performance_summary.get('profit_factor', 0), ""),
    ]
    
    for name, value, unit in metrics:
        if unit == "$":
            print(f"{name:.<20} ${value:>12,.2f}")
        elif unit == "%":
            print(f"{name:.<20} {value:>12.2f}%")
        else:
            print(f"{name:.<20} {value:>12.2f}")


def _print_exchange_statistics(exchange_stats: Dict[str, float]) -> None:
    """Print exchange-related statistics."""
    print(f"\n{'EXCHANGE STATISTICS':^40}")
    print("-" * 40)
    
    total_commission = exchange_stats.get('total_commission', 0)
    total_slippage = exchange_stats.get('total_slippage', 0)
    
    print(f"{'Total Commission':.<20} ${total_commission:>12.2f}")
    print(f"{'Total Slippage':.<20} ${total_slippage:>12.2f}")
    print(f"{'Total Costs':.<20} ${total_commission + total_slippage:>12.2f}")


def _plot_equity_curve(
    performance_history: List[Dict[str, Any]], 
    output_path: Path, 
    timestamp: str,
    save_plots: bool, 
    show_plots: bool
) -> None:
    """Generate equity curve plot."""
    df = pd.DataFrame(performance_history)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        ax.plot(df['timestamp'], df['balance'], linewidth=2, color='#2E86AB')
    else:
        ax.plot(df['balance'], linewidth=2, color='#2E86AB')
    
    ax.set_title('Equity Curve', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Balance ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(output_path / f"equity_curve_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def _plot_drawdown(
    performance_history: List[Dict[str, Any]], 
    output_path: Path, 
    timestamp: str,
    save_plots: bool, 
    show_plots: bool
) -> None:
    """Generate drawdown plot."""
    df = pd.DataFrame(performance_history)
    
    # Calculate drawdown if not present
    if 'drawdown' not in df.columns:
        peak = df['balance'].expanding().max()
        df['drawdown'] = (df['balance'] - peak) / peak * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        ax.fill_between(df['timestamp'], df['drawdown'], 0, 
                       color='#A23B72', alpha=0.7)
        ax.plot(df['timestamp'], df['drawdown'], 
               color='#A23B72', linewidth=1)
    else:
        ax.fill_between(range(len(df)), df['drawdown'], 0, 
                       color='#A23B72', alpha=0.7)
        ax.plot(df['drawdown'], color='#A23B72', linewidth=1)
    
    ax.set_title('Drawdown Analysis', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(output_path / f"drawdown_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def _plot_pnl_distribution(
    trade_history: List[Dict[str, Any]], 
    output_path: Path, 
    timestamp: str,
    save_plots: bool, 
    show_plots: bool
) -> None:
    """Generate PnL distribution histogram."""
    df = pd.DataFrame(trade_history)
    
    if 'pnl' not in df.columns:
        print("Warning: No PnL data available for distribution plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(df['pnl'], bins=50, alpha=0.7, color='#F18F01', edgecolor='black')
    ax1.axvline(df['pnl'].mean(), color='red', linestyle='--', 
               label=f'Mean: ${df["pnl"].mean():.2f}')
    ax1.axvline(df['pnl'].median(), color='green', linestyle='--', 
               label=f'Median: ${df["pnl"].median():.2f}')
    ax1.set_title('PnL Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('PnL ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(df['pnl'], vert=True, patch_artist=True,
               boxprops=dict(facecolor='#F18F01', alpha=0.7))
    ax2.set_title('PnL Box Plot', fontsize=14, fontweight='bold')
    ax2.set_ylabel('PnL ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(output_path / f"pnl_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def _plot_trade_timeline(
    trade_history: List[Dict[str, Any]], 
    output_path: Path, 
    timestamp: str,
    save_plots: bool, 
    show_plots: bool
) -> None:
    """Generate trade timeline plot."""
    df = pd.DataFrame(trade_history)
    
    if df.empty:
        print("Warning: No trade data available for timeline plot")
        return
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Separate winning and losing trades
    winners = df[df['pnl'] > 0] if 'pnl' in df.columns else pd.DataFrame()
    losers = df[df['pnl'] <= 0] if 'pnl' in df.columns else pd.DataFrame()
    
    if not winners.empty and 'timestamp' in winners.columns:
        ax.scatter(winners['timestamp'], winners['pnl'], 
                  c='green', alpha=0.6, s=30, label='Winning Trades')
    
    if not losers.empty and 'timestamp' in losers.columns:
        ax.scatter(losers['timestamp'], losers['pnl'], 
                  c='red', alpha=0.6, s=30, label='Losing Trades')
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('Trade Timeline', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('PnL ($)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(output_path / f"trade_timeline_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def _save_summary_to_file(
    results: Dict[str, Any], 
    output_path: Path, 
    timestamp: str
) -> None:
    """Save text summary of results to file."""
    summary_file = output_path / f"backtest_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"BACKTESTING REPORT - {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        
        # Performance summary
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        performance = results.get('performance_summary', {})
        for key, value in performance.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\n")
        
        # Exchange statistics
        if 'exchange_statistics' in results:
            f.write("EXCHANGE STATISTICS\n")
            f.write("-" * 40 + "\n")
            
            exchange_stats = results['exchange_statistics']
            for key, value in exchange_stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        # Trade summary
        if 'trade_history' in results and results['trade_history']:
            f.write(f"\nTRADE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Trades: {len(results['trade_history'])}\n")
            
            trade_df = pd.DataFrame(results['trade_history'])
            if 'pnl' in trade_df.columns:
                winning_trades = (trade_df['pnl'] > 0).sum()
                losing_trades = (trade_df['pnl'] <= 0).sum()
                f.write(f"Winning Trades: {winning_trades}\n")
                f.write(f"Losing Trades: {losing_trades}\n")
                f.write(f"Average PnL: ${trade_df['pnl'].mean():.2f}\n")
                f.write(f"Best Trade: ${trade_df['pnl'].max():.2f}\n")
                f.write(f"Worst Trade: ${trade_df['pnl'].min():.2f}\n")


def calculate_additional_metrics(
    performance_history: List[Dict[str, Any]], 
    trade_history: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate additional performance metrics not included in the main summary.
    
    Args:
        performance_history: List of performance data points
        trade_history: List of trade records
        
    Returns:
        Dictionary of additional metrics
    """
    metrics = {}
    
    if performance_history:
        perf_df = pd.DataFrame(performance_history)
        
        # Calculate volatility of returns
        if 'balance' in perf_df.columns and len(perf_df) > 1:
            returns = perf_df['balance'].pct_change().dropna()
            metrics['volatility_annualized'] = returns.std() * np.sqrt(252 * 24 * 60)  # Assuming minute data
            metrics['skewness'] = returns.skew()
            metrics['kurtosis'] = returns.kurtosis()
    
    if trade_history:
        trade_df = pd.DataFrame(trade_history)
        
        if 'pnl' in trade_df.columns:
            # Trade-level statistics
            metrics['avg_trade_pnl'] = trade_df['pnl'].mean()
            metrics['trade_pnl_std'] = trade_df['pnl'].std()
            metrics['best_trade'] = trade_df['pnl'].max()
            metrics['worst_trade'] = trade_df['pnl'].min()
            
            # Consecutive wins/losses
            trade_df['win'] = trade_df['pnl'] > 0
            consecutive_wins = []
            consecutive_losses = []
            current_win_streak = 0
            current_loss_streak = 0
            
            for win in trade_df['win']:
                if win:
                    current_win_streak += 1
                    if current_loss_streak > 0:
                        consecutive_losses.append(current_loss_streak)
                        current_loss_streak = 0
                else:
                    current_loss_streak += 1
                    if current_win_streak > 0:
                        consecutive_wins.append(current_win_streak)
                        current_win_streak = 0
            
            # Add final streaks
            if current_win_streak > 0:
                consecutive_wins.append(current_win_streak)
            if current_loss_streak > 0:
                consecutive_losses.append(current_loss_streak)
            
            metrics['max_consecutive_wins'] = max(consecutive_wins) if consecutive_wins else 0
            metrics['max_consecutive_losses'] = max(consecutive_losses) if consecutive_losses else 0
    
    return metrics