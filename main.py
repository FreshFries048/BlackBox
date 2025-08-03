"""
BlackBox Trading System - Main Application

This script demonstrates the full BlackBox Core Engine pipeline:
Phase 1: Strategy Node Parsing
Phase 2: Signal Detection  
Phase 3: Trade Execution

Shows end-to-end functionality from strategy files to completed trades.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Import all BlackBox components directly
from blackbox_core import NodeEngine
from node_detector import NodeDetectorEngine, SignalEvent
from trade_executor import TradeExecutorEngine, RiskManager, Position
from blackbox_config.config_loader import resolve_data_path, get_rr
from dataset_scanner import DatasetScanner

# Configuration
class Config:
    """Configuration settings for the BlackBox system."""
    
    # Directory paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    RESULTS_DIR = BASE_DIR / "results"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Data files
    EUR_USD_DATA = RAW_DATA_DIR / "EURUSD_1H_2020-2024.csv"
    STRATEGY_NODES_DIR = BASE_DIR / "blackbox_nodes"
    
    # Output files
    TRADES_CSV = RESULTS_DIR / "complete_system_trades.csv"
    SIGNALS_CSV = RESULTS_DIR / "complete_system_signals.csv"
    BACKTEST_LOG = LOGS_DIR / "backtest_results.txt"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        for dir_path in [cls.DATA_DIR, cls.PROCESSED_DATA_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(exist_ok=True)


def select_risk_reward_ratio() -> float:
    """
    Interactive selection of risk-reward ratio.
    
    Returns:
        Selected RR ratio as float
    """
    rr_options = [
        ("1", "0.5:1 (Conservative - Take profit at half the risk)", 0.5),
        ("2", "1:1 (Balanced - Equal risk/reward)", 1.0),
        ("3", "1:2 (Moderate - 2x risk for reward)", 2.0),
        ("4", "1:3 (Aggressive - 3x risk for reward)", 3.0),
        ("5", "1:4 (Very Aggressive - 4x risk for reward)", 4.0),
        ("6", "1:5 (Maximum - 5x risk for reward)", 5.0)
    ]
    
    print(f"\n⚖️  RISK-REWARD RATIO SELECTION")
    print("=" * 60)
    print("Choose your preferred risk-reward multiple:")
    print()
    
    for option, description, _ in rr_options:
        print(f"{option}. {description}")
    
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-6, or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("\n👋 Goodbye!")
                sys.exit(0)
                
            choice_num = int(choice)
            if 1 <= choice_num <= 6:
                selected = rr_options[choice_num - 1]
                print(f"\n✅ Selected: {selected[1]}")
                return selected[2]
            else:
                print(f"❌ Invalid choice. Please enter 1-6")
                
        except ValueError:
            print("❌ Invalid input. Please enter a number (1-6) or 'q'")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def load_real_eurusd_data(data_path: Path, num_points: int = 500, start_date: str = "2024-01-01"):
    """Load real EUR/USD data from CSV file and enhance with trading indicators."""
    
    print("📊 Loading real EUR/USD market data...")
    
    try:
        # Load the real EUR/USD data using provided path
        df = pd.read_csv(data_path)
        print(f"   ✅ Loaded {len(df)} rows of real EUR/USD data")
        
        # Convert timestamp column
        df['time'] = pd.to_datetime(df['time'])
        
        # Filter to recent data starting from specified date
        df_filtered = df[df['time'] >= start_date].copy()
        
        # Take subset for demo (last N points or from start_date)
        if len(df_filtered) > num_points:
            df_subset = df_filtered.tail(num_points).copy()
        else:
            df_subset = df_filtered.copy()
        
        df_subset = df_subset.reset_index(drop=True)
        print(f"   📈 Using {len(df_subset)} data points from {df_subset['time'].iloc[0]} to {df_subset['time'].iloc[-1]}")
        
        # Calculate additional technical indicators for signal detection
        df_subset['price_change'] = df_subset['close'].pct_change()
        df_subset['volatility'] = df_subset['high'] - df_subset['low']
        df_subset['volume_ma'] = df_subset['tick_volume'].rolling(window=10).mean()
        df_subset['volume_spike'] = df_subset['tick_volume'] / df_subset['volume_ma']
        
        # Create synthetic trading indicators based on real price action
        # These simulate the kind of institutional flow data the strategies look for
        np.random.seed(42)  # For consistent demo results
        
        # Gamma exposure (higher during volatile periods)
        df_subset['gamma'] = (df_subset['volatility'] * 10000 + 
                             np.random.randint(50, 300, len(df_subset))).astype(int)
        
        # Dark pool activity (correlated with volume spikes)
        df_subset['dark_prints'] = np.where(
            df_subset['volume_spike'].fillna(1) > 1.5,
            np.random.randint(3, 12, len(df_subset)),
            np.random.randint(0, 3, len(df_subset))
        )
        
        # Block trades (during high volatility periods)
        df_subset['block_size'] = np.where(
            df_subset['volatility'] > df_subset['volatility'].quantile(0.8),
            np.random.randint(5000, 20000, len(df_subset)),
            np.random.randint(0, 2000, len(df_subset))
        )
        
        # Footprint clusters (market structure analysis)
        df_subset['footprint_clusters'] = np.random.randint(0, 5, len(df_subset))
        
        # Add some strategic market events based on significant price movements
        significant_moves = df_subset[abs(df_subset['price_change']) > df_subset['price_change'].std() * 2].index
        
        print(f"   🎯 Enhanced data with synthetic institutional indicators")
        print(f"   📊 Identified {len(significant_moves)} significant price movements for enhanced signals")
        
        # Prepare data dictionary for the system
        data = {
            'timestamp': df_subset['time'],
            'price': df_subset['close'],  # Use close price as main price
            'open': df_subset['open'],
            'high': df_subset['high'],
            'low': df_subset['low'],
            'volume': df_subset['tick_volume'],
            'spread': df_subset['spread'],
            'gamma': df_subset['gamma'],
            'dark_prints': df_subset['dark_prints'],
            'footprint_clusters': df_subset['footprint_clusters'],
            'block_size': df_subset['block_size'],
            'volatility': df_subset['volatility'],
            'volume_spike': df_subset['volume_spike'].fillna(1.0)
        }
        
        return data
        
    except FileNotFoundError:
        print("❌ EUR/USD CSV file not found. Please ensure 'EURUSD_1H_2020-2024.csv' exists in the workspace.")
        raise
    except Exception as e:
        print(f"❌ Error loading EUR/USD data: {str(e)}")
        raise


def run_complete_blackbox_system(data_path: Path = None, rr_multiple: float = None):
    """Run the complete BlackBox trading system end-to-end."""
    
    print("🚀 BLACKBOX COMPLETE SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Ensure all required directories exist
    Config.ensure_directories()
    
    # PHASE 1: Load Strategy Nodes
    print("\n🔧 PHASE 1: Loading Strategy Knowledge Base")
    print("-" * 50)
    
    node_engine = NodeEngine()
    loaded_count = node_engine.load_nodes_from_folder(str(Config.STRATEGY_NODES_DIR))
    print(f"✅ Loaded {loaded_count} strategy nodes successfully")
    
    # Show strategy summary
    print("\nStrategy Portfolio:")
    for i, node in enumerate(node_engine.nodes, 1):
        print(f"  {i}. {node.name} ({node.metadata.get('confidence', 'Unknown')} confidence)")
    
    # PHASE 2: Generate Trading Signals
    print(f"\n🔍 PHASE 2: Real-Time Signal Detection")
    print("-" * 50)
    
    # Create market data from real EUR/USD data
    if data_path is None:
        data_path = Config.EUR_USD_DATA
    market_data = load_real_eurusd_data(data_path, 500, "2024-01-01")
    
    # Initialize signal detector
    detector = NodeDetectorEngine(node_engine, market_data)
    
    # Run signal detection
    print("\nRunning signal detection across market data...")
    signals = detector.run_detection(live_output=False)
    
    # Filter for tradeable signals
    tradeable_signals = [s for s in signals if any(conf in s.confidence for conf in ["High", "Highest"])]
    
    print(f"\n📊 Signal Generation Results:")
    print(f"   Total signals generated: {len(signals)}")
    print(f"   High/Highest confidence signals: {len(tradeable_signals)}")
    
    # Show signal breakdown
    signal_breakdown = {}
    for signal in tradeable_signals:
        node = signal.node_name
        signal_breakdown[node] = signal_breakdown.get(node, 0) + 1
    
    print(f"\nTradeable signals by strategy:")
    for node, count in signal_breakdown.items():
        print(f"   • {node}: {count} signals")
    
    # PHASE 3: Execute Trades
    print(f"\n💼 PHASE 3: Trade Execution & Management")
    print("-" * 50)
    
    # Initialize trade executor with forex-appropriate risk settings
    if rr_multiple is None:
        rr_multiple = 3.0  # Default fallback
    
    print(f"💼 Risk Management Configuration:")
    print(f"   Risk-Reward Multiple: {rr_multiple}")
    
    risk_manager = RiskManager(
        default_stop_loss_pct=50,    # 50 pips default (overridden by pip-based logic)
        default_take_profit_pct=60,  # 60 pips default (overridden by pip-based logic)
        rr_multiple=rr_multiple
    )
    
    executor = TradeExecutorEngine(market_data, risk_manager)
    
    # Run proper backtesting: execute signals chronologically and manage positions forward
    print("\nRunning chronological backtesting...")
    trades_opened = executor.run_backtest(tradeable_signals)
    
    # RESULTS & ANALYSIS
    print(f"\n📈 COMPLETE SYSTEM RESULTS")
    print("="*80)
    
    # Export trade log
    trade_log_path = executor.export_trades_to_csv(str(Config.TRADES_CSV))
    
    # Export side statistics
    side_stats_path = executor.export_side_statistics_to_csv(str(Config.RESULTS_DIR / "side_statistics.csv"))
    
    # Show performance summary
    executor.print_performance_summary()
    
    # Current portfolio status
    portfolio = executor.get_current_portfolio_value()
    print(f"\n💰 CURRENT PORTFOLIO STATUS")
    print("="*40)
    print(f"Active Positions: {portfolio['active_positions']}")
    print(f"Unrealized PnL: {portfolio['unrealized_pnl']:.2f} points")
    print(f"Realized PnL: {portfolio['realized_pnl']:.2f} points")
    print(f"Total PnL: {portfolio['total_pnl']:.2f} points")
    
    # System-wide statistics
    print(f"\n🔬 SYSTEM-WIDE STATISTICS")
    print("="*40)
    print(f"Strategy Nodes Loaded: {len(node_engine.nodes)}")
    print(f"Market Data Points: {len(market_data['timestamp'])}")
    print(f"Total Signals Generated: {len(signals)}")
    print(f"Tradeable Signals: {len(tradeable_signals)}")
    print(f"Trades Executed: {executor.execution_stats['trades_executed']}")
    print(f"Trades Completed: {len(executor.closed_trades)}")
    signal_to_trade_conversion = (trades_opened/len(tradeable_signals)*100) if len(tradeable_signals) > 0 else 0
    print(f"Signal-to-Trade Conversion: {signal_to_trade_conversion:.1f}%")
    
    # Export all data for analysis
    detector.export_signals_to_csv(str(Config.SIGNALS_CSV))
    
    return {
        'node_engine': node_engine,
        'detector': detector, 
        'executor': executor,
        'signals': signals,
        'tradeable_signals': tradeable_signals,
        'market_data': market_data
    }


def demonstrate_advanced_features(system_components):
    """Demonstrate advanced features of the complete system."""
    
    print(f"\n🚀 ADVANCED FEATURES DEMONSTRATION")
    print("="*60)
    
    executor = system_components['executor']
    signals = system_components['tradeable_signals']
    
    # 1. Risk Management Analysis
    print(f"\n⚠️  Risk Management Analysis")
    print("-" * 30)
    
    if executor.closed_trades:
        stop_loss_exits = [t for t in executor.closed_trades if 'STOP_LOSS' in t.exit_reason]
        take_profit_exits = [t for t in executor.closed_trades if 'TAKE_PROFIT' in t.exit_reason]
        timeout_exits = [t for t in executor.closed_trades if 'TIMEOUT' in t.exit_reason]
        
        print(f"Stop Loss Exits: {len(stop_loss_exits)}")
        print(f"Take Profit Exits: {len(take_profit_exits)}")
        print(f"Timeout Exits: {len(timeout_exits)}")
        
        if stop_loss_exits:
            avg_sl_loss = np.mean([t.pnl_points for t in stop_loss_exits])
            print(f"Average Stop Loss: {avg_sl_loss:.2f} points")
        
        if take_profit_exits:
            avg_tp_gain = np.mean([t.pnl_points for t in take_profit_exits])
            print(f"Average Take Profit: {avg_tp_gain:.2f} points")
    
    # 2. Signal Quality Analysis
    print(f"\n📊 Signal Quality Analysis")
    print("-" * 30)
    
    confidence_performance = {}
    for trade in executor.closed_trades:
        conf = trade.confidence
        if conf not in confidence_performance:
            confidence_performance[conf] = {'count': 0, 'total_pnl': 0, 'wins': 0}
        
        confidence_performance[conf]['count'] += 1
        confidence_performance[conf]['total_pnl'] += trade.pnl_points
        if trade.pnl_points > 0:
            confidence_performance[conf]['wins'] += 1
    
    for conf, stats in confidence_performance.items():
        win_rate = (stats['wins'] / stats['count']) * 100
        avg_pnl = stats['total_pnl'] / stats['count']
        print(f"{conf}: {stats['count']} trades, {win_rate:.1f}% win rate, {avg_pnl:.2f} avg PnL")
    
    # 3. Timing Analysis
    print(f"\n⏰ Trade Timing Analysis")
    print("-" * 30)
    
    if executor.closed_trades:
        durations = [t.duration_held for t in executor.closed_trades]
        print(f"Average Trade Duration: {np.mean(durations):.1f} ticks")
        print(f"Shortest Trade: {min(durations)} ticks")
        print(f"Longest Trade: {max(durations)} ticks")
        
        # Analyze profitability by duration
        short_trades = [t for t in executor.closed_trades if t.duration_held <= 10]
        long_trades = [t for t in executor.closed_trades if t.duration_held > 10]
        
        if short_trades:
            short_avg_pnl = np.mean([t.pnl_points for t in short_trades])
            print(f"Short trades (≤10 ticks) avg PnL: {short_avg_pnl:.2f}")
        
        if long_trades:
            long_avg_pnl = np.mean([t.pnl_points for t in long_trades])
            print(f"Long trades (>10 ticks) avg PnL: {long_avg_pnl:.2f}")


def show_sample_trades(executor):
    """Show detailed information for sample trades."""
    
    print(f"\n📋 SAMPLE TRADE DETAILS")
    print("="*60)
    
    if not executor.closed_trades:
        print("No completed trades to display")
        return
    
    # Show best and worst trades
    best_trade = max(executor.closed_trades, key=lambda t: t.pnl_points)
    worst_trade = min(executor.closed_trades, key=lambda t: t.pnl_points)
    
    print(f"\n🏆 BEST TRADE:")
    print(f"Strategy: {best_trade.node_name}")
    print(f"Entry: ${best_trade.entry_price:.2f} at {best_trade.timestamp}")
    print(f"Exit: ${best_trade.exit_price:.2f} at {best_trade.exit_timestamp}")
    print(f"PnL: {best_trade.pnl_points:.2f} points ({best_trade.pnl_percent:.1f}%)")
    print(f"Duration: {best_trade.duration_held} ticks")
    print(f"Exit Reason: {best_trade.exit_reason}")
    print(f"Confidence: {best_trade.confidence}")
    
    print(f"\n📉 WORST TRADE:")
    print(f"Strategy: {worst_trade.node_name}")
    print(f"Entry: ${worst_trade.entry_price:.2f} at {worst_trade.timestamp}")
    print(f"Exit: ${worst_trade.exit_price:.2f} at {worst_trade.exit_timestamp}")
    print(f"PnL: {worst_trade.pnl_points:.2f} points ({worst_trade.pnl_percent:.1f}%)")
    print(f"Duration: {worst_trade.duration_held} ticks")
    print(f"Exit Reason: {worst_trade.exit_reason}")
    print(f"Confidence: {worst_trade.confidence}")
    
    # Show first few trades for detailed analysis
    print(f"\n📝 FIRST 3 COMPLETED TRADES:")
    for i, trade in enumerate(executor.closed_trades[:3], 1):
        print(f"\n{i}. {trade.node_name}")
        print(f"   {trade.timestamp} → {trade.exit_timestamp}")
        print(f"   ${trade.entry_price:.2f} → ${trade.exit_price:.2f}")
        print(f"   PnL: {trade.pnl_points:.2f} pts | Reason: {trade.exit_reason}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BlackBox Trading System')
    parser.add_argument('--data', type=str, help='Path to data file (CSV/Parquet)')
    parser.add_argument('--rr', type=float, help='Risk-reward multiple')
    args = parser.parse_args()
    
    # Smart dataset selection
    if args.data:
        # Use specified data file
        data_path = resolve_data_path(args.data)
    else:
        # Use dataset scanner for interactive selection
        print("🔍 No dataset specified. Scanning available datasets...\n")
        scanner = DatasetScanner()
        datasets = scanner.scan_datasets()
        
        if not datasets:
            print("❌ No valid datasets found in data/csv_datasets/")
            print("💡 Please add CSV files to data/csv_datasets/ or use --data flag")
            sys.exit(1)
        
        scanner.display_datasets(datasets)
        selected_dataset = scanner.select_dataset(datasets)
        
        if not selected_dataset:
            print("\n👋 No dataset selected. Exiting.")
            sys.exit(0)
            
        data_path = Path(selected_dataset['path'])
        print(f"\n🎯 Using dataset: {selected_dataset['filename']}")
    
    # Interactive RR multiple selection
    if args.rr:
        # Use specified RR
        rr_multiple = get_rr(args.rr)
        print(f"\n⚖️  Using specified Risk-Reward Multiple: {rr_multiple}")
    else:
        # Use interactive RR selection
        rr_multiple = select_risk_reward_ratio()
    
    # Run the complete system
    system_components = run_complete_blackbox_system(data_path, rr_multiple)
    
    # Demonstrate advanced features
    demonstrate_advanced_features(system_components)
    
    # Show sample trades
    show_sample_trades(system_components['executor'])
    
    print(f"\n✅ BLACKBOX CORE ENGINE - COMPLETE SYSTEM DEMONSTRATION FINISHED")
    print("="*80)
    print("🎯 SYSTEM CAPABILITIES DEMONSTRATED:")
    print("   ✅ Strategy knowledge parsing from .txt files")
    print("   ✅ Real-time signal detection and evaluation")
    print("   ✅ Automated trade execution with risk management")
    print("   ✅ Position management with SL/TP/Duration limits")
    print("   ✅ Performance tracking and analysis")
    print("   ✅ LONG/SHORT position analytics and statistics")
    print("   ✅ CSV export for further analysis")
    print("   ✅ Modular architecture ready for production")
    print(f"\n🚀 Ready for:")
    print("   • Live broker API integration")
    print("   • Real-time data feeds")
    print("   • Advanced portfolio management")
    print("   • Machine learning optimization")
    print("   • Web-based monitoring dashboards")
