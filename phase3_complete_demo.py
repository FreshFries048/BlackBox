"""
BlackBox Phase 3 Demo - Complete Trading System Integration

This script demonstrates the full BlackBox Core Engine pipeline:
Phase 1: Strategy Node Parsing
Phase 2: Signal Detection  
Phase 3: Trade Execution

Shows end-to-end functionality from strategy files to completed trades.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import all BlackBox components
from blackbox_core import NodeEngine
from node_detector import NodeDetectorEngine, SignalEvent
from trade_executor import TradeExecutorEngine, RiskManager, Position


def create_realistic_market_data(num_points: int = 100):
    """Create realistic market data for comprehensive testing."""
    
    print("📊 Creating realistic market data...")
    
    # Create extended timeframe for proper trade management
    timestamps = pd.date_range('2024-08-02 09:30:00', periods=num_points, freq='30s')
    base_price = 4100.0
    
    # Create more realistic price action with trends and volatility
    np.random.seed(42)
    
    # Generate price movements with some trending behavior
    returns = np.random.randn(num_points) * 0.2
    
    # Add some trending periods
    trend_periods = [
        (10, 20, 0.5),   # Uptrend
        (30, 40, -0.3),  # Downtrend  
        (60, 80, 0.4),   # Another uptrend
    ]
    
    for start, end, trend_strength in trend_periods:
        if end <= num_points:
            for i in range(start, end):
                returns[i] += trend_strength * (i - start) / (end - start)
    
    # Create cumulative price series
    prices = base_price + np.cumsum(returns)
    
    # Create volume data with spikes
    volumes = np.random.randint(3000, 8000, num_points)
    volume_spikes = np.random.choice(range(num_points), size=15, replace=False)
    volumes[volume_spikes] = (volumes[volume_spikes] * np.random.uniform(2, 4, len(volume_spikes))).astype(int)
    
    data = {
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
        'gamma': np.random.randint(50, 800, num_points),
        'dark_prints': np.random.randint(0, 5, num_points),
        'footprint_clusters': np.random.randint(0, 3, num_points),
        'block_size': np.zeros(num_points),
    }
    
    # Add strategic market events for signal generation
    strategic_events = [
        (8, 'gamma_spike', {'gamma': 2500, 'volume': 15000}),
        (15, 'block_trade', {'block_size': 12000, 'dark_prints': 8}),
        (25, 'fusion_confluence', {'gamma': 1800, 'block_size': 8000, 'footprint_clusters': 5}),
        (35, 'dark_pool_activity', {'dark_prints': 12, 'volume': 18000}),
        (45, 'gamma_pin_setup', {'gamma': 3200, 'volume': 20000}),
        (55, 'block_footprint', {'block_size': 15000, 'footprint_clusters': 4}),
        (70, 'fusion_time_gamma', {'gamma': 2800, 'block_size': 6000, 'footprint_clusters': 3}),
        (85, 'final_spike', {'gamma': 4000, 'dark_prints': 15, 'volume': 25000}),
    ]
    
    for idx, event_type, values in strategic_events:
        if idx < num_points:
            for key, value in values.items():
                data[key][idx] = value
            print(f"   📍 {event_type} at {timestamps[idx].strftime('%H:%M:%S')}")
    
    return data


def run_complete_blackbox_system():
    """Run the complete BlackBox trading system end-to-end."""
    
    print("🚀 BLACKBOX COMPLETE SYSTEM DEMONSTRATION")
    print("="*80)
    
    # PHASE 1: Load Strategy Nodes
    print("\n🔧 PHASE 1: Loading Strategy Knowledge Base")
    print("-" * 50)
    
    node_engine = NodeEngine()
    loaded_count = node_engine.load_nodes_from_folder("/workspaces/BlackBox/blackbox_nodes")
    print(f"✅ Loaded {loaded_count} strategy nodes successfully")
    
    # Show strategy summary
    print("\nStrategy Portfolio:")
    for i, node in enumerate(node_engine.nodes, 1):
        print(f"  {i}. {node.name} ({node.metadata.get('confidence', 'Unknown')} confidence)")
    
    # PHASE 2: Generate Trading Signals
    print(f"\n🔍 PHASE 2: Real-Time Signal Detection")
    print("-" * 50)
    
    # Create market data
    market_data = create_realistic_market_data(100)
    
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
    
    # Initialize trade executor with custom risk settings
    risk_manager = RiskManager(
        default_stop_loss_pct=1.2,    # Tighter stops
        default_take_profit_pct=3.0   # Higher targets
    )
    
    executor = TradeExecutorEngine(market_data, risk_manager)
    
    # Execute signals
    print("\nExecuting tradeable signals...")
    trades_opened = executor.execute_signals(tradeable_signals, start_index=0)
    
    # Update positions through market data
    print(f"\nManaging positions through market movements...")
    update_stats = executor.update_positions()
    
    # RESULTS & ANALYSIS
    print(f"\n📈 COMPLETE SYSTEM RESULTS")
    print("="*80)
    
    # Export trade log
    trade_log_path = executor.export_trades_to_csv("complete_system_trades.csv")
    
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
    print(f"Signal-to-Trade Conversion: {(trades_opened/len(tradeable_signals)*100):.1f}%")
    
    # Export all data for analysis
    detector.export_signals_to_csv("complete_system_signals.csv")
    
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
    # Run the complete system
    system_components = run_complete_blackbox_system()
    
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
    print("   ✅ CSV export for further analysis")
    print("   ✅ Modular architecture ready for production")
    print(f"\n🚀 Ready for:")
    print("   • Live broker API integration")
    print("   • Real-time data feeds")
    print("   • Advanced portfolio management")
    print("   • Machine learning optimization")
    print("   • Web-based monitoring dashboards")
