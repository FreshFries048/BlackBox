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
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# Import all BlackBox components directly
import sys
import os
# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(__file__))
import blackbox_core  # Import the actual module file
NodeEngine = blackbox_core.NodeEngine  # Get the class from the module

from node_detector import NodeDetectorEngine, SignalEvent, MissingFeatureError
from trade_executor import TradeExecutorEngine, RiskManager, Position
from blackbox_config.config_loader import resolve_data_path, get_rr
from dataset_scanner import DatasetScanner

# Only import performance optimizations if fast-mode is requested
try:
    from performance_optimizer import (
        FastDataSampler, SignalOptimizer, BacktestAccelerator,
        apply_performance_optimizations, create_progress_tracker
    )
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False


def setup_environment():
    """Setup environment variables and directories."""
    # Create required directories
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("storage").mkdir(exist_ok=True)
    
    # Set default environment variables if not set
    if not os.getenv('BLACKBOX_COMMISSION'):
        os.environ['BLACKBOX_COMMISSION'] = '0.0'
    if not os.getenv('BLACKBOX_SPREAD'):
        os.environ['BLACKBOX_SPREAD'] = '0.0001'
    if not os.getenv('BLACKBOX_RR_MULTIPLE'):
        os.environ['BLACKBOX_RR_MULTIPLE'] = '2.0'


def load_market_data(data_file: str, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> Dict[str, Any]:
    """Load and prepare market data from CSV file."""
    # Setup logger locally
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading market data from {data_file}")
    
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load CSV data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} data points")
    
    # Validate required columns
    required_columns = ['time', 'open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Apply date filters
    if start_date:
        df = df[df['time'] >= start_date]
        logger.info(f"Filtered to {len(df)} points after start_date {start_date}")
    
    if end_date:
        df = df[df['time'] <= end_date]
        logger.info(f"Filtered to {len(df)} points after end_date {end_date}")
    
    # Prepare data dictionary
    market_data = {
        'timestamp': pd.to_datetime(df['time']),
        'price': df['close'],
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'volume': df.get('tick_volume', df.get('volume', 1000)),
        'spread': df.get('spread', 0.0001)
    }
    
    return market_data


def run_full_backtest(args):
    """Run complete backtest with all components and performance optimizations."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    setup_environment()
    
    # Only apply performance optimizations if fast-mode is enabled and available
    if args.fast_mode and PERFORMANCE_OPTIMIZER_AVAILABLE:
        progress = create_progress_tracker()
        optimizations = apply_performance_optimizations(args)
        logger.info("Starting BlackBox Trading System (Performance Optimized)")
        logger.info(f"Data Sampling: {optimizations['data_sample_rate']*100:.1f}%")
    else:
        # Use standard mode with core optimizations only
        optimizations = {'data_sample_rate': 1.0}  # No sampling
        progress = None
        logger.info("Starting BlackBox Trading System (Standard Mode with Core Optimizations)")
    
    logger.info(f"RR Multiple: {args.rr_multiple}")
    logger.info(f"Commission: {args.commission}%")
    logger.info(f"Spread: {args.spread} points")
    
    try:
        # Load market data
        if progress:
            progress.update("Loading market data")
        raw_market_data = load_market_data(
            args.data_file, 
            args.start_date, 
            args.end_date
        )
        
        # Apply performance optimization only in fast mode
        if args.fast_mode and optimizations['data_sample_rate'] < 1.0:
            if progress:
                progress.update("Optimizing data sample")
            market_data = FastDataSampler.sample_market_data(
                raw_market_data,
                sample_rate=optimizations['data_sample_rate'],
                preserve_volatility=optimizations['preserve_volatility_periods']
            )
            logger.info(f"Sampled data: {len(market_data['price'])} points "
                       f"(from {len(raw_market_data['price'])})")
        else:
            market_data = raw_market_data
            logger.info(f"Using full dataset: {len(market_data['price'])} points")
        
        # Load strategy nodes
        if progress:
            progress.update("Loading strategy nodes")
        nodes_dir = Path("blackbox_nodes")
        if not nodes_dir.exists():
            raise FileNotFoundError(f"Nodes directory not found: {nodes_dir}")
        
        node_engine = NodeEngine()
        node_engine.load_nodes_from_folder(str(nodes_dir))
        logger.info(f"Loaded {len(node_engine.nodes)} strategy nodes")
        
        # Initialize production-grade risk manager with trading costs
        risk_manager = RiskManager(
            rr_multiple=args.rr_multiple,
            commission_pct=args.commission,
            spread_points=args.spread
        )
        
        # Run signal detection with core optimizations
        if progress:
            progress.update("Running signal detection")
        detector = NodeDetectorEngine(node_engine, market_data)
        raw_signals = detector.run_detection(live_output=args.live_output)
        
        logger.info(f"Generated {len(raw_signals)} raw signals")
        
        # Apply signal optimizations only in fast mode
        if progress:
            progress.update("Processing signals")
        optimized_signals = raw_signals
        
        if args.fast_mode and PERFORMANCE_OPTIMIZER_AVAILABLE:
            if optimizations.get('high_quality_signals_only'):
                optimized_signals = SignalOptimizer.filter_high_quality_signals(
                    optimized_signals, min_confidence="High"
                )
                logger.info(f"High-quality signals: {len(optimized_signals)}")
            
            if optimizations.get('signal_deduplication_window', 0) > 0:
                optimized_signals = SignalOptimizer.deduplicate_signals(
                    optimized_signals, 
                    time_window_minutes=optimizations['signal_deduplication_window']
                )
                logger.info(f"Deduplicated signals: {len(optimized_signals)}")
            
            # Limit signals per node for performance
            if optimizations.get('max_signals_per_node'):
                node_signal_counts = {}
                final_signals = []
                for signal in optimized_signals:
                    node_name = signal.node_name
                    if node_name not in node_signal_counts:
                        node_signal_counts[node_name] = 0
                    
                    if node_signal_counts[node_name] < optimizations['max_signals_per_node']:
                        final_signals.append(signal)
                        node_signal_counts[node_name] += 1
                
                optimized_signals = final_signals
                logger.info(f"Limited signals: {len(optimized_signals)} (max {optimizations['max_signals_per_node']} per node)")
        
        # Filter tradeable signals (Medium and above confidence)
        tradeable_signals = []
        for signal in optimized_signals:
            if any(conf in signal.confidence for conf in ["Medium", "High", "Highest"]):
                tradeable_signals.append(signal)
        
        logger.info(f"Final tradeable signals: {len(tradeable_signals)}")
        
        # Run backtest
        if progress:
            progress.update("Running backtest")
        executor = TradeExecutorEngine(market_data, risk_manager)
        executor.run_backtest(tradeable_signals)
        
        # Prepare results
        if progress:
            progress.update("Preparing results")
        trades_data = []
        for trade in executor.closed_trades:
            trades_data.append({
                'node_name': trade.node_name,
                'timestamp': trade.timestamp,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl_points': trade.pnl_points,
                'pnl_percent': trade.pnl_percent,
                'exit_reason': trade.exit_reason,
                'duration_held': trade.duration_held,
                'confidence': trade.confidence
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        signals_data = []
        for signal in optimized_signals:
            signals_data.append({
                'node_name': signal.node_name,
                'timestamp': signal.timestamp,
                'confidence': signal.confidence,
                'signal_strength': getattr(signal, 'signal_strength', 1.0),
                'market_condition': getattr(signal, 'market_condition', 'normal')
            })
        
        signals_df = pd.DataFrame(signals_data)
        
        # Save results
        if progress:
            progress.update("Saving results")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"optimized_{timestamp}" if args.fast_mode else f"standard_{timestamp}"
        run_dir = results_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save as CSV
        trades_df.to_csv(run_dir / "trades.csv", index=False)
        signals_df.to_csv(run_dir / "signals.csv", index=False)
        
        # Save metadata
        metadata = {
            'rr_multiple': args.rr_multiple,
            'commission_perc': args.commission,
            'spread_points': args.spread,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'total_signals': len(optimized_signals),
            'tradeable_signals': len(tradeable_signals),
            'data_file': args.data_file,
            'mode': 'fast' if args.fast_mode else 'standard',
            'optimizations_applied': optimizations if args.fast_mode else None
        }
        
        import json
        with open(run_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Print summary
        total_trades = len(trades_df)
        
        # Complete progress tracking
        if progress:
            progress.complete(len(optimized_signals), total_trades)
        
        if total_trades > 0:
            winning_trades = len(trades_df[trades_df['pnl_points'] > 0])
            win_rate = (winning_trades / total_trades) * 100
            total_pnl = trades_df['pnl_points'].sum()
            
            logger.info("\n" + "="*50)
            logger.info(f"BLACKBOX BACKTEST RESULTS ({'FAST MODE' if args.fast_mode else 'STANDARD MODE'})")
            logger.info("="*50)
            logger.info(f"Run ID: {run_id}")
            if args.fast_mode:
                logger.info(f"Data Sample Rate: {optimizations.get('data_sample_rate', 1.0)*100:.1f}%")
            logger.info(f"Total Signals: {len(optimized_signals)}")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Winning Trades: {winning_trades}")
            logger.info(f"Win Rate: {win_rate:.2f}%")
            logger.info(f"Total PnL: {total_pnl:.2f} points")
            logger.info(f"Results saved to: results/{run_id}/")
            logger.info("="*50)
        else:
            logger.warning("No trades generated in this run")
        
        return run_id
        
    except MissingFeatureError as e:
        logger.error(f"Missing feature validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
        sys.exit(1)


def main():
    """Main entry point with enhanced CLI."""
    parser = argparse.ArgumentParser(
        description='BlackBox Trading System (Production Grade)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data-file', 
        type=str, 
        default='data/csv_datasets/EURUSD_1H_2020-2024.csv',
        help='Path to market data CSV file'
    )
    parser.add_argument(
        '--start-date', 
        type=str, 
        help='Start date for backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', 
        type=str, 
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    # Trading parameters
    parser.add_argument(
        '--rr-multiple', 
        type=float, 
        default=float(os.getenv('BLACKBOX_RR_MULTIPLE', '2.0')),
        help='Risk-reward multiple'
    )
    parser.add_argument(
        '--commission', 
        type=float, 
        default=float(os.getenv('BLACKBOX_COMMISSION', '0.0')),
        help='Commission percentage (e.g., 0.1 for 0.1%%)'
    )
    parser.add_argument(
        '--spread', 
        type=float, 
        default=float(os.getenv('BLACKBOX_SPREAD', '0.0001')),
        help='Spread in points'
    )
    
    # Output options
    parser.add_argument(
        '--live-output', 
        action='store_true',
        help='Show live signal detection output'
    )
    parser.add_argument(
        '--fast-mode', 
        action='store_true',
        help='Enable performance optimizations for faster execution'
    )
    parser.add_argument(
        '--api-mode', 
        action='store_true',
        help='Start API server instead of running backtest'
    )
    
    args = parser.parse_args()
    
    if args.api_mode:
        # Start API server (if available)
        try:
            logger.info("Starting BlackBox API server...")
            from api.main import app
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except ImportError:
            logger.error("API mode requires additional dependencies. Running backtest instead.")
            run_id = run_full_backtest(args)
            print(f"\nBacktest completed. Run ID: {run_id}")
    else:
        # Run backtest
        run_id = run_full_backtest(args)
        mode_msg = "Fast-Mode" if args.fast_mode else "Standard"
        print(f"\n{mode_msg} Backtest completed. Run ID: {run_id}")
        print(f"Results saved in: results/{run_id}/")


if __name__ == "__main__":
    main()


