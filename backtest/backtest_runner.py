#!/usr/bin/env python3
"""
BlackBox Backtest Runner - Walk-Forward Testing Framework

Provides comprehensive backtesting capabilities with walk-forward validation,
performance metrics, and equity curve generation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from blackbox_core import NodeEngine
from node_detector import NodeDetectorEngine
from trade_executor import TradeExecutorEngine, RiskManager
from data_feed import DataFeedProcessor


class BacktestRunner:
    """
    Walk-forward backtesting engine for BlackBox trading strategies.
    """
    
    def __init__(self, data_path: str, train_months: int = 3, test_months: int = 1, rr_multiple: float = 3.0):
        """
        Initialize backtest runner.
        
        Args:
            data_path: Path to market data file (.csv or .parquet)
            train_months: Training period in months
            test_months: Testing period in months
            rr_multiple: Risk-reward multiple for position sizing
        """
        self.data_path = Path(data_path)
        self.train_months = train_months
        self.test_months = test_months
        self.rr_multiple = rr_multiple
        
        # Initialize components
        self.node_engine = NodeEngine()
        self.data_processor = DataFeedProcessor()
        
        # Results storage
        self.walk_forward_results = []
        self.equity_curve = []
        self.all_trades = []
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare market data."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"Loading data from {self.data_path}")
        
        if self.data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix.lower() == '.csv':
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Ensure timestamp column
        if 'timestamp' not in df.columns and 'time' in df.columns:
            df['timestamp'] = df['time']
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Loaded {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def create_walk_forward_periods(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward train/test periods.
        
        Args:
            df: Full dataset
            
        Returns:
            List of (train_df, test_df) tuples
        """
        periods = []
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        current_date = start_date
        
        while current_date < end_date:
            # Define train period
            train_end = current_date + timedelta(days=30 * self.train_months)
            
            # Define test period
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.test_months)
            
            # Break if not enough data for test period
            if test_end > end_date:
                break
            
            # Extract train and test data
            train_df = df[(df['timestamp'] >= current_date) & (df['timestamp'] < train_end)].copy()
            test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)].copy()
            
            if len(train_df) > 0 and len(test_df) > 0:
                periods.append((train_df, test_df))
                print(f"Period {len(periods)}: Train {current_date.date()} - {train_end.date()}, "
                      f"Test {test_start.date()} - {test_end.date()}")
            
            # Move to next period
            current_date = test_start
        
        return periods
    
    def run_single_period(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                         period_num: int) -> Dict:
        """
        Run backtest for a single walk-forward period.
        
        Args:
            train_df: Training data (currently not used for optimization)
            test_df: Testing data for evaluation
            period_num: Period number for tracking
            
        Returns:
            Dictionary with period results
        """
        print(f"\nRunning period {period_num}...")
        
        # Prepare test data with enhanced features
        try:
            test_data_enhanced = self.data_processor.add_derived_features(test_df)
        except Exception as e:
            print(f"Warning: Could not add all enhanced features: {e}")
            test_data_enhanced = test_df.copy()
        
        # Convert to dictionary format for detector
        data_dict = {
            'timestamp': test_data_enhanced['timestamp'],
            'price': test_data_enhanced['close'],
            'open': test_data_enhanced['open'],
            'high': test_data_enhanced['high'], 
            'low': test_data_enhanced['low'],
            'volume': test_data_enhanced.get('volume', test_data_enhanced.get('tick_volume', 0)),
            'spread': test_data_enhanced.get('spread', 0.0001),
            # Enhanced features (will be 0 if not available)
            'gamma_exposure': test_data_enhanced.get('gamma_exposure', 0),
            'gamma_pin_distance': test_data_enhanced.get('gamma_pin_distance', 0.5),
            'oi': test_data_enhanced.get('oi', 0),
            'dark_prints': test_data_enhanced.get('dark_prints', 0),
            'hidden_volume': test_data_enhanced.get('hidden_volume', 0),
            'block_size': test_data_enhanced.get('block_size', 0),
            'atr_14': test_data_enhanced.get('atr_14', test_data_enhanced['close'].rolling(14).std() * 0.01)
        }
        
        # Initialize detector
        detector = NodeDetectorEngine(self.node_engine, data_dict)
        
        # Generate signals
        signals = detector.run_detection(live_output=False)
        high_conf_signals = [s for s in signals if 'High' in s.confidence]
        
        # Initialize executor with enhanced risk management
        risk_manager = RiskManager(risk_pct=0.02, account_equity=100000.0, rr_multiple=self.rr_multiple)
        executor = TradeExecutorEngine(data_dict, risk_manager)
        
        # Run backtest
        trades_opened = executor.run_backtest(high_conf_signals)
        
        # Print side performance for this period
        if executor.closed_trades:
            side_performance = executor.calculate_side_performance()
            if side_performance:
                long_trades = side_performance['overall_distribution']['long_trades']
                short_trades = side_performance['overall_distribution']['short_trades']
                long_win_rate = side_performance['long_performance']['win_rate']
                short_win_rate = side_performance['short_performance']['win_rate']
                print(f"   Period {period_num} Side Stats: LONG: {long_trades} trades ({long_win_rate:.1f}% win rate), SHORT: {short_trades} trades ({short_win_rate:.1f}% win rate)")
        
        # Calculate period performance
        period_stats = self.calculate_period_stats(executor, test_data_enhanced, period_num)
        
        # Store trades for overall analysis
        for trade in executor.closed_trades:
            trade_dict = {
                'period': period_num,
                'node_name': trade.node_name,
                'side': trade.side,  # Add side information
                'entry_time': trade.timestamp,
                'exit_time': trade.exit_timestamp,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl_points': trade.pnl_points,
                'pnl_percent': trade.pnl_percent,
                'duration': trade.duration_held,
                'exit_reason': trade.exit_reason,
                'confidence': trade.confidence
            }
            self.all_trades.append(trade_dict)
        
        return period_stats
    
    def calculate_period_stats(self, executor, test_df: pd.DataFrame, period_num: int) -> Dict:
        """Calculate performance statistics for a single period."""
        portfolio = executor.get_current_portfolio_value()
        
        # Basic stats
        total_trades = len(executor.closed_trades)
        winning_trades = len([t for t in executor.closed_trades if t.pnl_points > 0])
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        
        # PnL stats
        total_pnl = portfolio['total_pnl']
        avg_win = np.mean([t.pnl_points for t in executor.closed_trades if t.pnl_points > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl_points for t in executor.closed_trades if t.pnl_points < 0]) if (total_trades - winning_trades) > 0 else 0
        
        # Time period
        start_time = test_df['timestamp'].min()
        end_time = test_df['timestamp'].max()
        
        period_stats = {
            'period': period_num,
            'start_date': start_time.isoformat(),
            'end_date': end_time.isoformat(),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'data_points': len(test_df)
        }
        
        # Add to equity curve
        self.equity_curve.append({
            'period': period_num,
            'date': end_time.isoformat(),
            'cumulative_pnl': sum([r.get('total_pnl', 0) for r in self.walk_forward_results]) + total_pnl,
            'period_pnl': total_pnl,
            'period_trades': total_trades
        })
        
        return period_stats
    
    def run_walk_forward_backtest(self, strategies_folder: str) -> Dict:
        """
        Run complete walk-forward backtest.
        
        Args:
            strategies_folder: Path to folder containing strategy nodes
            
        Returns:
            Dictionary with overall results
        """
        print("="*80)
        print("BLACKBOX WALK-FORWARD BACKTEST")
        print("="*80)
        
        # Load strategies
        strategies_loaded = self.node_engine.load_nodes_from_folder(strategies_folder)
        print(f"Loaded {strategies_loaded} strategy nodes")
        
        # Load and prepare data
        df = self.load_data()
        
        # Create walk-forward periods
        periods = self.create_walk_forward_periods(df)
        print(f"Created {len(periods)} walk-forward periods")
        
        # Run backtest for each period
        for i, (train_df, test_df) in enumerate(periods, 1):
            period_result = self.run_single_period(train_df, test_df, i)
            self.walk_forward_results.append(period_result)
            
            print(f"Period {i} complete: {period_result['total_trades']} trades, "
                  f"{period_result['win_rate']:.1%} win rate, "
                  f"{period_result['total_pnl']:.2f} PnL")
        
        # Calculate overall statistics
        overall_stats = self.calculate_overall_stats()
        
        return overall_stats
    
    def calculate_overall_stats(self) -> Dict:
        """Calculate overall backtest statistics."""
        if not self.walk_forward_results:
            return {}
        
        # Aggregate statistics
        total_trades = sum([r['total_trades'] for r in self.walk_forward_results])
        total_winning = sum([r['winning_trades'] for r in self.walk_forward_results])
        total_pnl = sum([r['total_pnl'] for r in self.walk_forward_results])
        
        overall_win_rate = (total_winning / total_trades) if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        period_returns = [r['total_pnl'] for r in self.walk_forward_results]
        if len(period_returns) > 1:
            sharpe_ratio = np.mean(period_returns) / np.std(period_returns) if np.std(period_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        cumulative_pnl = [r['cumulative_pnl'] for r in self.equity_curve]
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = np.array(cumulative_pnl) - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        overall_stats = {
            'total_periods': len(self.walk_forward_results),
            'total_trades': total_trades,
            'overall_win_rate': overall_win_rate,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_periods': len([r for r in self.walk_forward_results if r['total_pnl'] > 0]),
            'avg_trades_per_period': total_trades / len(self.walk_forward_results) if self.walk_forward_results else 0
        }
        
        return overall_stats
    
    def _calculate_side_summary(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate side performance summary from trades data."""
        summary_data = []
        
        for side in ['LONG', 'SHORT']:
            side_trades = trades_df[trades_df['side'] == side]
            if len(side_trades) > 0:
                winning_trades = len(side_trades[side_trades['pnl_points'] > 0])
                win_rate = (winning_trades / len(side_trades)) * 100
                total_pnl = side_trades['pnl_points'].sum()
                avg_pnl = side_trades['pnl_points'].mean()
                
                summary_data.append({
                    'side': side,
                    'total_trades': len(side_trades),
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'percentage_of_total': (len(side_trades) / len(trades_df)) * 100
                })
        
        return pd.DataFrame(summary_data)
    
    def export_results(self, output_dir: str = "backtest_results"):
        """Export backtest results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(output_path / "equity_curve.csv", index=False)
        
        # Export all trades
        trades_df = pd.DataFrame(self.all_trades)
        if not trades_df.empty:
            trades_df.to_csv(output_path / "all_trades.csv", index=False)
            
            # Also export side statistics summary
            if 'side' in trades_df.columns:
                side_summary = self._calculate_side_summary(trades_df)
                side_summary.to_csv(output_path / "side_performance_summary.csv", index=False)
        
        # Export period results
        periods_df = pd.DataFrame(self.walk_forward_results)
        periods_df.to_csv(output_path / "period_results.csv", index=False)
        
        # Export summary JSON
        overall_stats = self.calculate_overall_stats()
        with open(output_path / "backtest_summary.json", 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        print(f"\nResults exported to {output_path}/")
        print(f"  - equity_curve.csv: Equity curve data")
        print(f"  - all_trades.csv: Individual trade details")
        print(f"  - period_results.csv: Period-by-period results") 
        print(f"  - backtest_summary.json: Overall statistics")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="BlackBox Walk-Forward Backtest Runner")
    parser.add_argument("--data", required=True, help="Path to market data file (.csv or .parquet)")
    parser.add_argument("--strategies", default="blackbox_nodes", 
                       help="Path to strategies folder")
    parser.add_argument("--train-months", type=int, default=3, 
                       help="Training period in months")
    parser.add_argument("--test-months", type=int, default=1,
                       help="Testing period in months") 
    parser.add_argument("--rr", type=float, default=3.0,
                       help="Risk-reward multiple (default: 3.0)")
    parser.add_argument("--output", default="backtest_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize and run backtest
    runner = BacktestRunner(args.data, args.train_months, args.test_months, args.rr)
    
    try:
        overall_stats = runner.run_walk_forward_backtest(args.strategies)
        
        # Print summary
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        print(f"Total Periods: {overall_stats['total_periods']}")
        print(f"Total Trades: {overall_stats['total_trades']}")
        print(f"Overall Win Rate: {overall_stats['overall_win_rate']:.1%}")
        print(f"Total PnL: {overall_stats['total_pnl']:.2f}")
        print(f"Sharpe Ratio: {overall_stats['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {overall_stats['max_drawdown']:.2f}")
        print(f"Profitable Periods: {overall_stats['profit_periods']}/{overall_stats['total_periods']}")
        
        # Export results
        runner.export_results(args.output)
        
    except Exception as e:
        print(f"Error during backtesting: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
