"""
BlackBox Core Engine - Phase 3: Trade Execution Layer

This module implements the trade execution engine that converts SignalEvents
into actual trading positions with stop-loss, take-profit, and risk management.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import csv
import os
from enum import Enum

# Import Phase 2 components
from node_detector import SignalEvent


class PositionStatus(Enum):
    """Enumeration for position status."""
    OPEN = "OPEN"
    CLOSED_TP = "CLOSED_TP"    # Take Profit
    CLOSED_SL = "CLOSED_SL"    # Stop Loss
    CLOSED_TIMEOUT = "CLOSED_TIMEOUT"
    CLOSED_MANUAL = "CLOSED_MANUAL"


@dataclass
class Position:
    """
    Represents a trading position opened from a SignalEvent.
    
    Attributes:
        node_name: Name of the strategy node that generated the signal
        entry_price: Price at which position was opened
        timestamp: When position was opened
        stop_loss: Stop loss price level
        take_profit: Take profit price level
        confidence: Confidence level from the original signal
        position_size: Size of the position (default 1.0 for simplicity)
        duration_limit: Maximum number of ticks to hold position
        signal_metadata: Original signal information for tracking
    """
    node_name: str
    entry_price: float
    timestamp: str
    stop_loss: float
    take_profit: float
    confidence: str
    position_size: float = 1.0
    duration_limit: Optional[int] = 20
    signal_metadata: Dict[str, Any] = None
    
    # Position tracking fields
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    exit_reason: Optional[str] = None
    pnl_points: float = 0.0
    pnl_percent: float = 0.0
    duration_held: int = 0
    current_price: float = 0.0
    entry_index: int = 0
    exit_index: Optional[int] = None
    
    def __post_init__(self):
        """Initialize computed fields after creation."""
        if self.signal_metadata is None:
            self.signal_metadata = {}
        self.current_price = self.entry_price
    
    def update_current_price(self, new_price: float, current_index: int, timestamp: str):
        """
        Update the current price and check for exit conditions.
        
        Args:
            new_price: Current market price
            current_index: Current data index
            timestamp: Current timestamp
            
        Returns:
            True if position should be closed, False otherwise
        """
        self.current_price = new_price
        self.duration_held = current_index - self.entry_index
        
        # Check stop loss
        if new_price <= self.stop_loss:
            self._close_position(new_price, timestamp, "STOP_LOSS", current_index)
            return True
        
        # Check take profit
        if new_price >= self.take_profit:
            self._close_position(new_price, timestamp, "TAKE_PROFIT", current_index)
            return True
        
        # Check duration limit
        if self.duration_limit and self.duration_held >= self.duration_limit:
            self._close_position(new_price, timestamp, "TIMEOUT", current_index)
            return True
        
        return False
    
    def _close_position(self, exit_price: float, exit_timestamp: str, reason: str, exit_index: int):
        """Close the position and calculate PnL."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.exit_reason = reason
        self.exit_index = exit_index
        
        # Calculate PnL
        self.pnl_points = (exit_price - self.entry_price) * self.position_size
        self.pnl_percent = (self.pnl_points / self.entry_price) * 100
        
        # Set status
        if reason == "STOP_LOSS":
            self.status = PositionStatus.CLOSED_SL
        elif reason == "TAKE_PROFIT":
            self.status = PositionStatus.CLOSED_TP
        elif reason == "TIMEOUT":
            self.status = PositionStatus.CLOSED_TIMEOUT
        else:
            self.status = PositionStatus.CLOSED_MANUAL
    
    def force_close(self, exit_price: float, exit_timestamp: str, exit_index: int):
        """Force close position (e.g., end of data)."""
        if self.status == PositionStatus.OPEN:
            self._close_position(exit_price, exit_timestamp, "FORCE_CLOSE", exit_index)
    
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.status == PositionStatus.OPEN
    
    def get_unrealized_pnl(self) -> float:
        """Get current unrealized PnL in points."""
        if self.is_open():
            return (self.current_price - self.entry_price) * self.position_size
        return self.pnl_points
    
    def get_unrealized_pnl_percent(self) -> float:
        """Get current unrealized PnL as percentage."""
        if self.is_open():
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        return self.pnl_percent


class RiskManager:
    """
    Enhanced risk management with ATR-based stops and position sizing.
    """
    
    def __init__(self, default_stop_loss_pct: float = 1.5, default_take_profit_pct: float = 2.5,
                 risk_pct: float = 0.02, account_equity: float = 100000.0):
        """
        Initialize risk manager.
        
        Args:
            default_stop_loss_pct: Default stop loss percentage (fallback)
            default_take_profit_pct: Default take profit percentage (fallback)
            risk_pct: Risk percentage per trade (default 2%)
            account_equity: Account equity for position sizing
        """
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.risk_pct = risk_pct
        self.account_equity = account_equity
        self.risk_multiple = 3.0  # Default risk:reward ratio
    
    def calculate_stop_loss(self, entry_price: float, confidence: str, 
                           atr: Optional[float] = None) -> float:
        """
        Calculate stop loss level using ATR or fallback to pip-based.
        
        Args:
            entry_price: Entry price for the position
            confidence: Confidence level of the signal
            atr: Average True Range value (if available)
            
        Returns:
            Stop loss price level
        """
        if atr is not None and atr > 0:
            # ATR-based stop loss (2 * ATR)
            stop_distance = 2.0 * atr
            return entry_price - stop_distance
        else:
            # Fallback to pip-based calculation for EUR/USD
            if "Highest" in confidence:
                sl_pips = 30  # 30 pips for highest confidence
            elif "High" in confidence:
                sl_pips = 50  # 50 pips for high confidence
            elif "Medium" in confidence:
                sl_pips = 80  # 80 pips for medium confidence
            else:
                sl_pips = 120  # 120 pips for low confidence
            
            return entry_price - (sl_pips * 0.0001)  # Assuming long positions
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                             confidence: str, signal_tags: List[str]) -> float:
        """
        Calculate take profit using risk multiple or confidence-based targets.
        
        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss level
            confidence: Confidence level of the signal
            signal_tags: Tags from the original signal
            
        Returns:
            Take profit price level
        """
        # Calculate stop distance for risk multiple calculation
        stop_distance = entry_price - stop_loss
        
        # Base risk multiple
        risk_multiple = self.risk_multiple
        
        # Adjust risk multiple based on confidence
        if "Highest" in confidence:
            risk_multiple = 4.0  # Higher targets for highest confidence
        elif "High" in confidence:
            risk_multiple = 3.5
        elif "Medium" in confidence:
            risk_multiple = 2.5
        else:
            risk_multiple = 2.0
        
        # Adjust based on signal type
        if any(tag.lower() in ['fusion', 'confluence'] for tag in signal_tags):
            risk_multiple *= 1.2  # Higher targets for confluence signals
        elif any(tag.lower() in ['gamma', 'pin'] for tag in signal_tags):
            risk_multiple *= 1.1  # Slightly higher for gamma signals
        
        return entry_price + (stop_distance * risk_multiple)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               confidence: str, signal_matches: int) -> float:
        """
        Calculate position size based on risk percentage and stop distance.
        
        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss level
            confidence: Confidence level of the signal
            signal_matches: Number of signal matches
            
        Returns:
            Position size
        """
        # Calculate stop distance in price units
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 1.0  # Default size if no stop distance
        
        # Calculate position size based on risk percentage
        risk_amount = self.account_equity * self.risk_pct
        position_size = risk_amount / stop_distance
        
        # Apply confidence multiplier
        confidence_multiplier = 1.0
        if "Highest" in confidence:
            confidence_multiplier = 1.5
        elif "High" in confidence:
            confidence_multiplier = 1.3
        elif "Medium" in confidence:
            confidence_multiplier = 1.1
        
        # Apply signal quality multiplier
        signal_multiplier = min(2.0, 1.0 + (signal_matches * 0.1))
        
        final_size = position_size * confidence_multiplier * signal_multiplier
        
        # Cap position size to reasonable limits
        return min(final_size, 10.0)  # Max 10x normal size


class TradeExecutorEngine:
    """
    Trade execution engine that converts SignalEvents into trading positions
    and manages them through their lifecycle with risk management.
    """
    
    def __init__(self, market_data: Union[pd.DataFrame, Dict], risk_manager: RiskManager = None):
        """
        Initialize the trade executor engine.
        
        Args:
            market_data: Market data for position management
            risk_manager: Risk management configuration
        """
        self.market_data = self._prepare_market_data(market_data)
        self.risk_manager = risk_manager or RiskManager()
        
        # Position tracking
        self.positions: List[Position] = []
        self.closed_trades: List[Position] = []
        self.active_positions: List[Position] = []
        
        # Performance tracking
        self.execution_stats = {
            'signals_received': 0,
            'trades_executed': 0,
            'trades_skipped': 0,
            'positions_closed': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
        
        # Current market state
        self.current_index = 0
        self.current_price = 0.0
        self.current_timestamp = ""
    
    def _prepare_market_data(self, market_data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Prepare and validate market data."""
        if isinstance(market_data, dict):
            df = pd.DataFrame(market_data)
        else:
            df = market_data.copy()
        
        # Ensure required columns
        if 'timestamp' not in df.columns or 'price' not in df.columns:
            raise ValueError("Market data must contain 'timestamp' and 'price' columns")
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def run_backtest(self, signals: List[SignalEvent]) -> int:
        """
        Run a proper chronological backtest with signals and position management.
        
        Args:
            signals: List of SignalEvent objects to process chronologically
            
        Returns:
            Number of trades executed
        """
        print(f"\n💼 RUNNING CHRONOLOGICAL BACKTEST")
        print("="*50)
        
        # Sort signals by timestamp for chronological processing
        sorted_signals = sorted(signals, key=lambda s: pd.to_datetime(s.timestamp))
        trades_executed = 0
        
        # Create a mapping of signal timestamps to market data indices
        signal_indices = {}
        for signal in sorted_signals:
            signal_time = pd.to_datetime(signal.timestamp)
            # Find the closest market data index for this signal
            closest_idx = 0
            min_diff = float('inf')
            for idx, row in self.market_data.iterrows():
                time_diff = abs((pd.to_datetime(row['timestamp']) - signal_time).total_seconds())
                if time_diff < min_diff:
                    min_diff = time_diff
                    closest_idx = idx
            signal_indices[signal.timestamp] = closest_idx
        
        # Process each market data point chronologically
        for current_idx in range(len(self.market_data)):
            current_row = self.market_data.iloc[current_idx]
            current_price = current_row['price']
            current_time = current_row['timestamp']
            current_timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Execute any signals that occur at this time point
            signals_at_this_time = [s for s in sorted_signals if signal_indices.get(s.timestamp) == current_idx]
            
            for signal in signals_at_this_time:
                if self._should_execute_signal(signal):
                    position = self._create_position_from_signal(signal, current_idx)
                    if position:
                        self.positions.append(position)
                        self.active_positions.append(position)
                        trades_executed += 1
                        self.execution_stats['trades_executed'] += 1
                        
                        print(f"✅ OPENED: {position.node_name} at index {current_idx}")
                        print(f"   Entry: ${position.entry_price:.4f} | SL: ${position.stop_loss:.4f} | TP: ${position.take_profit:.4f}")
                        print(f"   Time: {current_timestamp}")
                        print("-" * 50)
                    else:
                        self.execution_stats['trades_skipped'] += 1
                else:
                    self.execution_stats['trades_skipped'] += 1
            
            # Update all active positions with current price
            positions_to_close = []
            
            for position in self.active_positions:
                # Only update positions that were opened at or before this index
                if position.entry_index <= current_idx:
                    should_close = position.update_current_price(
                        current_price, current_idx, current_timestamp
                    )
                    
                    if should_close:
                        positions_to_close.append(position)
                        print(f"🔴 CLOSED: {position.node_name}")
                        print(f"   Entry: ${position.entry_price:.4f} → Exit: ${position.exit_price:.4f}")
                        print(f"   PnL: {position.pnl_points:.4f} pts ({position.pnl_percent:.1f}%)")
                        print(f"   Reason: {position.exit_reason} | Duration: {position.duration_held} ticks")
                        print("-" * 50)
            
            # Move closed positions
            for position in positions_to_close:
                self.active_positions.remove(position)
                self.closed_trades.append(position)
                self.execution_stats['positions_closed'] += 1
        
        # Force close any remaining positions at end of data
        if self.active_positions:
            final_row = self.market_data.iloc[-1]
            final_price = final_row['price']
            final_timestamp = final_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            remaining_positions = self.active_positions.copy()
            for position in remaining_positions:
                position.force_close(final_price, final_timestamp, len(self.market_data) - 1)
                self.active_positions.remove(position)
                self.closed_trades.append(position)
                print(f"🔒 FORCE CLOSED: {position.node_name} (End of data)")
        
        print(f"\n📊 Backtest Summary: {trades_executed} trades executed, {len(self.closed_trades)} completed")
        return trades_executed

    def execute_signals(self, signals: List[SignalEvent], start_index: int = 0) -> int:
        """
        Execute trades based on incoming signals.
        
        Args:
            signals: List of SignalEvent objects to process
            start_index: Starting index in market data
            
        Returns:
            Number of trades executed
        """
        trades_executed = 0
        
        print(f"\n💼 EXECUTING SIGNALS")
        print("="*50)
        
        for signal in signals:
            self.execution_stats['signals_received'] += 1
            
            # Check if we should trade this signal
            if self._should_execute_signal(signal):
                position = self._create_position_from_signal(signal, start_index)
                if position:
                    self.positions.append(position)
                    self.active_positions.append(position)
                    trades_executed += 1
                    self.execution_stats['trades_executed'] += 1
                    
                    print(f"✅ OPENED: {position.node_name}")
                    print(f"   Entry: ${position.entry_price:.2f} | SL: ${position.stop_loss:.2f} | TP: ${position.take_profit:.2f}")
                    print(f"   Confidence: {position.confidence} | Size: {position.position_size:.1f}")
                    print("-" * 50)
                else:
                    self.execution_stats['trades_skipped'] += 1
            else:
                self.execution_stats['trades_skipped'] += 1
                print(f"⏭️  SKIPPED: {signal.node_name} (Confidence: {signal.confidence})")
        
        print(f"\n📊 Execution Summary: {trades_executed} trades opened, {self.execution_stats['trades_skipped']} skipped")
        return trades_executed
    
    def _should_execute_signal(self, signal: SignalEvent) -> bool:
        """
        Determine if a signal should be executed based on confidence and other criteria.
        
        Args:
            signal: SignalEvent to evaluate
            
        Returns:
            True if signal should be executed
        """
        # Only trade High and Highest confidence signals
        confidence_levels = ["High", "Highest"]
        return any(level in signal.confidence for level in confidence_levels)
    
    def _create_position_from_signal(self, signal: SignalEvent, current_index: int) -> Optional[Position]:
        """
        Create a Position object from a SignalEvent.
        
        Args:
            signal: SignalEvent to convert
            current_index: Current index in market data
            
        Returns:
            Position object or None if creation failed
        """
        try:
            # Calculate risk levels
            stop_loss = self.risk_manager.calculate_stop_loss(signal.entry_price, signal.confidence)
            take_profit = self.risk_manager.calculate_take_profit(
                signal.entry_price, stop_loss, signal.confidence, signal.tags
            )
            position_size = self.risk_manager.calculate_position_size(
                signal.entry_price, stop_loss, signal.confidence, signal.workflow_matches
            )
            
            # Create position
            position = Position(
                node_name=signal.node_name,
                entry_price=signal.entry_price,
                timestamp=signal.timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=signal.confidence,
                position_size=position_size,
                signal_metadata={
                    'trigger_reason': signal.trigger_reason,
                    'tags': signal.tags,
                    'workflow_matches': signal.workflow_matches,
                    'node_type': signal.node_type
                }
            )
            
            position.entry_index = current_index
            return position
            
        except Exception as e:
            print(f"❌ Error creating position from signal {signal.node_name}: {e}")
            return None
    
    def update_positions(self, end_index: Optional[int] = None) -> Dict[str, int]:
        """
        Update all open positions with current market data.
        
        Args:
            end_index: End index for processing (None for all data)
            
        Returns:
            Dictionary with update statistics
        """
        if end_index is None:
            end_index = len(self.market_data)
        
        positions_closed = 0
        positions_updated = 0
        
        print(f"\n📈 UPDATING POSITIONS (Processing to index {end_index})")
        print("="*50)
        
        # Process each tick of market data from current position to end
        for idx in range(self.current_index, end_index):
            if idx >= len(self.market_data):
                break
                
            row = self.market_data.iloc[idx]
            current_price = row['price']
            current_timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            # Update all active positions that should be affected by this price tick
            positions_to_close = []
            
            for position in self.active_positions:
                # Only update positions that were opened at or before this index
                if position.entry_index <= idx:
                    should_close = position.update_current_price(
                        current_price, idx, current_timestamp
                    )
                    
                    if should_close:
                        positions_to_close.append(position)
                        positions_closed += 1
                        
                        print(f"🔴 CLOSED: {position.node_name}")
                        print(f"   Entry: ${position.entry_price:.2f} → Exit: ${position.exit_price:.2f}")
                        print(f"   PnL: {position.pnl_points:.2f} pts ({position.pnl_percent:.1f}%)")
                        print(f"   Reason: {position.exit_reason} | Duration: {position.duration_held} ticks")
                        print("-" * 50)
                    else:
                        positions_updated += 1
            
            # Move closed positions
            for position in positions_to_close:
                self.active_positions.remove(position)
                self.closed_trades.append(position)
                self.execution_stats['positions_closed'] += 1
        
        # Update current index
        self.current_index = end_index
        
        # Force close any remaining positions at end of data
        if end_index >= len(self.market_data) - 1:
            remaining_positions = self.active_positions.copy()
            final_row = self.market_data.iloc[-1]
            final_price = final_row['price']
            final_timestamp = final_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            for position in remaining_positions:
                position.force_close(final_price, final_timestamp, len(self.market_data) - 1)
                self.active_positions.remove(position)
                self.closed_trades.append(position)
                positions_closed += 1
                print(f"🔒 FORCE CLOSED: {position.node_name} (End of data)")
        
        print(f"\n📊 Update Summary: {positions_closed} closed, {len(self.active_positions)} still open")
        
        return {
            'positions_closed': positions_closed,
            'positions_updated': positions_updated,
            'active_positions': len(self.active_positions)
        }
    
    def export_trades_to_csv(self, filename: str = "trades.csv") -> str:
        """
        Export all closed trades to CSV.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Path to the created file
        """
        if not self.closed_trades:
            print("No closed trades to export")
            return ""
        
        # Prepare trade data
        trade_data = []
        for trade in self.closed_trades:
            trade_record = {
                'node_name': trade.node_name,
                'confidence': trade.confidence,
                'entry_timestamp': trade.timestamp,
                'entry_price': trade.entry_price,
                'exit_timestamp': trade.exit_timestamp,
                'exit_price': trade.exit_price,
                'exit_reason': trade.exit_reason,
                'position_size': trade.position_size,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'pnl_points': trade.pnl_points,
                'pnl_percent': trade.pnl_percent,
                'duration_held': trade.duration_held,
                'trigger_reason': trade.signal_metadata.get('trigger_reason', ''),
                'tags': ', '.join(trade.signal_metadata.get('tags', [])),
                'workflow_matches': trade.signal_metadata.get('workflow_matches', 0),
                'status': trade.status.value
            }
            trade_data.append(trade_record)
        
        # Write to CSV
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            if trade_data:
                fieldnames = trade_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trade_data)
        
        print(f"✅ Exported {len(self.closed_trades)} trades to {filepath}")
        return filepath
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.closed_trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t.pnl_points > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl_points < 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        total_pnl = sum(t.pnl_points for t in self.closed_trades)
        avg_win = np.mean([t.pnl_points for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_points for t in losing_trades]) if losing_trades else 0
        
        # Per-node metrics
        node_performance = {}
        for trade in self.closed_trades:
            node = trade.node_name
            if node not in node_performance:
                node_performance[node] = {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'avg_duration': 0
                }
            
            node_performance[node]['trades'] += 1
            if trade.pnl_points > 0:
                node_performance[node]['wins'] += 1
            node_performance[node]['total_pnl'] += trade.pnl_points
            node_performance[node]['avg_duration'] += trade.duration_held
        
        # Calculate node averages
        for node, stats in node_performance.items():
            stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
            stats['avg_duration'] = stats['avg_duration'] / stats['trades']
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'node_performance': node_performance
        }
    
    def print_performance_summary(self):
        """Print a comprehensive performance summary."""
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            print("No performance data available")
            return
        
        print(f"\n📊 TRADING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Total PnL: {metrics['total_pnl']:.2f} points")
        print(f"Average Win: {metrics['avg_win']:.2f} points")
        print(f"Average Loss: {metrics['avg_loss']:.2f} points")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        print(f"\n📈 NODE PERFORMANCE BREAKDOWN")
        print("="*60)
        for node, stats in metrics['node_performance'].items():
            print(f"\n{node}:")
            print(f"  Trades: {stats['trades']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Avg PnL: {stats['avg_pnl']:.2f} points")
            print(f"  Avg Duration: {stats['avg_duration']:.1f} ticks")
            print(f"  Total PnL: {stats['total_pnl']:.2f} points")
    
    def get_current_portfolio_value(self) -> Dict[str, float]:
        """Get current portfolio statistics."""
        unrealized_pnl = sum(pos.get_unrealized_pnl() for pos in self.active_positions)
        realized_pnl = sum(trade.pnl_points for trade in self.closed_trades)
        
        return {
            'active_positions': len(self.active_positions),
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_pnl': unrealized_pnl + realized_pnl
        }
