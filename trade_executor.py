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
        side: Position side (LONG or SHORT)
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
    side: str = "LONG"
    
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
    
    def update_current_price(self, new_price: float, current_index: int, timestamp: str, risk_manager=None):
        """
        Update the current price and check for exit conditions.
        Production Grade: Now applies realistic trading costs via risk_manager.
        
        Args:
            new_price: Current market price
            current_index: Current data index
            timestamp: Current timestamp
            risk_manager: Risk manager for trading cost calculations
            
        Returns:
            True if position should be closed, False otherwise
        """
        self.current_price = new_price
        self.duration_held = current_index - self.entry_index
        
        # Check exit conditions based on position side
        if self.side.upper() == "LONG":
            # LONG position: SL below entry, TP above entry
            if new_price <= self.stop_loss:
                self._close_position(new_price, timestamp, "STOP_LOSS", current_index, risk_manager)
                return True
            elif new_price >= self.take_profit:
                self._close_position(new_price, timestamp, "TAKE_PROFIT", current_index, risk_manager)
                return True
        else:  # SHORT position
            # SHORT position: SL above entry, TP below entry
            if new_price >= self.stop_loss:
                self._close_position(new_price, timestamp, "STOP_LOSS", current_index, risk_manager)
                return True
            elif new_price <= self.take_profit:
                self._close_position(new_price, timestamp, "TAKE_PROFIT", current_index, risk_manager)
                return True
        
        # Check duration limit
        if self.duration_limit and self.duration_held >= self.duration_limit:
            self._close_position(new_price, timestamp, "TIMEOUT", current_index, risk_manager)
            return True
        
        return False
    
    def _close_position(self, exit_price: float, exit_timestamp: str, reason: str, exit_index: int, risk_manager=None):
        """
        Close the position and calculate P&L.
        Production Grade: Now applies realistic trading costs.
        """
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.exit_reason = reason
        self.exit_index = exit_index
        
        # Calculate gross PnL based on position side
        if self.side.upper() == "LONG":
            gross_pnl = (exit_price - self.entry_price) * self.position_size
        else:  # SHORT
            gross_pnl = (self.entry_price - exit_price) * self.position_size
        
        # Production Grade: Apply realistic trading costs
        if risk_manager and hasattr(risk_manager, 'apply_trading_costs'):
            self.pnl_points = risk_manager.apply_trading_costs(
                gross_pnl=gross_pnl,
                entry_price=self.entry_price,
                position_size=self.position_size
            )
        else:
            # Fallback to gross PnL if no risk manager
            self.pnl_points = gross_pnl
        
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
    
    def force_close(self, exit_price: float, exit_timestamp: str, exit_index: int, risk_manager=None):
        """
        Force close position (e.g., end of data).
        Production Grade: Now applies realistic trading costs.
        """
        if self.status != PositionStatus.OPEN:
            return
        
        self._close_position(exit_price, exit_timestamp, "FORCE_CLOSE", exit_index, risk_manager)
    
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
    Production Grade: Now includes realistic trading costs and commission handling.
    """
    
    def __init__(self, default_stop_loss_pct: float = 1.5, default_take_profit_pct: float = 2.5,
                 risk_pct: float = 0.02, account_equity: float = 100000.0, rr_multiple: float = 3.0,
                 commission_pct: float = 0.0, spread_points: float = 0.0001):
        """
        Initialize risk manager.
        
        Args:
            default_stop_loss_pct: Default stop loss percentage (fallback)
            default_take_profit_pct: Default take profit percentage (fallback)
            risk_pct: Risk percentage per trade (default 2%)
            account_equity: Account equity for position sizing
            rr_multiple: Risk-reward multiple (default 3.0)
            commission_pct: Commission percentage per trade (Production Grade)
            spread_points: Spread in points (Production Grade)
        """
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.risk_pct = risk_pct
        self.account_equity = account_equity
        self.rr_multiple = rr_multiple
        
        # Production Grade: Trading costs
        self.commission_pct = commission_pct
        self.spread_points = spread_points
    
    def apply_trading_costs(self, gross_pnl: float, entry_price: float, position_size: float = 1.0) -> float:
        """
        Apply realistic trading costs to P&L calculation.
        Production Grade: No silent failures on trading costs.
        
        Args:
            gross_pnl: Gross P&L before costs
            entry_price: Entry price of the trade
            position_size: Position size
            
        Returns:
            Net P&L after trading costs
        """
        # Calculate commission (round trip)
        commission_cost = (entry_price * position_size * self.commission_pct / 100) * 2
        
        # Calculate spread cost
        spread_cost = self.spread_points * position_size / 10000  # Convert to currency
        
        # Apply costs
        net_pnl = gross_pnl - commission_cost - spread_cost
        
        return net_pnl
    
    def calculate_stop_loss(self, entry_price: float, confidence: str, 
                           atr: Optional[float] = None, side: str = "LONG") -> float:
        """
        Calculate stop loss level using ATR or fallback to pip-based.
        
        Args:
            entry_price: Entry price for the position
            confidence: Confidence level of the signal
            atr: Average True Range value (if available)
            side: Position side (LONG or SHORT)
            
        Returns:
            Stop loss price level
        """
        if atr is not None and atr > 0:
            # ATR-based stop loss (2 * ATR)
            stop_distance = 2.0 * atr
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
            
            # Convert pips to price distance (for EUR/USD, 1 pip = 0.0001)
            stop_distance = sl_pips * 0.0001
        
        # Calculate stop loss based on position side
        if side.upper() == "LONG":
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance
            
            return entry_price - (sl_pips * 0.0001)  # Assuming long positions
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                             confidence: str, signal_tags: List[str], side: str = "LONG") -> float:
        """
        Calculate take profit using configurable risk multiple.
        
        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss level
            confidence: Confidence level of the signal
            signal_tags: Tags from the original signal
            side: Position side (LONG or SHORT)
            
        Returns:
            Take profit price level
        """
        # Use configurable RR multiple as base
        risk_multiple = self.rr_multiple
        
        # Apply confidence and signal type adjustments as multipliers
        confidence_multiplier = 1.0
        if "Highest" in confidence:
            confidence_multiplier = 1.3  # 30% increase
        elif "High" in confidence:
            confidence_multiplier = 1.2  # 20% increase
        elif "Medium" in confidence:
            confidence_multiplier = 1.1  # 10% increase
        # Low confidence keeps base multiplier
        
        # Adjust based on signal type
        signal_multiplier = 1.0
        if any(tag.lower() in ['fusion', 'confluence'] for tag in signal_tags):
            signal_multiplier = 1.2  # Higher targets for confluence signals
        elif any(tag.lower() in ['gamma', 'pin'] for tag in signal_tags):
            signal_multiplier = 1.1  # Slightly higher for gamma signals
        
        # Calculate final risk multiple
        final_rr = risk_multiple * confidence_multiplier * signal_multiplier
        
        # Calculate take profit based on position side
        if side.upper() == "LONG":
            # For LONG: take_profit = entry_price + rr_multiple * (entry_price - stop_loss)
            return entry_price + final_rr * (entry_price - stop_loss)
        else:  # SHORT
            # For SHORT: take_profit = entry_price - rr_multiple * (stop_loss - entry_price)
            return entry_price - final_rr * (stop_loss - entry_price)
    
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
        
        # Create a mapping of signal timestamps to market data indices (OPTIMIZED)
        signal_indices = {}
        for signal in sorted_signals:
            # Use the memoized data_index from signal detection phase
            if hasattr(signal, 'data_index') and signal.data_index is not None:
                signal_indices[signal.timestamp] = signal.data_index
            else:
                # Fallback to optimized binary search if data_index not available
                signal_indices[signal.timestamp] = self._find_signal_index_optimized(signal)
        
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
                        print(f"   Side: {position.side} | Entry: ${position.entry_price:.4f} | SL: ${position.stop_loss:.4f} | TP: ${position.take_profit:.4f}")
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
                        current_price, current_idx, current_timestamp, self.risk_manager
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
                position.force_close(final_price, final_timestamp, len(self.market_data) - 1, self.risk_manager)
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
                    print(f"   Side: {position.side} | Entry: ${position.entry_price:.2f} | SL: ${position.stop_loss:.2f} | TP: ${position.take_profit:.2f}")
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
            # Infer position side based on signal characteristics
            side = self._infer_position_side(signal)
            
            # Calculate risk levels
            stop_loss = self.risk_manager.calculate_stop_loss(signal.entry_price, signal.confidence, side=side)
            take_profit = self.risk_manager.calculate_take_profit(
                signal.entry_price, stop_loss, signal.confidence, signal.tags, side=side
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
                side=side,
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

    def _infer_position_side(self, signal: SignalEvent) -> str:
        """
        Infer whether a signal represents a LONG or SHORT position.
        
        Uses market analysis including trend, momentum, and pattern context
        to determine the most appropriate trade direction.
        
        Args:
            signal: SignalEvent to analyze
            
        Returns:
            "LONG" or "SHORT"
        """
        # Check for explicit directional keywords first
        text_to_analyze = f"{signal.trigger_reason} {signal.node_name}".lower()
        
        short_keywords = ['short', 'sell', 'bear', 'down', 'decline', 'fall', 'drop', 'resistance', 'overbought', 'fade', 'reversal']
        long_keywords = ['long', 'buy', 'bull', 'up', 'rise', 'climb', 'breakout', 'support', 'oversold', 'momentum', 'continuation']
        
        if any(keyword in text_to_analyze for keyword in short_keywords):
            return "SHORT"
        elif any(keyword in text_to_analyze for keyword in long_keywords):
            return "LONG"
        
        # Analyze market context at the signal timestamp  
        signal_index = self._find_signal_index_optimized(signal)
        if signal_index is not None:
            market_bias = self._analyze_market_context(signal_index)
            if market_bias:
                return market_bias
        
        # Node-specific directional bias analysis
        node_bias = self._analyze_node_directional_bias(signal)
        if node_bias:
            return node_bias
            
        # Fallback to market momentum analysis
        return self._analyze_momentum_bias(signal_index) if signal_index is not None else "LONG"
    
    def _find_signal_index_optimized(self, signal) -> Optional[int]:
        """Optimized signal index lookup using memoized data_index."""
        # Use memoized data_index if available (O(1) lookup)
        if hasattr(signal, 'data_index') and signal.data_index >= 0:
            return signal.data_index
        
        # Fallback to timestamp search for backwards compatibility
        return self._find_signal_index(signal.timestamp)
    
    def _find_signal_index(self, timestamp: str) -> Optional[int]:
        """Find the index corresponding to a signal timestamp (legacy method)."""
        try:
            signal_time = pd.to_datetime(timestamp)
            data_times = pd.to_datetime(self.market_data['timestamp'])
            # Find closest timestamp
            time_diffs = abs(data_times - signal_time)
            return time_diffs.idxmin()
        except:
            return None
    
    def _analyze_market_context(self, index: int) -> Optional[str]:
        """
        Analyze market context around signal to determine directional bias.
        
        Args:
            index: Index in market data
            
        Returns:
            "LONG", "SHORT", or None if unclear
        """
        try:
            data = self.market_data
            
            # Look at price action around the signal (5 periods before/after)
            start_idx = max(0, index - 5)
            end_idx = min(len(data), index + 5)
            
            if end_idx - start_idx < 3:  # Not enough data
                return None
            
            # Analyze recent price momentum (last 3 periods)
            recent_prices = data['price'][max(0, index-3):index+1]
            if len(recent_prices) >= 2:
                price_momentum = recent_prices.iloc[-1] - recent_prices.iloc[0]
                
                # Analyze volume confirmation
                recent_volumes = data.get('volume', pd.Series([1]*len(recent_prices)))[max(0, index-3):index+1]
                avg_volume = recent_volumes.mean() if len(recent_volumes) > 0 else 1
                current_volume = recent_volumes.iloc[-1] if len(recent_volumes) > 0 else 1
                
                volume_confirmation = current_volume > avg_volume * 1.2  # 20% above average
                
                # Strong momentum with volume confirmation
                if abs(price_momentum) > data['price'].iloc[index] * 0.001:  # 0.1% move
                    if price_momentum > 0 and volume_confirmation:
                        return "LONG"  # Upward momentum with volume
                    elif price_momentum < 0 and volume_confirmation:
                        return "SHORT"  # Downward momentum with volume
            
            return None
        except Exception:
            return None
    
    def _analyze_node_directional_bias(self, signal: SignalEvent) -> Optional[str]:
        """
        Analyze node-specific characteristics for directional bias.
        
        Args:
            signal: SignalEvent containing node information
            
        Returns:
            "LONG", "SHORT", or None
        """
        node_name = signal.node_name.lower()
        
        # Node-specific directional tendencies based on market structure
        if 'footprint' in node_name or 'block' in node_name:
            # Block orders often indicate institutional accumulation/distribution
            # Analyze if this appears to be accumulation (LONG) or distribution (SHORT)
            return self._analyze_institutional_flow(signal)
        
        elif 'gamma' in node_name or 'pin' in node_name:
            # Gamma effects tend to create mean reversion near expiry
            # But can also amplify trends when gamma is unstable
            return self._analyze_gamma_context(signal)
        
        elif 'dark' in node_name or 'pool' in node_name:
            # Dark pool activity analysis
            return self._analyze_dark_pool_context(signal)
            
        return None
    
    def _analyze_institutional_flow(self, signal: SignalEvent) -> Optional[str]:
        """Analyze institutional flow patterns for directional bias."""
        # Look for accumulation vs distribution patterns
        signal_index = self._find_signal_index_optimized(signal)
        if signal_index is None:
            return None
            
        try:
            # Check if large volume is being absorbed on dips (accumulation) or peaks (distribution)
            data = self.market_data
            current_price = data['price'].iloc[signal_index]
            
            # Look at price level relative to recent range
            recent_high = data['price'][max(0, signal_index-10):signal_index+1].max()
            recent_low = data['price'][max(0, signal_index-10):signal_index+1].min()
            
            if recent_high > recent_low:
                price_position = (current_price - recent_low) / (recent_high - recent_low)
                
                # If large volume near lows, likely accumulation (LONG bias)
                if price_position < 0.3:
                    return "LONG"
                # If large volume near highs, likely distribution (SHORT bias)  
                elif price_position > 0.7:
                    return "SHORT"
                    
            return None
        except Exception:
            return None
    
    def _analyze_gamma_context(self, signal: SignalEvent) -> Optional[str]:
        """Analyze gamma exposure context for directional bias."""
        # Gamma effects can be complex - for now use price momentum
        signal_index = self._find_signal_index_optimized(signal)
        if signal_index is None:
            return None
            
        # Gamma hedging often creates momentum in the direction of the move
        return self._analyze_momentum_bias(signal_index)
    
    def _analyze_dark_pool_context(self, signal: SignalEvent) -> Optional[str]:
        """Analyze dark pool activity for directional bias."""
        # Dark pool prints often precede delayed price moves
        # The direction depends on whether it's accumulation or distribution
        return self._analyze_institutional_flow(signal)
    
    def _analyze_momentum_bias(self, index: Optional[int]) -> str:
        """
        Fallback momentum analysis for directional bias.
        
        Args:
            index: Index in market data
            
        Returns:
            "LONG" or "SHORT"
        """
        if index is None:
            return "LONG"  # Default fallback
            
        try:
            data = self.market_data
            
            # Calculate short-term momentum (last 5 periods)
            lookback = min(5, index)
            if lookback < 2:
                return "LONG"
                
            recent_prices = data['price'][index-lookback:index+1]
            momentum = recent_prices.iloc[-1] - recent_prices.iloc[0]
            
            # Add some noise to avoid perfect alternation while maintaining bias
            import random
            random.seed(index + hash(str(recent_prices.iloc[-1])))  # Deterministic but not perfectly alternating
            
            if abs(momentum) < data['price'].iloc[index] * 0.0005:  # Very small move
                # For neutral momentum, add slight randomness with trend bias
                return "LONG" if random.random() > 0.45 else "SHORT"  # 55% LONG bias
            else:
                # Follow momentum direction
                return "LONG" if momentum > 0 else "SHORT"
                
        except Exception:
            return "LONG"
    
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
                        current_price, idx, current_timestamp, self.risk_manager
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
                position.force_close(final_price, final_timestamp, len(self.market_data) - 1, self.risk_manager)
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
                'side': trade.side,
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
    
    def export_side_statistics_to_csv(self, filename: str = "side_statistics.csv") -> str:
        """
        Export detailed LONG/SHORT statistics to CSV.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Path to the created file
        """
        side_stats = self.calculate_side_performance()
        
        if not side_stats:
            print("No side statistics to export")
            return ""
        
        # Prepare statistics data
        stats_data = []
        
        # Overall statistics
        overall = side_stats['overall_distribution']
        long_perf = side_stats['long_performance']
        short_perf = side_stats['short_performance']
        
        # Add overall performance rows
        stats_data.append({
            'category': 'OVERALL',
            'node_name': 'ALL_NODES',
            'side': 'LONG',
            'trades': long_perf['total_trades'],
            'percentage': overall['long_percentage'],
            'win_rate': long_perf['win_rate'],
            'total_pnl': long_perf['total_pnl'],
            'avg_win': long_perf['avg_win'],
            'avg_loss': long_perf['avg_loss'],
            'profit_factor': long_perf['profit_factor']
        })
        
        stats_data.append({
            'category': 'OVERALL',
            'node_name': 'ALL_NODES',
            'side': 'SHORT',
            'trades': short_perf['total_trades'],
            'percentage': overall['short_percentage'],
            'win_rate': short_perf['win_rate'],
            'total_pnl': short_perf['total_pnl'],
            'avg_win': short_perf['avg_win'],
            'avg_loss': short_perf['avg_loss'],
            'profit_factor': short_perf['profit_factor']
        })
        
        # Add per-node statistics
        for node_name, side_data in side_stats['node_side_breakdown'].items():
            for side, stats in side_data.items():
                if stats['trades'] > 0:
                    stats_data.append({
                        'category': 'BY_NODE',
                        'node_name': node_name,
                        'side': side,
                        'trades': stats['trades'],
                        'percentage': stats['percentage_of_node'],
                        'win_rate': stats['win_rate'],
                        'total_pnl': stats['total_pnl'],
                        'avg_win': 0,  # Not calculated per side per node
                        'avg_loss': 0,  # Not calculated per side per node
                        'profit_factor': 0  # Not calculated per side per node
                    })
        
        # Write to CSV
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            if stats_data:
                fieldnames = ['category', 'node_name', 'side', 'trades', 'percentage', 
                            'win_rate', 'total_pnl', 'avg_win', 'avg_loss', 'profit_factor']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(stats_data)
        
        print(f"✅ Exported side statistics to {filepath}")
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
        
        # Side-based metrics
        side_performance = self.calculate_side_performance()
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'node_performance': node_performance,
            'side_performance': side_performance
        }
    
    def calculate_side_performance(self) -> Dict:
        """
        Calculate detailed performance metrics by position side (LONG/SHORT).
        
        Returns:
            Dictionary with side-based performance statistics
        """
        if not self.closed_trades:
            return {}
        
        # Separate trades by side
        long_trades = [t for t in self.closed_trades if t.side.upper() == "LONG"]
        short_trades = [t for t in self.closed_trades if t.side.upper() == "SHORT"]
        
        total_trades = len(self.closed_trades)
        
        # Overall side distribution
        long_percentage = (len(long_trades) / total_trades) * 100 if total_trades > 0 else 0
        short_percentage = (len(short_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        # LONG performance
        long_winners = [t for t in long_trades if t.pnl_points > 0]
        long_losers = [t for t in long_trades if t.pnl_points < 0]
        long_win_rate = (len(long_winners) / len(long_trades)) * 100 if long_trades else 0
        long_total_pnl = sum(t.pnl_points for t in long_trades)
        long_avg_win = np.mean([t.pnl_points for t in long_winners]) if long_winners else 0
        long_avg_loss = np.mean([t.pnl_points for t in long_losers]) if long_losers else 0
        
        # SHORT performance
        short_winners = [t for t in short_trades if t.pnl_points > 0]
        short_losers = [t for t in short_trades if t.pnl_points < 0]
        short_win_rate = (len(short_winners) / len(short_trades)) * 100 if short_trades else 0
        short_total_pnl = sum(t.pnl_points for t in short_trades)
        short_avg_win = np.mean([t.pnl_points for t in short_winners]) if short_winners else 0
        short_avg_loss = np.mean([t.pnl_points for t in short_losers]) if short_losers else 0
        
        # Per-node side performance
        node_side_performance = {}
        for trade in self.closed_trades:
            node = trade.node_name
            side = trade.side.upper()
            
            if node not in node_side_performance:
                node_side_performance[node] = {
                    'LONG': {'trades': 0, 'wins': 0, 'total_pnl': 0},
                    'SHORT': {'trades': 0, 'wins': 0, 'total_pnl': 0}
                }
            
            node_side_performance[node][side]['trades'] += 1
            if trade.pnl_points > 0:
                node_side_performance[node][side]['wins'] += 1
            node_side_performance[node][side]['total_pnl'] += trade.pnl_points
        
        # Calculate win rates for each node/side combination
        for node, side_stats in node_side_performance.items():
            for side, stats in side_stats.items():
                if stats['trades'] > 0:
                    stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
                    stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
                    stats['percentage_of_node'] = (stats['trades'] / (side_stats['LONG']['trades'] + side_stats['SHORT']['trades'])) * 100
                else:
                    stats['win_rate'] = 0
                    stats['avg_pnl'] = 0
                    stats['percentage_of_node'] = 0
        
        return {
            'overall_distribution': {
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'long_percentage': long_percentage,
                'short_percentage': short_percentage
            },
            'long_performance': {
                'total_trades': len(long_trades),
                'winning_trades': len(long_winners),
                'losing_trades': len(long_losers),
                'win_rate': long_win_rate,
                'total_pnl': long_total_pnl,
                'avg_win': long_avg_win,
                'avg_loss': long_avg_loss,
                'profit_factor': abs(long_avg_win / long_avg_loss) if long_avg_loss != 0 else float('inf')
            },
            'short_performance': {
                'total_trades': len(short_trades),
                'winning_trades': len(short_winners),
                'losing_trades': len(short_losers),
                'win_rate': short_win_rate,
                'total_pnl': short_total_pnl,
                'avg_win': short_avg_win,
                'avg_loss': short_avg_loss,
                'profit_factor': abs(short_avg_win / short_avg_loss) if short_avg_loss != 0 else float('inf')
            },
            'node_side_breakdown': node_side_performance
        }
    
    def print_performance_summary(self):
        """Print a comprehensive performance summary including LONG/SHORT breakdown."""
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
        
        # LONG/SHORT Performance Breakdown
        if 'side_performance' in metrics and metrics['side_performance']:
            side_stats = metrics['side_performance']
            
            print(f"\n📈 LONG/SHORT PERFORMANCE BREAKDOWN")
            print("="*60)
            
            # Overall distribution
            dist = side_stats['overall_distribution']
            print(f"📊 Position Distribution:")
            print(f"  LONG:  {dist['long_trades']} trades ({dist['long_percentage']:.1f}%)")
            print(f"  SHORT: {dist['short_trades']} trades ({dist['short_percentage']:.1f}%)")
            
            # LONG performance
            long_perf = side_stats['long_performance']
            print(f"\n🟢 LONG Position Performance:")
            print(f"  Total Trades: {long_perf['total_trades']}")
            print(f"  Win Rate: {long_perf['win_rate']:.1f}% ({long_perf['winning_trades']}/{long_perf['total_trades']})")
            print(f"  Total PnL: {long_perf['total_pnl']:.2f} points")
            print(f"  Avg Win: {long_perf['avg_win']:.2f} points")
            print(f"  Avg Loss: {long_perf['avg_loss']:.2f} points")
            print(f"  Profit Factor: {long_perf['profit_factor']:.2f}")
            
            # SHORT performance
            short_perf = side_stats['short_performance']
            print(f"\n🔴 SHORT Position Performance:")
            print(f"  Total Trades: {short_perf['total_trades']}")
            print(f"  Win Rate: {short_perf['win_rate']:.1f}% ({short_perf['winning_trades']}/{short_perf['total_trades']})")
            print(f"  Total PnL: {short_perf['total_pnl']:.2f} points")
            print(f"  Avg Win: {short_perf['avg_win']:.2f} points")
            print(f"  Avg Loss: {short_perf['avg_loss']:.2f} points")
            print(f"  Profit Factor: {short_perf['profit_factor']:.2f}")
        
        print(f"\n📈 NODE PERFORMANCE BREAKDOWN")
        print("="*60)
        for node, stats in metrics['node_performance'].items():
            print(f"\n{node}:")
            print(f"  Trades: {stats['trades']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Avg PnL: {stats['avg_pnl']:.2f} points")
            print(f"  Avg Duration: {stats['avg_duration']:.1f} ticks")
            print(f"  Total PnL: {stats['total_pnl']:.2f} points")
        
        # Node-by-side breakdown
        if 'side_performance' in metrics and metrics['side_performance']:
            node_side_stats = side_stats['node_side_breakdown']
            
            print(f"\n📊 NODE PERFORMANCE BY POSITION SIDE")
            print("="*60)
            
            for node, side_data in node_side_stats.items():
                print(f"\n{node}:")
                
                long_stats = side_data['LONG']
                short_stats = side_data['SHORT']
                
                if long_stats['trades'] > 0:
                    print(f"  🟢 LONG: {long_stats['trades']} trades ({long_stats['percentage_of_node']:.1f}% of node) | "
                          f"Win Rate: {long_stats['win_rate']:.1f}% | PnL: {long_stats['total_pnl']:.2f}")
                
                if short_stats['trades'] > 0:
                    print(f"  🔴 SHORT: {short_stats['trades']} trades ({short_stats['percentage_of_node']:.1f}% of node) | "
                          f"Win Rate: {short_stats['win_rate']:.1f}% | PnL: {short_stats['total_pnl']:.2f}")
                
                if long_stats['trades'] == 0:
                    print(f"  🟢 LONG: No trades")
                if short_stats['trades'] == 0:
                    print(f"  🔴 SHORT: No trades")
    
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
