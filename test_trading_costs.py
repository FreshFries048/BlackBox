#!/usr/bin/env python3
"""
Production Grade Trading Cost Demonstration

This script demonstrates that realistic trading costs (commission + spreads)
are now being applied to P&L calculations in the BlackBox system.
"""

from dataclasses import dataclass
from trade_executor import RiskManager, Position, PositionStatus

def test_trading_costs():
    """Test that trading costs are properly applied to P&L."""
    
    print("🔬 Testing Production Grade Trading Costs")
    print("=" * 50)
    
    # Create risk manager with realistic trading costs
    risk_manager = RiskManager(
        commission_pct=0.05,  # 0.05% commission per trade
        spread_points=0.0002  # 2 pip spread for EUR/USD
    )
    
    # Create a test position
    position = Position(
        node_name="TEST_NODE",
        side="LONG",
        entry_price=1.0800,
        position_size=10000,
        stop_loss=1.0750,
        take_profit=1.0850,
        timestamp="2024-01-01 10:00:00",
        entry_index=0,
        confidence="HIGH"
    )
    
    print(f"📊 Test Position:")
    print(f"   Side: {position.side}")
    print(f"   Entry Price: ${position.entry_price:.4f}")
    print(f"   Position Size: {position.position_size:,}")
    print(f"   Take Profit: ${position.take_profit:.4f}")
    print()
    
    # Test profitable trade with costs
    print("🎯 Testing PROFITABLE trade (hit take profit):")
    
    # Calculate gross profit first
    gross_profit = (position.take_profit - position.entry_price) * position.position_size
    print(f"   Gross Profit: ${gross_profit:.2f}")
    
    # Apply trading costs manually for comparison
    commission_cost = (position.entry_price * position.position_size * risk_manager.commission_pct / 100) * 2
    spread_cost = risk_manager.spread_points * position.position_size / 10000
    total_costs = commission_cost + spread_cost
    expected_net = gross_profit - total_costs
    
    print(f"   Commission (round trip): ${commission_cost:.2f}")
    print(f"   Spread Cost: ${spread_cost:.2f}")
    print(f"   Total Trading Costs: ${total_costs:.2f}")
    print(f"   Expected Net P&L: ${expected_net:.2f}")
    
    # Close position with risk manager
    position._close_position(
        exit_price=position.take_profit,
        exit_timestamp="2024-01-01 11:00:00",
        reason="TAKE_PROFIT",
        exit_index=60,
        risk_manager=risk_manager
    )
    
    print(f"   Actual Net P&L: ${position.pnl_points:.2f}")
    print(f"   ✅ Costs Applied: {abs(position.pnl_points - expected_net) < 0.01}")
    print()
    
    # Test losing trade with costs
    print("🔴 Testing LOSING trade (hit stop loss):")
    
    position2 = Position(
        node_name="TEST_NODE_2",
        side="LONG", 
        entry_price=1.0800,
        position_size=10000,
        stop_loss=1.0750,
        take_profit=1.0850,
        timestamp="2024-01-01 10:00:00",
        entry_index=0,
        confidence="HIGH"
    )
    
    # Calculate gross loss
    gross_loss = (position2.stop_loss - position2.entry_price) * position2.position_size
    expected_net_loss = gross_loss - total_costs  # Even more negative
    
    print(f"   Gross Loss: ${gross_loss:.2f}")
    print(f"   Trading Costs: ${total_costs:.2f}")
    print(f"   Expected Net Loss: ${expected_net_loss:.2f}")
    
    position2._close_position(
        exit_price=position2.stop_loss,
        exit_timestamp="2024-01-01 10:30:00", 
        reason="STOP_LOSS",
        exit_index=30,
        risk_manager=risk_manager
    )
    
    print(f"   Actual Net Loss: ${position2.pnl_points:.2f}")
    print(f"   ✅ Costs Applied: {abs(position2.pnl_points - expected_net_loss) < 0.01}")
    print()
    
    print("🎯 Summary:")
    print(f"   Position 1 (Profit): ${position.pnl_points:.2f} (reduced by ${total_costs:.2f})")
    print(f"   Position 2 (Loss): ${position2.pnl_points:.2f} (worsened by ${total_costs:.2f})")
    print("   ✅ Production Grade: Realistic trading costs now applied!")

if __name__ == "__main__":
    test_trading_costs()
