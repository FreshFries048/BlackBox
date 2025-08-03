#!/usr/bin/env python3
"""Test script for new BlackBox features"""

# Test config loader
try:
    from blackbox_config.config_loader import resolve_data_path, get_rr
    print(f"✅ Config loader working - Data: {resolve_data_path()}, RR: {get_rr()}")
except Exception as e:
    print(f"❌ Config loader error: {e}")

# Test RiskManager
try:
    from trade_executor import RiskManager
    rm = RiskManager(rr_multiple=2.5)
    print(f"✅ RiskManager working - RR: {rm.rr_multiple}")
except Exception as e:
    print(f"❌ RiskManager error: {e}")

# Test Position 
try:
    from trade_executor import Position
    pos = Position(
        node_name='test', 
        entry_price=1.0, 
        timestamp='test', 
        stop_loss=0.9, 
        take_profit=1.3, 
        confidence='High', 
        position_size=10000, 
        side='LONG'
    )
    print(f"✅ Position working - Side: {pos.side}")
except Exception as e:
    print(f"❌ Position error: {e}")

print("Test complete!")
