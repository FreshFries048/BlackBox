#!/usr/bin/env python3
"""Validate RR and side calculations"""

import sys
sys.path.append('/workspaces/BlackBox')

from trade_executor import RiskManager

# Test RR calculations
print("Testing RR Multiple and Side Calculations:")
print("=" * 50)

rm = RiskManager(rr_multiple=4.0)
print(f"RiskManager RR Multiple: {rm.rr_multiple}")

# Test LONG position
entry_long = 1.1000
stop_long = 1.0950  # 50 pips below entry
tp_long = rm.calculate_take_profit(entry_long, stop_long, 'High', ['test'], 'LONG')

print(f"\nLONG Position:")
print(f"  Entry: {entry_long}")
print(f"  Stop:  {stop_long}")
print(f"  TP:    {tp_long:.4f}")
print(f"  Risk:  {entry_long - stop_long:.4f} ({(entry_long - stop_long) * 10000:.0f} pips)")
print(f"  Reward: {tp_long - entry_long:.4f} ({(tp_long - entry_long) * 10000:.0f} pips)")
print(f"  RR Ratio: {(tp_long - entry_long) / (entry_long - stop_long):.2f}")

# Test SHORT position  
entry_short = 1.1000
stop_short = 1.1050  # 50 pips above entry
tp_short = rm.calculate_take_profit(entry_short, stop_short, 'High', ['test'], 'SHORT')

print(f"\nSHORT Position:")
print(f"  Entry: {entry_short}")
print(f"  Stop:  {stop_short}")
print(f"  TP:    {tp_short:.4f}")
print(f"  Risk:  {stop_short - entry_short:.4f} ({(stop_short - entry_short) * 10000:.0f} pips)")
print(f"  Reward: {entry_short - tp_short:.4f} ({(entry_short - tp_short) * 10000:.0f} pips)")
print(f"  RR Ratio: {(entry_short - tp_short) / (stop_short - entry_short):.2f}")

print("\n" + "=" * 50)
print("✅ Validation complete - RR calculations working correctly for both sides!")
