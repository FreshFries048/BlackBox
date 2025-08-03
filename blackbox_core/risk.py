"""
Enhanced Risk Management with Realistic Trading Costs
"""
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class TradingCosts:
    """Trading cost configuration."""
    commission_perc: float = 0.0  # Commission as percentage of trade value
    spread_points: float = 0.0    # Bid-ask spread in points
    
    @classmethod
    def from_env(cls) -> 'TradingCosts':
        """Load trading costs from environment variables."""
        commission = float(os.getenv('BLACKBOX_COMMISSION', '0.0'))
        spread = float(os.getenv('BLACKBOX_SPREAD', '0.0'))
        return cls(commission_perc=commission, spread_points=spread)


class EnhancedRiskManager:
    """Risk manager with realistic trading costs."""
    
    def __init__(self, 
                 default_stop_loss_pct: float = 50,
                 default_take_profit_pct: float = 60,
                 rr_multiple: float = 1.0,
                 trading_costs: Optional[TradingCosts] = None):
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.rr_multiple = rr_multiple
        self.trading_costs = trading_costs or TradingCosts()
    
    def calculate_position_size(self, account_balance: float, risk_percent: float,
                              entry_price: float, stop_loss_price: float) -> float:
        """Calculate position size accounting for commission."""
        # Calculate raw position size
        risk_amount = account_balance * (risk_percent / 100)
        price_risk = abs(entry_price - stop_loss_price)
        raw_position_size = risk_amount / price_risk if price_risk > 0 else 0
        
        # Adjust for commission costs
        commission_cost = raw_position_size * entry_price * (self.trading_costs.commission_perc / 100)
        adjusted_risk = risk_amount - commission_cost
        
        return max(0, adjusted_risk / price_risk if price_risk > 0 else 0)
    
    def apply_spread_to_entry(self, price: float, side: str) -> float:
        """Apply bid-ask spread to entry price."""
        half_spread = self.trading_costs.spread_points / 2
        if side == "LONG":
            return price + half_spread  # Buy at ask
        else:  # SHORT
            return price - half_spread  # Sell at bid
    
    def apply_spread_to_exit(self, price: float, side: str) -> float:
        """Apply bid-ask spread to exit price.""" 
        half_spread = self.trading_costs.spread_points / 2
        if side == "LONG":
            return price - half_spread  # Sell at bid
        else:  # SHORT
            return price + half_spread  # Cover at ask
    
    def calculate_net_pnl(self, gross_pnl: float, trade_value: float) -> float:
        """Calculate net PnL after commission."""
        commission = trade_value * (self.trading_costs.commission_perc / 100)
        return gross_pnl - (2 * commission)  # Entry + exit commission
    
    def get_costs_metadata(self) -> dict:
        """Return trading costs for run metadata."""
        return {
            "commission_perc": self.trading_costs.commission_perc,
            "spread_points": self.trading_costs.spread_points
        }
