"""
BlackBox Trading System - Core Components

This package contains the core trading system components:
- Strategy knowledge base (blackbox_core)
- Signal detection engine (node_detector) 
- Trade execution engine (trade_executor)
"""

from .blackbox_core import NodeEngine, StrategyNode, StrategyNodeParser
from .node_detector import NodeDetectorEngine, SignalEvent, DataFeedProcessor
from .trade_executor import TradeExecutorEngine, Position, RiskManager, PositionStatus

__all__ = [
    'NodeEngine',
    'StrategyNode', 
    'StrategyNodeParser',
    'NodeDetectorEngine',
    'SignalEvent',
    'DataFeedProcessor',
    'TradeExecutorEngine',
    'Position',
    'RiskManager',
    'PositionStatus'
]
