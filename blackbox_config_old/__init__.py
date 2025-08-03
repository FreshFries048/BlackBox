"""
BlackBox Core Engine Module
"""

# Import classes from blackbox_core.py for easier access
import sys
import os

# Add the parent directory to sys.path to access blackbox_core.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from the main blackbox_core.py file
try:
    from blackbox_core import NodeEngine, StrategyNode, StrategyNodeParser, MissingFeatureError
    __all__ = ['NodeEngine', 'StrategyNode', 'StrategyNodeParser', 'MissingFeatureError']
except ImportError:
    # Fallback to config_loader only
    from .config_loader import *