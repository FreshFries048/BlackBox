"""
Configuration loader for BlackBox trading system.
Handles precedence: CLI args > env vars > YAML defaults.
"""

import os
import yaml
from pathlib import Path
from typing import Optional


def resolve_data_path(cli_arg: Optional[str] = None) -> Path:
    """
    Resolve data file path with precedence: CLI --data, env BLACKBOX_DATA_PATH, YAML default.
    
    Args:
        cli_arg: Command line argument for data file path
        
    Returns:
        Path object to data file
    """
    # CLI argument takes highest precedence
    if cli_arg:
        return Path(cli_arg)
    
    # Environment variable takes second precedence
    env_path = os.getenv('BLACKBOX_DATA_PATH')
    if env_path:
        return Path(env_path)
    
    # Fall back to YAML config default
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return Path(config.get('data_file', 'data/raw/EURUSD_1H_2020-2024.csv'))
    
    # Ultimate fallback
    return Path('data/raw/EURUSD_1H_2020-2024.csv')


def get_rr(cli_arg: Optional[float] = None) -> float:
    """
    Get risk-reward multiple with precedence: CLI --rr, env BLACKBOX_RR, YAML default.
    
    Args:
        cli_arg: Command line argument for RR multiple
        
    Returns:
        Risk-reward multiple as float
    """
    # CLI argument takes highest precedence
    if cli_arg is not None:
        return float(cli_arg)
    
    # Environment variable takes second precedence
    env_rr = os.getenv('BLACKBOX_RR')
    if env_rr:
        return float(env_rr)
    
    # Fall back to YAML config default
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return float(config.get('rr_multiple', 3.0))
    
    # Ultimate fallback
    return 3.0
