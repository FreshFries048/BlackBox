"""
Data Feed Processor - Enhanced Market Data Features

This module provides data preprocessing and feature engineering
for the BlackBox trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import sys
import os

# Import the actual blackbox_core.py module file
sys.path.insert(0, os.path.dirname(__file__))
import blackbox_core

# Get exception from the module
MissingFeatureError = blackbox_core.MissingFeatureError


class DataFeedProcessor:
    """
    Processor for market data with enhanced feature engineering.
    
    Handles calculation of derived features like gamma exposure,
    dark pool metrics, and technical indicators.
    """
    
    def __init__(self):
        self.required_base_columns = ['open', 'high', 'low', 'close', 'volume']
        self.enhanced_features = [
            'gamma_exposure', 'gamma_pin_distance', 'oi',
            'dark_prints', 'hidden_volume', 'block_size',
            'atr_14'
        ]
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to market data DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with enhanced features
            
        Raises:
            MissingFeatureError: If required base columns are missing
        """
        # Validate required columns
        missing_cols = [col for col in self.required_base_columns if col not in df.columns]
        if missing_cols:
            raise MissingFeatureError(f"Missing required columns: {missing_cols}")
        
        df_enhanced = df.copy()
        
        # Calculate ATR (14-period) if not present
        if 'atr_14' not in df_enhanced.columns:
            df_enhanced['atr_14'] = self._calculate_atr(df_enhanced, period=14)
        
        # Calculate gamma-related features if not present
        if 'gamma_exposure' not in df_enhanced.columns:
            df_enhanced['gamma_exposure'] = self._calculate_gamma_exposure(df_enhanced)
        
        if 'gamma_pin_distance' not in df_enhanced.columns:
            df_enhanced['gamma_pin_distance'] = self._calculate_gamma_pin_distance(df_enhanced)
        
        # Calculate dark pool metrics if not present
        if 'dark_prints' not in df_enhanced.columns:
            df_enhanced['dark_prints'] = self._calculate_dark_prints(df_enhanced)
        
        if 'hidden_volume' not in df_enhanced.columns:
            df_enhanced['hidden_volume'] = self._calculate_hidden_volume(df_enhanced)
        
        if 'block_size' not in df_enhanced.columns:
            df_enhanced['block_size'] = self._calculate_block_size(df_enhanced)
        
        # Calculate open interest if not present
        if 'oi' not in df_enhanced.columns:
            df_enhanced['oi'] = self._estimate_open_interest(df_enhanced)
        
        return df_enhanced
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_gamma_exposure(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate gamma exposure based on price volatility and volume.
        
        This is a synthetic calculation for demonstration.
        In production, this would come from options data.
        """
        volatility = (df['high'] - df['low']) / df['close']
        volume_normalized = df['volume'] / df['volume'].rolling(20).mean()
        
        # Synthetic gamma exposure calculation
        gamma_exposure = volatility * volume_normalized * 1000
        return gamma_exposure.fillna(100)
    
    def _calculate_gamma_pin_distance(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate distance to nearest gamma pin level.
        
        This estimates how close price is to major options strikes.
        """
        # Synthetic calculation - in production would use real options data
        price_rounded = (df['close'] / 50).round() * 50  # Round to nearest $50
        distance = abs(df['close'] - price_rounded) / df['close']
        
        return distance
    
    def _calculate_dark_prints(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate dark pool prints based on volume patterns.
        
        High volume with small price movement suggests dark pool activity.
        """
        volume_ma = df['volume'].rolling(10).mean()
        volume_spike = df['volume'] / volume_ma
        price_change = abs(df['close'].pct_change())
        
        # High volume, low price change = potential dark pool activity
        dark_prints = np.where(
            (volume_spike > 1.5) & (price_change < 0.005),
            np.random.randint(5, 15, len(df)),  # Synthetic for demo
            np.random.randint(0, 3, len(df))
        )
        
        return pd.Series(dark_prints, index=df.index)
    
    def _calculate_hidden_volume(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate hidden volume from iceberg orders and dark pools.
        """
        # Synthetic calculation based on volume patterns
        volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
        hidden_volume = df['volume'] * np.where(volume_ratio > 1.2, 0.3, 0.1)
        
        return hidden_volume.fillna(0)
    
    def _calculate_block_size(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate block trade sizes based on volume and price impact.
        """
        volume_ma = df['volume'].rolling(5).mean()
        price_impact = abs(df['close'].pct_change())
        
        # Large volume with price impact suggests block trades
        block_size = np.where(
            (df['volume'] > volume_ma * 2) & (price_impact > 0.001),
            np.random.randint(5000, 20000, len(df)),  # Synthetic for demo
            np.random.randint(0, 2000, len(df))
        )
        
        return pd.Series(block_size, index=df.index)
    
    def _estimate_open_interest(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate open interest based on volume and volatility patterns.
        """
        # Synthetic calculation - in production would use real options/futures data
        volume_trend = df['volume'].rolling(10).mean()
        volatility = (df['high'] - df['low']) / df['close']
        
        oi = volume_trend * volatility * 1000
        return oi.fillna(0)
    
    def validate_features(self, df: pd.DataFrame, required_features: List[str]) -> None:
        """
        Validate that required features are present in DataFrame.
        
        Args:
            df: DataFrame to validate
            required_features: List of required feature names
            
        Raises:
            MissingFeatureError: If any required features are missing
        """
        missing_features = [feat for feat in required_features if feat not in df.columns]
        if missing_features:
            raise MissingFeatureError(
                f"Missing required features: {missing_features}. "
                f"Available features: {list(df.columns)}"
            )
    
    def prepare_market_data(self, df: pd.DataFrame, required_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        Args:
            df: Raw market data DataFrame
            required_features: Optional list of features that must be present
            
        Returns:
            Enhanced DataFrame ready for strategy evaluation
        """
        # Add derived features
        df_enhanced = self.add_derived_features(df)
        
        # Validate required features if specified
        if required_features:
            self.validate_features(df_enhanced, required_features)
        
        return df_enhanced


if __name__ == "__main__":
    # Test the data feed processor
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    processor = DataFeedProcessor()
    enhanced_data = processor.prepare_market_data(sample_data)
    
    print("Enhanced features added:")
    for col in enhanced_data.columns:
        if col not in sample_data.columns:
            print(f"  - {col}")
    
    print(f"\nDataFrame shape: {enhanced_data.shape}")
    print(f"Features: {list(enhanced_data.columns)}")
