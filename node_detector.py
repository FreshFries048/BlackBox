"""
BlackBox Core Engine - Phase 2: NodeDetectorEngine

Real-time trade detection and strategy simulation system that evaluates
parsed strategy nodes against market data to generate trading signals.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import csv
import os
import sys
import pandas as pd
import re

# Optional Numba JIT acceleration
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    njit = lambda **kwargs: lambda func: func

# Import the actual blackbox_core.py module file
sys.path.insert(0, os.path.dirname(__file__))

# Import blackbox_core.py module directly to avoid package conflicts
import importlib.util
spec = importlib.util.spec_from_file_location("blackbox_core_module", 
                                               os.path.join(os.path.dirname(__file__), "blackbox_core.py"))
blackbox_core_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blackbox_core_module)

# Production Grade: Custom exception for missing features (no silent failures)
class MissingFeatureError(Exception):
    """Raised when required features are missing from dataset."""
    pass

# Get classes from the module
NodeEngine = blackbox_core_module.NodeEngine
StrategyNode = blackbox_core_module.StrategyNode


@dataclass
class SignalEvent:
    """
    Represents a triggered trading signal from a strategy node.
    
    Attributes:
        node_name: Name of the strategy node that triggered
        timestamp: When the signal was generated
        trigger_reason: Description of why the signal triggered
        confidence: Confidence level from the node metadata
        entry_price: Price at signal generation
        node_type: Type of strategy node
        tags: Tags associated with the node
        workflow_matches: Number of workflow steps that matched
        validation_status: Current validation state
        metadata: Additional signal metadata
    """
    node_name: str
    timestamp: str
    trigger_reason: str
    confidence: str
    entry_price: float
    node_type: str = ""
    tags: List[str] = None
    workflow_matches: int = 0
    validation_status: str = "pending"
    metadata: Dict[str, Any] = None
    data_index: int = -1  # Memoised bar index for O(1) trade executor lookup
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class DataFeedProcessor:
    """
    Processes and validates market data feeds for the detector engine.
    """
    
    @staticmethod
    def validate_data_feed(data_feed: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and convert data feed to standardized DataFrame format.
        
        Args:
            data_feed: Price/time/volume data as dict or DataFrame
            
        Returns:
            Standardized DataFrame with required columns
        """
        if isinstance(data_feed, dict):
            df = pd.DataFrame(data_feed)
        elif isinstance(data_feed, pd.DataFrame):
            df = data_feed.copy()
        else:
            raise ValueError("Data feed must be a dictionary or pandas DataFrame")
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'price', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from data feed")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features that strategy nodes commonly use.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            DataFrame with additional derived features
        """
        # Price-based features
        df['price_change'] = df['price'].pct_change()
        df['price_range'] = df['price'].rolling(window=5).max() - df['price'].rolling(window=5).min()
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * 2)
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['time_of_day'] = df['hour'] + df['minute'] / 60
        
        # Liquidity proxies
        df['bid_ask_spread'] = df.get('ask', df['price'] * 1.001) - df.get('bid', df['price'] * 0.999)
        df['liquidity_proxy'] = df['volume'] / (df['bid_ask_spread'] + 0.001)
        
        return df


class NodeDetectorEngine:
    """
    Real-time detection engine that evaluates strategy nodes against market data
    and generates trading signals when conditions are met.
    
    This engine:
    1. Accepts market data feeds (CSV, dict, or DataFrame)
    2. Evaluates all loaded strategy nodes against each data point
    3. Triggers SignalEvents when node conditions are satisfied
    4. Maintains active signals and logs all activity
    """
    
    def __init__(self, node_engine: NodeEngine, data_feed: Union[Dict, pd.DataFrame] = None):
        """
        Initialize the detector engine.
        
        Args:
            node_engine: Loaded NodeEngine with strategy nodes
            data_feed: Market data (optional, can be set later)
        """
        self.node_engine = node_engine
        self.data_feed = None
        self.processed_data = None
        
        # Performance optimizations
        self._cached_arrays = {}  # Cache numpy arrays for vectorized operations
        self._compiled_regex = {}  # Cache compiled regex patterns
        self._condition_cache = {}  # Cache condition evaluation results
        
        self.active_signals: List[SignalEvent] = []
        self.signal_history: List[SignalEvent] = []
        self.detection_stats = {
            'total_ticks_processed': 0,
            'signals_generated': 0,
            'nodes_evaluated': 0,
            'last_processing_time': None
        }
        
        # Data field mappings for strategy nodes
        self.field_mappings = self._initialize_field_mappings()
        
        if data_feed is not None:
            self.set_data_feed(data_feed)
    
    def set_data_feed(self, data_feed: Union[Dict, pd.DataFrame]):
        """
        Set and validate the market data feed.
        
        Args:
            data_feed: Market data as dict or DataFrame
        """
        self.data_feed = DataFeedProcessor.validate_data_feed(data_feed)
        self.processed_data = DataFeedProcessor.add_derived_features(self.data_feed.copy())
        
        # Validate all nodes have required metrics
        data_columns = set(self.processed_data.columns)
        for node in self.node_engine.nodes:
            self._validate_metrics_exist(node, data_columns)
        
        print(f"✓ Data feed loaded: {len(self.processed_data)} data points")
        print(f"  Columns: {list(self.processed_data.columns)}")
        print(f"  Time range: {self.processed_data['timestamp'].min()} to {self.processed_data['timestamp'].max()}")
        
        # Performance optimization: Cache arrays for vectorized operations
        self._cache_arrays_for_vectorization()
    
    def _cache_arrays_for_vectorization(self):
        """Cache frequently accessed DataFrame columns as NumPy arrays."""
        if self.processed_data is None:
            return
        
        # Cache numeric columns that are frequently accessed
        numeric_cols = ['price', 'volume', 'price_change', 'volume_spike', 'time_of_day']
        for col in numeric_cols:
            if col in self.processed_data.columns:
                self._cached_arrays[col] = self.processed_data[col].values
        
        # Cache datetime index for timestamp operations
        if 'timestamp' in self.processed_data.columns:
            self._cached_arrays['timestamp'] = pd.to_datetime(self.processed_data['timestamp'])
    
    @staticmethod
    @njit(cache=True, fastmath=True) if NUMBA_AVAILABLE else lambda x: x
    def _evaluate_numeric_condition_vectorized(values: np.ndarray, threshold: float, operator: str) -> np.ndarray:
        """JIT-compiled vectorized numeric condition evaluation."""
        if operator == '>':
            return values > threshold
        elif operator == '<':
            return values < threshold
        elif operator == '>=':
            return values >= threshold
        elif operator == '<=':
            return values <= threshold
        elif operator == '==':
            return np.abs(values - threshold) < 1e-10
        else:
            return np.zeros(len(values), dtype=np.bool_)
    
    def _initialize_field_mappings(self) -> Dict[str, List[str]]:
        """
        Initialize field mappings to help match workflow steps to data columns.
        
        Returns:
            Dictionary mapping strategy concepts to data field names
        """
        return {
            'gamma': ['gamma', 'gamma_exposure', 'oi', 'open_interest', 'gamma_pin'],
            'block': ['block_size', 'large_prints', 'dark_prints', 'institutional_flow'],
            'volume': ['volume', 'volume_ma', 'volume_spike', 'cumulative_volume'],
            'footprint': ['footprint_clusters', 'volume_profile', 'order_flow'],
            'liquidity': ['liquidity_proxy', 'bid_ask_spread', 'market_depth'],
            'price': ['price', 'price_change', 'price_range', 'breakout'],
            'time': ['time_of_day', 'hour', 'minute', 'session_time'],
            'dealer': ['dealer_flow', 'market_maker_activity', 'hedging_flow'],
            'dark_pool': ['dark_prints', 'hidden_volume', 'iceberg_orders'],
            'expiry': ['expiry_date', 'days_to_expiry', 'opex_proximity']
        }
    
    def _validate_metrics_exist(self, node: StrategyNode, data_columns: Set[str]) -> None:
        """
        Validate that all required metrics for a node exist in the data.
        Production Grade: No silent failures on missing features.
        
        Args:
            node: Strategy node to validate
            data_columns: Set of available column names in data
            
        Raises:
            MissingFeatureError: If required metric is missing
        """
        # For now, assume basic market data columns are required
        required_metrics = {'price', 'timestamp'}  # Basic requirements
        
        # Check if basic metrics exist
        for metric in required_metrics:
            if metric not in data_columns:
                raise MissingFeatureError(f"Node '{node.name}' requires missing metric: {metric}")
    
    def run_detection(self, start_idx: int = 0, end_idx: Optional[int] = None, 
                     live_output: bool = True) -> List[SignalEvent]:
        """
        Run signal detection across the data feed.
        
        Args:
            start_idx: Starting index in the data
            end_idx: Ending index (None for all data)
            live_output: Whether to print signals as they're generated
            
        Returns:
            List of generated SignalEvents
        """
        if self.processed_data is None:
            raise ValueError("No data feed set. Use set_data_feed() first.")
        
        if end_idx is None:
            end_idx = len(self.processed_data)
        
        new_signals = []
        
        if not live_output:
            print(f"\n🔍 Starting optimized signal detection...")
        else:
            print(f"\n🔍 Starting signal detection...")
        print(f"   Processing {end_idx - start_idx} data points")
        print(f"   Evaluating {len(self.node_engine.nodes)} strategy nodes")
        print("="*60)
        
        # Performance optimization: Cache DataFrame columns as Series for faster access
        cached_series = {}
        for col in self.processed_data.columns:
            cached_series[col] = self.processed_data[col]
        
        for idx in range(start_idx, end_idx):
            # Use cached series instead of iloc for faster access
            current_data = {col: series.iloc[idx] for col, series in cached_series.items()}
            tick_signals = self._evaluate_all_nodes_optimized(current_data, idx)
            
            for signal in tick_signals:
                # Memoize data_index for O(1) trade executor lookup
                signal.data_index = idx
                new_signals.append(signal)
                self.active_signals.append(signal)
                self.signal_history.append(signal)
                
                if live_output:
                    self._print_signal(signal)
            
            self.detection_stats['total_ticks_processed'] += 1
        
        self.detection_stats['signals_generated'] += len(new_signals)
        self.detection_stats['last_processing_time'] = datetime.now()
        
        # Performance optimization: Clear cache if it gets too large
        if len(self._condition_cache) > 10000:
            self._condition_cache.clear()
        
        print(f"\n✓ Detection complete: {len(new_signals)} signals generated")
        return new_signals
    
    def _evaluate_all_nodes_optimized(self, data_dict: Dict[str, Any], row_idx: int) -> List[SignalEvent]:
        """
        Optimized evaluation of all strategy nodes against a single data point.
        Uses cached data and early exit optimizations.
        
        Args:
            data_dict: Current market data as dictionary (faster than Series)
            row_idx: Index of the current row
            
        Returns:
            List of triggered signals for this data point
        """
        signals = []
        
        for node in self.node_engine.nodes:
            self.detection_stats['nodes_evaluated'] += 1
            
            # Early exit: Check cache first
            cache_key = f"{node.name}_{row_idx}"
            if cache_key in self._condition_cache:
                match_result = self._condition_cache[cache_key]
            else:
                # Check if this node should trigger a signal with early exit
                match_result = self._evaluate_node_conditions_optimized(node, data_dict, row_idx)
                self._condition_cache[cache_key] = match_result
            
            if match_result['should_trigger']:
                signal = self._create_signal_event_optimized(node, data_dict, match_result)
                signals.append(signal)
        
        return signals
    
    def _evaluate_node_conditions_optimized(self, node: StrategyNode, data_dict: Dict[str, Any], 
                                          row_idx: int) -> Dict[str, Any]:
        """
        Optimized evaluation of node conditions with early exit and cached regex.
        
        Args:
            node: Strategy node to evaluate
            data_dict: Current market data as dictionary (faster access)
            row_idx: Current row index
            
        Returns:
            Dictionary with evaluation results
        """
        # Early exit: Check if node has mathematical steps for optimized path
        has_mathematical_steps = any('>' in step or '<' in step or '=' in step 
                                   for step in node.critical_steps + node.optional_steps)
        
        if has_mathematical_steps:
            # Use optimized confluence logic with early exit
            confluence_result = self._evaluate_confluence_optimized(node, data_dict)
            
            if confluence_result['confluence_met']:
                return {
                    'should_trigger': True,
                    'confluence_score': confluence_result['score'],
                    'matched_conditions': confluence_result['matched_conditions'],
                    'trigger_reason': f"Optimized confluence: {confluence_result['score']:.1%}",
                    'evaluation_method': 'optimized_confluence'
                }
            else:
                # Early exit - no need to continue evaluation
                return {'should_trigger': False, 'evaluation_method': 'optimized_confluence'}
        else:
            # Fall back to cached regex evaluation for legacy workflow
            return self._evaluate_legacy_workflow_cached(node, data_dict, row_idx)
    
    def _evaluate_confluence_optimized(self, node: StrategyNode, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized confluence evaluation with early exit on critical steps."""
        critical_matches = 0
        optional_matches = 0
        matched_conditions = []
        
        # Early exit on critical steps - if any fail, stop immediately
        for step in node.critical_steps:
            if self._evaluate_single_condition_optimized(step, data_dict):
                critical_matches += 1
                matched_conditions.append(step)
            else:
                # Early exit: critical step failed
                return {
                    'confluence_met': False,
                    'score': 0.0,
                    'matched_conditions': matched_conditions
                }
        
        # Only evaluate optional steps if all critical steps passed
        for step in node.optional_steps:
            if self._evaluate_single_condition_optimized(step, data_dict):
                optional_matches += 1
                matched_conditions.append(step)
        
        # Calculate score
        total_critical = len(node.critical_steps)
        total_optional = len(node.optional_steps)
        
        if total_critical + total_optional == 0:
            score = 0.0
        else:
            critical_score = (critical_matches / total_critical) if total_critical > 0 else 1.0
            optional_score = (optional_matches / total_optional) if total_optional > 0 else 0.0
            score = 0.7 * critical_score + 0.3 * optional_score
        
        confluence_met = (critical_matches == total_critical and score >= 0.6)
        
        return {
            'confluence_met': confluence_met,
            'score': score,
            'matched_conditions': matched_conditions
        }
    
    def _evaluate_single_condition_optimized(self, condition: str, data_dict: Dict[str, Any]) -> bool:
        """Optimized single condition evaluation with cached regex and JIT numeric ops."""
        condition = condition.strip()
        
        # Use cached regex for pattern matching
        pattern_key = f"numeric_{hash(condition)}"
        if pattern_key not in self._compiled_regex:
            self._compiled_regex[pattern_key] = re.compile(r'(\w+)\s*([><=]+)\s*([\d.]+)')
        
        numeric_match = self._compiled_regex[pattern_key].search(condition)
        
        if numeric_match:
            field, operator, threshold_str = numeric_match.groups()
            try:
                threshold = float(threshold_str)
                
                if field in data_dict:
                    value = data_dict[field]
                    
                    # Use JIT-compiled numeric evaluation if available and value is numeric
                    if NUMBA_AVAILABLE and isinstance(value, (int, float)):
                        return self._evaluate_numeric_condition_fast(float(value), threshold, operator)
                    else:
                        # Fallback to standard evaluation
                        if operator == '>':
                            return value > threshold
                        elif operator == '<':
                            return value < threshold
                        elif operator == '>=':
                            return value >= threshold
                        elif operator == '<=':
                            return value <= threshold
                        elif operator == '==':
                            return abs(float(value) - threshold) < 1e-10
                return False
            except (ValueError, TypeError):
                return False
        
        # Fallback to text matching for non-numeric conditions
        return self._evaluate_text_condition_cached(condition, data_dict)
    
    @staticmethod
    @njit(cache=True, fastmath=True) if NUMBA_AVAILABLE else lambda x: x
    def _evaluate_numeric_condition_fast(value: float, threshold: float, operator: str) -> bool:
        """JIT-compiled fast numeric condition evaluation."""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 1e-10
        return False
    
    def _evaluate_text_condition_cached(self, condition: str, data_dict: Dict[str, Any]) -> bool:
        """Evaluate text conditions using cached regex patterns."""
        # Initialize cached text patterns if not exists
        if 'text_patterns' not in self._compiled_regex:
            self._compiled_regex['text_patterns'] = {
                'volume_spike': re.compile(r'volume.*spike|spike.*volume', re.IGNORECASE),
                'price_movement': re.compile(r'price.*(up|down|move)', re.IGNORECASE),
                'trend': re.compile(r'trend.*(up|down|bull|bear)', re.IGNORECASE)
            }
        
        # Quick pattern matching
        for pattern_name, regex in self._compiled_regex['text_patterns'].items():
            if regex.search(condition):
                return self._evaluate_pattern_condition(pattern_name, data_dict)
        
        return False
    
    def _evaluate_pattern_condition(self, pattern_name: str, data_dict: Dict[str, Any]) -> bool:
        """Evaluate common pattern conditions efficiently."""
        if pattern_name == 'volume_spike' and 'volume_spike' in data_dict:
            return bool(data_dict['volume_spike'])
        elif pattern_name == 'price_movement' and 'price_change' in data_dict:
            return abs(data_dict.get('price_change', 0)) > 0.001
        elif pattern_name == 'trend' and 'price_change' in data_dict:
            return abs(data_dict.get('price_change', 0)) > 0.002
        return False
    
    def _evaluate_legacy_workflow_cached(self, node: StrategyNode, data_dict: Dict[str, Any], row_idx: int) -> Dict[str, Any]:
        """Cached evaluation of legacy workflow steps."""
        # Convert dict back to Series for legacy compatibility
        data_row = pd.Series(data_dict)
        return self._evaluate_node_conditions(node, data_row, row_idx)
    
    def _create_signal_event_optimized(self, node: StrategyNode, data_dict: Dict[str, Any], match_result: Dict[str, Any]) -> SignalEvent:
        """Optimized signal event creation."""
        return SignalEvent(
            node_name=node.name,
            timestamp=str(data_dict.get('timestamp', '')),
            trigger_reason=match_result.get('trigger_reason', 'Optimized detection'),
            confidence=node.metadata.get('confidence', 'Medium'),
            entry_price=float(data_dict.get('price', 0.0)),
            node_type=getattr(node, 'type', ''),
            workflow_matches=len(match_result.get('matched_conditions', [])),
            metadata={
                'confluence_score': match_result.get('confluence_score', 0.0),
                'evaluation_method': match_result.get('evaluation_method', 'optimized'),
                'matched_conditions': match_result.get('matched_conditions', [])
            }
        )
    
    def _evaluate_node_conditions_optimized(self, node: StrategyNode, data_dict: Dict[str, Any], 
                                          row_idx: int) -> Dict[str, Any]:
        """
        Optimized evaluation of node conditions with early exit and cached regex.
        
        Args:
            node: Strategy node to evaluate
            data_dict: Current market data as dictionary (faster access)
            row_idx: Current row index
            
        Returns:
            Dictionary with evaluation results
        """
        # Early exit: Check if node has mathematical steps for optimized path
        has_mathematical_steps = any('>' in step or '<' in step or '=' in step 
                                   for step in node.critical_steps + node.optional_steps)
        
        if has_mathematical_steps:
            # Use optimized confluence logic with early exit
            confluence_result = self._evaluate_confluence_optimized(node, data_dict)
            
            if confluence_result['confluence_met']:
                return {
                    'should_trigger': True,
                    'confluence_score': confluence_result['score'],
                    'matched_conditions': confluence_result['matched_conditions'],
                    'trigger_reason': f"Optimized confluence: {confluence_result['score']:.1%}",
                    'evaluation_method': 'optimized_confluence'
                }
            else:
                # Early exit - no need to continue evaluation
                return {'should_trigger': False, 'evaluation_method': 'optimized_confluence'}
        else:
            # Fall back to cached regex evaluation for legacy workflow
            return self._evaluate_legacy_workflow_cached(node, data_dict, row_idx)
    
    def _evaluate_confluence_optimized(self, node: StrategyNode, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized confluence evaluation with vectorized operations where possible."""
        critical_matches = 0
        optional_matches = 0
        matched_conditions = []
        
        # Early exit on critical steps - if any fail, stop immediately
        for step in node.critical_steps:
            if self._evaluate_single_condition_optimized(step, data_dict):
                critical_matches += 1
                matched_conditions.append(step)
            else:
                # Early exit: critical step failed
                return {
                    'confluence_met': False,
                    'score': 0.0,
                    'matched_conditions': matched_conditions
                }
        
        # Only evaluate optional steps if all critical steps passed
        for step in node.optional_steps:
            if self._evaluate_single_condition_optimized(step, data_dict):
                optional_matches += 1
                matched_conditions.append(step)
        
        # Calculate score
        total_critical = len(node.critical_steps)
        total_optional = len(node.optional_steps)
        
        if total_critical + total_optional == 0:
            score = 0.0
        else:
            critical_score = (critical_matches / total_critical) if total_critical > 0 else 1.0
            optional_score = (optional_matches / total_optional) if total_optional > 0 else 0.0
            score = 0.7 * critical_score + 0.3 * optional_score
        
        confluence_met = (critical_matches == total_critical and score >= 0.6)
        
        return {
            'confluence_met': confluence_met,
            'score': score,
            'matched_conditions': matched_conditions
        }
    
    def _evaluate_single_condition_optimized(self, condition: str, data_dict: Dict[str, Any]) -> bool:
        """Optimized single condition evaluation with JIT compilation for numeric expressions."""
        condition = condition.strip()
        
        # Use cached regex for pattern matching
        pattern_key = f"numeric_{condition}"
        if pattern_key not in self._compiled_regex:
            self._compiled_regex[pattern_key] = re.compile(r'(\w+)\s*([><=]+)\s*([\d.]+)')
        
        numeric_match = self._compiled_regex[pattern_key].search(condition)
        
        if numeric_match:
            field, operator, threshold_str = numeric_match.groups()
            threshold = float(threshold_str)
            
            if field in data_dict:
                value = data_dict[field]
                
                # Use JIT-compiled numeric evaluation if available
                if NUMBA_AVAILABLE and isinstance(value, (int, float)):
                    return self._evaluate_numeric_condition_fast(value, threshold, operator)
                else:
                    # Fallback to standard evaluation
                    if operator == '>':
                        return value > threshold
                    elif operator == '<':
                        return value < threshold
                    elif operator == '>=':
                        return value >= threshold
                    elif operator == '<=':
                        return value <= threshold
                    elif operator == '==':
                        return abs(value - threshold) < 1e-10
            return False
        
        # Fallback to text matching with cached regex
        return self._evaluate_text_condition_cached(condition, data_dict)
    
    @staticmethod
    @njit(cache=True, fastmath=True) if NUMBA_AVAILABLE else lambda x: x
    def _evaluate_numeric_condition_fast(value: float, threshold: float, operator: str) -> bool:
        """JIT-compiled fast numeric condition evaluation."""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 1e-10
        return False
    
    def _evaluate_text_condition_cached(self, condition: str, data_dict: Dict[str, Any]) -> bool:
        """Evaluate text conditions using cached regex patterns."""
        # Cache common text pattern regex
        if 'text_patterns' not in self._compiled_regex:
            self._compiled_regex['text_patterns'] = {
                'volume_spike': re.compile(r'volume.*spike|spike.*volume', re.IGNORECASE),
                'price_movement': re.compile(r'price.*(up|down|move)', re.IGNORECASE),
                'trend': re.compile(r'trend.*(up|down|bull|bear)', re.IGNORECASE)
            }
        
        # Quick pattern matching
        for pattern_name, regex in self._compiled_regex['text_patterns'].items():
            if regex.search(condition):
                return self._evaluate_pattern_condition(pattern_name, data_dict)
        
        return False
    
    def _evaluate_pattern_condition(self, pattern_name: str, data_dict: Dict[str, Any]) -> bool:
        """Evaluate common pattern conditions efficiently."""
        if pattern_name == 'volume_spike' and 'volume_spike' in data_dict:
            return bool(data_dict['volume_spike'])
        elif pattern_name == 'price_movement' and 'price_change' in data_dict:
            return abs(data_dict['price_change']) > 0.001
        elif pattern_name == 'trend' and 'price_change' in data_dict:
            return abs(data_dict['price_change']) > 0.002
        return False
    
    def _evaluate_legacy_workflow_cached(self, node: StrategyNode, data_dict: Dict[str, Any], row_idx: int) -> Dict[str, Any]:
        """Cached evaluation of legacy workflow steps."""
        # Convert dict back to Series for legacy compatibility
        data_row = pd.Series(data_dict)
        return self._evaluate_node_conditions(node, data_row, row_idx)
    
    def _create_signal_event_optimized(self, node: StrategyNode, data_dict: Dict[str, Any], match_result: Dict[str, Any]) -> SignalEvent:
        """Optimized signal event creation."""
        return SignalEvent(
            node_name=node.name,
            timestamp=str(data_dict.get('timestamp', '')),
            trigger_reason=match_result.get('trigger_reason', 'Optimized detection'),
            confidence=node.metadata.get('confidence', 'Medium'),
            entry_price=float(data_dict.get('price', 0.0)),
            node_type=getattr(node, 'type', ''),
            workflow_matches=len(match_result.get('matched_conditions', [])),
            metadata={
                'confluence_score': match_result.get('confluence_score', 0.0),
                'evaluation_method': match_result.get('evaluation_method', 'optimized'),
                'matched_conditions': match_result.get('matched_conditions', [])
            }
        )
    
    def _evaluate_all_nodes(self, data_row: pd.Series, row_idx: int) -> List[SignalEvent]:
        """
        Evaluate all strategy nodes against a single data point.
        
        Args:
            data_row: Current market data row
            row_idx: Index of the current row
            
        Returns:
            List of triggered signals for this data point
        """
        signals = []
        
        for node in self.node_engine.nodes:
            self.detection_stats['nodes_evaluated'] += 1
            
            # Check if this node should trigger a signal
            match_result = self._evaluate_node_conditions(node, data_row, row_idx)
            
            if match_result['should_trigger']:
                signal = self._create_signal_event(node, data_row, match_result)
                signals.append(signal)
        
        return signals
    
    def _evaluate_node_conditions(self, node: StrategyNode, data_row: pd.Series, 
                                 row_idx: int) -> Dict[str, Any]:
        """
        Evaluate whether a specific node's conditions are met.
        
        Uses enhanced confluence logic if YAML critical_steps exist,
        otherwise falls back to legacy workflow evaluation.
        
        Args:
            node: Strategy node to evaluate
            data_row: Current market data
            row_idx: Current row index
            
        Returns:
            Dictionary with evaluation results
        """
        # Check if node has YAML-defined critical steps (mathematical expressions)
        has_mathematical_steps = any('>' in step or '<' in step or '=' in step 
                                   for step in node.critical_steps + node.optional_steps)
        
        if has_mathematical_steps:
            # Use new confluence logic for YAML-enhanced nodes
            data_dict = data_row.to_dict()
            confluence_result = self.node_engine.evaluate_node_confluence(node, data_dict)
            
            result = {
                'should_trigger': confluence_result['fired'],
                'matched_steps': confluence_result['critical_passed'] + confluence_result['optional_passed'],
                'total_steps': confluence_result['critical_total'] + confluence_result['optional_total'],
                'trigger_reason': "",
                'matched_conditions': [],
                'confidence_score': confluence_result['confidence_score']
            }
            
            if result['should_trigger']:
                critical_desc = f"{confluence_result['critical_passed']}/{confluence_result['critical_total']} critical"
                optional_desc = f"{confluence_result['optional_passed']}/{confluence_result['optional_total']} optional" if confluence_result['optional_total'] > 0 else ""
                
                if optional_desc:
                    result['trigger_reason'] = f"Confluence achieved: {critical_desc}, {optional_desc} (confidence: {result['confidence_score']:.1%})"
                else:
                    result['trigger_reason'] = f"Critical confluence: {critical_desc} (confidence: {result['confidence_score']:.1%})"
        else:
            # Fall back to legacy evaluation for existing strategy files
            result = self._evaluate_legacy_workflow(node, data_row, row_idx)
        
        return result
    
    def _evaluate_legacy_workflow(self, node: StrategyNode, data_row: pd.Series, 
                                 row_idx: int) -> Dict[str, Any]:
        """
        Legacy evaluation method for existing strategy files.
        
        Args:
            node: Strategy node to evaluate
            data_row: Current market data
            row_idx: Current row index
            
        Returns:
            Dictionary with evaluation results
        """
        # Initialize result
        result = {
            'should_trigger': False,
            'matched_steps': 0,
            'total_steps': len(node.workflow),
            'trigger_reason': "",
            'matched_conditions': []
        }
        
        # Evaluate each workflow step using legacy logic
        for step_idx, workflow_step in enumerate(node.workflow):
            step_matches = self._evaluate_workflow_step(workflow_step, node, data_row, row_idx)
            
            if step_matches['matches']:
                result['matched_steps'] += 1
                result['matched_conditions'].append({
                    'step': step_idx + 1,
                    'description': workflow_step,
                    'reason': step_matches['reason']
                })
        
        # Determine if signal should trigger (3+ steps matched or 75% of steps)
        min_matches = max(3, int(0.75 * result['total_steps']))
        result['should_trigger'] = result['matched_steps'] >= min_matches
        
        if result['should_trigger']:
            reasons = [cond['reason'] for cond in result['matched_conditions']]
            result['trigger_reason'] = f"{result['matched_steps']}/{result['total_steps']} conditions met: " + "; ".join(reasons[:3])
        
        return result
    
    def _evaluate_workflow_step(self, workflow_step: str, node: StrategyNode, 
                               data_row: pd.Series, row_idx: int) -> Dict[str, Any]:
        """
        Evaluate a single workflow step against current market data.
        
        Args:
            workflow_step: Individual step from node's workflow
            node: Parent strategy node
            data_row: Current market data
            row_idx: Current row index
            
        Returns:
            Dictionary indicating if step matches and why
        """
        step_lower = workflow_step.lower()
        matches = False
        reason = ""
        
        # Check for gamma-related conditions
        if any(keyword in step_lower for keyword in ['gamma', 'pin', 'strike', 'oi']):
            if any(col in data_row.index for col in self.field_mappings['gamma']):
                gamma_value = data_row.get('gamma', data_row.get('oi', 0))
                if gamma_value > 0:
                    matches = True
                    reason = f"Gamma/OI detected ({gamma_value})"
        
        # Check for block/large print conditions
        elif any(keyword in step_lower for keyword in ['block', 'large', 'print', 'institutional']):
            block_size = data_row.get('block_size', data_row.get('large_prints', 0))
            volume = data_row.get('volume', 0)
            volume_ma = data_row.get('volume_ma', volume)
            
            if block_size > 0 or volume > volume_ma * 1.5:
                matches = True
                reason = f"Large print/block detected (size: {block_size}, vol: {volume:.0f})"
        
        # Check for volume/footprint conditions
        elif any(keyword in step_lower for keyword in ['volume', 'footprint', 'cluster', 'absorption']):
            volume_spike = data_row.get('volume_spike', False)
            footprint = data_row.get('footprint_clusters', 0)
            
            if volume_spike or footprint > 0:
                matches = True
                reason = f"Volume/footprint activity (spike: {volume_spike}, clusters: {footprint})"
        
        # Check for price/breakout conditions
        elif any(keyword in step_lower for keyword in ['price', 'breakout', 'expansion', 'move']):
            price_change = abs(data_row.get('price_change', 0))
            price_range = data_row.get('price_range', 0)
            
            if price_change > 0.002 or price_range > 0:  # 0.2% move or range expansion
                matches = True
                reason = f"Price movement detected (change: {price_change:.1%})"
        
        # Check for time-based conditions
        elif any(keyword in step_lower for keyword in ['time', 'window', '10am', '2:30pm', 'session']):
            time_of_day = data_row.get('time_of_day', data_row.get('hour', 12))
            
            # Key trading times: market open (9.5), mid-morning (10), lunch (12), close (15.5)
            key_times = [9.5, 10, 12, 15.5]
            if any(abs(time_of_day - kt) < 0.5 for kt in key_times):
                matches = True
                reason = f"Key time window (hour: {time_of_day})"
        
        # Check for liquidity conditions
        elif any(keyword in step_lower for keyword in ['liquidity', 'level', 'node', 'support', 'resistance']):
            liquidity_proxy = data_row.get('liquidity_proxy', 0)
            spread = data_row.get('bid_ask_spread', 0.01)
            
            if liquidity_proxy > 1000 or spread < 0.005:  # High liquidity indicators
                matches = True
                reason = f"Liquidity conditions met (proxy: {liquidity_proxy:.0f})"
        
        # Check for dark pool conditions
        elif any(keyword in step_lower for keyword in ['dark', 'hidden', 'vwap', 'tape']):
            dark_prints = data_row.get('dark_prints', 0)
            hidden_volume = data_row.get('hidden_volume', 0)
            
            if dark_prints > 0 or hidden_volume > 0:
                matches = True
                reason = f"Dark pool activity (prints: {dark_prints}, hidden: {hidden_volume})"
        
        # Check for confluence/alignment conditions
        elif any(keyword in step_lower for keyword in ['confluence', 'alignment', 'overlay', 'combine']):
            # Multiple factors present
            factors = 0
            if data_row.get('gamma', 0) > 0: factors += 1
            if data_row.get('volume_spike', False): factors += 1
            if abs(data_row.get('price_change', 0)) > 0.001: factors += 1
            if data_row.get('liquidity_proxy', 0) > 500: factors += 1
            
            if factors >= 2:
                matches = True
                reason = f"Multiple factor confluence ({factors} factors)"
        
        # Default fallback - check for any relevant data
        elif not matches:
            # If step mentions any mapped concepts, check for basic data presence
            for concept, fields in self.field_mappings.items():
                if concept in step_lower:
                    for field in fields:
                        if field in data_row.index and data_row[field] is not None:
                            value = data_row[field]
                            if isinstance(value, (int, float)) and value != 0:
                                matches = True
                                reason = f"{concept.title()} data present ({field}: {value})"
                                break
                    if matches:
                        break
        
        return {'matches': matches, 'reason': reason}
    
    def _create_signal_event(self, node: StrategyNode, data_row: pd.Series, 
                           match_result: Dict[str, Any]) -> SignalEvent:
        """
        Create a SignalEvent from a triggered node.
        
        Args:
            node: Strategy node that triggered
            data_row: Current market data
            match_result: Results from condition evaluation
            
        Returns:
            SignalEvent object
        """
        timestamp = data_row['timestamp']
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        tags = node.metadata.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',')]
        
        signal = SignalEvent(
            node_name=node.name,
            timestamp=timestamp,
            trigger_reason=match_result['trigger_reason'],
            confidence=node.metadata.get('confidence', 'Unknown'),
            entry_price=float(data_row['price']),
            node_type=node.type,
            tags=tags,
            workflow_matches=match_result['matched_steps'],
            validation_status="triggered",
            metadata={
                'total_workflow_steps': match_result['total_steps'],
                'matched_conditions': match_result['matched_conditions'],
                'data_row_index': data_row.name,
                'volume': data_row.get('volume', 0),
                'time_of_day': data_row.get('time_of_day', 0)
            }
        )
        
        return signal
    
    def _print_signal(self, signal: SignalEvent):
        """Print a signal event in a formatted way."""
        print(f"🚨 SIGNAL: {signal.node_name}")
        print(f"   Time: {signal.timestamp} | Price: ${signal.entry_price:.2f}")
        print(f"   Confidence: {signal.confidence} | Matches: {signal.workflow_matches}")
        print(f"   Reason: {signal.trigger_reason}")
        print(f"   Tags: {', '.join(signal.tags)}")
        print("-" * 60)
    
    def export_signals_to_csv(self, filename: str = "signal_log.csv") -> str:
        """
        Export all signals to a CSV file.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Path to the created file
        """
        if not self.signal_history:
            print("No signals to export")
            return ""
        
        # Convert signals to dictionaries
        signal_dicts = [asdict(signal) for signal in self.signal_history]
        
        # Flatten nested metadata
        for signal_dict in signal_dicts:
            metadata = signal_dict.pop('metadata', {})
            for key, value in metadata.items():
                signal_dict[f'meta_{key}'] = value
            
            # Convert tags list to string
            signal_dict['tags'] = ', '.join(signal_dict['tags'])
        
        # Write to CSV
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            if signal_dicts:
                fieldnames = signal_dicts[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(signal_dicts)
        
        print(f"✓ Exported {len(self.signal_history)} signals to {filepath}")
        return filepath
    
    def get_active_signals(self, confidence_filter: str = None) -> List[SignalEvent]:
        """
        Get currently active signals, optionally filtered by confidence.
        
        Args:
            confidence_filter: Filter by confidence level (e.g., "High", "Highest")
            
        Returns:
            List of active signals
        """
        if confidence_filter is None:
            return self.active_signals.copy()
        
        return [s for s in self.active_signals if confidence_filter.lower() in s.confidence.lower()]
    
    def print_detection_stats(self):
        """Print detection engine statistics."""
        print(f"\n📊 DETECTION ENGINE STATISTICS")
        print("="*40)
        print(f"Total ticks processed: {self.detection_stats['total_ticks_processed']:,}")
        print(f"Signals generated: {self.detection_stats['signals_generated']:,}")
        print(f"Nodes evaluated: {self.detection_stats['nodes_evaluated']:,}")
        print(f"Active signals: {len(self.active_signals)}")
        print(f"Signal history: {len(self.signal_history)}")
        if self.detection_stats['last_processing_time']:
            print(f"Last run: {self.detection_stats['last_processing_time']}")
        
        # Signal breakdown by confidence
        if self.signal_history:
            print(f"\nSignal breakdown by confidence:")
            confidence_counts = {}
            for signal in self.signal_history:
                conf = signal.confidence
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
            
            for conf, count in sorted(confidence_counts.items()):
                print(f"  {conf}: {count} signals")


# Example usage and testing
if __name__ == "__main__":
    # NodeEngine is already imported at the top
    
    print("🔧 BLACKBOX PHASE 2: NODE DETECTOR ENGINE TEST")
    print("="*60)
    
    # Load strategy nodes
    print("\n1. Loading strategy nodes...")
    node_engine = NodeEngine()
    node_engine.load_nodes_from_folder("/workspaces/BlackBox/blackbox_nodes")
    
    # Create sample market data
    print("\n2. Creating sample market data...")
    np.random.seed(42)  # For reproducible results
    
    timestamps = pd.date_range('2024-08-02 09:30:00', periods=100, freq='1min')
    base_price = 4100.0
    
    sample_data = {
        'timestamp': timestamps,
        'price': base_price + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100),
        'gamma': np.random.randint(0, 1000, 100),
        'dark_prints': np.random.randint(0, 5, 100),
        'footprint_clusters': np.random.randint(0, 3, 100),
        'block_size': np.random.choice([0, 0, 0, 0, 1000, 2000, 5000], 100),
    }
    
    # Add some strategic spikes for testing
    sample_data['gamma'][20] = 5000  # Large gamma at index 20
    sample_data['block_size'][35] = 10000  # Large block at index 35
    sample_data['dark_prints'][50] = 15  # Dark pool activity at index 50
    
    print(f"   Created {len(timestamps)} data points")
    
    # Initialize detector engine
    print("\n3. Initializing detector engine...")
    detector = NodeDetectorEngine(node_engine, sample_data)
    
    # Run detection
    print("\n4. Running signal detection...")
    signals = detector.run_detection(live_output=True)
    
    # Print results
    print(f"\n5. Detection Results Summary")
    print("="*40)
    detector.print_detection_stats()
    
    # Export to CSV
    print(f"\n6. Exporting signals...")
    csv_path = detector.export_signals_to_csv("test_signal_log.csv")
    
    # Show high confidence signals
    high_conf_signals = detector.get_active_signals("High")
    print(f"\n7. High confidence signals: {len(high_conf_signals)}")
    for signal in high_conf_signals[:3]:  # Show first 3
        print(f"   • {signal.node_name} at {signal.timestamp}")
        print(f"     Price: ${signal.entry_price:.2f}, Matches: {signal.workflow_matches}")
    
    print("\n✅ Phase 2 NodeDetectorEngine test complete!")
