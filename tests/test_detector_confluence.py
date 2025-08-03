"""
Test suite for NodeDetectorEngine confluence logic.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from blackbox_core import NodeEngine, StrategyNode
from node_detector import NodeDetectorEngine


class TestDetectorConfluence:
    """Test confluence logic for critical-only firing."""
    
    def create_mock_node(self, critical_steps: list, optional_steps: list = None, 
                        quorum: str = "critical") -> StrategyNode:
        """Create a mock strategy node for testing."""
        if optional_steps is None:
            optional_steps = []
            
        return StrategyNode(
            name="Test Node",
            type="Test",
            description="Test node for confluence",
            workflow=[],
            validation={},
            qa_pair={},
            metadata={"confidence": "High"},
            critical_steps=critical_steps,
            optional_steps=optional_steps,
            quorum=quorum
        )
    
    def create_mock_data(self) -> dict:
        """Create mock market data for testing."""
        return {
            'liquidity_spike': 3.0,
            'gamma_pin_distance': 0.2,
            'volume': 1500,
            'price_change': 0.015,
            'dark_prints': 8,
            'block_size': 3000,
            'session_clock': 10.0  # 10:00 AM
        }
    
    def test_critical_only_quorum(self):
        """Test that critical quorum requires all critical steps to pass."""
        node_engine = NodeEngine()
        
        # Create node with critical steps
        test_node = self.create_mock_node(
            critical_steps=[
                "liquidity_spike >= 2",
                "gamma_pin_distance <= 0.25"
            ],
            optional_steps=[
                "dark_prints >= 10"  # This will fail
            ],
            quorum="critical"
        )
        
        node_engine.nodes = [test_node]
        
        # Test data where critical passes but optional fails
        test_data = self.create_mock_data()
        test_data['dark_prints'] = 3  # Fails optional condition
        
        # Evaluate confluence
        result = node_engine.evaluate_node_confluence(test_node, test_data)
        
        # Should fire because all critical steps pass
        assert result['fired'] == True
        assert result['critical_passed'] == 2
        assert result['critical_total'] == 2
        assert result['optional_passed'] == 0
        assert result['optional_total'] == 1
    
    def test_critical_failure_prevents_firing(self):
        """Test that failing critical steps prevents firing."""
        node_engine = NodeEngine()
        
        test_node = self.create_mock_node(
            critical_steps=[
                "liquidity_spike >= 5",  # This will fail
                "gamma_pin_distance <= 0.25"
            ],
            optional_steps=[
                "dark_prints >= 5"  # This will pass
            ],
            quorum="critical"
        )
        
        node_engine.nodes = [test_node]
        
        test_data = self.create_mock_data()
        test_data['liquidity_spike'] = 2.0  # Fails critical condition
        
        result = node_engine.evaluate_node_confluence(test_node, test_data)
        
        # Should not fire because critical step failed
        assert result['fired'] == False
        assert result['critical_passed'] == 1
        assert result['critical_total'] == 2
        assert result['optional_passed'] == 1
    
    def test_majority_quorum(self):
        """Test majority quorum logic."""
        node_engine = NodeEngine()
        
        test_node = self.create_mock_node(
            critical_steps=[
                "liquidity_spike >= 2",  # Pass
                "gamma_pin_distance <= 0.1"  # Fail
            ],
            optional_steps=[
                "dark_prints >= 5",  # Pass
                "volume >= 1000"  # Pass
            ],
            quorum="majority"
        )
        
        node_engine.nodes = [test_node]
        
        test_data = self.create_mock_data()
        test_data['gamma_pin_distance'] = 0.3  # Fails condition
        
        result = node_engine.evaluate_node_confluence(test_node, test_data)
        
        # Should fire because 3/4 steps pass (majority)
        assert result['fired'] == True
        assert result['critical_passed'] == 1
        assert result['optional_passed'] == 2
    
    def test_all_quorum(self):
        """Test 'all' quorum requires every step to pass."""
        node_engine = NodeEngine()
        
        test_node = self.create_mock_node(
            critical_steps=[
                "liquidity_spike >= 2",
                "gamma_pin_distance <= 0.25"
            ],
            optional_steps=[
                "dark_prints >= 5",
                "volume >= 2000"  # This will fail
            ],
            quorum="all"
        )
        
        node_engine.nodes = [test_node]
        
        test_data = self.create_mock_data()
        test_data['volume'] = 800  # Fails condition
        
        result = node_engine.evaluate_node_confluence(test_node, test_data)
        
        # Should not fire because not all steps pass
        assert result['fired'] == False
    
    def test_step_evaluation_operators(self):
        """Test different operators in step evaluation."""
        node_engine = NodeEngine()
        
        # Test various operators
        test_cases = [
            ("volume >= 1000", {"volume": 1500}, True),
            ("volume >= 1000", {"volume": 800}, False),
            ("volume <= 1000", {"volume": 800}, True),
            ("volume <= 1000", {"volume": 1200}, False),
            ("volume == 1000", {"volume": 1000}, True),
            ("volume == 1000", {"volume": 1001}, False),
            ("volume != 1000", {"volume": 1001}, True),
            ("volume != 1000", {"volume": 1000}, False),
            ("volume > 1000", {"volume": 1001}, True),
            ("volume > 1000", {"volume": 1000}, False),
            ("volume < 1000", {"volume": 999}, True),
            ("volume < 1000", {"volume": 1000}, False),
        ]
        
        for step, data, expected in test_cases:
            result = node_engine.evaluate_workflow_step(step, data)
            assert result == expected, f"Failed for {step} with data {data}"
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation with critical and optional steps."""
        node_engine = NodeEngine()
        
        test_node = self.create_mock_node(
            critical_steps=[
                "liquidity_spike >= 2",  # Pass
                "gamma_pin_distance <= 0.25"  # Pass
            ],
            optional_steps=[
                "dark_prints >= 5",  # Pass
                "volume >= 1000"  # Pass
            ],
            quorum="critical"
        )
        
        test_data = self.create_mock_data()
        
        result = node_engine.evaluate_node_confluence(test_node, test_data)
        
        # All steps pass, confidence should be high
        assert result['fired'] == True
        assert result['confidence_score'] > 0.8  # Should be high with all steps passing
        assert result['confidence_score'] <= 1.0
    
    def test_missing_data_handling(self):
        """Test handling of missing data fields."""
        node_engine = NodeEngine()
        
        test_node = self.create_mock_node(
            critical_steps=[
                "missing_field >= 100"  # Field not in data
            ],
            quorum="critical"
        )
        
        test_data = {"volume": 1000}  # Missing the required field
        
        result = node_engine.evaluate_node_confluence(test_node, test_data)
        
        # Should not fire due to missing data
        assert result['fired'] == False
        assert result['critical_passed'] == 0
    
    def test_integration_with_detector_engine(self):
        """Test integration with NodeDetectorEngine."""
        # Create mock DataFrame
        df_data = {
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'price': np.random.randn(10).cumsum() + 100,
            'close': np.random.randn(10).cumsum() + 100,
            'liquidity_spike': [3.0] * 10,
            'gamma_pin_distance': [0.2] * 10,
            'volume': [1500] * 10
        }
        
        # Create NodeEngine with test node
        node_engine = NodeEngine()
        test_node = self.create_mock_node(
            critical_steps=["liquidity_spike >= 2", "gamma_pin_distance <= 0.25"],
            quorum="critical"
        )
        node_engine.nodes = [test_node]
        
        # Create detector
        detector = NodeDetectorEngine(node_engine, df_data)
        
        # Run detection
        signals = detector.run_detection(live_output=False)
        
        # Should generate signals since critical conditions are met
        assert len(signals) > 0
        
        # Check signal properties
        first_signal = signals[0]
        assert first_signal.node_name == "Test Node"
        assert "Confluence achieved" in first_signal.trigger_reason


if __name__ == "__main__":
    pytest.main([__file__])
