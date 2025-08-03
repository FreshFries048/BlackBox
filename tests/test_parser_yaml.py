"""
Test suite for BlackBox YAML parser and critical/optional step separation.
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from blackbox_core import StrategyNodeParser, NodeEngine, StrategyNode


class TestYamlParser:
    """Test YAML front-matter parsing in strategy nodes."""
    
    def create_temp_strategy_file(self, content: str) -> str:
        """Create a temporary strategy file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_yaml_frontmatter_parsing(self):
        """Test that YAML front-matter is correctly parsed."""
        yaml_content = """---
id: LBlack_TestStrategy
critical_steps:
  - liquidity_spike >= 2
  - gamma_pin_distance <= 0.25
optional_steps:
  - tape_delta >= 1000
quorum: critical
---
Title: Test Strategy
Type: Test Node
Version: 1.0
Date: 2025-01-01
Status: Test

---

Mentor-Grade Definition:
This is a test strategy for YAML parsing.

---

Atomic Workflow / Rules:
1. Check liquidity spike condition
2. Verify gamma pin distance
3. Optional tape delta confirmation

---

Validation / Falsification:
- Valid: All critical conditions met
- Invalid: Missing critical conditions

---

QA Pair (Mentor-Level):
Q: How does this work?
A: It tests YAML parsing.

---

Metadata:
Confidence: High
Tags: Test, YAML
"""
        
        temp_file = self.create_temp_strategy_file(yaml_content)
        
        try:
            parser = StrategyNodeParser()
            node = parser.parse_file(temp_file)
            
            assert node is not None
            assert node.name == "Test Strategy"
            assert len(node.critical_steps) == 2
            assert "liquidity_spike >= 2" in node.critical_steps
            assert "gamma_pin_distance <= 0.25" in node.critical_steps
            assert len(node.optional_steps) == 1
            assert "tape_delta >= 1000" in node.optional_steps
            assert node.quorum == "critical"
            
        finally:
            os.unlink(temp_file)
    
    def test_legacy_format_fallback(self):
        """Test that legacy format without YAML still works."""
        legacy_content = """Title: Legacy Strategy
Type: Legacy Node
Version: 1.0
Date: 2025-01-01
Status: Test

---

Mentor-Grade Definition:
This is a legacy strategy without YAML.

---

Atomic Workflow / Rules:
1. Legacy workflow step one
2. Legacy workflow step two

---

Validation / Falsification:
- Valid: Legacy validation
- Invalid: Legacy invalidation

---

QA Pair (Mentor-Level):
Q: How does legacy work?
A: It falls back to workflow.

---

Metadata:
Confidence: Medium
Tags: Legacy, Test
"""
        
        temp_file = self.create_temp_strategy_file(legacy_content)
        
        try:
            parser = StrategyNodeParser()
            node = parser.parse_file(temp_file)
            
            assert node is not None
            assert node.name == "Legacy Strategy"
            # Legacy format should use workflow as critical steps
            assert len(node.critical_steps) == 2
            assert "Legacy workflow step one" in node.critical_steps
            assert "Legacy workflow step two" in node.critical_steps
            assert len(node.optional_steps) == 0
            assert node.quorum == "critical"
            
        finally:
            os.unlink(temp_file)
    
    def test_critical_optional_separation(self):
        """Test that critical and optional steps are properly separated."""
        yaml_content = """---
critical_steps:
  - volume >= 1000
  - price_change >= 0.01
optional_steps:
  - dark_prints >= 5
  - block_size >= 2000
quorum: critical
---
Title: Separation Test
Type: Test Node

---

Mentor-Grade Definition:
Tests critical/optional separation.

---

Atomic Workflow / Rules:
1. Check volume threshold
2. Verify price movement
3. Look for dark prints
4. Check block size

---

Metadata:
Confidence: High
Tags: Test
"""
        
        temp_file = self.create_temp_strategy_file(yaml_content)
        
        try:
            parser = StrategyNodeParser()
            node = parser.parse_file(temp_file)
            
            assert node is not None
            assert len(node.critical_steps) == 2
            assert len(node.optional_steps) == 2
            
            # Verify critical steps
            assert "volume >= 1000" in node.critical_steps
            assert "price_change >= 0.01" in node.critical_steps
            
            # Verify optional steps
            assert "dark_prints >= 5" in node.optional_steps
            assert "block_size >= 2000" in node.optional_steps
            
        finally:
            os.unlink(temp_file)
    
    def test_different_quorum_rules(self):
        """Test different quorum rule parsing."""
        quorum_tests = [
            ("critical", "critical"),
            ("majority", "majority"),
            ("all", "all")
        ]
        
        for test_quorum, expected_quorum in quorum_tests:
            yaml_content = f"""---
critical_steps:
  - test_condition >= 1
quorum: {test_quorum}
---
Title: Quorum Test
Type: Test Node

---

Mentor-Grade Definition:
Tests quorum rules.

---

Metadata:
Confidence: High
Tags: Test
"""
            
            temp_file = self.create_temp_strategy_file(yaml_content)
            
            try:
                parser = StrategyNodeParser()
                node = parser.parse_file(temp_file)
                
                assert node is not None
                assert node.quorum == expected_quorum
                
            finally:
                os.unlink(temp_file)
    
    def test_malformed_yaml_fallback(self):
        """Test that malformed YAML falls back gracefully."""
        malformed_content = """---
critical_steps:
  - invalid: yaml: structure
  malformed
---
Title: Malformed YAML Test
Type: Test Node

---

Mentor-Grade Definition:
Tests malformed YAML handling.

---

Atomic Workflow / Rules:
1. Fallback workflow step

---

Metadata:
Confidence: Low
Tags: Test
"""
        
        temp_file = self.create_temp_strategy_file(malformed_content)
        
        try:
            parser = StrategyNodeParser()
            node = parser.parse_file(temp_file)
            
            # Should still parse successfully, falling back to workflow
            assert node is not None
            assert node.name == "Malformed YAML Test"
            # Should fall back to using workflow as critical steps
            assert len(node.critical_steps) == 1
            assert "Fallback workflow step" in node.critical_steps
            
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
