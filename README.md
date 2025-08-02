# BlackBox Core Engine - Foundation Documentation

## Overview
The BlackBox Core Engine provides the foundation for a real-time, event-based trading system by parsing and structuring microstructure trading strategy nodes from text files.

## Core Components

### 1. StrategyNode Class
```python
@dataclass
class StrategyNode:
    name: str                    # Strategy title
    type: str                    # Node type and classification
    description: str             # Mentor-grade definition
    workflow: List[str]          # Atomic workflow steps
    validation: Dict[str, str]   # Valid/invalid criteria
    qa_pair: Dict[str, str]      # Mentor Q&A
    metadata: Dict[str, Any]     # Tags, relationships, confidence
    raw_content: str             # Original file content
```

### 2. NodeEngine Class
The main engine for loading and querying strategy nodes:

```python
class NodeEngine:
    def load_nodes_from_folder(folder_path: str) -> int
    def get_node_by_tag(tag: str) -> List[StrategyNode]
    def get_node_by_name(name: str) -> Optional[StrategyNode]
    def list_all_nodes() -> List[Dict[str, Any]]
    def get_high_confidence_nodes() -> List[StrategyNode]
```

### 3. StrategyNodeParser Class
Handles parsing of .txt files into structured StrategyNode objects.

## Parsed Strategy Nodes

### Successfully Loaded (4 nodes):

1. **Block Order Footprint Detection**
   - Type: Black Box / Microstructure
   - Confidence: High
   - Tags: Block, Footprint, Breakout, Absorption
   - Source: Bookmap/Footprint

2. **Fusion Node—Liquidity + Gamma + Time**
   - Type: Black Box / Microstructure  
   - Confidence: Highest
   - Tags: Fusion, Gamma, Liquidity, Time, Confluence
   - Source: DR, macro overlay backtest

3. **Dark Pool Signature / Block Print Node**
   - Type: Black Box / Microstructure
   - Confidence: High
   - Tags: Dark Pool, Block, VWAP, Tape
   - Source: DR / Tape Reading

4. **Gamma Pin / Dealer Magnet Edge**
   - Type: Black Box / Microstructure
   - Confidence: High (with data)
   - Tags: Gamma, OI, Pin, Dealer, Expiry
   - Source: Black Box DR, OI analytics

## File Structure Parsed

Each .txt file contains these sections (parsed automatically):
- **Title**: Strategy name
- **Type**: Classification and metadata
- **Mentor-Grade Definition**: Strategy description
- **Atomic Workflow / Rules**: Step-by-step implementation
- **Validation / Falsification**: Success/failure criteria
- **QA Pair (Mentor-Level)**: Knowledge validation
- **Metadata**: Tags, relationships, confidence levels

## Usage Examples

### Basic Loading
```python
from blackbox_core import NodeEngine

engine = NodeEngine()
engine.load_nodes_from_folder("/path/to/blackbox_nodes")
```

### Querying Strategies
```python
# Get gamma-related strategies
gamma_nodes = engine.get_node_by_tag("gamma")

# Get high confidence strategies
high_conf = engine.get_high_confidence_nodes()

# Get specific strategy
node = engine.get_node_by_name("Block Order Footprint Detection")
```

### Accessing Strategy Data
```python
for node in engine.nodes:
    print(f"Strategy: {node.name}")
    print(f"Workflow steps: {len(node.workflow)}")
    print(f"First step: {node.workflow[0]}")
    print(f"Success criteria: {node.validation.get('valid')}")
```

## Query Capabilities

### Tag-Based Search
- **"gamma"**: 2 nodes (Fusion Node, Gamma Pin)
- **"block"**: 2 nodes (Block Order Footprint, Dark Pool Signature)
- **"liquidity"**: 1 node (Fusion Node)
- **"dark"**: 1 node (Dark Pool Signature)

### Confidence Filtering
- **Highest**: 1 node (Fusion Node)
- **High**: 3 nodes (Block Order, Dark Pool, Gamma Pin)

### Relationship Mapping
The engine tracks parent-child and related node relationships for building strategy hierarchies.

## Integration Points

### For Strategy Detectors
```python
# Get strategies by type/tag
gamma_strategies = engine.get_node_by_tag("gamma")
for strategy in gamma_strategies:
    # Extract workflow for implementation
    steps = strategy.workflow
    confidence = strategy.metadata.get('confidence')
```

### For Risk Management
```python
# Extract exit rules and stop conditions
for node in engine.nodes:
    exit_rules = [step for step in node.workflow if 'exit' in step.lower()]
    stop_condition = node.validation.get('invalid')
```

### For Backtesting
```python
# Group strategies by confidence for testing priority
high_conf_strategies = engine.get_high_confidence_nodes()
```

## Future Extensions

The foundation is prepared for:
- Real-time strategy execution engines
- Pattern recognition modules
- Risk management systems
- Backtesting frameworks
- Strategy performance analytics

## Files Created

1. **`blackbox_core.py`** - Main engine and classes
2. **`demo.py`** - Comprehensive demonstration
3. **`examples.py`** - Integration examples
4. **`README.md`** - This documentation

All 4 strategy node .txt files have been successfully parsed and are ready for use by other trading system modules.
