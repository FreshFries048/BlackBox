"""
Example: Using BlackBox Core Engine in Other Modules

This demonstrates how other parts of the trading system can
leverage the parsed strategy nodes.
"""

from blackbox_core import NodeEngine


def example_strategy_detector():
    """Example of how a detector module might use the engine."""
    
    # Initialize and load nodes
    engine = NodeEngine()
    engine.load_nodes_from_folder("/workspaces/BlackBox/blackbox_nodes")
    
    print("🎯 EXAMPLE: Building a Strategy Detector")
    print("="*50)
    
    # Get gamma-related strategies for options trading
    gamma_strategies = engine.get_node_by_tag("gamma")
    print(f"\nFound {len(gamma_strategies)} gamma-related strategies:")
    
    for strategy in gamma_strategies:
        print(f"\n📊 {strategy.name}")
        print(f"   Confidence: {strategy.metadata.get('confidence')}")
        print(f"   Key workflow steps:")
        for i, step in enumerate(strategy.workflow[:2], 1):  # Show first 2 steps
            print(f"     {i}. {step}")
        
        # Example: Extract specific data for detector
        validation_criteria = strategy.validation
        print(f"   Success criteria: {validation_criteria.get('valid', 'N/A')}")
    
    # Example: Get high confidence block trade strategies
    print(f"\n🔍 High-confidence block trade strategies:")
    block_strategies = engine.get_node_by_tag("block")
    high_conf_blocks = [s for s in block_strategies if 'high' in s.metadata.get('confidence', '').lower()]
    
    for strategy in high_conf_blocks:
        print(f"   • {strategy.name}")
        print(f"     First action: {strategy.workflow[0] if strategy.workflow else 'N/A'}")
    
    return engine


def example_risk_manager():
    """Example of how a risk manager might use the nodes."""
    
    engine = NodeEngine()
    engine.load_nodes_from_folder("/workspaces/BlackBox/blackbox_nodes")
    
    print(f"\n⚠️  EXAMPLE: Risk Management Rules")
    print("="*50)
    
    # Extract exit rules from all strategies
    print("\nExit rules from all strategies:")
    for node in engine.nodes:
        exit_steps = [step for step in node.workflow if 'exit' in step.lower()]
        if exit_steps:
            print(f"\n{node.name}:")
            for step in exit_steps:
                print(f"   • {step}")
        
        # Extract invalidation criteria
        invalid_criteria = node.validation.get('invalid', '')
        if invalid_criteria:
            print(f"   ❌ Stop condition: {invalid_criteria}")


def example_backtester():
    """Example of how a backtester might use the nodes."""
    
    engine = NodeEngine()
    engine.load_nodes_from_folder("/workspaces/BlackBox/blackbox_nodes")
    
    print(f"\n📈 EXAMPLE: Backtesting Framework")
    print("="*50)
    
    # Group strategies by confidence for testing priority
    confidence_groups = {}
    for node in engine.nodes:
        conf = node.metadata.get('confidence', 'unknown')
        if conf not in confidence_groups:
            confidence_groups[conf] = []
        confidence_groups[conf].append(node)
    
    print("\nStrategies grouped by confidence (testing priority):")
    for confidence, strategies in sorted(confidence_groups.items(), reverse=True):
        print(f"\n{confidence.upper()}:")
        for strategy in strategies:
            print(f"   • {strategy.name}")
            print(f"     Source: {strategy.metadata.get('source', 'Unknown')}")
            print(f"     Steps: {len(strategy.workflow)}")


if __name__ == "__main__":
    print("🚀 BLACKBOX CORE ENGINE - INTEGRATION EXAMPLES")
    print("="*60)
    
    # Run examples
    engine = example_strategy_detector()
    example_risk_manager()
    example_backtester()
    
    print(f"\n✅ INTEGRATION EXAMPLES COMPLETE")
    print("="*60)
    print("These examples show how different modules can:")
    print("• Access parsed strategy knowledge")
    print("• Filter strategies by tags and confidence")
    print("• Extract specific rules for implementation")
    print("• Build detectors based on structured workflows")
