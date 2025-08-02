"""
BlackBox Phase 2 Demo - Simplified NodeDetectorEngine Test

This script demonstrates the key features of the NodeDetectorEngine
with focused examples and cleaner output.
"""

import pandas as pd
import numpy as np
from blackbox_core import NodeEngine
from node_detector import NodeDetectorEngine, SignalEvent


def create_strategic_market_data():
    """Create realistic market data with strategic patterns for testing."""
    
    print("📊 Creating strategic market data...")
    
    # Create 50 data points for focused testing
    timestamps = pd.date_range('2024-08-02 09:30:00', periods=50, freq='2min')
    base_price = 4100.0
    
    # Create realistic price movement
    np.random.seed(42)
    price_changes = np.random.randn(50) * 0.3  # Smaller moves
    prices = base_price + np.cumsum(price_changes)
    
    data = {
        'timestamp': timestamps,
        'price': prices,
        'volume': np.random.randint(2000, 8000, 50),
        'gamma': np.random.randint(10, 500, 50),
        'dark_prints': np.random.randint(0, 3, 50),
        'footprint_clusters': np.random.randint(0, 2, 50),
        'block_size': np.zeros(50),  # Start with no blocks
    }
    
    # Add strategic events at specific times
    strategic_events = [
        (5, 'gamma_spike', {'gamma': 2000, 'block_size': 0}),
        (12, 'block_trade', {'block_size': 8000, 'dark_prints': 10}),
        (20, 'fusion_event', {'gamma': 1500, 'block_size': 5000, 'footprint_clusters': 3}),
        (30, 'dark_pool', {'dark_prints': 15, 'block_size': 3000}),
        (40, 'gamma_pin', {'gamma': 3000, 'volume': 15000}),
    ]
    
    for idx, event_type, values in strategic_events:
        for key, value in values.items():
            data[key][idx] = value
        print(f"   Added {event_type} at index {idx} (time: {timestamps[idx].strftime('%H:%M')})")
    
    return data


def demonstrate_detection_engine():
    """Demonstrate the NodeDetectorEngine with focused output."""
    
    print("🔧 BLACKBOX PHASE 2: FOCUSED DETECTOR DEMO")
    print("="*60)
    
    # 1. Load strategy nodes
    print("\n1. Loading strategy nodes...")
    node_engine = NodeEngine()
    node_engine.load_nodes_from_folder("/workspaces/BlackBox/blackbox_nodes")
    
    # 2. Create strategic market data
    print("\n2. Creating strategic market data...")
    market_data = create_strategic_market_data()
    
    # 3. Initialize detector
    print("\n3. Initializing NodeDetectorEngine...")
    detector = NodeDetectorEngine(node_engine, market_data)
    
    # 4. Run detection with focused output
    print("\n4. Running focused signal detection...")
    print("   (Showing only key signals for clarity)")
    print("-" * 60)
    
    # Run detection but capture signals without live output
    signals = detector.run_detection(live_output=False)
    
    # Filter and show only the most interesting signals
    high_value_signals = []
    for signal in signals:
        # Show signals with 4+ matches or from strategic events
        if signal.workflow_matches >= 4 or 'Large print/block detected' in signal.trigger_reason:
            high_value_signals.append(signal)
    
    print(f"\nFiltered to {len(high_value_signals)} high-value signals:")
    print("="*60)
    
    for i, signal in enumerate(high_value_signals[:10], 1):  # Show first 10
        print(f"{i}. 🚨 {signal.node_name}")
        print(f"   Time: {signal.timestamp}")
        print(f"   Price: ${signal.entry_price:.2f}")
        print(f"   Confidence: {signal.confidence}")
        print(f"   Workflow Matches: {signal.workflow_matches}/{signal.metadata.get('total_workflow_steps', 'N/A')}")
        print(f"   Reason: {signal.trigger_reason[:80]}...")
        print(f"   Tags: {', '.join(signal.tags[:3])}{'...' if len(signal.tags) > 3 else ''}")
        print("-" * 60)
    
    # 5. Show statistics
    print("\n5. Detection Statistics")
    print("="*40)
    detector.print_detection_stats()
    
    # 6. Node performance analysis
    print("\n6. Node Performance Analysis")
    print("="*40)
    
    node_performance = {}
    for signal in signals:
        node_name = signal.node_name
        if node_name not in node_performance:
            node_performance[node_name] = {
                'count': 0,
                'avg_matches': 0,
                'confidence': signal.confidence
            }
        node_performance[node_name]['count'] += 1
        node_performance[node_name]['avg_matches'] += signal.workflow_matches
    
    # Calculate averages
    for node_name, stats in node_performance.items():
        stats['avg_matches'] = stats['avg_matches'] / stats['count']
    
    print("Node trigger summary:")
    for node_name, stats in sorted(node_performance.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"• {node_name}")
        print(f"  Signals: {stats['count']}")
        print(f"  Avg workflow matches: {stats['avg_matches']:.1f}")
        print(f"  Confidence: {stats['confidence']}")
        print()
    
    # 7. Export results
    print("7. Exporting results...")
    csv_path = detector.export_signals_to_csv("focused_signal_log.csv")
    
    return detector, signals


def demonstrate_signal_structure():
    """Show the structure of a SignalEvent."""
    
    print("\n📋 SIGNAL EVENT STRUCTURE")
    print("="*40)
    
    # Create a sample signal for demonstration
    sample_signal = SignalEvent(
        node_name="Block Order Footprint Detection",
        timestamp="2024-08-02 10:30:00",
        trigger_reason="4/4 conditions met: Large print detected, Volume spike, Price movement",
        confidence="High",
        entry_price=4102.25,
        node_type="Black Box / Microstructure",
        tags=["Block", "Footprint", "Breakout"],
        workflow_matches=4,
        validation_status="triggered",
        metadata={
            'total_workflow_steps': 4,
            'volume': 12500,
            'block_size': 8000
        }
    )
    
    print("SignalEvent attributes:")
    print(f"• node_name: {sample_signal.node_name}")
    print(f"• timestamp: {sample_signal.timestamp}")
    print(f"• trigger_reason: {sample_signal.trigger_reason}")
    print(f"• confidence: {sample_signal.confidence}")
    print(f"• entry_price: ${sample_signal.entry_price}")
    print(f"• node_type: {sample_signal.node_type}")
    print(f"• tags: {sample_signal.tags}")
    print(f"• workflow_matches: {sample_signal.workflow_matches}")
    print(f"• validation_status: {sample_signal.validation_status}")
    print(f"• metadata: {sample_signal.metadata}")


if __name__ == "__main__":
    # Run the main demonstration
    detector, signals = demonstrate_detection_engine()
    
    # Show signal structure
    demonstrate_signal_structure()
    
    print(f"\n✅ PHASE 2 DEMONSTRATION COMPLETE")
    print("="*60)
    print("The NodeDetectorEngine successfully demonstrates:")
    print("• Real-time strategy node evaluation against market data")
    print("• Signal generation when node conditions are met")
    print("• Structured SignalEvent objects with full metadata")
    print("• CSV export for further analysis")
    print("• Modular design ready for execution integration")
    print("\nReady for Phase 3: Trade Execution Engine!")
