"""
BlackBox Phase 2 Demo - Simplified NodeDetectorEngine Test

This script demonstrates the key features of the NodeDetectorEngine
with focused examples and cleaner output.
"""

import pandas as pd
import numpy as np
from blackbox_core import NodeEngine
from node_detector import NodeDetectorEngine, SignalEvent


def load_real_eurusd_data_focused(num_points: int = 100):
    """Load focused real EUR/USD data for signal detection testing."""
    
    print("📊 Loading real EUR/USD data for focused testing...")
    
    try:
        # Load the real EUR/USD data
        df = pd.read_csv('/workspaces/BlackBox/EURUSD_1H_2020-2024.csv')
        print(f"   ✅ Loaded {len(df)} rows of real EUR/USD data")
        
        # Convert timestamp column
        df['time'] = pd.to_datetime(df['time'])
        
        # Take a focused subset for testing (recent volatile period)
        df_subset = df.tail(num_points).copy().reset_index(drop=True)
        
        print(f"   📈 Using {len(df_subset)} data points from {df_subset['time'].iloc[0]} to {df_subset['time'].iloc[-1]}")
        
        # Calculate enhanced indicators for signal detection
        df_subset['price_change'] = df_subset['close'].pct_change()
        df_subset['volatility'] = df_subset['high'] - df_subset['low']
        df_subset['volume_ma'] = df_subset['tick_volume'].rolling(window=5).mean()
        df_subset['volume_ratio'] = df_subset['tick_volume'] / df_subset['volume_ma']
        
        # Create enhanced synthetic indicators based on real market patterns
        np.random.seed(42)  # Consistent results
        
        # Gamma exposure based on volatility and volume
        base_gamma = (df_subset['volatility'] * 5000).fillna(100)
        df_subset['gamma'] = (base_gamma + np.random.randint(10, 200, len(df_subset))).astype(int)
        
        # Dark pool activity during volume spikes
        df_subset['dark_prints'] = np.where(
            df_subset['volume_ratio'].fillna(1) > 1.3,
            np.random.randint(5, 15, len(df_subset)),
            np.random.randint(0, 3, len(df_subset))
        )
        
        # Block trades during significant movements
        significant_moves = abs(df_subset['price_change'].fillna(0)) > df_subset['price_change'].std()
        df_subset['block_size'] = np.where(
            significant_moves,
            np.random.randint(3000, 12000, len(df_subset)),
            np.random.randint(0, 1000, len(df_subset))
        )
        
        # Footprint clusters
        df_subset['footprint_clusters'] = np.random.randint(0, 4, len(df_subset))
        
        # Add strategic events at high volatility points
        high_vol_indices = df_subset.nlargest(5, 'volatility').index.tolist()
        for i, idx in enumerate(high_vol_indices[:3]):
            if i == 0:
                df_subset.loc[idx, 'gamma'] = 2500
                print(f"   🎯 Enhanced gamma spike at {df_subset.loc[idx, 'time']}")
            elif i == 1:
                df_subset.loc[idx, 'block_size'] = 15000
                df_subset.loc[idx, 'dark_prints'] = 12
                print(f"   🎯 Enhanced block trade at {df_subset.loc[idx, 'time']}")
            else:
                df_subset.loc[idx, 'gamma'] = 1800
                df_subset.loc[idx, 'block_size'] = 8000
                df_subset.loc[idx, 'footprint_clusters'] = 4
                print(f"   🎯 Enhanced fusion event at {df_subset.loc[idx, 'time']}")
        
        # Prepare data dictionary
        data = {
            'timestamp': df_subset['time'],
            'price': df_subset['close'],
            'volume': df_subset['tick_volume'],
            'gamma': df_subset['gamma'],
            'dark_prints': df_subset['dark_prints'],
            'footprint_clusters': df_subset['footprint_clusters'],
            'block_size': df_subset['block_size'],
            'volatility': df_subset['volatility'],
            'spread': df_subset['spread']
        }
        
        return data
        
    except FileNotFoundError:
        print("❌ EUR/USD CSV file not found. Please ensure 'EURUSD_1H_2020-2024.csv' exists.")
        raise
    except Exception as e:
        print(f"❌ Error loading EUR/USD data: {str(e)}")
        raise
    
    return data


def demonstrate_detection_engine():
    """Demonstrate the NodeDetectorEngine with focused output."""
    
    print("🔧 BLACKBOX PHASE 2: FOCUSED DETECTOR DEMO")
    print("="*60)
    
    # 1. Load strategy nodes
    print("\n1. Loading strategy nodes...")
    node_engine = NodeEngine()
    node_engine.load_nodes_from_folder("/workspaces/BlackBox/blackbox_nodes")
    
    # 2. Load real EUR/USD market data
    print("\n2. Loading real EUR/USD market data...")
    market_data = load_real_eurusd_data_focused(100)
    
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
