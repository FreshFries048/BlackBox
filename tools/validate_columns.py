#!/usr/bin/env python3
"""
CLI tool to validate that all strategy nodes have required metrics in dataset.
"""
import sys
import argparse
from pathlib import Path
from typing import Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
# Import directly from the module file
import blackbox_core as bc
from blackbox_core_pkg.exceptions import MissingFeatureError

NodeEngine = bc.NodeEngine


def validate_dataset_columns(data_path: Path, nodes_dir: Path) -> bool:
    """
    Validate that dataset contains all required metrics for strategy nodes.
    
    Args:
        data_path: Path to dataset file (CSV/Parquet)
        nodes_dir: Path to strategy nodes directory
        
    Returns:
        True if all metrics are present, False otherwise
    """
    print(f"🔍 Validating dataset: {data_path}")
    print(f"📁 Using strategy nodes from: {nodes_dir}")
    
    # Load dataset
    try:
        if data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        
        data_columns = set(df.columns)
        print(f"📊 Dataset columns ({len(data_columns)}): {sorted(data_columns)}")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Load strategy nodes
    try:
        node_engine = NodeEngine()
        loaded_count = node_engine.load_nodes_from_folder(str(nodes_dir))
        print(f"🎯 Loaded {loaded_count} strategy nodes")
        
    except Exception as e:
        print(f"❌ Failed to load strategy nodes: {e}")
        return False
    
    # Validate each node
    validation_passed = True
    missing_metrics = {}
    
    for node in node_engine.nodes:
        print(f"\n🔧 Validating node: {node.name}")
        
        # Extract required metrics from workflow steps
        required_metrics = set()
        for step in node.workflow_steps:
            for condition in step.conditions:
                required_metrics.add(condition.metric)
        
        print(f"   Required metrics: {sorted(required_metrics)}")
        
        # Check for missing metrics
        node_missing = required_metrics - data_columns
        if node_missing:
            validation_passed = False
            missing_metrics[node.name] = list(node_missing)
            print(f"   ❌ Missing metrics: {sorted(node_missing)}")
        else:
            print(f"   ✅ All required metrics present")
    
    # Summary
    print(f"\n{'='*60}")
    if validation_passed:
        print("✅ VALIDATION PASSED: All strategy nodes have required metrics")
        return True
    else:
        print("❌ VALIDATION FAILED: Missing metrics detected")
        print("\nMissing metrics by node:")
        for node_name, metrics in missing_metrics.items():
            print(f"  • {node_name}: {', '.join(sorted(metrics))}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate dataset columns against strategy node requirements'
    )
    parser.add_argument(
        'data_path', 
        type=Path,
        help='Path to dataset file (CSV or Parquet)'
    )
    parser.add_argument(
        '--nodes-dir',
        type=Path,
        default=Path('blackbox_nodes'),
        help='Path to strategy nodes directory (default: blackbox_nodes)'
    )
    
    args = parser.parse_args()
    
    if not args.data_path.exists():
        print(f"❌ Dataset file not found: {args.data_path}")
        sys.exit(1)
    
    if not args.nodes_dir.exists():
        print(f"❌ Strategy nodes directory not found: {args.nodes_dir}")
        sys.exit(1)
    
    success = validate_dataset_columns(args.data_path, args.nodes_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
