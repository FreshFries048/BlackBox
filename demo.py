"""
BlackBox Core Engine - Demonstration Script

This script demonstrates the capabilities of the BlackBox Core Engine
by showing detailed information about parsed strategy nodes.
"""

from blackbox_core import NodeEngine, StrategyNode
import json


def demonstrate_node_parsing():
    """Demonstrate the node parsing and querying capabilities."""
    
    print("🔧 BLACKBOX CORE ENGINE DEMONSTRATION")
    print("="*60)
    
    # Initialize the engine
    engine = NodeEngine()
    
    # Load nodes
    print("\n📁 Loading strategy nodes from folder...")
    folder_path = "/workspaces/BlackBox/blackbox_nodes"
    loaded_count = engine.load_nodes_from_folder(folder_path)
    
    # Show basic summary
    print(f"\n📊 Successfully loaded {loaded_count} strategy nodes")
    
    # Demonstrate detailed node information
    print("\n🔍 DETAILED NODE ANALYSIS")
    print("="*60)
    
    for i, node in enumerate(engine.nodes, 1):
        print(f"\n{i}. {node.name}")
        print("-" * (len(node.name) + 3))
        
        print(f"Type: {node.type}")
        print(f"Status: {node.metadata.get('status', 'Unknown')}")
        print(f"Confidence: {node.metadata.get('confidence', 'Unknown')}")
        print(f"Source: {node.metadata.get('source', 'Unknown')}")
        
        # Show description
        print(f"\nDescription:")
        print(f"  {node.description}")
        
        # Show workflow
        print(f"\nWorkflow ({len(node.workflow)} steps):")
        for j, step in enumerate(node.workflow, 1):
            print(f"  {j}. {step}")
        
        # Show validation criteria
        print(f"\nValidation:")
        for key, value in node.validation.items():
            print(f"  {key.title()}: {value}")
        
        # Show QA pair
        print(f"\nMentor Q&A:")
        if 'question' in node.qa_pair:
            print(f"  Q: {node.qa_pair['question']}")
        if 'answer' in node.qa_pair:
            print(f"  A: {node.qa_pair['answer']}")
        
        # Show relationships
        print(f"\nRelationships:")
        parent = node.metadata.get('parent', '')
        if parent:
            print(f"  Parent: {parent}")
        
        children = node.metadata.get('children', [])
        if children:
            if isinstance(children, list):
                print(f"  Children: {', '.join(children)}")
            else:
                print(f"  Children: {children}")
        
        related = node.metadata.get('related', [])
        if related:
            if isinstance(related, list):
                print(f"  Related: {', '.join(related)}")
            else:
                print(f"  Related: {related}")
        
        # Show tags
        tags = node.metadata.get('tags', [])
        if tags:
            if isinstance(tags, list):
                print(f"  Tags: {', '.join(tags)}")
            else:
                print(f"  Tags: {tags}")
        
        print()  # Add spacing between nodes
    
    # Demonstrate querying capabilities
    print("\n🔎 QUERYING CAPABILITIES")
    print("="*60)
    
    # Search by tags
    print("\n1. Searching by tags:")
    test_tags = ["gamma", "block", "liquidity", "dark"]
    
    for tag in test_tags:
        matching_nodes = engine.get_node_by_tag(tag)
        print(f"   '{tag}' tag: {len(matching_nodes)} nodes")
        for node in matching_nodes:
            print(f"     - {node.name}")
    
    # High confidence nodes
    print("\n2. High confidence strategies:")
    high_conf_nodes = engine.get_high_confidence_nodes()
    for node in high_conf_nodes:
        confidence = node.metadata.get('confidence', 'Unknown')
        print(f"   - {node.name} ({confidence})")
    
    # Node relationships
    print("\n3. Node relationship mapping:")
    for node in engine.nodes:
        parent = node.metadata.get('parent', '')
        children = node.metadata.get('children', [])
        related = node.metadata.get('related', [])
        
        if parent or children or related:
            print(f"   {node.name}:")
            if parent:
                print(f"     ↑ Parent: {parent}")
            if children:
                if isinstance(children, list):
                    for child in children:
                        print(f"     ↓ Child: {child}")
                else:
                    print(f"     ↓ Child: {children}")
            if related:
                if isinstance(related, list):
                    for rel in related:
                        print(f"     ↔ Related: {rel}")
                else:
                    print(f"     ↔ Related: {related}")
    
    print("\n✅ DEMONSTRATION COMPLETE")
    print("="*60)
    print("The BlackBox Core Engine successfully:")
    print("• Parsed 4 strategy node files")
    print("• Extracted structured data (workflow, validation, metadata)")
    print("• Enabled tag-based and relationship-based querying")
    print("• Prepared foundation for future trading system development")


if __name__ == "__main__":
    demonstrate_node_parsing()
