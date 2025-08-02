"""
BlackBox Core Engine - Strategy Node Parser and Engine

This module provides the foundation for a real-time, event-based trading system
by parsing and structuring microstructure trading strategy nodes.
"""

import os
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StrategyNode:
    """
    Represents a single microstructure trading strategy node.
    
    Attributes:
        name: Strategy title/name
        type: Node type and classification
        description: Mentor-grade definition of the strategy
        workflow: List of atomic workflow steps/rules
        validation: Validation and falsification criteria
        qa_pair: Question and answer pair for mentor-level understanding
        metadata: Dictionary containing tags, relationships, confidence, etc.
        raw_content: Original file content for reference
    """
    name: str
    type: str
    description: str
    workflow: List[str]
    validation: Dict[str, str]
    qa_pair: Dict[str, str]
    metadata: Dict[str, Any]
    raw_content: str = ""
    
    def __post_init__(self):
        """Ensure workflow is a list and validation/qa_pair are dicts"""
        if isinstance(self.workflow, str):
            self.workflow = [self.workflow]
        if not isinstance(self.validation, dict):
            self.validation = {"criteria": str(self.validation)}
        if not isinstance(self.qa_pair, dict):
            self.qa_pair = {"qa": str(self.qa_pair)}


class NodeEngine:
    """
    Engine for loading, storing, and accessing strategy nodes.
    
    This class manages the collection of strategy nodes and provides
    methods for querying and filtering them.
    """
    
    def __init__(self):
        self.nodes: List[StrategyNode] = []
        self._parser = StrategyNodeParser()
    
    def load_nodes_from_folder(self, folder_path: str) -> int:
        """
        Load all strategy nodes from .txt files in the specified folder.
        
        Args:
            folder_path: Path to folder containing .txt node files
            
        Returns:
            Number of nodes successfully loaded
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        loaded_count = 0
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        
        for filename in txt_files:
            file_path = os.path.join(folder_path, filename)
            try:
                node = self._parser.parse_file(file_path)
                if node:
                    self.nodes.append(node)
                    loaded_count += 1
                    print(f"✓ Loaded node: {node.name}")
                else:
                    print(f"✗ Failed to parse: {filename}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {str(e)}")
        
        print(f"\nSuccessfully loaded {loaded_count} strategy nodes.")
        return loaded_count
    
    def get_node_by_tag(self, tag: str) -> List[StrategyNode]:
        """
        Return nodes that contain the specified tag in their metadata.
        
        Args:
            tag: Tag to search for (case-insensitive)
            
        Returns:
            List of matching strategy nodes
        """
        matching_nodes = []
        tag_lower = tag.lower()
        
        for node in self.nodes:
            node_tags = node.metadata.get('tags', [])
            if isinstance(node_tags, str):
                node_tags = [t.strip() for t in node_tags.split(',')]
            
            # Check if any tag matches (case-insensitive)
            if any(tag_lower in node_tag.lower() for node_tag in node_tags):
                matching_nodes.append(node)
        
        return matching_nodes
    
    def get_node_by_name(self, name: str) -> Optional[StrategyNode]:
        """
        Get a node by its exact name.
        
        Args:
            name: Exact name of the strategy node
            
        Returns:
            Strategy node if found, None otherwise
        """
        for node in self.nodes:
            if node.name.lower() == name.lower():
                return node
        return None
    
    def list_all_nodes(self) -> List[Dict[str, Any]]:
        """
        Return a summary list of all loaded nodes.
        
        Returns:
            List of dictionaries with node summaries
        """
        summaries = []
        for node in self.nodes:
            summary = {
                'name': node.name,
                'type': node.type,
                'confidence': node.metadata.get('confidence', 'Unknown'),
                'tags': node.metadata.get('tags', ''),
                'workflow_steps': len(node.workflow),
                'status': node.metadata.get('status', 'Unknown')
            }
            summaries.append(summary)
        
        return summaries
    
    def get_high_confidence_nodes(self) -> List[StrategyNode]:
        """Get nodes with high or highest confidence ratings."""
        high_conf_nodes = []
        for node in self.nodes:
            confidence = node.metadata.get('confidence', '').lower()
            if 'high' in confidence:
                high_conf_nodes.append(node)
        return high_conf_nodes
    
    def print_summary(self):
        """Print a formatted summary of all loaded nodes."""
        print(f"\n{'='*60}")
        print(f"BLACKBOX CORE ENGINE - NODE SUMMARY")
        print(f"{'='*60}")
        print(f"Total Nodes Loaded: {len(self.nodes)}")
        print(f"{'='*60}")
        
        for i, node in enumerate(self.nodes, 1):
            tags = node.metadata.get('tags', [])
            if isinstance(tags, list):
                tags_str = ', '.join(tags)
            else:
                tags_str = str(tags)
            
            print(f"\n{i}. {node.name}")
            print(f"   Type: {node.type}")
            print(f"   Confidence: {node.metadata.get('confidence', 'Unknown')}")
            print(f"   Tags: {tags_str if tags_str else 'None'}")
            print(f"   Workflow Steps: {len(node.workflow)}")
            print(f"   Status: {node.metadata.get('status', 'Unknown')}")


class StrategyNodeParser:
    """
    Parser for extracting structured data from strategy node .txt files.
    
    This class handles the parsing logic to convert raw text files into
    structured StrategyNode objects.
    """
    
    def __init__(self):
        self.section_patterns = {
            'title': r'^Title:\s*(.+)$',
            'type': r'^Type:\s*(.+)$',
            'version': r'^Version:\s*(.+)$',
            'date': r'^Date:\s*(.+)$',
            'status': r'^Status:\s*(.+)$',
        }
    
    def parse_file(self, file_path: str) -> Optional[StrategyNode]:
        """
        Parse a strategy node file into a StrategyNode object.
        
        Args:
            file_path: Path to the .txt file to parse
            
        Returns:
            StrategyNode object if successful, None otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._parse_content(content)
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {str(e)}")
            return None
    
    def _parse_content(self, content: str) -> Optional[StrategyNode]:
        """Parse the content of a strategy node file."""
        try:
            # Split content into sections based on --- separators
            sections = content.split('---')
            
            # Extract basic metadata from the first section
            header_section = sections[0] if sections else ""
            metadata = self._parse_header(header_section)
            
            # Extract main content sections
            name = metadata.get('title', 'Unknown Strategy')
            node_type = metadata.get('type', 'Unknown Type')
            
            # Parse each section
            description = self._extract_section(content, "Mentor-Grade Definition:")
            workflow = self._extract_workflow(content)
            validation = self._extract_validation(content)
            qa_pair = self._extract_qa_pair(content)
            
            # Extract additional metadata
            metadata.update(self._extract_metadata_section(content))
            
            return StrategyNode(
                name=name,
                type=node_type,
                description=description,
                workflow=workflow,
                validation=validation,
                qa_pair=qa_pair,
                metadata=metadata,
                raw_content=content
            )
            
        except Exception as e:
            print(f"Error parsing content: {str(e)}")
            return None
    
    def _parse_header(self, header_section: str) -> Dict[str, str]:
        """Parse the header section for basic metadata."""
        metadata = {}
        
        for line in header_section.split('\n'):
            line = line.strip()
            for key, pattern in self.section_patterns.items():
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    metadata[key] = match.group(1).strip()
        
        return metadata
    
    def _extract_section(self, content: str, section_header: str) -> str:
        """Extract content from a specific section."""
        pattern = rf"{re.escape(section_header)}\s*\n(.+?)(?=\n\s*---|\n\s*[A-Z][^:]*:|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_workflow(self, content: str) -> List[str]:
        """Extract and parse the atomic workflow steps."""
        workflow_text = self._extract_section(content, "Atomic Workflow / Rules:")
        
        if not workflow_text:
            return []
        
        # Split by numbered steps and clean up
        steps = []
        lines = workflow_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and leading dashes
                step = re.sub(r'^\d+\.\s*', '', line)
                step = re.sub(r'^-\s*', '', step)
                if step:
                    steps.append(step)
        
        return steps
    
    def _extract_validation(self, content: str) -> Dict[str, str]:
        """Extract validation and falsification criteria."""
        validation_text = self._extract_section(content, "Validation / Falsification:")
        
        if not validation_text:
            return {}
        
        validation = {}
        lines = validation_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('- Valid:'):
                validation['valid'] = line.replace('- Valid:', '').strip()
            elif line.startswith('- Invalid:'):
                validation['invalid'] = line.replace('- Invalid:', '').strip()
        
        return validation
    
    def _extract_qa_pair(self, content: str) -> Dict[str, str]:
        """Extract the QA pair from the mentor-level section."""
        qa_text = self._extract_section(content, "QA Pair (Mentor-Level):")
        
        if not qa_text:
            return {}
        
        qa_pair = {}
        lines = qa_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                qa_pair['question'] = line.replace('Q:', '').strip()
            elif line.startswith('A:'):
                qa_pair['answer'] = line.replace('A:', '').strip()
        
        return qa_pair
    
    def _extract_metadata_section(self, content: str) -> Dict[str, Any]:
        """Extract the metadata section with tags, relationships, etc."""
        # Use a more specific pattern to capture the Metadata section
        pattern = r"Metadata:\s*\n(.*?)(?=\n\s*---|\n\s*Pending|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if not match:
            return {}
        
        metadata_text = match.group(1).strip()
        metadata = {}
        lines = metadata_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith('---'):
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Handle special cases
                if key == 'tags':
                    metadata[key] = [tag.strip() for tag in value.split(',')]
                elif key in ['related', 'children', 'parent']:
                    metadata[key] = [item.strip() for item in value.split(',')]
                else:
                    metadata[key] = value
        
        return metadata


# Example usage and testing
if __name__ == "__main__":
    # Create the engine and load nodes
    engine = NodeEngine()
    
    # Load nodes from the blackbox_nodes folder
    folder_path = "/workspaces/BlackBox/blackbox_nodes"
    loaded_count = engine.load_nodes_from_folder(folder_path)
    
    # Print detailed summary
    engine.print_summary()
    
    # Test some queries
    print(f"\n{'='*60}")
    print("TESTING NODE QUERIES")
    print(f"{'='*60}")
    
    # Test tag search
    gamma_nodes = engine.get_node_by_tag("gamma")
    print(f"\nNodes with 'gamma' tag: {len(gamma_nodes)}")
    for node in gamma_nodes:
        print(f"  - {node.name}")
    
    # Test high confidence nodes
    high_conf = engine.get_high_confidence_nodes()
    print(f"\nHigh confidence nodes: {len(high_conf)}")
    for node in high_conf:
        print(f"  - {node.name} ({node.metadata.get('confidence')})")
    
    # Test specific node lookup
    block_node = engine.get_node_by_name("Block Order Footprint Detection")
    if block_node:
        print(f"\nFound node: {block_node.name}")
        print(f"Workflow steps: {len(block_node.workflow)}")
        print("First workflow step:", block_node.workflow[0] if block_node.workflow else "None")
