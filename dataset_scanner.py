#!/usr/bin/env python3
"""
Dataset Scanner for BlackBox Trading System

Automatically discovers and manages CSV datasets in the data/csv_datasets directory.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re


class DatasetScanner:
    """Smart scanner for CSV datasets with automatic validation and selection."""
    
    def __init__(self, datasets_dir: str = "data/csv_datasets"):
        """
        Initialize the dataset scanner.
        
        Args:
            datasets_dir: Directory to scan for CSV datasets
        """
        self.datasets_dir = Path(datasets_dir)
        self.discovered_datasets = []
        
    def scan_datasets(self) -> List[Dict[str, any]]:
        """
        Scan the datasets directory for valid CSV files.
        
        Returns:
            List of dataset metadata dictionaries
        """
        datasets = []
        
        if not self.datasets_dir.exists():
            print(f"❌ Datasets directory not found: {self.datasets_dir}")
            return datasets
            
        # Scan for CSV files
        csv_files = list(self.datasets_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                # Analyze the dataset
                metadata = self._analyze_dataset(csv_file)
                if metadata:
                    datasets.append(metadata)
                    
            except Exception as e:
                print(f"⚠️  Skipping {csv_file.name}: {e}")
                
        self.discovered_datasets = datasets
        return datasets
    
    def _analyze_dataset(self, csv_path: Path) -> Optional[Dict[str, any]]:
        """
        Analyze a CSV file to extract metadata.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Dataset metadata dictionary or None if invalid
        """
        try:
            # Read a sample to validate structure
            df_sample = pd.read_csv(csv_path, nrows=10)
            
            # Extract instrument and timeframe from filename
            filename = csv_path.stem
            instrument, timeframe, date_range = self._parse_filename(filename)
            
            # Validate required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df_sample.columns]
            
            # Try alternative column names
            alt_mappings = {
                'price': 'close',
                'Price': 'close',
                'CLOSE': 'close',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Timestamp': 'timestamp',
                'Time': 'timestamp',
                'time': 'timestamp',  # Common alternative
                'DateTime': 'timestamp'
            }
            
            # Check for alternative column names
            available_cols = list(df_sample.columns)
            mapped_cols = {}
            for alt_name, standard_name in alt_mappings.items():
                if alt_name in available_cols and standard_name in missing_cols:
                    mapped_cols[alt_name] = standard_name
                    missing_cols.remove(standard_name)
            
            # Get full dataset info
            df_full = pd.read_csv(csv_path)
            total_rows = len(df_full)
            
            # Try to parse timestamps to get date range
            try:
                if 'timestamp' in df_full.columns:
                    timestamps = pd.to_datetime(df_full['timestamp'])
                elif 'Timestamp' in df_full.columns:
                    timestamps = pd.to_datetime(df_full['Timestamp'])
                elif 'Time' in df_full.columns:
                    timestamps = pd.to_datetime(df_full['Time'])
                else:
                    timestamps = None
                    
                if timestamps is not None:
                    start_date = timestamps.min().strftime('%Y-%m-%d')
                    end_date = timestamps.max().strftime('%Y-%m-%d')
                    actual_date_range = f"{start_date} to {end_date}"
                else:
                    actual_date_range = date_range or "Unknown"
                    
            except Exception:
                actual_date_range = date_range or "Unknown"
            
            return {
                'filename': csv_path.name,
                'path': str(csv_path),
                'instrument': instrument,
                'timeframe': timeframe,
                'date_range': actual_date_range,
                'total_rows': total_rows,
                'columns': available_cols,
                'missing_columns': missing_cols,
                'column_mappings': mapped_cols,
                'is_valid': len(missing_cols) == 0 or len(mapped_cols) > 0,
                'file_size_mb': round(csv_path.stat().st_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {csv_path.name}: {e}")
            return None
    
    def _parse_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        Parse instrument, timeframe, and date range from filename.
        
        Args:
            filename: CSV filename without extension
            
        Returns:
            Tuple of (instrument, timeframe, date_range)
        """
        # Common patterns for forex datasets
        patterns = [
            r'([A-Z]{6})_(\d+[HMD])_(\d{4}-\d{4})',  # EURUSD_1H_2020-2024
            r'([A-Z]{6})_(\d+[HMD])',                # EURUSD_1H
            r'([A-Z]{3}[A-Z]{3})_(\d+[HMD])_(\d{4}-\d{4})',  # Alternative format
            r'([A-Z]+)_(\w+)_(\d{4}-\d{4})',        # Generic instrument_timeframe_dates
            r'([A-Z]+)_(\w+)',                       # Generic instrument_timeframe
        ]
        
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                groups = match.groups()
                instrument = groups[0] if len(groups) > 0 else "Unknown"
                timeframe = groups[1] if len(groups) > 1 else "Unknown"
                date_range = groups[2] if len(groups) > 2 else "Unknown"
                return instrument, timeframe, date_range
        
        # Fallback: try to extract any currency pairs
        currency_match = re.search(r'([A-Z]{6})', filename)
        if currency_match:
            return currency_match.group(1), "Unknown", "Unknown"
            
        return filename, "Unknown", "Unknown"
    
    def display_datasets(self, datasets: List[Dict[str, any]]) -> None:
        """
        Display discovered datasets in a formatted table.
        
        Args:
            datasets: List of dataset metadata
        """
        if not datasets:
            print("❌ No valid datasets found in data/csv_datasets/")
            return
            
        print(f"\n📊 DISCOVERED DATASETS ({len(datasets)} found)")
        print("=" * 80)
        
        for i, dataset in enumerate(datasets, 1):
            status = "✅ VALID" if dataset['is_valid'] else "⚠️  NEEDS MAPPING"
            
            print(f"{i}. {dataset['filename']}")
            print(f"   Instrument: {dataset['instrument']} | Timeframe: {dataset['timeframe']}")
            print(f"   Date Range: {dataset['date_range']}")
            print(f"   Rows: {dataset['total_rows']:,} | Size: {dataset['file_size_mb']} MB")
            print(f"   Status: {status}")
            
            if not dataset['is_valid']:
                print(f"   Missing: {', '.join(dataset['missing_columns'])}")
                if dataset['column_mappings']:
                    print(f"   Can map: {dataset['column_mappings']}")
            
            print("-" * 40)
    
    def select_dataset(self, datasets: List[Dict[str, any]]) -> Optional[Dict[str, any]]:
        """
        Interactive dataset selection.
        
        Args:
            datasets: List of available datasets
            
        Returns:
            Selected dataset metadata or None
        """
        if not datasets:
            return None
            
        if len(datasets) == 1:
            print(f"\n🎯 Auto-selecting only available dataset: {datasets[0]['filename']}")
            return datasets[0]
        
        while True:
            try:
                print(f"\n🎯 SELECT DATASET (1-{len(datasets)}, or 'q' to quit):")
                choice = input("Enter your choice: ").strip()
                
                if choice.lower() == 'q':
                    return None
                    
                idx = int(choice) - 1
                if 0 <= idx < len(datasets):
                    selected = datasets[idx]
                    print(f"\n✅ Selected: {selected['filename']}")
                    return selected
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(datasets)}")
                    
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'q'")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                return None


def discover_and_select_dataset() -> Optional[str]:
    """
    Main function to discover and select a dataset.
    
    Returns:
        Path to selected dataset or None
    """
    scanner = DatasetScanner()
    datasets = scanner.scan_datasets()
    
    scanner.display_datasets(datasets)
    selected = scanner.select_dataset(datasets)
    
    return selected['path'] if selected else None


if __name__ == "__main__":
    # Test the scanner
    path = discover_and_select_dataset()
    if path:
        print(f"\n🚀 Would load dataset: {path}")
    else:
        print("\n❌ No dataset selected")
