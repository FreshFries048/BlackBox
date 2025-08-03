"""
Loss-Free Result Storage with Parquet and Database
"""
import os
import json
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import uuid


class ResultWriter:
    """Handles persistent storage of backtest results."""
    
    def __init__(self, base_results_dir: str = "results", db_path: str = "storage/blackbox.db"):
        self.base_results_dir = Path(base_results_dir)
        self.db_path = Path(db_path)
        self._ensure_directories()
        self._init_database()
    
    def _ensure_directories(self):
        """Ensure results and storage directories exist."""
        self.base_results_dir.mkdir(exist_ok=True)
        self.db_path.parent.mkdir(exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database with schema."""
        schema_path = self.db_path.parent / "schema.sql"
        if schema_path.exists():
            with sqlite3.connect(self.db_path) as conn:
                with open(schema_path) as f:
                    conn.executescript(f.read())
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _compute_node_hashes(self, node_dir: Path) -> Dict[str, str]:
        """Compute hashes of all node files."""
        node_hashes = {}
        if node_dir.exists():
            for node_file in node_dir.glob("*.txt"):
                node_hashes[node_file.name] = self._compute_file_hash(node_file)
        return node_hashes
    
    def save_run(self, 
                 trades_df: pd.DataFrame,
                 signals_df: pd.DataFrame,
                 metadata: Dict[str, Any],
                 data_file_path: Optional[Path] = None,
                 node_dir_path: Optional[Path] = None) -> str:
        """
        Save a complete backtest run to Parquet and database.
        
        Args:
            trades_df: DataFrame containing trade results
            signals_df: DataFrame containing signal data
            metadata: Run metadata (RR multiple, costs, etc.)
            data_file_path: Path to source data file
            node_dir_path: Path to strategy nodes directory
            
        Returns:
            Unique run ID
        """
        run_id = str(uuid.uuid4())
        run_dir = self.base_results_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save to Parquet
        trades_path = run_dir / "trades.parquet"
        signals_path = run_dir / "signals.parquet"
        
        trades_df.to_parquet(trades_path, index=False)
        signals_df.to_parquet(signals_path, index=False)
        
        # Compute hashes
        data_sha256 = self._compute_file_hash(data_file_path) if data_file_path else None
        node_hashes = self._compute_node_hashes(node_dir_path) if node_dir_path else {}
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO runs (
                    run_id, timestamp, commission, spread, node_hashes, 
                    rr_multiple, data_sha256, trades_path, signals_path, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                datetime.now().isoformat(),
                metadata.get('commission_perc', 0.0),
                metadata.get('spread_points', 0.0),
                json.dumps(node_hashes),
                metadata.get('rr_multiple', 1.0),
                data_sha256,
                str(trades_path),
                str(signals_path),
                json.dumps(metadata)
            ))
        
        return run_id
    
    def load_run(self, run_id: str) -> Dict[str, Any]:
        """Load a complete run by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Run {run_id} not found")
            
            # Load Parquet files
            trades_df = pd.read_parquet(row['trades_path'])
            signals_df = pd.read_parquet(row['signals_path'])
            
            return {
                'run_id': row['run_id'],
                'timestamp': row['timestamp'],
                'trades_df': trades_df,
                'signals_df': signals_df,
                'metadata': json.loads(row['metadata']),
                'node_hashes': json.loads(row['node_hashes'])
            }
    
    def export_csv(self, run_id: str, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Export run data to CSV for human inspection."""
        run_data = self.load_run(run_id)
        
        if output_dir is None:
            output_dir = self.base_results_dir / run_id
        
        trades_csv = output_dir / "trades.csv"
        signals_csv = output_dir / "signals.csv"
        
        run_data['trades_df'].to_csv(trades_csv, index=False)
        run_data['signals_df'].to_csv(signals_csv, index=False)
        
        return {'trades': trades_csv, 'signals': signals_csv}
