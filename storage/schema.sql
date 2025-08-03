-- BlackBox Results Database Schema

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    commission REAL DEFAULT 0.0,
    spread REAL DEFAULT 0.0,
    node_hashes TEXT,  -- JSON object with node file names and their hashes
    rr_multiple REAL DEFAULT 1.0,
    data_sha256 TEXT,
    trades_path TEXT NOT NULL,
    signals_path TEXT NOT NULL,
    metadata TEXT,  -- JSON object with additional run metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_rr_multiple ON runs(rr_multiple);
CREATE INDEX IF NOT EXISTS idx_runs_data_sha256 ON runs(data_sha256);
