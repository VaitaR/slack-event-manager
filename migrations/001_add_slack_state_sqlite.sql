CREATE TABLE IF NOT EXISTS slack_ingestion_state (
  channel_id TEXT PRIMARY KEY,
  max_processed_ts REAL NOT NULL DEFAULT 0,
  resume_cursor TEXT,
  resume_min_ts REAL,
  updated_at TEXT
);
