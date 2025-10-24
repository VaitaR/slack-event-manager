CREATE TABLE IF NOT EXISTS slack_ingestion_state (
  channel_id TEXT PRIMARY KEY,
  max_processed_ts DOUBLE PRECISION NOT NULL DEFAULT 0,
  resume_cursor TEXT,
  resume_min_ts DOUBLE PRECISION,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
