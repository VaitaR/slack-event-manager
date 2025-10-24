CREATE TABLE IF NOT EXISTS slack_users_cache (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS slack_permalink_cache (
    channel_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    url TEXT NOT NULL,
    updated_at TEXT,
    PRIMARY KEY (channel_id, ts)
);
