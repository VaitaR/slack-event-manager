# Slack Ingestion State Migrations

Apply manually using your preferred client.

```bash
# SQLite
sqlite3 data/app.db < migrations/001_add_slack_state_sqlite.sql

# PostgreSQL
psql "$DATABASE_URL" -f migrations/001_add_slack_state_pg.sql
```
