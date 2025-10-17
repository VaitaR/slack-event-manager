# PostgreSQL Migration Guide

This guide explains how to switch from SQLite (development) to PostgreSQL (production).

## Prerequisites

- **PostgreSQL 16+** installed (locally or in Docker)
- **Python dependencies** installed: `alembic`, `psycopg2-binary`
- **Environment variables** configured (see below)

## Configuration

### 1. Environment Variables

Add to your `.env` file:

```bash
# Required for PostgreSQL
DATABASE_TYPE=postgres
POSTGRES_PASSWORD=your_secure_password

# Optional (defaults shown)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=slack_events
POSTGRES_USER=postgres
```

### 2. Config File (Optional)

You can also configure in `config.yaml` (environment variables take precedence):

```yaml
database:
  type: postgres  # or "sqlite" for development
  path: data/slack_events.db  # Used only for SQLite
  
  postgres:
    host: localhost
    port: 5432
    database: slack_events
    user: postgres
    # Password MUST be in .env as POSTGRES_PASSWORD
```

## Migration Steps

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start PostgreSQL (if using Docker)
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=slack_events \
  -p 5432:5432 \
  postgres:16-alpine

# 3. Run migrations
alembic upgrade head

# 4. Test the application
python scripts/run_pipeline.py
```

### Docker Compose (Recommended)

```bash
# 1. Set environment variables in .env
echo "DATABASE_TYPE=postgres" >> .env
echo "POSTGRES_PASSWORD=your_password" >> .env

# 2. Build and start services
docker compose up -d

# Migrations run automatically via docker-entrypoint.sh
```

## Verification

### Check Database Connection

```bash
# Using psql
psql -h localhost -U postgres -d slack_events

# List tables
\dt

# Expected tables:
# - raw_slack_messages
# - event_candidates
# - events
# - llm_calls
# - channel_watermarks
# - ingestion_state
```

### Check Application Logs

```bash
# Docker
docker compose logs -f slack-bot

# Look for:
# 🐘 PostgreSQL database mode: postgres@localhost:5432/slack_events
# ✅ Migrations completed!
```

### Run Tests

```bash
# All tests (uses SQLite by default)
make test

# Type checking
make typecheck

# Linting
make lint
```

## Rollback to SQLite

If you need to rollback:

```bash
# 1. Update .env
DATABASE_TYPE=sqlite

# 2. Restart services
docker compose restart slack-bot streamlit-ui
```

## Schema Compatibility

PostgreSQL schema is **100% compatible** with SQLite schema:
- ✅ Same table names
- ✅ Same column names
- ✅ Same data types (TEXT → String, INTEGER → Integer, JSONB → JSON)
- ✅ Same indexes
- ✅ Same constraints

**Key Differences:**
- PostgreSQL uses **JSONB** for better performance
- PostgreSQL has **timezone-aware timestamps**
- PostgreSQL supports **concurrent access** (multiple workers)

## Common Issues

### Issue: Connection refused

**Solution:**
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check environment variables
echo $POSTGRES_HOST
echo $POSTGRES_PORT
```

### Issue: Password authentication failed

**Solution:**
```bash
# Verify password in .env
cat .env | grep POSTGRES_PASSWORD

# Try connecting manually
psql -h localhost -U postgres -d slack_events
```

### Issue: Migrations fail

**Solution:**
```bash
# Check current migration status
alembic current

# Check migration history
alembic history

# Reset migrations (DANGER: drops all tables!)
alembic downgrade base
alembic upgrade head
```

## Performance Tuning

### Connection Pooling

PostgresRepository uses connection pooling by default:
- **Min connections**: 1
- **Max connections**: 10

To adjust:

```python
from src.adapters.postgres_repository import PostgresRepository

repo = PostgresRepository(
    host="localhost",
    port=5432,
    database="slack_events",
    user="postgres",
    password="password",
    minconn=2,  # Minimum connections
    maxconn=20,  # Maximum connections
)
```

### Indexes

All critical indexes are created automatically:
- `idx_raw_messages_channel_ts` - Message queries by channel and timestamp
- `idx_events_dedup_key` - Fast deduplication lookups
- `idx_events_date` - Event queries by date range

## Backup & Recovery

### Backup

```bash
# Full database dump
pg_dump -h localhost -U postgres slack_events > backup.sql

# Specific table
pg_dump -h localhost -U postgres -t events slack_events > events_backup.sql
```

### Restore

```bash
# Full restore
psql -h localhost -U postgres slack_events < backup.sql

# Specific table
psql -h localhost -U postgres slack_events < events_backup.sql
```

## Monitoring

### Check Database Size

```sql
SELECT 
    pg_database.datname,
    pg_size_pretty(pg_database_size(pg_database.datname)) AS size
FROM pg_database
WHERE datname = 'slack_events';
```

### Check Table Sizes

```sql
SELECT 
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

### Check Active Connections

```sql
SELECT 
    count(*),
    state
FROM pg_stat_activity
WHERE datname = 'slack_events'
GROUP BY state;
```

## Production Checklist

- ✅ Set strong `POSTGRES_PASSWORD`
- ✅ Configure connection pooling
- ✅ Set up regular backups
- ✅ Monitor database size
- ✅ Configure PostgreSQL logging
- ✅ Set up alerts for connection errors
- ✅ Test rollback procedure
- ✅ Document recovery process

## Support

For issues or questions:
1. Check logs: `docker compose logs -f`
2. Verify configuration: `make test`
3. Review this guide
4. Check `DATABASE_CONFIG_ANALYSIS.md` for advanced configuration details

