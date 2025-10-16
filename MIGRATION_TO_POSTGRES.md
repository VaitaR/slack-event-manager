# Migration Guide: SQLite to PostgreSQL

**Last Updated:** 2025-10-16  
**Status:** Production Ready

## Overview

This guide explains how to migrate the Slack Event Manager from SQLite to PostgreSQL for production deployment. The migration maintains backward compatibility with SQLite for local development.

## Why PostgreSQL?

### Advantages over SQLite

- **Concurrent Access**: PostgreSQL supports multiple writers without database locks
- **Scalability**: Better performance for large datasets and high traffic
- **Production Ready**: Industry-standard database for microservices
- **Advanced Features**: JSONB fields, full-text search, connection pooling
- **ACID Compliance**: Stronger transaction guarantees

### Use Cases

- **SQLite**: Local development, testing, small deployments
- **PostgreSQL**: Production microservices, Docker deployments, high availability

## Prerequisites

### Required Software

- PostgreSQL 16+ (PostgreSQL 16-alpine recommended for Docker)
- Python 3.11+
- All existing project dependencies (see `requirements.txt`)

### Environment Setup

1. **Install PostgreSQL** (if not using Docker):
   ```bash
   # macOS
   brew install postgresql@16
   
   # Ubuntu/Debian
   sudo apt-get install postgresql-16
   
   # Start PostgreSQL service
   brew services start postgresql@16  # macOS
   sudo systemctl start postgresql    # Linux
   ```

2. **Create Database**:
   ```bash
   # Create database
   createdb slack_events
   
   # Or via psql
   psql -U postgres
   CREATE DATABASE slack_events;
   \q
   ```

## Configuration

### Step 1: Update config.yaml

Edit `config.yaml` to use PostgreSQL:

```yaml
database:
  type: postgres  # Changed from sqlite
  postgres:
    host: localhost
    port: 5432
    database: slack_events
    user: postgres
```

### Step 2: Set Environment Variables

Add PostgreSQL password to `.env`:

```bash
# Existing secrets
SLACK_BOT_TOKEN=xoxb-your-token
OPENAI_API_KEY=sk-your-key

# PostgreSQL password (new)
POSTGRES_PASSWORD=your_secure_password
```

**Security Note**: Never commit `.env` to version control!

### Step 3: Run Database Migrations

```bash
# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DATABASE=slack_events
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password

# Run Alembic migrations
alembic upgrade head
```

Expected output:
```
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 001, Initial schema
```

### Step 4: Verify Installation

```bash
# Test database connection
python -c "from src.adapters.repository_factory import create_repository; \
           from src.config.settings import get_settings; \
           repo = create_repository(get_settings()); \
           print('✅ PostgreSQL connection successful')"
```

## Docker Deployment

### Using Docker Compose

The easiest way to deploy with PostgreSQL:

```bash
# 1. Update .env with PostgreSQL password
echo "POSTGRES_PASSWORD=your_password" >> .env

# 2. Build and start services
docker compose build
docker compose up -d

# 3. View logs
docker compose logs -f slack-bot

# 4. Check PostgreSQL
docker compose exec postgres psql -U postgres -d slack_events -c "\dt"
```

### Docker Compose Services

The `docker-compose.yml` includes:

1. **postgres**: PostgreSQL 16-alpine with persistent volume
2. **slack-bot**: Main application with automatic migrations
3. **streamlit-ui**: Dashboard interface

## Migration Verification

### Test Checklist

Run these commands to verify the migration:

```bash
# 1. Test configuration loading
python -c "from src.config.settings import get_settings; \
           s = get_settings(); \
           print(f'Database type: {s.database_type}'); \
           assert s.database_type == 'postgres'"

# 2. Test repository creation
python -c "from src.adapters.repository_factory import create_repository; \
           from src.config.settings import get_settings; \
           repo = create_repository(get_settings()); \
           print('Repository created:', type(repo).__name__)"

# 3. Run tests
pytest tests/test_postgres_repository.py -v

# 4. Run quick pipeline test
python scripts/quick_test.py
```

### Database Inspection

```bash
# Connect to PostgreSQL
psql -h localhost -U postgres -d slack_events

# List tables
\dt

# Check table schemas
\d raw_slack_messages
\d event_candidates
\d events

# Count records
SELECT COUNT(*) FROM raw_slack_messages;
SELECT COUNT(*) FROM events;

# Exit
\q
```

## Running the Application

### Manual Start

```bash
# Run pipeline once
python scripts/run_pipeline.py

# Run continuously (every hour)
python scripts/run_pipeline.py --interval-seconds 3600

# Generate digest
python scripts/generate_digest.py --channel YOUR_CHANNEL_ID
```

### Docker Start

```bash
# Start all services
docker compose up -d

# View application logs
docker compose logs -f slack-bot

# Access Streamlit UI
open http://localhost:8501
```

## Switching Between SQLite and PostgreSQL

The system supports both databases simultaneously through configuration:

### Use SQLite (Development)

```yaml
# config.yaml
database:
  type: sqlite
  path: data/slack_events.db
```

### Use PostgreSQL (Production)

```yaml
# config.yaml
database:
  type: postgres
  postgres:
    host: localhost
    port: 5432
    database: slack_events
    user: postgres
```

No code changes required! The `create_repository()` factory handles database selection.

## Troubleshooting

### Connection Errors

**Problem**: `RepositoryError: Failed to create connection pool`

**Solutions**:
```bash
# Check PostgreSQL is running
pg_isready -h localhost -U postgres

# Check credentials
psql -h localhost -U postgres -d slack_events

# Check firewall/network
telnet localhost 5432
```

### Migration Errors

**Problem**: `alembic.util.exc.CommandError: Can't locate revision identified by '001'`

**Solutions**:
```bash
# Check alembic version table
psql -U postgres -d slack_events -c "SELECT * FROM alembic_version"

# Reset migrations (WARNING: drops all tables)
alembic downgrade base
alembic upgrade head
```

### Permission Errors

**Problem**: `psycopg2.OperationalError: FATAL: role "postgres" does not exist`

**Solutions**:
```bash
# Create PostgreSQL user
createuser -s postgres

# Or set different user in config.yaml
database:
  postgres:
    user: your_username
```

### Docker Issues

**Problem**: Services fail to start or can't connect to PostgreSQL

**Solutions**:
```bash
# Check service status
docker compose ps

# View PostgreSQL logs
docker compose logs postgres

# Restart services
docker compose restart slack-bot

# Check network
docker compose exec slack-bot ping postgres
```

### Port Conflicts

**Problem**: `bind: address already in use`

**Solutions**:
```bash
# Check what's using port 5432
lsof -i :5432

# Stop conflicting service
brew services stop postgresql@16  # macOS
sudo systemctl stop postgresql    # Linux

# Or change port in docker-compose.yml
services:
  postgres:
    ports:
      - "5433:5432"  # Use port 5433 instead
```

## Performance Tuning

### Connection Pool Settings

Adjust in repository initialization (see `postgres_repository.py`):

```python
PostgresRepository(
    host=host,
    port=port,
    database=database,
    user=user,
    password=password,
    min_connections=2,   # Increase for high traffic
    max_connections=20,  # Increase for concurrency
)
```

### PostgreSQL Configuration

For production, tune PostgreSQL settings in `/etc/postgresql/*/main/postgresql.conf`:

```conf
# Memory
shared_buffers = 256MB
effective_cache_size = 1GB

# Connections
max_connections = 100

# Performance
work_mem = 4MB
maintenance_work_mem = 64MB

# Logging
log_statement = 'mod'  # Log modifications
log_duration = on      # Log query duration
```

After changes:
```bash
sudo systemctl reload postgresql
```

### Indexes

The initial migration creates two indexes:
- `idx_events_dedup_key` on `events(dedup_key)`
- `idx_events_date` on `events(event_date)`

Add more indexes for common queries:
```sql
-- If you frequently filter by category
CREATE INDEX idx_events_category ON events(category);

-- If you frequently filter by confidence
CREATE INDEX idx_events_confidence ON events(confidence);
```

## Backup and Recovery

### Backup PostgreSQL Database

```bash
# Full backup
pg_dump -U postgres slack_events > backup_$(date +%Y%m%d).sql

# Compressed backup
pg_dump -U postgres slack_events | gzip > backup_$(date +%Y%m%d).sql.gz

# Docker backup
docker compose exec postgres pg_dump -U postgres slack_events > backup.sql
```

### Restore from Backup

```bash
# Restore from backup
psql -U postgres slack_events < backup_20251016.sql

# Restore compressed backup
gunzip -c backup_20251016.sql.gz | psql -U postgres slack_events

# Docker restore
cat backup.sql | docker compose exec -T postgres psql -U postgres slack_events
```

### Automated Backups

Add to crontab:
```bash
# Daily backup at 2 AM
0 2 * * * pg_dump -U postgres slack_events | gzip > /backups/slack_events_$(date +\%Y\%m\%d).sql.gz

# Keep only 7 days of backups
0 3 * * * find /backups -name "slack_events_*.sql.gz" -mtime +7 -delete
```

## Rollback Plan

### Switch Back to SQLite

If you encounter critical issues:

1. **Update config.yaml**:
   ```yaml
   database:
     type: sqlite
     path: data/slack_events.db
   ```

2. **Restart services**:
   ```bash
   # Docker
   docker compose restart slack-bot
   
   # Manual
   pkill -f "python scripts/run_pipeline.py"
   python scripts/run_pipeline.py
   ```

3. **Verify**:
   ```bash
   python -c "from src.config.settings import get_settings; \
              print(get_settings().database_type)"
   # Output: sqlite
   ```

No data is lost during rollback since SQLite database remains unchanged.

## Production Checklist

Before deploying to production:

- [ ] PostgreSQL 16+ installed and running
- [ ] Database created (`slack_events`)
- [ ] Migrations executed successfully (`alembic upgrade head`)
- [ ] Environment variables set (`.env` file with `POSTGRES_PASSWORD`)
- [ ] Configuration updated (`config.yaml` with `type: postgres`)
- [ ] Connection pooling configured appropriately
- [ ] Backup strategy implemented
- [ ] Monitoring and alerts configured
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Docker Compose tested (`docker compose up -d`)
- [ ] Performance tuning applied
- [ ] Security review completed

## Support

For issues or questions:

1. Check this migration guide first
2. Review troubleshooting section
3. Check logs: `docker compose logs -f` or `logs/pipeline_*.log`
4. Consult [AGENTS.md](AGENTS.md) for detailed project documentation
5. Contact the platform team

## Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/16/)
- [psycopg2 Documentation](https://www.psycopg.org/docs/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

---

**Migration completed successfully?** Update `AGENTS.md` with your production deployment notes!

