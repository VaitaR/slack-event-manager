# Docker Deployment Guide

This guide explains how to deploy the Slack Event Manager as a containerized service using Docker.

## Quick Start

```bash
# 1. Build the Docker image
docker compose build

# 2. Start the services
docker compose up -d

# 3. View logs
docker compose logs -f slack-bot

# 4. Access Streamlit UI
open http://localhost:8501
```

## Architecture

The deployment consists of two services:

1. **slack-bot**: Main pipeline service that runs continuously
   - Fetches messages from Slack
   - Processes with LLM
   - Stores in SQLite database
   - Runs every hour by default

2. **streamlit-ui**: Web interface for viewing data
   - Accessible at http://localhost:8501
   - Read-only access to database (application level)
   - Note: Volume mounted without `:ro` flag because SQLite needs to create temporary files (WAL, SHM) even for read operations
   - Real-time visualization

## Configuration

### Environment Variables

The services use the existing `.env` file in the project root. No changes needed.

Required variables:
- `SLACK_BOT_TOKEN`: Slack bot token
- `SLACK_CHANNELS`: JSON array of channels to monitor
- `SLACK_DIGEST_CHANNEL_ID`: Channel for posting digests
- `OPENAI_API_KEY`: OpenAI API key
- `LLM_MODEL`: Model to use (e.g., gpt-5-nano)
- `DB_PATH`: Database path (default: data/slack_events.db)

### First Run Behavior

On the first run, the bot will:
1. Check `ingestion_state` table for each channel
2. If no state exists: backfill last 30 days of messages
3. Process and store all messages
4. Update `ingestion_state` with latest timestamp

On subsequent runs:
1. Read last processed timestamp from `ingestion_state`
2. Fetch only new messages since last run
3. Process incrementally
4. Update state after successful processing

### Custom Backfill Date

To backfill from a specific date on first run:

```bash
docker compose run --rm slack-bot python scripts/run_pipeline.py --backfill-from 2025-09-01
```

## Data Persistence

### SQLite Database

The database is stored in `./data/slack_events.db` on the host machine via bind mount.

**Important**: Data persists even if containers are removed:
- Container crash: Data safe, restarts automatically
- `docker compose down`: Data safe
- `docker compose up`: Continues from last state
- Server reboot: Data safe, services auto-restart

### Logs

Logs are stored in `./logs/` directory on the host:
- `pipeline_YYYYMMDD.log`: Daily log files
- Accessible from host and container
- Rotate automatically by date

## Common Operations

### View Logs

```bash
# Follow logs in real-time
docker compose logs -f slack-bot

# View last 100 lines
docker compose logs --tail=100 slack-bot

# View Streamlit logs
docker compose logs -f streamlit-ui
```

### Check Service Status

```bash
# View running services
docker compose ps

# Check health
docker compose exec slack-bot python -c "from src.config.settings import get_settings; print('OK')"
```

### Inspect Database

```bash
# Open SQLite shell
docker compose exec slack-bot sqlite3 /app/data/slack_events.db

# Quick queries
docker compose exec slack-bot sqlite3 /app/data/slack_events.db "SELECT COUNT(*) FROM events;"
docker compose exec slack-bot sqlite3 /app/data/slack_events.db "SELECT title, category, event_date FROM events ORDER BY event_date DESC LIMIT 10;"

# Check ingestion state
docker compose exec slack-bot sqlite3 /app/data/slack_events.db "SELECT * FROM ingestion_state;"
```

### Restart Services

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart slack-bot

# Stop and remove containers (data persists)
docker compose down

# Start fresh
docker compose up -d
```

### Change Run Interval

Edit `docker-compose.yml` and modify the `command`:

```yaml
# Run every 30 minutes (1800 seconds)
command: python scripts/run_pipeline.py --interval-seconds 1800

# Run every 6 hours (21600 seconds)
command: python scripts/run_pipeline.py --interval-seconds 21600
```

Then restart:

```bash
docker compose up -d
```

### Run Pipeline Once

To run the pipeline once without starting the continuous service:

```bash
docker compose run --rm slack-bot python scripts/run_pipeline.py
```

### Enable Digest Publishing

Edit `docker-compose.yml`:

```yaml
command: python scripts/run_pipeline.py --interval-seconds 3600 --publish
```

## Monitoring

### Health Checks

Both services have health checks configured:

- **slack-bot**: Validates settings can be loaded every 5 minutes
- **streamlit-ui**: HTTP health check on `/_stcore/health` every 30 seconds

Check health status:

```bash
docker compose ps
# Look for "(healthy)" status
```

### Resource Usage

```bash
# View resource usage
docker stats

# Typical usage:
# - slack-bot: ~200MB RAM, <5% CPU (idle), ~50% CPU (during LLM calls)
# - streamlit-ui: ~150MB RAM, <5% CPU
```

### Cost Monitoring

Query LLM costs from database:

```bash
docker compose exec slack-bot sqlite3 /app/data/slack_events.db "
SELECT 
  DATE(ts) as date,
  COUNT(*) as calls,
  SUM(tokens_in) as total_tokens_in,
  SUM(tokens_out) as total_tokens_out,
  SUM(cost_usd) as total_cost_usd
FROM llm_calls
GROUP BY DATE(ts)
ORDER BY date DESC
LIMIT 7;"
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs for errors
docker compose logs slack-bot

# Common issues:
# 1. Missing .env file
# 2. Invalid API keys
# 3. Database permissions

# Test configuration
docker compose run --rm slack-bot python -c "from src.config.settings import get_settings; print(get_settings())"
```

### Database Locked

If you see "database is locked" errors:

```bash
# Stop all services
docker compose down

# Check for stale locks
ls -la data/

# Start services
docker compose up -d
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean old Docker images
docker system prune -a

# Archive old logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
rm logs/*.log
```

### Reset Everything

To start completely fresh:

```bash
# Stop services
docker compose down

# Remove database (WARNING: deletes all data!)
rm data/slack_events.db

# Remove logs
rm -rf logs/*

# Rebuild and start
docker compose build --no-cache
docker compose up -d
```

## Production Deployment

### Server Setup

1. **Install Docker and Docker Compose**

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

2. **Clone Repository**

```bash
git clone <repo-url>
cd slack_event_manager
```

3. **Configure Environment**

```bash
# .env should already exist with production values
# Verify it exists and has correct values
cat .env
```

4. **Deploy**

```bash
# Build and start
docker compose build
docker compose up -d

# Verify
docker compose ps
docker compose logs --tail=50 slack-bot
```

### Automatic Restart on Reboot

Services are configured with `restart: unless-stopped`, so they will automatically start after server reboot.

To ensure Docker itself starts on boot:

```bash
sudo systemctl enable docker
```

### Backup Strategy

Recommended backup schedule:

```bash
# Daily backup script (save as backup.sh)
#!/bin/bash
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d)

# Backup database
cp data/slack_events.db "$BACKUP_DIR/slack_events_$DATE.db"

# Backup logs (optional)
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" logs/

# Keep only last 30 days
find "$BACKUP_DIR" -name "slack_events_*.db" -mtime +30 -delete
find "$BACKUP_DIR" -name "logs_*.tar.gz" -mtime +30 -delete
```

Add to crontab:

```bash
0 2 * * * /path/to/backup.sh
```

### Updating the Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose build
docker compose down
docker compose up -d

# Verify
docker compose logs --tail=50 slack-bot
```

## CLI Reference

The `run_pipeline.py` script supports the following options:

```
--interval-seconds INT    Run interval in seconds (0=once, >0=continuous)
--backfill-from DATE      Backfill from YYYY-MM-DD (first run only)
--publish                 Publish digest after processing
--skip-llm                Skip LLM extraction step
--dry-run                 Dry run mode (no digest posting)
--lookback-hours INT      Deprecated, use --backfill-from
```

### Examples

```bash
# Run once
docker compose run --rm slack-bot python scripts/run_pipeline.py

# Run continuously every 2 hours
docker compose run --rm slack-bot python scripts/run_pipeline.py --interval-seconds 7200

# Backfill from Jan 1, 2025
docker compose run --rm slack-bot python scripts/run_pipeline.py --backfill-from 2025-01-01

# Run with digest publishing
docker compose run --rm slack-bot python scripts/run_pipeline.py --interval-seconds 3600 --publish

# Skip LLM for testing
docker compose run --rm slack-bot python scripts/run_pipeline.py --skip-llm
```

## Security Notes

1. **API Keys**: Never commit `.env` file to version control
2. **Database Access**: SQLite file contains all message data
3. **Network**: Services run on isolated Docker network
4. **Streamlit Port**: Only exposed on localhost by default

To expose Streamlit publicly, edit `docker-compose.yml`:

```yaml
ports:
  - "0.0.0.0:8501:8501"  # Expose on all interfaces
```

**Warning**: Add authentication if exposing publicly!

## Support

For issues or questions:
- Check logs: `docker compose logs -f slack-bot`
- Run health check: `docker compose exec slack-bot python scripts/quick_test.py`
- Review AGENTS.md for architecture details

