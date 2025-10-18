# Telegram Integration Guide

**Last Updated:** 2025-10-18  
**Status:** ✅ Phase 7 Complete - Telegram Text Extraction

## Overview

This document describes the Telegram integration for the Slack Event Manager. The integration uses **Telethon** library with **user client** authentication to fetch text messages from public Telegram channels.

**Key Features:**
- Text message extraction from public channels
- Historical backfill (configurable from_date)
- Incremental ingestion (only new messages)
- FloodWait error handling with automatic retry
- URL and anchor extraction
- Integration with existing multi-source pipeline

**Scope (V1):**
- ✅ Text messages only
- ✅ Public channels (by @username)
- ✅ 1 day backfill by default
- ✅ User client authentication (API_ID/API_HASH)
- ❌ Media (photos/videos) - out of scope
- ❌ Reactions/views - out of scope
- ❌ Private channels - out of scope

## Architecture

### Components

```
Telegram Channel → TelegramClient (Telethon wrapper) → 
Message Processing → raw_telegram_messages table → 
Candidate Building → LLM Extraction → Events
```

**New Files:**
- `src/adapters/telegram_client.py` - Telethon wrapper with async→sync conversion
- `src/use_cases/ingest_telegram_messages.py` - Telegram ingestion use case
- `scripts/telegram_auth.py` - Interactive authentication helper
- `config/telegram_channels.yaml` - Channel configuration
- `tests/test_telegram_client.py` - Client tests (17 test cases)
- `tests/test_telegram_message_processing.py` - Processing tests (10+ test cases)
- `tests/test_telegram_e2e.py` - E2E tests (6 test cases)

**Modified Files:**
- `src/config/settings.py` - Added telegram_api_id, telegram_api_hash, telegram_channels
- `src/adapters/message_client_factory.py` - Added Telegram client creation
- `scripts/run_multi_source_pipeline.py` - Added Telegram ingestion branch
- `requirements.txt` - Added telethon>=1.36.0

### Database Schema

**New Table:** `raw_telegram_messages`
```sql
CREATE TABLE raw_telegram_messages (
    message_id TEXT PRIMARY KEY,  -- SHA1 hash of channel|telegram_id
    channel TEXT NOT NULL,
    message_date TEXT NOT NULL,
    sender_id TEXT,
    sender_name TEXT,
    text TEXT,
    text_norm TEXT,
    forward_from_channel TEXT,
    forward_from_message_id TEXT,
    media_type TEXT,
    links_raw TEXT,  -- JSON array
    links_norm TEXT,  -- JSON array
    anchors TEXT,  -- JSON array
    views INTEGER DEFAULT 0,
    ingested_at TEXT
);
```

**State Tracking:** `ingestion_state_telegram`
```sql
CREATE TABLE ingestion_state_telegram (
    channel_id TEXT PRIMARY KEY,
    last_processed_ts REAL NOT NULL,  -- Stores Telegram message_id as float
    updated_at TEXT NOT NULL
);
```

## Setup

### Prerequisites

1. **Telegram Account** - Active Telegram account with phone number
2. **API Credentials** - From https://my.telegram.org
3. **Python 3.11+** - With telethon library

### Step 1: Get Telegram API Credentials

1. Go to https://my.telegram.org
2. Log in with your phone number
3. Navigate to "API development tools"
4. Create an application (any name/description)
5. Copy **API ID** and **API hash**

### Step 2: Configure Environment

Add to `.env`:
```bash
# Telegram User Client (NOT bot token!)
TELEGRAM_API_ID=12345
TELEGRAM_API_HASH=abc123def456...
```

### Step 3: Authenticate

Run interactive authentication script:
```bash
python scripts/telegram_auth.py
```

This will:
1. Prompt for your phone number (with country code, e.g., +1234567890)
2. Send verification code via Telegram or SMS
3. Prompt for verification code
4. Create session file: `data/telegram_session.session`

**Important:** Session file must exist before running ingestion!

### Step 4: Configure Channels

Edit `config/telegram_channels.yaml`:
```yaml
telegram_channels:
  - channel_id: "@crypto_news"
    channel_name: "Crypto News"
    from_date: "2025-10-17T00:00:00Z"  # Start date for backfill
    enabled: true
  
  - channel_id: "@tech_updates"
    channel_name: "Tech Updates"
    from_date: "2025-10-16T00:00:00Z"
    enabled: true
```

**Channel ID formats:**
- Public channel: `@username` (e.g., `@crypto_news`)
- Numeric ID: `-1001234567890` (for private channels - not supported in V1)

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `telethon>=1.36.0`.

## Usage

### Test Telegram Ingestion

```bash
python scripts/test_telegram_ingestion.py
```

This will:
- Verify credentials and session file
- Fetch messages from configured channels
- Save to test database: `data/test_telegram_ingestion.db`
- Print statistics and sample messages

### Run Full Pipeline

```bash
# Run only Telegram source
python scripts/run_multi_source_pipeline.py --source telegram

# Run with backfill from specific date
python scripts/run_multi_source_pipeline.py --source telegram --backfill-from 2025-10-01

# Run continuously (every hour)
python scripts/run_multi_source_pipeline.py --source telegram --interval-seconds 3600
```

### Run All Sources (Slack + Telegram)

```bash
# Process both Slack and Telegram
python scripts/run_multi_source_pipeline.py

# With digest publishing
python scripts/run_multi_source_pipeline.py --publish
```

## Configuration

### Settings (src/config/settings.py)

**Telegram Credentials:**
```python
telegram_api_id: int | None  # From .env
telegram_api_hash: SecretStr | None  # From .env
telegram_session_path: str = "data/telegram_session"  # Session file path
```

**Telegram Channels:**
```python
telegram_channels: list[dict[str, Any]]  # Loaded from config/telegram_channels.yaml
```

### Channel Configuration

```yaml
telegram_channels:
  - channel_id: "@channel_username"  # Required
    channel_name: "Display Name"  # Optional
    from_date: "2025-10-17T00:00:00Z"  # Optional, default: 1 day ago
    enabled: true  # Optional, default: true
```

## Technical Details

### Telethon vs Bot API

**Why Telethon (User Client)?**
- ✅ Access to channel history (bot API doesn't support)
- ✅ No need to add bot to channels
- ✅ Full MTProto API access
- ❌ Requires phone number authentication
- ❌ More complex setup

**Bot API Limitations:**
- Cannot fetch channel history
- Cannot access messages before bot was added
- Limited to bot-specific methods

### Async→Sync Wrapper

Telethon is fully async, but our protocol is synchronous. We wrap async calls:

```python
def fetch_messages(self, channel_id: str, limit: int = 100) -> list[dict]:
    return asyncio.run(self._fetch_messages_async(channel_id, limit))

async def _fetch_messages_async(self, channel_id: str, limit: int):
    async for message in client.iter_messages(channel_id, limit=limit):
        # Process message
        yield message
```

### FloodWait Handling

Telegram enforces rate limits with `FloodWaitError`. We handle this automatically:

```python
try:
    messages = await client.iter_messages(channel_id)
except FloodWaitError as e:
    logger.warning(f"FloodWait: must wait {e.seconds}s")
    await asyncio.sleep(e.seconds)
    # Retry
```

**Retry Strategy:**
- Max 3 retries
- Wait exactly `error.seconds` (as required by Telegram)
- After 3 failures, raise `RateLimitError`

### Message ID Format

Telegram message IDs are integers, but we store as strings for consistency:

```python
# Telegram native: 123456 (int)
# Stored in DB: "abc123..." (SHA1 hash of channel|telegram_id)
# State tracking: 123456 (float in ingestion_state_telegram)
```

### URL Extraction

URLs are extracted from:
1. **Entities** (MessageEntityUrl, MessageEntityTextUrl)
2. **Regex** (naked URLs in text)

```python
from telethon.tl.types import MessageEntityUrl, MessageEntityTextUrl

for entity in message.entities:
    if isinstance(entity, MessageEntityUrl):
        url = text[entity.offset:entity.offset + entity.length]
    elif isinstance(entity, MessageEntityTextUrl):
        url = entity.url
```

### Post URL Construction

For public channels with username:
```
https://t.me/{username}/{message_id}
```

Example: `https://t.me/crypto_news/12345`

For private channels (numeric ID): Not supported in V1.

## Testing

### Unit Tests

```bash
# Run all Telegram tests
pytest tests/test_telegram_client.py -v
pytest tests/test_telegram_message_processing.py -v

# Run E2E tests
pytest tests/test_telegram_e2e.py -v
```

**Test Coverage:**
- 17 tests for TelegramClient
- 10+ tests for message processing
- 6 E2E tests

### Manual Testing

```bash
# Test with real API
python scripts/test_telegram_ingestion.py

# Check database
sqlite3 data/test_telegram_ingestion.db
SELECT * FROM raw_telegram_messages LIMIT 10;
```

## Troubleshooting

### Session File Not Found

**Error:** `Session file not found: data/telegram_session.session`

**Solution:**
```bash
python scripts/telegram_auth.py
```

### API Credentials Missing

**Error:** `TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env`

**Solution:**
1. Get credentials from https://my.telegram.org
2. Add to `.env`:
   ```bash
   TELEGRAM_API_ID=12345
   TELEGRAM_API_HASH=abc123...
   ```

### FloodWait Errors

**Error:** `FloodWaitError: must wait 3600s`

**Solution:**
- Wait as requested (automatic in code)
- Reduce fetch frequency
- Use smaller `limit` parameter

### Channel Not Found

**Error:** `Channel not found: @channel_name`

**Solution:**
- Verify channel username is correct
- Check if channel is public
- Try joining channel manually first

### No Messages Fetched

**Possible causes:**
1. Channel has no messages in backfill window
2. `from_date` is too recent
3. Channel is disabled in config

**Solution:**
- Check `from_date` in config
- Verify `enabled: true`
- Check channel actually has messages

## Limitations (V1)

### Out of Scope

1. **Media** - Photos, videos, documents not extracted
2. **Reactions** - Not extracted (Telegram API limitation)
3. **Views** - Extracted but not used in scoring
4. **Private Channels** - Numeric IDs not supported
5. **Comments** - Reply threads not extracted
6. **Edits** - Message edits not tracked

### Future Enhancements (V2+)

- [ ] Media extraction (photos, videos)
- [ ] Reaction tracking
- [ ] Private channel support
- [ ] Edit history tracking
- [ ] Comment/reply extraction
- [ ] View count analytics
- [ ] Bot API support (for simpler setup)

## Performance

### Rate Limits

**Telegram Limits:**
- ~5 requests per second (configurable)
- FloodWait enforced dynamically
- Backfill: ~10k messages/day

**Our Limits:**
- Default: 100 messages per fetch
- Backfill: 1 day by default
- Incremental: Only new messages

### Cost

**Telegram API:**
- ✅ Free (no costs)
- ❌ Rate limits apply

**LLM Processing:**
- Same as Slack (gpt-5-nano)
- ~$0.0005 per event extracted

## Security

### Session File

**Location:** `data/telegram_session.session`

**Security:**
- Contains authentication tokens
- ❌ Never commit to git
- ✅ Add to `.gitignore`
- ✅ Set file permissions: `chmod 600 data/telegram_session.session`

### API Credentials

**Storage:** `.env` file only

**Best Practices:**
- Never commit `.env` to git
- Rotate credentials periodically
- Use separate credentials for dev/prod

### Data Privacy

**PII Considerations:**
- Sender IDs stored (but not names in V1)
- Message text stored as-is
- No user profile data extracted

## Migration

### From Slack-Only to Multi-Source

No migration needed! Telegram integration is additive:

1. Existing Slack ingestion continues working
2. Add Telegram credentials to `.env`
3. Configure channels in `config/telegram_channels.yaml`
4. Run authentication: `python scripts/telegram_auth.py`
5. Run pipeline: `python scripts/run_multi_source_pipeline.py`

### Database

New tables created automatically:
- `raw_telegram_messages`
- `ingestion_state_telegram`

Existing tables unchanged.

## Support

### Logs

```bash
# View pipeline logs
tail -f logs/pipeline_*.log

# Check for errors
grep ERROR logs/pipeline_*.log
```

### Debug Mode

```bash
export LOG_LEVEL=DEBUG
python scripts/test_telegram_ingestion.py
```

### Common Issues

See **Troubleshooting** section above.

### Contact

For issues or questions:
1. Check this documentation
2. Review test scripts for examples
3. Check logs for error details

## References

- **Telethon Documentation:** https://docs.telethon.dev/
- **Telegram API:** https://core.telegram.org/api
- **MTProto Protocol:** https://core.telegram.org/mtproto
- **My Telegram:** https://my.telegram.org (get API credentials)

---

**Next Steps:**
1. ✅ Complete Phase 7 (Telegram Integration)
2. ⏭️ Optional: Add media extraction (V2)
3. ⏭️ Optional: Add reaction tracking (V2)
4. ⏭️ Optional: Add private channel support (V2)

