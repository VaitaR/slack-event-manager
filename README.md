# Slack Event Manager MVP

AI-powered event extraction and digest system for Slack channels. Automatically processes Slack messages to extract structured events, deduplicate them, and publish daily digests.

## Features

- 🤖 **LLM-powered extraction**: Uses OpenAI GPT to extract 0-5 events per message
- 📊 **Intelligent scoring**: Configurable per-channel scoring for candidate selection
- 🔗 **Anchor detection**: Extracts Jira keys, GitHub issues, meeting links, document IDs
- 📅 **Smart date resolution**: Handles absolute, relative dates, ranges, and timezones
- 🔄 **Deduplication**: Merges similar events across messages with fuzzy matching
- 💾 **Dual database**: PostgreSQL (production) or SQLite (development) with seamless switching
- 💰 **Budget control**: Daily LLM cost tracking with graceful degradation
- 🌍 **Multi-channel**: Whitelist channels with per-channel configurations
- 📨 **Digest publishing**: Beautiful Slack Block Kit digests
- 🐳 **Docker-ready**: Full Docker Compose setup with PostgreSQL, auto-migrations, and Streamlit UI

## Architecture

Clean architecture with clear separation of concerns and enterprise patterns:

```
src/
├── domain/              # Pure business logic
│   ├── models.py       # Pydantic models
│   ├── protocols.py    # Abstract interfaces
│   ├── exceptions.py   # Custom exceptions
│   ├── specifications.py         # Specification pattern (NEW)
│   ├── deduplication_constants.py # Business rules
│   └── scoring_constants.py      # Scoring limits
├── services/           # Domain services
│   ├── text_normalizer.py
│   ├── link_extractor.py
│   ├── date_resolver.py
│   ├── scoring_engine.py
│   └── deduplicator.py
├── adapters/           # External integrations
│   ├── slack_client.py
│   ├── llm_client.py
│   ├── sqlite_repository.py
│   ├── postgres_repository.py    # PostgreSQL adapter (NEW)
│   ├── repository_factory.py     # DB selection (NEW)
│   └── query_builders.py         # Type-safe queries
├── use_cases/          # Application orchestration
│   ├── ingest_messages.py
│   ├── build_candidates.py
│   ├── extract_events.py
│   ├── deduplicate_events.py
│   └── publish_digest.py
└── config/             # Settings management
    └── settings.py
```

**Design Patterns:**
- **Repository Pattern**: Abstract data access with dual SQLite/PostgreSQL support
- **Factory Pattern**: Database selection based on configuration
- **Specification Pattern**: Composable business rules with AND/OR/NOT logic
- **Query Builder (Criteria)**: Type-safe database queries without string literals
- **Use Case Pattern**: Clean orchestration of business logic

## Prerequisites

- Python 3.11+
- Slack Bot Token with permissions:
  - `channels:read`, `channels:history`
  - `users:read`, `reactions:read`
  - `chat:write` (for digest posting)
- OpenAI API Key

## Quick Start

### 1. Installation

```bash
# Clone repository
cd /path/to/slack_event_manager

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

**Automated Setup (Recommended):**
```bash
# Run setup script - creates all config files from examples
./scripts/setup_config.sh

# Edit files with your values
# 1. .env - Add your API tokens
# 2. config/main.yaml - Adjust settings (optional, uses good defaults)
# 3. config/object_registry.yaml - Add your internal systems
# 4. config/channels.yaml - Add your Slack channels
```

**Manual Setup:**
```bash
# 1. Copy config files from examples
cp config/defaults/*.example.yaml config/

# 2. Create .env file
cat > .env << 'EOF'
SLACK_BOT_TOKEN=xoxb-your-token
OPENAI_API_KEY=sk-your-key
EOF

# 3. Edit all files with your actual values
# - config/main.yaml
# - config/object_registry.yaml
# - config/channels.yaml
```

**Configuration System:**
- All `config/*.yaml` files are automatically loaded and merged
- Validated against JSON schemas in `config/schemas/`
- See [CONFIG.md](CONFIG.md) for detailed documentation

**Example config structure:**
```yaml
llm:
  model: gpt-5-nano
  temperature: 1.0
  timeout_seconds: 120
  daily_budget_usd: 10.0

database:
  type: sqlite  # or postgres for production
  path: data/slack_events.db  # for SQLite
  postgres:  # for PostgreSQL
    host: localhost
    port: 5432
    database: slack_events
    user: postgres

slack:
  digest_channel_id: C789012

processing:
  tz_default: Europe/Amsterdam
  threshold_score_default: 0.0

deduplication:
  date_window_hours: 48
  title_similarity: 0.8

logging:
  level: INFO
```

**Note:**
- `.env` contains ONLY secrets (never committed to git)
- `config/*.yaml` files contain application settings (in `.gitignore`, created from `config/defaults/*.example.yaml`)
- `config/defaults/*.example.yaml` are templates with example values (committed to git)

### 3. Run Pipeline

```bash
# Run complete pipeline
python scripts/run_pipeline.py

# With optional digest publication
python scripts/run_pipeline.py --publish

# Dry run (don't post digest)
python scripts/run_pipeline.py --publish --dry-run
```

## Usage

### Main Pipeline

Process messages end-to-end:

```bash
python scripts/run_pipeline.py [OPTIONS]

Options:
  --publish              Publish digest after processing
  --lookback-hours N     Hours to look back (default: from settings)
  --skip-llm            Skip LLM extraction
  --dry-run             Build but don't post digest
```

### Backfill Historical Data

Process historical messages:

```bash
python scripts/backfill.py \
  --start-date 2025-10-01 \
  --end-date 2025-10-10 \
  --budget-per-day 5.0
```

### Generate Digest

Standalone digest generation:

```bash
python scripts/generate_digest.py [OPTIONS]

Options:
  --date DATE            Date (YYYY-MM-DD, "yesterday", "today")
  --lookback-hours N     Hours to look back (default: 48)
  --channel ID           Override target channel
  --dry-run             Don't post to Slack
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_date_resolver.py -v

# Run with verbose output
pytest tests/ -vv -s
```

Test coverage target: >90% for core services.

## Demo & Testing

### End-to-End Demo

Run a complete demonstration with mock data:

```bash
# Basic demo with mock data
python scripts/demo_e2e.py

# Demo with custom time window
python scripts/demo_e2e.py --hours 72

# Try with real Slack API (requires valid tokens)
python scripts/demo_e2e.py --real --channel C1234567890
```

The demo shows:
- 📥 Message ingestion and processing
- 🎯 Candidate scoring and filtering
- 🤖 LLM event extraction
- 🔗 Event deduplication
- 📋 Beautiful terminal digest display

## Pipeline Stages

### 1. Ingest Messages

Fetches messages from Slack channels:
- Watermark-based incremental fetch
- Root messages only (no threads)
- Extracts links, anchors, reactions
- Normalizes text

### 2. Build Candidates

Scores messages for event extraction:
- Configurable scoring weights
- Keyword matching
- @channel/@here mentions
- Reactions, replies, anchors, links
- Bot penalty

### 3. Extract Events

LLM-powered event extraction:
- 0-5 events per message
- Structured JSON output
- Date/time resolution
- Category classification
- Confidence scoring
- Budget enforcement

### 4. Deduplicate

Merges similar events:
- **Rule 1**: Same message events never merge
- **Rule 2**: Inter-message merge if:
  - Anchor/link overlap
  - Date delta ≤ 48 hours
  - Title similarity ≥ 0.8 (fuzzy)

### 5. Publish Digest

Daily Slack digest:
- Sorted by date, category, confidence
- Slack Block Kit formatting
- Chunking for long digests
- Localized date/time display

## Configuration

### Channel Configuration

Each monitored channel can have custom settings:

```json
{
  "channel_id": "C123456",
  "channel_name": "releases",
  "threshold_score": 0.0,
  "whitelist_keywords": ["release", "deploy", "launch", "update"],
  "keyword_weight": 10.0,
  "mention_weight": 8.0,
  "reply_weight": 5.0,
  "reaction_weight": 3.0,
  "anchor_weight": 4.0,
  "link_weight": 2.0,
  "file_weight": 3.0,
  "bot_penalty": -15.0
}
```

**Important:** By default, `threshold_score` is set to `0.0`, which means **all messages** from the channel will be processed by LLM. This ensures maximum event capture but may increase costs. You can increase this threshold to filter out low-quality messages if needed.

### Scoring System

Messages are scored based on features:

| Feature | Default Weight | Max Contribution |
|---------|----------------|------------------|
| Keywords | 10.0 per keyword | Unlimited |
| @channel/@here | 8.0 | 8.0 |
| Replies ≥1 | 5.0 | 5.0 |
| Reactions ≥2 | 3.0 | 3.0 |
| Anchors | 4.0 per anchor | 12.0 (capped) |
| Links | 2.0 per link | 6.0 (capped) |
| Files | 3.0 | 3.0 |
| Bot | -15.0 | -15.0 |

### Event Categories

- `product`: Releases, features, deployments, launches
- `process`: Internal processes, workflows, policies
- `marketing`: Campaigns, promotions, announcements
- `risk`: Incidents, issues, compliance, security
- `org`: Organizational changes, hiring, team updates
- `unknown`: Unclear or doesn't fit

## Data Model

### SQLite Schema

Three main tables:

1. **raw_slack_messages**: Raw ingested messages
2. **event_candidates**: Scored candidates for LLM processing
3. **events**: Extracted and deduplicated events

Plus auxiliary tables:
- **llm_calls**: LLM API call metadata and costs
- **channel_watermarks**: Incremental processing state

Supports both SQLite (development) and PostgreSQL (production) through repository factory pattern.

## Development

### Code Quality

```bash
# Format code (using ruff for speed)
ruff format src/ tests/ scripts/

# Lint
ruff check src/ tests/ scripts/

# Type check
mypy src/ --config-file pyproject.toml
```

**CI/CD:**
- ⚡ Fast CI with uv (10-100x faster than pip)
- 🔄 Parallel jobs: lint (8s), typecheck (23s), tests (19s)
- 🎯 Total CI time: ~30s (was 2-3min)
- ✅ Pre-commit hooks for automatic formatting

### Project Standards

- **PEP 8** compliance with Black formatting (line length: 88)
- **Type hints** required for all functions
- **Google-style docstrings** for public APIs
- **100% type coverage** with mypy strict mode
- **Specification Pattern** for domain filtering (no string literals)
- **Query Builder** for type-safe database queries
- **Domain constants** with `Final` type hints
- **Ruff PLR2004** enforced (no magic numbers)

**Code Quality Score:** 7/7 (100%)
- ✅ Constants in domain layer
- ✅ Final + type hints
- ✅ Enum/StrEnum usage
- ✅ Config vs domain separation
- ✅ Criteria/Specification pattern
- ✅ PLR2004 compliance
- ✅ Compiled regex patterns

## Troubleshooting

### Slack API Errors

```bash
# Test Slack connection
python -c "from slack_sdk import WebClient; client = WebClient(token='YOUR_TOKEN'); print(client.auth_test())"
```

### OpenAI API Errors

```bash
# Test OpenAI connection
python -c "from openai import OpenAI; client = OpenAI(api_key='YOUR_KEY'); print(client.models.list())"
```

### Budget Exceeded

If daily budget is reached:
- Pipeline stops LLM processing
- Only high-score candidates (P90+) are processed
- Error logged in results

### Database Issues

```bash
# Check database
sqlite3 data/slack_events.db ".tables"

# View recent events
sqlite3 data/slack_events.db "SELECT title, event_date FROM events ORDER BY event_date DESC LIMIT 10"
```

## Recent Updates

### 2025-10-17: PostgreSQL Support ✅

**Production-Ready Database:**
- ✅ Full PostgreSQL integration with Alembic migrations
- ✅ Repository factory pattern for seamless DB switching
- ✅ Docker Compose with PostgreSQL 16 Alpine
- ✅ Auto-migration on container startup
- ✅ 100% backward compatible with SQLite
- ✅ Streamlit UI supports both databases
- ✅ See `MIGRATION_TO_POSTGRES.md` for migration guide

**Key Features:**
- Configuration via `DATABASE_TYPE` environment variable
- Identical schema for SQLite and PostgreSQL
- JSONB support for structured data in PostgreSQL
- Health checks and connection pooling
- 84 tests passing (13 PostgreSQL-specific)

### 2025-10-10: Configuration Refactoring ✅

**Secrets vs Config Separation:**
- ✅ `.env` - Only SLACK_BOT_TOKEN and OPENAI_API_KEY
- ✅ `config.yaml` - All non-sensitive application settings
- ✅ Added PyYAML dependency
- ✅ Backward compatible with `.env` overrides
- ✅ See `CONFIG_REFACTORING.md` for details

### 2025-10-10: Code Quality Enhancement ✅

**Major improvements:**
- ✅ Specification Pattern implementation (330 lines)
- ✅ Query Builder pattern (371 lines)
- ✅ Domain constants layer with Final type hints
- ✅ Ruff PLR2004 enforcement (no magic numbers)
- ✅ Type-safe database queries (no string literals)
- ✅ 100% backward compatibility

**Quality metrics:**
- Tests: 79/79 passing ✅
- Linters: All checks passed ✅
- Code quality: 7/7 (100%) ✅
- Zero breaking changes ✅

### 2025-10-09: Production Validation ✅

- ✅ Tested with 20 real messages
- ✅ 100% LLM extraction success rate
- ✅ Cost: $0.0031 for 20 messages
- ✅ Average latency: 13.5s per LLM call
- ✅ Rate limiting handled gracefully
- ✅ Comprehensive logging added

## Database Configuration

### SQLite (Default - Development)
Perfect for local development and testing. No additional setup required.

```yaml
# config.yaml
database:
  type: sqlite
  path: data/slack_events.db
```

### PostgreSQL (Production)
Recommended for production deployment with Docker.

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

Set password in `.env`:
```bash
POSTGRES_PASSWORD=your_secure_password
```

Run migrations:
```bash
alembic upgrade head
```

See [MIGRATION_TO_POSTGRES.md](MIGRATION_TO_POSTGRES.md) for complete migration guide.

## Future Enhancements

Planned for future releases:

- [ ] Thread/reply processing
- [ ] Edit/delete event handling
- [ ] Semantic search with embeddings
- [ ] Calendar export (Google Calendar, ICS)
- [ ] Real-time streaming mode
- [ ] Enhanced web dashboard with analytics
- [ ] Multi-workspace support

## License

This project is internal tooling. All rights reserved.

## Support

For issues or questions, contact the platform team.

