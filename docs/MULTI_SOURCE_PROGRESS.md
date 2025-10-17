# Multi-Source Architecture Implementation Progress

**Last Updated:** 2025-10-17  
**Status:** Phase 4 Complete with Prompt Loading (~75% of total implementation)

## Overview

This document tracks progress on the multi-source architecture implementation that enables the Slack Event Manager to support multiple message sources (Slack, Telegram, etc.) with strict isolation and shared processing pipelines.

## Completed Phases

### ✅ Phase 1: Domain Layer (Complete)
**Status:** 100% complete with all tests passing

**Deliverables:**
- ✅ `MessageSource` enum (SLACK, TELEGRAM)
- ✅ `MessageSourceConfig` model for source configuration
- ✅ `TelegramMessage` model for Telegram-specific fields
- ✅ `source_id` field added to `SlackMessage`, `EventCandidate`, `Event`
- ✅ `MessageClientProtocol` for generic message sources
- ✅ Updated `RepositoryProtocol` with source-specific methods
- ✅ 28 passing tests for domain models and protocols

**Key Files:**
- `src/domain/models.py` - Core data models with multi-source support
- `src/domain/protocols.py` - Abstract interfaces for adapters
- `tests/test_domain_models_multi_source.py` - Domain model tests
- `tests/test_protocols_multi_source.py` - Protocol tests

### ✅ Phase 2: Repository Layer (Complete)
**Status:** 100% complete with 100% backward compatibility

**Deliverables:**
- ✅ `raw_telegram_messages` table for Telegram-specific fields
- ✅ Source-specific ingestion state tables (`ingestion_state_slack`, `ingestion_state_telegram`)
- ✅ `source_id` column added to `event_candidates` and `events` tables
- ✅ `save_telegram_messages()` and `get_telegram_messages()` methods
- ✅ `get_candidates_by_source()` and `get_events_by_source()` filtering
- ✅ Updated state tracking with `source_id` parameter
- ✅ Backward compatibility maintained for legacy Slack-only tables
- ✅ 19 new repository tests, all passing (total: 204 tests)

**Key Files:**
- `src/adapters/sqlite_repository.py` - Multi-source repository implementation
- `tests/test_repository_multi_source.py` - Repository multi-source tests

**Database Schema Changes:**
```sql
-- New tables
CREATE TABLE raw_telegram_messages (...)
CREATE TABLE ingestion_state_slack (...)
CREATE TABLE ingestion_state_telegram (...)

-- Updated tables
ALTER TABLE event_candidates ADD COLUMN source_id TEXT DEFAULT 'slack'
ALTER TABLE events ADD COLUMN source_id TEXT DEFAULT 'slack'
```

### ✅ Phase 3: Adapters Layer (Complete)
**Status:** 100% complete with all tests passing

**Deliverables:**
- ✅ `TelegramClient` stub that returns empty message lists
- ✅ `message_client_factory.py` for source-based client instantiation
- ✅ Factory pattern implementation with `get_message_client()`
- ✅ Protocol compliance verification
- ✅ 20 new adapter tests, all passing (total: 224 tests)

**Key Files:**
- `src/adapters/telegram_client.py` - Telegram client stub
- `src/adapters/message_client_factory.py` - Client factory
- `tests/test_telegram_client.py` - TelegramClient tests
- `tests/test_message_client_factory.py` - Factory tests

**Factory Usage:**
```python
from src.adapters.message_client_factory import get_message_client
from src.domain.models import MessageSource

# Get appropriate client based on source
slack_client = get_message_client(MessageSource.SLACK, bot_token="xoxb-...")
telegram_client = get_message_client(MessageSource.TELEGRAM, bot_token="...")
```

### ✅ Phase 4: Configuration Layer (Complete)
**Status:** 100% complete with auto-migration

**Deliverables:**
- ✅ `MessageSourceConfig` Pydantic model
- ✅ `message_sources` field in `Settings` class
- ✅ Auto-migration from legacy `channels` config to `message_sources` format
- ✅ `get_source_config()` and `get_enabled_sources()` helper methods
- ✅ Per-source LLM settings (temperature, timeout, prompt file)
- ✅ `config/prompts/slack.txt` - Slack-specific extraction prompt
- ✅ `config/prompts/telegram.txt` - Telegram-specific extraction prompt
- ✅ 16 new configuration tests, all passing (total: 240 tests)
- ✅ 100% backward compatibility with existing deployments

**Key Files:**
- `src/config/settings.py` - Settings with multi-source support and auto-migration
- `src/domain/models.py` - MessageSourceConfig model
- `config/prompts/slack.txt` - Slack extraction prompt
- `config/prompts/telegram.txt` - Telegram extraction prompt
- `tests/test_config_integration.py` - Configuration integration tests
- `tests/test_settings_multi_source.py` - Settings tests

**Configuration Format:**
```yaml
# New format (multi-source)
message_sources:
  - source_id: slack
    enabled: true
    bot_token_env: SLACK_BOT_TOKEN
    raw_table: raw_slack_messages
    state_table: ingestion_state_slack
    prompt_file: config/prompts/slack.txt
    llm_settings:
      temperature: 1.0
      timeout_seconds: 30
    channels:
      - C123
      - C456

  - source_id: telegram
    enabled: false
    bot_token_env: TELEGRAM_BOT_TOKEN
    raw_table: raw_telegram_messages
    state_table: ingestion_state_telegram
    prompt_file: config/prompts/telegram.txt
    llm_settings:
      temperature: 0.7
      timeout_seconds: 30
    channels: []

# Legacy format (auto-migrated to message_sources)
channels:
  - channel_id: C123
    channel_name: releases
  - channel_id: C456
    channel_name: updates
```

**Auto-Migration:**
- Existing configs with `channels` are automatically converted to `message_sources` format
- Slack source is created with defaults from global `llm` settings
- Zero breaking changes for existing deployments

### ✅ Phase 4.1: LLM Prompt Loading (Complete)
**Status:** 100% complete with all tests passing

**Deliverables:**
- ✅ `load_prompt_from_file()` helper function in LLMClient
- ✅ LLMClient accepts `prompt_template` and `prompt_file` parameters
- ✅ Prompt file takes precedence over template over default  
- ✅ `self.system_prompt` attribute set dynamically based on parameters
- ✅ All LLM extraction calls use `self.system_prompt` instead of global constant
- ✅ 10 new prompt loading tests, all passing (total: 65 tests)

**Key Changes:**
- `src/adapters/llm_client.py`:
  - Added `load_prompt_from_file()` function
  - Added `prompt_template` and `prompt_file` parameters to `__init__`
  - Updated `extract_events()` to use `self.system_prompt`
- `tests/test_llm_prompt_loading.py`: Comprehensive prompt loading tests

**Usage:**
```python
# With prompt file
client = LLMClient(
    api_key="sk-...",
    prompt_file="config/prompts/slack.txt"
)

# With custom template
client = LLMClient(
    api_key="sk-...",
    prompt_template="Custom extraction prompt..."
)
```

## Remaining Phases

### ✅ Phase 5: Use Case Layer (Complete)
**Status:** 100% complete

**Completed:**
- ✅ Update `LLMClient` to accept `prompt_template` parameter and load from files
- ✅ Add `source_id` filtering to deduplication (prevent cross-source merging)
- ✅ Create multi-source orchestrator (`scripts/run_multi_source_pipeline.py`)

**Key Changes:**
- `src/use_cases/deduplicate_events.py`: Added optional `source_id` parameter
- Deduplication now supports strict source isolation
- Backward compatible: `source_id=None` deduplicates all sources (legacy behavior)
- `scripts/run_multi_source_pipeline.py`: Complete orchestrator for all enabled sources

**Orchestrator Features:**
- Loops through all enabled sources from configuration
- Creates source-specific clients (message client, LLM client with custom prompts)
- Runs full pipeline for each source independently
- Aggregates statistics across all sources
- Supports graceful shutdown, continuous mode, backfill
- Strict source isolation in deduplication

**Note:** `ingest_messages.py` refactoring deferred (works with existing Slack-only logic, orchestrator handles multi-source routing)

### 📝 Phase 6: CLI & Scripts (Planned)
**Status:** Not started

**Planned Deliverables:**
- [ ] Add `--source` CLI flag to `run_pipeline.py`
- [ ] Create source-specific pipeline scripts
- [ ] Create `migrate_multi_source.py` for schema migration
- [ ] Update existing scripts for multi-source support

## Test Coverage

**Total Tests:** 240 (100% passing)
- Domain models: 28 tests
- Protocols: 10 tests  
- Repository: 19 tests
- Adapters: 20 tests
- Configuration: 16 tests
- Other tests: 147 tests

**Test Coverage:** 59% overall, 97% on new multi-source code

## Architecture Decisions

### Source Isolation Strategy
**Decision:** Strict isolation (separate event streams, no cross-source merging)

**Rationale:**
- Each source maintains independent data and processing
- Deduplication only within source boundaries
- Clearer ownership and debugging

### LLM Prompt Configuration
**Decision:** Prompt files in `config/prompts/` with YAML references

**Rationale:**
- Easier to edit and version control prompts
- Per-source customization without code changes
- Clear separation of configuration and code

### Ingestion State Tracking
**Decision:** Separate state tables per source

**Rationale:**
- Independent progress tracking
- No state collisions between sources
- Easier to reset/debug individual sources

### Pipeline Orchestration
**Decision:** Separate scripts per source for independent scheduling

**Rationale:**
- Different cadences for different sources
- Independent failure handling
- Simpler deployment and monitoring

### Backward Compatibility
**Decision:** Auto-migrate Slack settings to `message_sources` format on load

**Rationale:**
- Zero breaking changes for existing deployments
- Seamless transition to multi-source architecture
- Preserves all existing functionality

## Next Steps

1. **Phase 5: Use Case Layer**
   - Update LLMClient for prompt file loading
   - Refactor ingestion and extraction use cases
   - Implement source-specific orchestration

2. **Phase 6: CLI & Scripts**
   - Add CLI arguments for source selection
   - Create migration scripts
   - Update documentation

3. **Documentation**
   - Create comprehensive `MULTI_SOURCE.md` guide
   - Update `README.md` with multi-source examples
   - Document configuration options

## Timeline

- **Phase 1-2:** Completed 2025-10-17 (Domain + Repository)
- **Phase 3:** Completed 2025-10-17 (Adapters)
- **Phase 4:** Completed 2025-10-17 (Configuration)
- **Phase 5:** In progress (Use Cases)
- **Phase 6:** Planned (CLI & Scripts)

**Estimated Completion:** Phase 5-6 require approximately 2-3 hours of additional work.

## Related Documentation

- `docs/MULTI_SOURCE_IMPLEMENTATION_SUMMARY.md` - Overall implementation summary
- `docs/MULTI_SOURCE_NEXT_STEPS.md` - Detailed next steps guide
- `docs/PHASE2_REPOSITORY_SUMMARY.md` - Phase 2 detailed summary
- `AGENTS.md` - Project overview with recent changes
