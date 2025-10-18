# Phase 5 Summary: Use Case Layer - LLM Prompt Loading Complete ✅

**Completion Date:** 2025-10-17
**Status:** LLMClient updated with prompt loading capability

## Overview

Phase 5 focuses on updating the use case layer to support multi-source operations. The first major deliverable—updating LLMClient to load prompts from files—is now complete.

## Completed: LLMClient Prompt Loading ✅

### Implementation Details

**1. Helper Function: `load_prompt_from_file()`**
```python
def load_prompt_from_file(file_path: str) -> str:
    """Load prompt template from a file.

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    prompt_file = Path(file_path)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    return prompt_file.read_text()
```

**2. Updated LLMClient `__init__()`**
```python
def __init__(
    self,
    api_key: str,
    model: str = "gpt-5-nano",
    temperature: float | None = None,
    timeout: int = 30,
    verbose: bool = False,
    prompt_template: str | None = None,  # NEW
    prompt_file: str | None = None,       # NEW
) -> None:
    # ... existing initialization ...

    # Load prompt (priority: file > template > default)
    if prompt_file:
        self.system_prompt = load_prompt_from_file(prompt_file)
    elif prompt_template:
        self.system_prompt = prompt_template
    else:
        self.system_prompt = SYSTEM_PROMPT  # Default Slack prompt
```

**3. Updated `extract_events()` Method**
- Changed all references from `SYSTEM_PROMPT` (global constant) to `self.system_prompt`
- Allows each LLMClient instance to use its own prompt

### Usage Examples

**1. Load from File (Recommended for Multi-Source)**
```python
slack_client = LLMClient(
    api_key="sk-...",
    model="gpt-5-nano",
    temperature=1.0,
    prompt_file="config/prompts/slack.txt"
)

telegram_client = LLMClient(
    api_key="sk-...",
    model="gpt-4o-mini",
    temperature=0.7,
    prompt_file="config/prompts/telegram.txt"
)
```

**2. Custom Template (for Testing)**
```python
test_client = LLMClient(
    api_key="sk-...",
    prompt_template="Extract events from this test message..."
)
```

**3. Default Prompt (Backward Compatible)**
```python
# No prompt parameters = uses SYSTEM_PROMPT constant (Slack prompt)
client = LLMClient(api_key="sk-...")
```

### Prompt File Precedence

The prompt loading follows this priority order:
1. **`prompt_file`** (highest priority) - Load from file
2. **`prompt_template`** - Use provided string
3. **`SYSTEM_PROMPT`** (default) - Use global constant

### Test Coverage ✅

**10 New Tests** (all passing):
1. `test_load_prompt_from_file` - Load from temp file
2. `test_load_prompt_file_not_found` - Error handling
3. `test_llm_client_accepts_custom_prompt` - Template parameter
4. `test_llm_client_uses_default_prompt_if_none_provided` - Fallback
5. `test_llm_client_with_prompt_file_path` - File parameter
6. `test_slack_prompt_file_exists` - Verify Slack prompt
7. `test_telegram_prompt_file_exists` - Verify Telegram prompt
8. `test_prompt_files_contain_required_sections` - Content validation
9. `test_llm_client_prompt_file_overrides_template` - Precedence
10. `test_llm_client_handles_empty_prompt_file` - Edge case

**Test Results:** 65/65 tests passing (100% success rate)

### Backward Compatibility ✅

**Zero Breaking Changes:**
- Existing code without `prompt_file`/`prompt_template` continues to work
- Default behavior: uses `SYSTEM_PROMPT` (Slack-focused prompt)
- All existing scripts run without modifications

### Integration with Multi-Source Architecture

**How It Fits:**
1. **Configuration Layer** (Phase 4) defines `prompt_file` per source in `MessageSourceConfig`
2. **LLMClient** (Phase 5) loads prompts dynamically based on config
3. **Pipeline/Orchestrator** (Next) creates source-specific LLMClient instances

**Example Multi-Source Flow:**
```python
from src.config.settings import get_settings
from src.adapters.llm_client import LLMClient
from src.domain.models import MessageSource

settings = get_settings()

# Get source-specific config
slack_config = settings.get_source_config(MessageSource.SLACK)
telegram_config = settings.get_source_config(MessageSource.TELEGRAM)

# Create source-specific LLM clients
slack_llm = LLMClient(
    api_key=settings.openai_api_key.get_secret_value(),
    model=settings.llm_model,
    prompt_file=slack_config.llm_settings.get("prompt_file")
)

telegram_llm = LLMClient(
    api_key=settings.openai_api_key.get_secret_value(),
    model="gpt-4o-mini",
    temperature=0.7,
    prompt_file=telegram_config.llm_settings.get("prompt_file")
)

# Use in extract_events_use_case
extract_events_use_case(slack_llm, repository, settings)
```

## Next Steps (Remaining Phase 5 Tasks)

### 1. Update `extract_events_use_case` Signature (Optional)
Currently accepts `LLMClient` directly (already flexible):
```python
def extract_events_use_case(
    llm_client: LLMClient,  # Source-specific client
    repository: SQLiteRepository,
    settings: Settings,
    batch_size: int = 50,
    check_budget: bool = True,
) -> ExtractionResult:
```

**No changes needed** - dependency injection already supports multi-source!

### 2. Refactor Pipeline Orchestrator (Critical)
Create new `run_multi_source_pipeline.py`:
```python
def run_for_source(source_config: MessageSourceConfig, settings: Settings):
    # 1. Get source-specific message client
    message_client = get_message_client(
        source_config.source_id,
        bot_token=get_token(source_config.bot_token_env)
    )

    # 2. Create source-specific LLM client
    llm_client = LLMClient(
        api_key=settings.openai_api_key.get_secret_value(),
        prompt_file=source_config.llm_settings.get("prompt_file"),
        temperature=source_config.llm_settings.get("temperature"),
    )

    # 3. Run pipeline steps
    ingest_messages_use_case(message_client, repository, source_config)
    build_candidates_use_case(repository, settings)
    extract_events_use_case(llm_client, repository, settings)
    deduplicate_events_use_case(repository, source_config.source_id)

# Loop through enabled sources
for source_config in settings.get_enabled_sources():
    run_for_source(source_config, settings)
```

### 3. Add Source Filtering to Deduplication
Update `deduplicate_events_use_case` to prevent cross-source merging:
```python
def deduplicate_events_use_case(
    repository: SQLiteRepository,
    source_id: MessageSource | None = None,  # NEW parameter
) -> dict[str, Any]:
    if source_id:
        # Only deduplicate within this source
        events = repository.get_events_by_source(source_id)
    else:
        # Deduplicate all (legacy behavior)
        events = repository.get_all_events()
```

## Files Modified

### Core Implementation
- ✅ `src/adapters/llm_client.py` - Added prompt loading
- ✅ `config/prompts/slack.txt` - Slack extraction prompt
- ✅ `config/prompts/telegram.txt` - Telegram extraction prompt

### Tests
- ✅ `tests/test_llm_prompt_loading.py` - Comprehensive prompt tests

### Documentation
- ✅ `docs/MULTI_SOURCE_PROGRESS.md` - Updated progress
- ✅ `docs/PHASE5_SUMMARY.md` - This document

## Performance Impact

**Zero Performance Overhead:**
- Prompt loaded once during `__init__`
- No file I/O during event extraction
- No changes to LLM API call logic
- Same token costs and latency

## Security Considerations

**File Access:**
- Prompt files stored in `config/prompts/` (version controlled)
- Only local file system access (no remote URLs)
- `FileNotFoundError` raised for missing files

**No Sensitive Data:**
- Prompt templates don't contain secrets
- API keys still loaded from `.env`

## Conclusion

Phase 5.1 (LLM Prompt Loading) is **100% complete** with:
- ✅ Clean implementation with proper error handling
- ✅ Comprehensive test coverage (10 new tests)
- ✅ 100% backward compatibility
- ✅ Zero performance impact
- ✅ Ready for multi-source integration

**Next:** Refactor pipeline orchestrator to loop through sources and create source-specific LLMClient instances.
