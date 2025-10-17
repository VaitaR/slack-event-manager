# Database Configuration Analysis & Fix Plan

## 🔍 Problem Analysis

### Current Issue
The PostgreSQL migration implementation has a **critical configuration bug** that makes the documented migration path fail:

1. **Documentation says**: Edit `config.yaml` to switch databases
   ```yaml
   database:
     type: postgres
     postgres:
       host: localhost
       port: 5432
       database: slack_events
       user: postgres
   ```

2. **Code does**: Ignores `database` section in `config.yaml` completely
   - Lines 82-85 in `settings.py` only load `db_path` for SQLite
   - PostgreSQL settings (`database_type`, `postgres_host`, etc.) are NOT loaded from YAML
   - Only environment variables work

3. **Result**: Users edit `config.yaml` as documented → nothing happens → SQLite is still used

### Root Cause
In commit `c004770`, I intentionally removed database settings loading with this comment:
```python
# NOTE: database settings (database_type, postgres_*) are NOT loaded from config.yaml
# to allow environment variables to take precedence. Use defaults in Field() or env vars.
```

**This was wrong** because:
- Pydantic Settings **already** handles env var precedence correctly
- Using `setdefault()` doesn't prevent env vars from overriding
- This breaks the project's philosophy: secrets in `.env`, config in `config.yaml`
- Makes the documented migration path fail

## ✅ Correct Approach

### How Pydantic Settings Works
```python
# Priority (highest to lowest):
# 1. Environment variables (e.g., DATABASE_TYPE=postgres)
# 2. Values passed to __init__(**data)
# 3. Field defaults (e.g., Field(default="sqlite"))

# Using setdefault() in __init__:
data.setdefault("database_type", "postgres")  # Only sets if not in data
# If DATABASE_TYPE env var exists, it's already in data → setdefault does nothing
# If DATABASE_TYPE env var missing, use config.yaml value
```

### Solution
Load database settings from `config.yaml` using `setdefault()`, **just like all other settings**:

```python
if "database" in config:
    # Load database type (sqlite or postgres)
    data.setdefault("database_type", config["database"].get("type", "sqlite"))
    
    # Load SQLite path
    data.setdefault("db_path", config["database"].get("path", "data/slack_events.db"))
    
    # Load PostgreSQL settings if present
    if "postgres" in config["database"]:
        pg = config["database"]["postgres"]
        data.setdefault("postgres_host", pg.get("host", "localhost"))
        data.setdefault("postgres_port", pg.get("port", 5432))
        data.setdefault("postgres_database", pg.get("database", "slack_events"))
        data.setdefault("postgres_user", pg.get("user", "postgres"))
```

## 📋 Implementation Plan

### 1. Update `src/config/settings.py`
- [ ] Add PostgreSQL field definitions (database_type, postgres_*)
- [ ] Add `postgres_password` secret field
- [ ] Load all database settings from `config.yaml` in `__init__`
- [ ] Maintain env var override capability
- [ ] Add digest settings (max_events, min_confidence, category_priorities)

### 2. Update `config.yaml`
- [ ] Add `database.type` field with default "sqlite"
- [ ] Add `database.postgres` section with all connection details
- [ ] Add comments explaining env var overrides
- [ ] Add digest configuration section

### 3. Add PostgreSQL Components
- [ ] Re-create `src/adapters/postgres_repository.py`
- [ ] Re-create `src/adapters/repository_factory.py`
- [ ] Re-create Alembic configuration files
- [ ] Re-create `scripts/docker-entrypoint.sh`

### 4. Update Docker
- [ ] Update `docker-compose.yml` with PostgreSQL service
- [ ] Update `Dockerfile` with entrypoint
- [ ] Update `.dockerignore`

### 5. Update Tests
- [ ] Re-create `tests/test_postgres_repository.py`
- [ ] Update `tests/conftest.py` with PostgreSQL fixtures
- [ ] Update `Makefile` with test-postgres target

### 6. Update Scripts
- [ ] Update `scripts/run_pipeline.py` to use factory
- [ ] Update `scripts/generate_digest.py` to use factory
- [ ] Update `scripts/backfill.py` to use factory
- [ ] Update `app.py` to use factory

### 7. Documentation
- [ ] Create `MIGRATION_TO_POSTGRES.md` with correct config examples
- [ ] Update `AGENTS.md` with database configuration section
- [ ] Update `README.md` with database configuration section
- [ ] Add clear examples of env var overrides

## 🎯 Key Principles

1. **Config YAML First**: Default configuration in `config.yaml`
2. **Env Vars Override**: Environment variables can override any setting
3. **Consistency**: Database settings work like LLM, slack, processing settings
4. **Documentation**: Config examples must match code behavior
5. **Testing**: Test both config.yaml and env var approaches

## 🔬 Testing Strategy

### Test Config Priority
```python
# Test 1: config.yaml only
config.yaml: database.type = "postgres"
env vars: none
expected: postgres_repository

# Test 2: env var override
config.yaml: database.type = "sqlite"
env vars: DATABASE_TYPE=postgres
expected: postgres_repository (env wins)

# Test 3: defaults
config.yaml: no database section
env vars: none
expected: sqlite_repository (Field default)
```

### Test Real Scenarios
1. Local dev with SQLite (no config changes)
2. Docker with PostgreSQL (env vars only)
3. Docker with PostgreSQL (config.yaml + secrets in .env)
4. Migration path (edit config.yaml, restart)

## 📊 Expected Behavior After Fix

| Scenario | config.yaml | .env | DATABASE_TYPE env | Result |
|----------|-------------|------|-------------------|--------|
| Default | `type: sqlite` | - | - | SQLite |
| Config only | `type: postgres` | `POSTGRES_PASSWORD` | - | PostgreSQL |
| Env override | `type: sqlite` | - | `DATABASE_TYPE=postgres` | PostgreSQL |
| Docker | `type: postgres` | - | `DATABASE_TYPE=postgres` | PostgreSQL |

## 🚀 Benefits of Correct Implementation

1. ✅ **Documentation works**: Users can follow migration guide
2. ✅ **Flexibility**: Both config.yaml and env vars work
3. ✅ **Consistency**: Database config works like other settings
4. ✅ **Docker-friendly**: Env vars override config for containers
5. ✅ **Local dev**: Simple config.yaml changes, no env vars needed
6. ✅ **Production ready**: Secrets in env, config in YAML

## 🔧 Implementation Order

1. **Phase 1**: Fix Settings class (config loading)
2. **Phase 2**: Add PostgreSQL adapter and factory
3. **Phase 3**: Add Alembic migrations
4. **Phase 4**: Update Docker setup
5. **Phase 5**: Update scripts and tests
6. **Phase 6**: Update documentation
7. **Phase 7**: Test all scenarios
8. **Phase 8**: Create PR with proper description

---

**Status**: Analysis complete, ready for implementation
**Estimated LOC**: ~1500 lines (same as before, but correct this time)
**Breaking Changes**: None (fixing bug, not changing interface)

