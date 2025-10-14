# Configuration Refactoring - Secrets vs Config

**Date:** 2025-10-10  
**Status:** ✅ Complete

## 🎯 Motivation

Previously, `.env` file contained both **secrets** (tokens, API keys) and **non-sensitive configuration** (model names, timeouts, paths). This violated the principle of separation of concerns.

## ✅ New Structure

### **`.env` - Secrets Only**

Contains ONLY sensitive data:
```bash
# Slack Bot Token (required)
SLACK_BOT_TOKEN=xoxb-your-token-here

# OpenAI API Key (required)
OPENAI_API_KEY=sk-your-key-here
```

### **`config.yaml` - Application Configuration**

Contains all non-sensitive settings:
```yaml
# LLM Configuration
llm:
  model: gpt-5-nano
  temperature: 1.0
  timeout_seconds: 120
  daily_budget_usd: 10.0
  max_events_per_msg: 5

# Database Configuration
database:
  path: data/slack_events.db

# Slack Configuration
slack:
  digest_channel_id: D07T451C1KK
  lookback_hours_default: 24

# Processing Configuration
processing:
  tz_default: Europe/Amsterdam
  threshold_score_default: 0.0

# Deduplication Configuration
deduplication:
  date_window_hours: 48
  title_similarity: 0.8
  message_lookback_days: 30

# Observability
logging:
  level: INFO
```

## 📝 Changes Made

### 1. Added `config.yaml`
- Created new configuration file for non-sensitive settings
- Structured by logical sections (llm, database, slack, etc.)
- Easy to read and modify without touching secrets

### 2. Updated `settings.py`
- Added `load_config_yaml()` function
- Modified `Settings.__init__()` to load from `config.yaml`
- Maintains backward compatibility with `.env` overrides
- **Priority:** `.env` > `config.yaml` > defaults

### 3. Simplified `.env`
- Removed all non-sensitive settings
- Kept only `SLACK_BOT_TOKEN` and `OPENAI_API_KEY`
- Created `.env.example` with clear instructions

### 4. Added PyYAML dependency
- Updated `requirements.txt` with `pyyaml>=6.0.0`
- Installed in Docker images

## 🔄 Migration Guide

### For Existing Deployments

**Option 1: Keep current .env (works as before)**
```bash
# Your existing .env with all settings will continue to work
# .env values override config.yaml
```

**Option 2: Migrate to new structure (recommended)**
```bash
# 1. Create config.yaml with your settings
cp config.yaml.example config.yaml
vim config.yaml  # Edit your non-sensitive settings

# 2. Simplify .env to only secrets
cat > .env << 'EOF'
SLACK_BOT_TOKEN=your-token
OPENAI_API_KEY=your-key
EOF

# 3. Rebuild Docker (if using Docker)
docker compose down
docker compose build
docker compose up -d
```

## 🎯 Benefits

### Security
- ✅ Secrets are isolated in `.env` (gitignored)
- ✅ Config can be safely committed to git
- ✅ Easier to audit what's sensitive vs what's not

### Maintainability
- ✅ Clear separation of concerns
- ✅ Easier to modify configuration
- ✅ Better for team collaboration
- ✅ Config changes don't require touching secrets

### Flexibility
- ✅ Different configs for dev/staging/prod
- ✅ Easy to version control configuration
- ✅ Can override with environment variables if needed

## 📦 Files Changed

```
✅ config.yaml                    # NEW - Application configuration
✅ .env                           # MODIFIED - Secrets only
✅ .env.example                   # NEW - Template for secrets
✅ src/config/settings.py         # MODIFIED - Loads from config.yaml
✅ requirements.txt               # MODIFIED - Added pyyaml>=6.0.0
✅ Dockerfile                     # No changes needed (copies everything)
✅ docker-compose.yml             # No changes needed
```

## ✅ Verification

### Test Configuration Loading
```bash
# Test locally
python -c "from src.config.settings import get_settings; s = get_settings(); print(f'LLM Model: {s.llm_model}'); print(f'Temperature: {s.llm_temperature}'); print(f'Timeout: {s.llm_timeout_seconds}s')"

# Expected output:
# LLM Model: gpt-5-nano
# Temperature: 1.0
# Timeout: 120s
```

### Test Docker
```bash
# Build and run
docker compose build
docker compose up -d

# Check logs
docker compose logs slack-bot --tail=20

# Should see successful pipeline execution
```

## 🔐 Security Best Practices

### ✅ DO
- Keep `.env` in `.gitignore`
- Use `.env.example` for documentation
- Commit `config.yaml` to git (no secrets!)
- Rotate tokens/keys regularly

### ❌ DON'T
- Don't commit `.env` to git
- Don't put secrets in `config.yaml`
- Don't hardcode secrets in code
- Don't share `.env` files via Slack/email

## 📚 Documentation Updated

- ✅ `CONFIG_REFACTORING.md` (this file)
- ✅ `AGENTS.md` - Updated deployment section
- ✅ `.env.example` - Template for secrets
- ✅ `config.yaml` - Inline comments

## 🎊 Result

**Before:**
```bash
.env (25 lines, mixed secrets and config)
```

**After:**
```bash
.env (5 lines, secrets only)
config.yaml (30 lines, configuration only)
```

**Status:** ✅ Production-ready, tested, and deployed!




