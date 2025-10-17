#!/bin/bash
set -e

echo "🔄 Migrating config.yaml to config/main.yaml..."

if [ -f config.yaml ]; then
    # Backup
    cp config.yaml config.yaml.old-format-backup
    echo "✅ Backed up config.yaml → config.yaml.old-format-backup"

    # Move to config/
    mv config.yaml config/main.yaml
    echo "✅ Moved to config/main.yaml"

    echo ""
    echo "⚠️  IMPORTANT: Review config/main.yaml"
    echo "    The 'channels:' section should be in config/channels.yaml"
    echo "    (config/channels.yaml was created earlier from your config.yaml)"
    echo ""
    echo "Run: python scripts/quick_test.py to verify everything works"
else
    echo "⏭️  No config.yaml found in root, nothing to migrate"
    echo "    (Configuration is already in config/ directory)"
fi
