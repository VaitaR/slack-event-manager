#!/bin/bash
set -e

echo "🔧 Setting up configuration files..."

# Create config directory if missing
mkdir -p config

# Copy all example files from defaults/ to config/
if [ -d config/defaults ]; then
    for example_file in config/defaults/*.example.yaml; do
        if [ -f "$example_file" ]; then
            filename=$(basename "$example_file" .example.yaml)
            target="config/${filename}.yaml"

            if [ ! -f "$target" ]; then
                cp "$example_file" "$target"
                echo "✅ Created $target"
            else
                echo "⏭️  $target already exists"
            fi
        fi
    done
else
    echo "⚠️  config/defaults/ directory not found"
fi

# Create .env if missing
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# Slack Bot Token
SLACK_BOT_TOKEN=xoxb-your-token-here

# OpenAI API Key
OPENAI_API_KEY=sk-your-key-here
EOF
    echo "✅ Created .env"
else
    echo "⏭️  .env already exists"
fi

echo ""
echo "📝 Next steps:"
echo "   1. Edit .env with your API tokens"
echo "   2. Edit config/main.yaml with your settings"
echo "   3. Edit config/object_registry.yaml with your internal systems"
echo "   4. Edit config/channels.yaml with your Slack channels"
echo ""
echo "🚀 Ready to go! Run: python scripts/quick_test.py"
