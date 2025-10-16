#!/bin/bash
set -e

echo "🚀 Starting Slack Event Manager..."

# Check if we're using PostgreSQL
if [ "${DATABASE_TYPE:-sqlite}" = "postgres" ]; then
    echo "📊 PostgreSQL database detected"

    # Wait for PostgreSQL to be ready
    echo "⏳ Waiting for PostgreSQL to be ready..."
    until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -c '\q' 2>/dev/null; do
        echo "   PostgreSQL is unavailable - sleeping"
        sleep 2
    done

    echo "✅ PostgreSQL is ready!"

    # Run Alembic migrations
    echo "🔄 Running database migrations..."
    alembic upgrade head
    echo "✅ Migrations completed!"
else
    echo "📁 SQLite database mode"
fi

# Execute the main command
echo "▶️  Executing: $@"
exec "$@"
