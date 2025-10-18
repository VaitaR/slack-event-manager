FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    postgresql-client \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application
COPY . .

# Copy docker entrypoint script
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Create config files from examples if they don't exist
# This ensures the image works even if built from a fresh git clone
RUN for example_file in config/defaults/*.example.yaml; do \
        if [ -f "$example_file" ]; then \
            filename=$(basename "$example_file" .example.yaml); \
            target="config/${filename}.yaml"; \
            if [ ! -f "$target" ]; then \
                cp "$example_file" "$target"; \
                echo "Created $target from example"; \
            fi; \
        fi; \
    done

# Healthcheck: validate settings can be loaded
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from src.config.settings import get_settings; get_settings(); print('OK')" || exit 1

# Set entrypoint for automatic migrations
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command: run pipeline with 1-hour interval
CMD ["python", "scripts/run_pipeline.py", "--interval-seconds", "3600"]
