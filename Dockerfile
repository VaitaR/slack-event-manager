FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        postgresql-client \
        sqlite3 \
    && apt-get clean \
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

# Healthcheck: validate settings can be loaded
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from src.config.settings import get_settings; get_settings(); print('OK')" || exit 1

# Set entrypoint for automatic migrations
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command: run pipeline with 1-hour interval
CMD ["python", "scripts/run_pipeline.py", "--interval-seconds", "3600"]
