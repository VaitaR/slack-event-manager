FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application
COPY . .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Copy and set up entrypoint script
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Healthcheck: validate settings can be loaded
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from src.config.settings import get_settings; get_settings(); print('OK')" || exit 1

# Set entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command: run pipeline with 1-hour interval
CMD ["python", "scripts/run_pipeline.py", "--interval-seconds", "3600"]
