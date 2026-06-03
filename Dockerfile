FROM python:3.11-slim

LABEL maintainer="Purplle Tech Challenge 2026"
LABEL description="Store Intelligence API — CCTV Analytics"

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY pipeline/ ./pipeline/
COPY data/ ./data/
COPY resources/*.csv ./resources/ 2>/dev/null || true

# Create events directory
RUN mkdir -p data/events

# Environment variables with defaults
ENV DATABASE_URL=sqlite:///data/events.db
ENV PYTHONPATH=/app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
