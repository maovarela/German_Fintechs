FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by scikit-learn's OpenMP-linked binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY api/ ./api/

# src and api are importable as packages
ENV PYTHONPATH=/app

# Persistent directories
RUN mkdir -p /app/data /app/mlruns

# MODE=train runs ingest + train
# MODE=serve starts the FastAPI server
ENV MODE=train

CMD ["sh", "-c", "\
    if [ \"$MODE\" = 'serve' ]; then \
        uvicorn api.app:app --host 0.0.0.0 --port 8000; \
    else \
        python src/ingest.py && python src/train.py; \
    fi"]

EXPOSE 8000
