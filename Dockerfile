FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2, chromadb (chroma-hnswlib), tesseract OCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libpq-dev build-essential tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest yfinance

COPY . .

# Keep container running for interactive use
CMD ["tail", "-f", "/dev/null"]
