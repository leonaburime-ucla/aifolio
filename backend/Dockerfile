# ---------- Cloud Run: FastAPI backend ----------
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080

EXPOSE ${PORT}

# Run FastAPI via uvicorn
CMD ["sh", "-c", "uvicorn server.http:app --host 0.0.0.0 --port ${PORT}"]
