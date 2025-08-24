FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git curl build-essential \
    && pip install --no-cache-dir torch transformers fastapi uvicorn[standard] sentencepiece

COPY app.py .

# Expose port
EXPOSE 8085

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8085"]
