FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Install python dependencies
RUN pip install --no-cache-dir torch transformers fastapi uvicorn

# Set working dir
WORKDIR /app

# Copy service code
COPY app.py /app/

# Expose port
EXPOSE 8085

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8085"]
