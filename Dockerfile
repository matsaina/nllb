FROM python:3.10-slim

RUN apt-get update && apt-get install -y git

# Install dependencies including sentencepiece
RUN pip install --no-cache-dir torch transformers fastapi uvicorn sentencepiece

WORKDIR /app
COPY app.py /app/

EXPOSE 8085

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8085"]
