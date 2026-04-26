FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Environment variables for GCP
ENV GOOGLE_APPLICATION_CREDENTIALS=""

# Entrypoint will be overridden by Vertex AI but we can set a default
ENTRYPOINT ["python", "src/train.py"]
