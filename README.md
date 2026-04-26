# Vertex AI Fraud: Real-Time Fraud Risk Scoring System

This project demonstrates a production-style MLOps pipeline for fraud detection using Google Cloud Vertex AI.

## Project Structure

- `src/`
    - `data_gen.py`: Generates synthetic fraud data.
    - `train.py`: Trains a RandomForest model locally or in a container.
    - `predict.py`: Core inference logic and monitoring placeholders.
    - `api.py`: Flask API for local or custom serving.
    - `endpoint_caller.py`: Script to call a deployed Vertex AI endpoint.
- `pipeline/`
    - `pipeline.py`: Kubeflow Pipelines definition for Vertex AI.
- `Dockerfile`: For custom training and serving containers.
- `requirements.txt`: Python dependencies.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **GCP Configuration**:
   Ensure you have the Google Cloud SDK installed and authenticated:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project [YOUR_PROJECT_ID]
   ```

## Local Development

### 1. Generate Synthetic Data
```bash
python src/data_gen.py --output-path data/fraud_data.csv --n-rows 10000
```

### 2. Train Model Locally
```bash
python src/train.py --data-path data/fraud_data.csv --model-dir model/
```

### 3. Run Inference API Locally
```bash
export MODEL_PATH=model/model.joblib
python src/api.py
```
Test it:
```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{
    "transaction_amount": 4500.0,
    "transaction_type": "atm",
    "location": "New York",
    "device_type": "mobile",
    "account_age_days": 5
}'
```

## MLOps Pipeline (Vertex AI)

### 1. Compile Pipeline
```bash
python pipeline/pipeline.py
```
This generates `fraud_pipeline.yaml`.

### 2. Run Pipeline on Vertex AI
You can upload the `fraud_pipeline.yaml` to the Vertex AI UI or run it via the Python SDK:

```python
from google.cloud import aiplatform

aiplatform.init(project="[YOUR_PROJECT_ID]", location="us-central1")

job = aiplatform.PipelineJob(
    display_name="fraud-detection-pipeline",
    template_path="fraud_pipeline.yaml",
    parameter_values={
        "project": "[YOUR_PROJECT_ID]",
        "region": "us-central1"
    }
)

job.run()
```

## Features & Monitoring

- **Input Features**: `transaction_amount`, `transaction_type`, `location`, `device_type`, `account_age_days`.
- **Output**: `fraud_probability` (0 to 1).
- **Deployment**: Deployed to Vertex AI Endpoint with autoscaling enabled (min 1, max 3 replicas).
- **Monitoring**: Inference logs are printed to stdout, which Cloud Logging captures. A placeholder for drift detection is included in `src/predict.py`.
