# Vertex AI Fraud Detection System

This project implements a professional MLOps pipeline for real-time fraud risk scoring using Google Cloud Vertex AI and Scikit-Learn.

## Project Structure

- **src/**: Core source code
    - `data_gen.py`: Generates synthetic fraud datasets for training.
    - `train.py`: Trains a RandomForest model using index-based feature mapping.
    - `predict.py`: Contains the inference logic and monitoring placeholders.
    - `api.py`: Local Flask API for testing predictions.
    - `endpoint_caller.py`: Client script for interacting with Vertex AI endpoints.
- **pipeline/**: MLOps orchestration
    - `pipeline.py`: Definition for the Vertex AI Pipeline (Kubeflow).
- **Dockerfile**: Container configuration for custom environments.
- **requirements.txt**: Project dependencies.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **GCP Configuration**:
   Authenticate and set your project details:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project [YOUR_PROJECT_ID]
   ```

## Local Development

### 1. Generate Data
```bash
python src/data_gen.py --output-path data/fraud_data.csv --n-rows 10000
```

### 2. Train Locally
```bash
python src/train.py --data-path data/fraud_data.csv --model-dir model/
```

### 3. Run Local API
```bash
export MODEL_PATH=model/model.joblib
python src/api.py
```
The API will be available at `http://localhost:8080/predict`.

Example request for local testing:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 100.0,
    "transaction_type": "online",
    "location": "London",
    "device_type": "mobile",
    "account_age_days": 30
  }'
```

## Vertex AI Pipeline (Production)

The pipeline automates the entire lifecycle: Data -> Training -> Registry -> Endpoint.

### 1. Execute Pipeline
```bash
python pipeline/pipeline.py
```
This script compiles the pipeline and submits it to Vertex AI.

### 2. Test Deployed Endpoint
You can use the caller script or a direct CURL command.

**Using Python:**
```bash
python src/endpoint_caller.py --endpoint-id [YOUR_ENDPOINT_ID]
```

**Using CURL:**
```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://[REGION]-aiplatform.googleapis.com/v1/projects/[PROJECT_ID]/locations/[REGION]/endpoints/[ENDPOINT_ID]:predict" \
  -d '{
    "instances": [
      [100.0, "online", "London", "mobile", 30]
    ]
  }'
```

## Local vs Production Comparison

| Feature | Local Environment | Production (Vertex AI) |
|---------|-------------------|------------------------|
| **Compute** | Your local machine | Google Cloud (n1-standard-4) |
| **Scaling** | Single process | Autoscaling (1 to 3 nodes) |
| **API Format** | JSON Dictionary | JSON List (NumPy Array style) |
| **Auth** | None (Internal access) | OAuth2 Bearer Token |
| **Performance** | Limited by local hardware | High-availability, low-latency |

## Technical Specifications

- **Model**: RandomForestClassifier (Scikit-Learn 1.3.2)
- **Feature Order**:
    1. `transaction_amount` (Numerical)
    2. `transaction_type` (Categorical)
    3. `location` (Categorical)
    4. `device_type` (Categorical)
    5. `account_age_days` (Numerical)
- **Deployment**: Autoscaling enabled (1-3 replicas) on `n1-standard-4` instances.
