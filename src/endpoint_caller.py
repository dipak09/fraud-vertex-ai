import argparse
import json
import os
from dotenv import load_dotenv
from google.cloud import aiplatform

def predict_fraud(project, location, endpoint_id, instance):
    """
    Calls the Vertex AI Endpoint for fraud risk scoring.
    """
    print(f"Connecting to Vertex AI Endpoint: {endpoint_id} in {project} ({location})...")
    aiplatform.init(project=project, location=location)
    
    endpoint = aiplatform.Endpoint(endpoint_id)
    
    # Vertex AI expects a list of instances
    if isinstance(instance, dict):
        instances = [instance]
    else:
        instances = instance
        
    response = endpoint.predict(instances=instances)
    
    print("✅ Prediction Response:")
    for prediction in response.predictions:
        print(json.dumps(prediction, indent=2))
    
    return response.predictions

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser()
    # Read from .env if available, otherwise require arguments
    parser.add_argument('--project', type=str, default=os.getenv("PROJECT_ID"), help="GCP Project ID")
    parser.add_argument('--location', type=str, default=os.getenv("REGION", "us-central1"), help="GCP Region")
    parser.add_argument('--endpoint-id', type=str, required=True, help="Vertex AI Endpoint ID")
    args = parser.parse_args()
    
    if not args.project:
        print("❌ Error: GCP Project ID is required. Set PROJECT_ID in .env or pass --project")
        exit(1)

    # Example instance
    sample_instance = {
        "transaction_amount": 1250.0,
        "transaction_type": "online",
        "location": "London",
        "device_type": "desktop",
        "account_age_days": 120
    }
    
    predict_fraud(args.project, args.location, args.endpoint_id, sample_instance)
