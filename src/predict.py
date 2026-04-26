import joblib
import pandas as pd
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudPredictor:
    def __init__(self, model_path):
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
    def predict(self, input_json):
        """
        Expects input_json to be a list of dictionaries or a single dictionary.
        """
        if isinstance(input_json, dict):
            input_json = [input_json]
            
        df = pd.DataFrame(input_json)
        
        # Ensure all required features are present
        required_features = ['transaction_amount', 'transaction_type', 'location', 'device_type', 'account_age_days']
        for feat in required_features:
            if feat not in df.columns:
                raise ValueError(f"Missing required feature: {feat}")
        
        # Reorder columns to match training
        df = df[required_features]
        
        probabilities = self.model.predict_proba(df)[:, 1]
        predictions = (probabilities > 0.5).astype(int)
        
        results = []
        for prob, pred in zip(probabilities, predictions):
            results.append({
                "fraud_probability": float(prob),
                "is_fraud": int(pred)
            })
            
        # Log for monitoring simulation
        self._log_prediction(input_json, results)
        
        return results

    def _log_prediction(self, inputs, results):
        """Simulates logging for monitoring and drift detection."""
        logger.info(f"Logging prediction: Input={inputs}, Output={results}")
        
        # Placeholder for drift detection
        # In a real system, you'd send this to Vertex AI Model Monitoring or a custom DB
        pass

def main():
    # Simple local test
    model_path = 'model/model.joblib'
    if not os.path.exists(model_path):
        logger.error("Model file not found. Run training first.")
        return

    predictor = FraudPredictor(model_path)
    test_data = {
        "transaction_amount": 4500.0,
        "transaction_type": "atm",
        "location": "New York",
        "device_type": "mobile",
        "account_age_days": 5
    }
    
    result = predictor.predict(test_data)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
