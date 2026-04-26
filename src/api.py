from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
import logging
import sys

# Add project root to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import FraudPredictor

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model globally
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")
predictor = None

try:
    if os.path.exists(MODEL_PATH):
        predictor = FraudPredictor(MODEL_PATH)
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. Prediction endpoint will fail.")
except Exception as e:
    logger.error(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not predictor:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        results = predictor.predict(data)
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": predictor is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
