import pandas as pd
import numpy as np
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_data(n_rows=10000, fraud_rate=0.05):
    """Generates synthetic fraud detection dataset."""
    logger.info(f"Generating {n_rows} rows of synthetic data with {fraud_rate*100}% fraud rate.")
    
    np.random.seed(42)
    
    # Features
    transaction_amount = np.random.uniform(1, 5000, n_rows)
    transaction_type = np.random.choice(['online', 'in-store', 'atm'], n_rows)
    location = np.random.choice(['New York', 'London', 'Mumbai', 'Tokyo', 'Berlin'], n_rows)
    device_type = np.random.choice(['mobile', 'desktop', 'tablet'], n_rows)
    account_age_days = np.random.randint(1, 3650, n_rows)
    
    # Synthetic target based on some rules + noise
    # More fraud for high amounts, specific types, and new accounts
    fraud_score = (
        (transaction_amount / 5000) * 2 +
        (account_age_days < 30).astype(int) * 3 +
        (transaction_type == 'atm').astype(int) * 1.5 +
        np.random.normal(0, 1, n_rows)
    )
    
    # Threshold to get ~fraud_rate
    threshold = np.percentile(fraud_score, 100 * (1 - fraud_rate))
    is_fraud = (fraud_score >= threshold).astype(int)
    
    df = pd.DataFrame({
        'transaction_amount': transaction_amount,
        'transaction_type': transaction_type,
        'location': location,
        'device_type': device_type,
        'account_age_days': account_age_days,
        'is_fraud': is_fraud
    })
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default='data/fraud_data.csv', help='Path to save the CSV')
    parser.add_argument('--n-rows', type=int, default=10000, help='Number of rows to generate')
    args = parser.parse_args()

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    df = generate_data(n_rows=args.n_rows)
    df.to_csv(args.output_path, index=False)
    logger.info(f"Data saved to {args.output_path}")

if __name__ == "__main__":
    main()
