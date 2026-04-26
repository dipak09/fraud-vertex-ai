import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Mock data
n_rows = 100
df = pd.DataFrame({
    'transaction_amount': np.random.uniform(1, 5000, n_rows),
    'transaction_type': np.random.choice(['online', 'in-store', 'atm'], n_rows),
    'location': np.random.choice(['New York', 'London', 'Mumbai', 'Tokyo', 'Berlin'], n_rows),
    'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_rows),
    'account_age_days': np.random.randint(1, 3650, n_rows),
    'is_fraud': np.random.randint(0, 2, n_rows)
})

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

categorical_features = ['transaction_type', 'location', 'device_type']
numeric_features = ['transaction_amount', 'account_age_days']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

clf.fit(X_train, y_train)
print("Fit successful")
