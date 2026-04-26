import kfp
from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, Input, Output, component

# Define Components
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy"]
)
def data_generation(
    n_rows: int,
    fraud_rate: float,
    dataset: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    transaction_amount = np.random.uniform(1, 5000, n_rows)
    transaction_type = np.random.choice(['online', 'in-store', 'atm'], n_rows)
    location = np.random.choice(['New York', 'London', 'Mumbai', 'Tokyo', 'Berlin'], n_rows)
    device_type = np.random.choice(['mobile', 'desktop', 'tablet'], n_rows)
    account_age_days = np.random.randint(1, 3650, n_rows)
    
    fraud_score = (
        (transaction_amount / 5000) * 2 +
        (account_age_days < 30).astype(int) * 3 +
        (transaction_type == 'atm').astype(int) * 1.5 +
        np.random.normal(0, 1, n_rows)
    )
    
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
    
    df.to_csv(dataset.path, index=False)

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn==1.3.2", "joblib", "numpy", "scipy"]
)
def train_model(
    dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics]
):
    import pandas as pd
    import joblib
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    
    print("Starting train_model component...")
    df = pd.read_csv(dataset.path)
    print(f"Loaded dataset with {len(df)} rows")
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Use indices instead of names for compatibility with Vertex AI pre-built container
    categorical_features = [1, 2, 3]
    numeric_features = [0, 4]
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    clf.fit(X_train, y_train)
    
    # Eval
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    metrics.log_metric("accuracy", acc)
    metrics.log_metric("precision", prec)
    metrics.log_metric("recall", rec)
    metrics.log_metric("roc_auc", auc)
    
    # Vertex AI Registry expects the file to be named 'model.joblib'
    model_dir = os.path.dirname(model.path)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, 'model.joblib'))

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform"]
)
def deploy_model(
    model: Input[Model],
    project: str,
    region: str,
    serving_container_image: str,
    endpoint_name: str,
    model_name: str
):
    from google.cloud import aiplatform
    import os
    
    aiplatform.init(project=project, location=region)
    
    # Upload model
    uploaded_model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=os.path.dirname(model.uri),
        serving_container_image_uri=serving_container_image,
    )
    
    # Deploy to endpoint
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    
    uploaded_model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3,
    )

# Define Pipeline
@dsl.pipeline(
    name="fraud-detection-pipeline",
    description="MLOps pipeline for fraud detection"
)
def fraud_pipeline(
    project: str,
    region: str,
    n_rows: int = 10000,
    fraud_rate: float = 0.05,
    serving_container_image: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
):
    data_task = data_generation(n_rows=n_rows, fraud_rate=fraud_rate)
    
    train_task = train_model(dataset=data_task.outputs['dataset'])
    
    deploy_task = deploy_model(
        model=train_task.outputs['model'],
        project=project,
        region=region,
        serving_container_image=serving_container_image,
        endpoint_name="fraud-detection-endpoint",
        model_name="fraud-risk-model"
    )

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from kfp import compiler
    from google.cloud import aiplatform

    # Load environment variables from .env file
    load_dotenv()
    
    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "us-central1")
    bucket_name = os.getenv("BUCKET_NAME")

    # 1. Compile the pipeline to a YAML file
    compiler.Compiler().compile(pipeline_func=fraud_pipeline, package_path="fraud_pipeline.yaml")
    print("Pipeline compiled to fraud_pipeline.yaml")

    # 2. If GCP credentials are set, submit the job to Vertex AI automatically
    if project_id and bucket_name and project_id != "your-project-id":
        print(f"Submitting pipeline to Vertex AI project: {project_id} in {region}")
        aiplatform.init(project=project_id, location=region, staging_bucket=f"gs://{bucket_name}")
        
        job = aiplatform.PipelineJob(
            display_name="fraud-detection-pipeline",
            template_path="fraud_pipeline.yaml",
            parameter_values={
                "project": project_id,
                "region": region
            },
            enable_caching=False
        )
        job.submit()
        print(f"View your pipeline run: https://console.cloud.google.com/vertex-ai/pipelines?project={project_id}")
    else:
        print("Valid PROJECT_ID or BUCKET_NAME not found in .env. Skipping Vertex AI submission.")
