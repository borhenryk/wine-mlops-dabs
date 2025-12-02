# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Train Wine Classifier Model
# MAGIC This notebook trains a RandomForest classifier on the wine dataset with MLflow tracking.

# COMMAND ----------

# MAGIC %pip install scikit-learn mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Get parameters
try:
    catalog = dbutils.widgets.get("catalog")
    schema = dbutils.widgets.get("schema")
    experiment_name = dbutils.widgets.get("experiment_name")
except:
    catalog = "mcp_dabs_test"
    schema = "wine_mlops_dev"
    experiment_name = "/Shared/wine_mlops_experiment"

print(f"Catalog: {catalog}, Schema: {schema}")
print(f"Experiment: {experiment_name}")

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

# Set MLflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# Load data from Delta table
table_name = f"{catalog}.{schema}.wine_data"
df = spark.table(table_name).toPandas()

# Prepare features and target
feature_cols = [c for c in df.columns if c not in ['target', 'target_name']]
X = df[feature_cols]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# COMMAND ----------

# Train model with MLflow tracking
with mlflow.start_run(run_name="wine_rf_classifier") as run:
    # Parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    mlflow.log_params(params)
    
    # Train
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
    
    # Log model
    signature = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(
        model,
        "wine_classifier",
        signature=signature,
        input_example=X_train.iloc[:3]
    )
    
    run_id = run.info.run_id
    print(f"✅ Model trained successfully!")
    print(f"   Run ID: {run_id}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

# COMMAND ----------

# Store run_id for downstream tasks
dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="accuracy", value=accuracy)
print(f"Task values set: run_id={run_id}, accuracy={accuracy}")
