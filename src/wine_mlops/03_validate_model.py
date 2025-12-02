# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Validate Model
# MAGIC This notebook validates the trained model meets quality thresholds before deployment.

# COMMAND ----------

# MAGIC %pip install mlflow scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Get parameters
try:
    catalog = dbutils.widgets.get("catalog")
    schema = dbutils.widgets.get("schema")
except:
    catalog = "mcp_dabs_test"
    schema = "wine_mlops_dev"

print(f"Catalog: {catalog}, Schema: {schema}")

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# Get run_id and accuracy from upstream task
try:
    run_id = dbutils.jobs.taskValues.get(taskKey="train_model", key="run_id")
    accuracy = dbutils.jobs.taskValues.get(taskKey="train_model", key="accuracy")
except:
    # Fallback for testing - get latest run
    experiment = mlflow.get_experiment_by_name("/Shared/wine_mlops_experiment")
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
        if len(runs) > 0:
            run_id = runs.iloc[0]['run_id']
            accuracy = runs.iloc[0]['metrics.accuracy']
        else:
            raise Exception("No runs found in experiment")
    else:
        raise Exception("Experiment not found")

print(f"Validating run: {run_id}")
print(f"Accuracy: {accuracy}")

# COMMAND ----------

# Validation thresholds
MIN_ACCURACY = 0.85
MIN_F1_SCORE = 0.80

# Get run metrics
run = client.get_run(run_id)
metrics = run.data.metrics

print("Model Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")

# COMMAND ----------

# Validate metrics
validation_passed = True
validation_messages = []

if metrics.get('accuracy', 0) < MIN_ACCURACY:
    validation_passed = False
    validation_messages.append(f"❌ Accuracy {metrics['accuracy']:.4f} < {MIN_ACCURACY}")
else:
    validation_messages.append(f"✅ Accuracy {metrics['accuracy']:.4f} >= {MIN_ACCURACY}")

if metrics.get('f1_score', 0) < MIN_F1_SCORE:
    validation_passed = False
    validation_messages.append(f"❌ F1 Score {metrics['f1_score']:.4f} < {MIN_F1_SCORE}")
else:
    validation_messages.append(f"✅ F1 Score {metrics['f1_score']:.4f} >= {MIN_F1_SCORE}")

print("\nValidation Results:")
for msg in validation_messages:
    print(f"  {msg}")

# COMMAND ----------

if validation_passed:
    print("\n✅ MODEL VALIDATION PASSED - Ready for deployment")
    dbutils.jobs.taskValues.set(key="validation_passed", value=True)
    dbutils.jobs.taskValues.set(key="run_id", value=run_id)
else:
    print("\n❌ MODEL VALIDATION FAILED - Stopping pipeline")
    dbutils.jobs.taskValues.set(key="validation_passed", value=False)
    raise Exception("Model validation failed. See messages above.")
