# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Deploy Model to Unity Catalog
# MAGIC This notebook registers the validated model to Unity Catalog and sets the Champion alias.

# COMMAND ----------

# MAGIC %pip install mlflow

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

model_name = f"{catalog}.{schema}.wine_classifier"
print(f"Deploying model: {model_name}")

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# Get run_id from validation task
try:
    run_id = dbutils.jobs.taskValues.get(taskKey="validate_model", key="run_id")
    validation_passed = dbutils.jobs.taskValues.get(taskKey="validate_model", key="validation_passed")
except:
    # Fallback for testing
    experiment = mlflow.get_experiment_by_name("/Shared/wine_mlops_experiment")
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
        if len(runs) > 0:
            run_id = runs.iloc[0]['run_id']
            validation_passed = True
        else:
            raise Exception("No runs found")
    else:
        raise Exception("Experiment not found")

if not validation_passed:
    raise Exception("Model validation did not pass. Cannot deploy.")

print(f"Deploying run: {run_id}")

# COMMAND ----------

# Register model to Unity Catalog
model_uri = f"runs:/{run_id}/wine_classifier"

try:
    registered_model = mlflow.register_model(model_uri, model_name)
    version = registered_model.version
    print(f"✅ Model registered: {model_name} version {version}")
except Exception as e:
    print(f"Registration error: {e}")
    raise

# COMMAND ----------

# Compare with current Champion (if exists)
try:
    current_champion = client.get_model_version_by_alias(model_name, "Champion")
    print(f"Current Champion: version {current_champion.version}")
    
    # Get metrics for comparison
    current_run = client.get_run(current_champion.run_id)
    current_accuracy = current_run.data.metrics.get('accuracy', 0)
    
    new_run = client.get_run(run_id)
    new_accuracy = new_run.data.metrics.get('accuracy', 0)
    
    print(f"Current Champion accuracy: {current_accuracy:.4f}")
    print(f"New model accuracy: {new_accuracy:.4f}")
    
    if new_accuracy >= current_accuracy:
        print("New model is better or equal - promoting to Champion")
        promote = True
    else:
        print("⚠️ New model is worse - keeping current Champion")
        promote = False
        
except Exception as e:
    print(f"No current Champion found - this will be the first: {e}")
    promote = True

# COMMAND ----------

if promote:
    # Set Champion alias
    client.set_registered_model_alias(model_name, "Champion", version)
    print(f"\n✅ MODEL DEPLOYED SUCCESSFULLY!")
    print(f"   Model: {model_name}")
    print(f"   Version: {version}")
    print(f"   Alias: Champion")
else:
    # Set Challenger alias for the new model
    client.set_registered_model_alias(model_name, "Challenger", version)
    print(f"\n⚠️ Model registered as Challenger (not promoted)")
    print(f"   Model: {model_name}")
    print(f"   Version: {version}")
    print(f"   Alias: Challenger")

# COMMAND ----------

# Verify deployment
print("\nDeployed Model Info:")
model_info = client.get_registered_model(model_name)
print(f"  Name: {model_info.name}")
print(f"  Latest Versions: {[v.version for v in model_info.latest_versions]}")

for alias_info in client.get_registered_model(model_name).aliases:
    print(f"  Alias '{alias_info}' -> version {client.get_model_version_by_alias(model_name, alias_info).version}")
