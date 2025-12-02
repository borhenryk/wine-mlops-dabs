# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Load Wine Dataset
# MAGIC This notebook downloads the Wine Quality dataset and saves it as a Delta table.

# COMMAND ----------

# MAGIC %pip install scikit-learn

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

print(f"Using catalog: {catalog}, schema: {schema}")

# COMMAND ----------

from sklearn.datasets import load_wine
import pandas as pd

# Load wine dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df['target_name'] = df['target'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})

print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
df.head()

# COMMAND ----------

# Create schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# Convert to Spark DataFrame and save as Delta table
spark_df = spark.createDataFrame(df)
table_name = f"{catalog}.{schema}.wine_data"

spark_df.write.mode("overwrite").saveAsTable(table_name)

print(f"✅ Data saved to {table_name}")
print(f"   Rows: {spark_df.count()}")

# COMMAND ----------

# Verify the table
display(spark.sql(f"SELECT * FROM {table_name} LIMIT 5"))
