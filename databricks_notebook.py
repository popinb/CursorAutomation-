# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Notebook Demo
# MAGIC 
# MAGIC This notebook demonstrates common Databricks functionality including:
# MAGIC - Data loading and manipulation
# MAGIC - Data visualization
# MAGIC - Machine learning workflows
# MAGIC - Delta Lake operations
# MAGIC 
# MAGIC ## Setup and Configuration

# COMMAND ----------

# Import common libraries
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Spark session
spark = SparkSession.builder \
    .appName("Databricks Notebook Demo") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

print(f"Spark version: {spark.version}")
print(f"Spark UI: {spark.sparkContext.uiWebUrl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading and Exploration

# COMMAND ----------

# Sample data creation
data = [
    (1, "Alice", 25, "Engineering"),
    (2, "Bob", 30, "Sales"),
    (3, "Charlie", 35, "Marketing"),
    (4, "Diana", 28, "Engineering"),
    (5, "Eve", 32, "HR")
]

schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), False),
    StructField("age", IntegerType(), False),
    StructField("department", StringType(), False)
])

# Create DataFrame
df = spark.createDataFrame(data, schema)
print("Sample DataFrame:")
df.show()

# COMMAND ----------

# Basic DataFrame operations
print("DataFrame Schema:")
df.printSchema()

print("\nDataFrame Statistics:")
df.describe().show()

print("\nDepartment Count:")
df.groupBy("department").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualization

# COMMAND ----------

# Convert to Pandas for visualization
pdf = df.toPandas()

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Age distribution
axes[0].hist(pdf['age'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')

# Department count
dept_counts = pdf['department'].value_counts()
axes[1].bar(dept_counts.index, dept_counts.values, color='lightcoral', alpha=0.7)
axes[1].set_title('Department Distribution')
axes[1].set_xlabel('Department')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Lake Operations

# COMMAND ----------

# Write to Delta format
df.write.format("delta").mode("overwrite").save("/tmp/employee_data")

# Read from Delta format
delta_df = spark.read.format("delta").load("/tmp/employee_data")
print("Data loaded from Delta format:")
delta_df.show()

# Show Delta table history
print("\nDelta table history:")
spark.sql("DESCRIBE HISTORY '/tmp/employee_data'").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Machine Learning Example

# COMMAND ----------

# Create sample ML data
np.random.seed(42)
n_samples = 1000

ml_data = [
    (float(np.random.normal(25, 5)),  # age
     float(np.random.normal(50000, 15000)),  # salary
     int(np.random.choice([0, 1], p=[0.7, 0.3])))  # target (promotion)
    for _ in range(n_samples)
]

ml_schema = StructType([
    StructField("age", FloatType(), False),
    StructField("salary", FloatType(), False),
    StructField("promotion", IntegerType(), False)
])

ml_df = spark.createDataFrame(ml_data, ml_schema)
print("ML Dataset created:")
ml_df.show(5)

# COMMAND ----------

# Feature engineering and model training
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Prepare features
assembler = VectorAssembler(inputCols=["age", "salary"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Create and train model
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="promotion")

pipeline = Pipeline(stages=[assembler, scaler, lr])

# Split data
train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)

# Train model
model = pipeline.fit(train_df)

# Make predictions
predictions = model.transform(test_df)

# Evaluate model
evaluator = BinaryClassificationEvaluator(labelCol="promotion")
auc = evaluator.evaluate(predictions)
print(f"Model AUC: {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Operations

# COMMAND ----------

# Register DataFrame as temp view for SQL
df.createOrReplaceTempView("employees")

# SQL query
sql_result = spark.sql("""
    SELECT 
        department,
        COUNT(*) as employee_count,
        AVG(age) as avg_age
    FROM employees 
    GROUP BY department
    ORDER BY employee_count DESC
""")

print("SQL Query Result:")
sql_result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup and Summary

# COMMAND ----------

# Clean up temporary files
import shutil
try:
    shutil.rmtree("/tmp/employee_data")
    print("Temporary files cleaned up")
except:
    print("Cleanup completed")

# Summary
print("\n=== Notebook Summary ===")
print(f"Total records processed: {df.count()}")
print(f"DataFrame columns: {', '.join(df.columns)}")
print(f"Spark version used: {spark.version}")
print("Notebook execution completed successfully!")