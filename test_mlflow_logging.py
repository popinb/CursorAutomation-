# Databricks notebook source
# MAGIC %md
# MAGIC # Test MLflow Logging
# MAGIC 
# MAGIC This notebook tests MLflow logging functionality to ensure it's working before integrating into the evaluation system.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Import and Setup MLflow

# COMMAND ----------

import mlflow
import pandas as pd
from datetime import datetime
import time

# Get current user
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
print(f"Current user: {username}")

# Set experiment name
experiment_name = f"/Users/{username}/test_mlflow_logging"
mlflow.set_experiment(experiment_name)

print(f"✅ MLflow experiment set: {experiment_name}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Create a Simple Test Run

# COMMAND ----------

# Start a test MLflow run
with mlflow.start_run(run_name=f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
    print(f"🚀 Started MLflow run: {run.info.run_id}")
    
    # Log some test parameters
    mlflow.log_param("test_param_1", "value1")
    mlflow.log_param("test_param_2", 42)
    mlflow.log_param("model_type", "test_model")
    
    # Log some test metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("precision", 0.92)
    mlflow.log_metric("recall", 0.89)
    
    # Log multiple values for a metric (like tracking over time)
    for i in range(5):
        mlflow.log_metric("iteration_score", 0.8 + i * 0.02, step=i)
        time.sleep(0.1)  # Small delay to simulate processing
    
    # Create and log a test artifact (CSV file)
    test_data = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall'],
        'value': [0.95, 0.92, 0.89]
    })
    
    artifact_path = f"/tmp/test_results_{run.info.run_id}.csv"
    test_data.to_csv(artifact_path, index=False)
    mlflow.log_artifact(artifact_path)
    
    # Log a dictionary as JSON
    mlflow.log_dict(
        {"test_config": {"param1": "value1", "param2": 42}}, 
        "test_config.json"
    )
    
    print(f"✅ Logged parameters, metrics, and artifacts")
    print(f"\nRun URL: {mlflow.get_tracking_uri()}#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Verify the Run Was Logged

# COMMAND ----------

# Get MLflow client
client = mlflow.tracking.MlflowClient()

# Get the experiment
experiment = client.get_experiment_by_name(experiment_name)

if experiment:
    print(f"✅ Found experiment: {experiment.name}")
    print(f"   Experiment ID: {experiment.experiment_id}")
    print(f"   Artifact Location: {experiment.artifact_location}")
    
    # Search for runs in this experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=5
    )
    
    print(f"\n📊 Found {len(runs)} runs in experiment:")
    
    for i, run in enumerate(runs):
        print(f"\n   Run {i+1}:")
        print(f"   - Run ID: {run.info.run_id}")
        print(f"   - Run Name: {run.info.run_name}")
        print(f"   - Status: {run.info.status}")
        print(f"   - Start Time: {datetime.fromtimestamp(run.info.start_time/1000)}")
        
        # Show some metrics
        if run.data.metrics:
            print(f"   - Metrics: {list(run.data.metrics.keys())}")
            for key, value in run.data.metrics.items():
                print(f"     • {key}: {value}")
        
        # Show parameters
        if run.data.params:
            print(f"   - Parameters: {list(run.data.params.keys())}")
else:
    print("❌ Experiment not found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Test Creating Multiple Runs (Simulating Multiple Evaluations)

# COMMAND ----------

# Create multiple test runs to simulate multiple evaluations
print("Creating multiple test runs...")

for i in range(3):
    with mlflow.start_run(run_name=f"evaluation_run_{i+1}") as run:
        # Simulate different evaluation results
        accuracy = 0.85 + i * 0.03
        completeness = 0.80 + i * 0.05
        style_score = 0.75 + i * 0.04
        
        # Log parameters
        mlflow.log_param("evaluation_id", f"eval_{i+1}")
        mlflow.log_param("num_samples", 100)
        mlflow.log_param("model", "gpt-3.5-turbo")
        
        # Log metrics
        mlflow.log_metric("accuracy_mean", accuracy)
        mlflow.log_metric("completeness_mean", completeness)
        mlflow.log_metric("style_mean", style_score)
        mlflow.log_metric("overall_score", (accuracy + completeness + style_score) / 3)
        
        print(f"✅ Run {i+1} completed: {run.info.run_id}")
        time.sleep(0.5)  # Small delay between runs

print("\n✅ Created 3 test evaluation runs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Query and Display All Runs

# COMMAND ----------

# Query all runs and display as a summary table
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

# Create a summary dataframe
summary_data = []
for run in runs:
    run_data = {
        'run_name': run.info.run_name,
        'run_id': run.info.run_id[:8] + "...",  # Shortened ID
        'status': run.info.status,
        'start_time': datetime.fromtimestamp(run.info.start_time/1000).strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add metrics
    for key, value in run.data.metrics.items():
        run_data[key] = round(value, 3)
    
    summary_data.append(run_data)

summary_df = pd.DataFrame(summary_data)
print("📊 Summary of all runs in experiment:")
display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Clean Up Test Runs (Optional)

# COMMAND ----------

# Optionally delete test runs to keep experiment clean
# WARNING: This will delete the runs permanently!

# Uncomment the following lines to delete all test runs:
# for run in runs:
#     if run.info.run_name and "test_run" in run.info.run_name:
#         client.delete_run(run.info.run_id)
#         print(f"Deleted test run: {run.info.run_name}")

print("ℹ️ Test runs preserved. Uncomment the code above to delete them.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 MLflow Logging Test Complete!
# MAGIC 
# MAGIC ### ✅ What we tested:
# MAGIC - Created MLflow experiment
# MAGIC - Logged parameters, metrics, and artifacts
# MAGIC - Created multiple runs
# MAGIC - Queried and displayed run data
# MAGIC 
# MAGIC ### 🔍 Next Steps:
# MAGIC 1. Check the MLflow UI to see your runs visually
# MAGIC 2. The experiment is located at: `/Users/{your_username}/test_mlflow_logging`
# MAGIC 3. Once this works, we can integrate MLflow logging into the evaluation notebook
# MAGIC 
# MAGIC ### 📊 Dashboard Requirements:
# MAGIC For the dashboard to work, we need:
# MAGIC - ✅ MLflow experiment with runs (we just created these!)
# MAGIC - ✅ Metrics logged for each evaluation
# MAGIC - ✅ Parameters to track evaluation configuration
# MAGIC - ✅ Artifacts for detailed results
# MAGIC 
# MAGIC **MLflow is working correctly! We can now integrate it into the evaluation system.**