# Databricks notebook source
# COMMAND ----------
# CELL 1: Import the module

from databricks_metrics_editor import MetricsEditor

# Initialize the editor
editor = MetricsEditor()

# COMMAND ----------
# CELL 2: Load your data

editor.load_data(
    evaluation_data_path="evaluation_data.csv",
    metrics_config_path="sample_metrics_config_simplified.csv",
    ground_truth_files="ground_truth_accuracy.csv;ground_truth_safety.csv"
)

# Display summary
editor.display_summary()

# COMMAND ----------
# CELL 3: Show interactive editor

editor.show_editor()

# COMMAND ----------
# CELL 4: Apply changes (Run after editing)

updated_metrics = editor.apply_changes()

# Display the updated data
display(updated_metrics)

# COMMAND ----------
# CELL 5: Save to file

editor.save_metrics("metrics_config_edited.csv")

# COMMAND ----------
# CELL 6: Access the data in your code

# Access evaluation data
eval_df = editor.evaluation_data
print(f"Evaluation data: {len(eval_df)} rows")

# Access metrics configuration
metrics_df = editor.metrics_config_data
print(f"Metrics config: {len(metrics_df)} rows")

# Access ground truth files
for filename, df in editor.ground_truth_data.items():
    print(f"Ground truth {filename}: {len(df)} rows")

# COMMAND ----------
# CELL 7: Cleanup (optional)

editor.cleanup()
