# DATABRICKS NOTEBOOK - METRICS EDITOR WIDGET
# Simple, working solution for non-technical users

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? Metrics Configuration Editor
# MAGIC 
# MAGIC **Simple 3-step process:**
# MAGIC 1. See your current metrics in the table below
# MAGIC 2. Edit the CSV text in the widget
# MAGIC 3. Re-run this cell to reload

# COMMAND ----------

import pandas as pd
import json

# Default metrics (your Cinderella example)
DEFAULT_METRICS = [
    {
        "name": "Story_Accuracy",
        "type": "binary",
        "description": "Evaluate if the response is factually accurate about the Cinderella story",
        "grading_rubric": "Score 1 if all story facts are correct and align with the classic Cinderella tale. Score 0 if any facts are incorrect.",
        "threshold": "1",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    },
    {
        "name": "Response_Completeness",
        "type": "1-5_scale",
        "description": "Evaluate how complete and thorough the response is",
        "grading_rubric": "5=Fully complete; 4=Mostly complete; 3=Partially complete; 2=Barely complete; 1=Incomplete",
        "threshold": "4",
        "ground_truth_file_path": "",
        "ground_truth_column": ""
    },
    {
        "name": "Child_Friendliness",
        "type": "percentage",
        "description": "Evaluate what percentage of the response is appropriate for children",
        "grading_rubric": "100%=Perfectly child-friendly; 75%=Mostly appropriate; 50%=Somewhat appropriate; 0%=Not suitable",
        "threshold": "75",
        "ground_truth_file_path": "",
        "ground_truth_column": ""
    }
]

# Initialize widget if not exists
try:
    metrics_csv = dbutils.widgets.get("metrics_csv")
    if not metrics_csv:
        raise Exception("Empty")
except:
    # First run - create widget with defaults
    default_df = pd.DataFrame(DEFAULT_METRICS)
    dbutils.widgets.text("metrics_csv", default_df.to_csv(index=False), "?? Edit Metrics Here (CSV format)")
    metrics_csv = dbutils.widgets.get("metrics_csv")

# Parse CSV
try:
    from io import StringIO
    METRICS_CONFIG_DATA = pd.read_csv(StringIO(metrics_csv))
    
    print("? Successfully loaded metrics configuration!")
    print("="*80)
    print(f"?? Total metrics: {len(METRICS_CONFIG_DATA)}")
    print("="*80)
    
    # Display as nice table
    display(METRICS_CONFIG_DATA)
    
    # Show summary
    print("\n?? Metrics Summary:")
    for idx, row in METRICS_CONFIG_DATA.iterrows():
        metric_type = row['type']
        threshold = row['threshold']
        has_gt = "?" if row.get('ground_truth_file_path') else "?"
        print(f"  {idx+1}. {row['name']} ({metric_type}, threshold={threshold}, ground_truth={has_gt})")
    
    print("\n" + "="*80)
    print("?? To edit:")
    print("   1. Copy the CSV from the widget above")
    print("   2. Edit in Excel/Google Sheets or text editor")
    print("   3. Paste back into the widget")
    print("   4. Re-run this cell")
    print("\n?? To add a metric: Add a new row to the CSV")
    print("?? To delete a metric: Remove the row from CSV")
    print("="*80)
    
except Exception as e:
    print(f"? Error parsing metrics CSV: {e}")
    print("\nUsing default metrics instead.")
    print("Please check your CSV format in the widget above.")
    METRICS_CONFIG_DATA = pd.DataFrame(DEFAULT_METRICS)
    display(METRICS_CONFIG_DATA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? Quick Add Metric (Optional)
# MAGIC 
# MAGIC **Don't want to edit CSV?** Use this form instead!

# COMMAND ----------

# Create individual widgets for quick adding
dbutils.widgets.text("new_metric_name", "", "?? Metric Name")
dbutils.widgets.dropdown("new_metric_type", "binary", ["binary", "1-5_scale", "percentage"], "?? Metric Type")
dbutils.widgets.text("new_metric_description", "", "?? Description")
dbutils.widgets.text("new_metric_rubric", "", "?? Grading Rubric")
dbutils.widgets.text("new_metric_threshold", "", "?? Threshold")
dbutils.widgets.text("new_metric_gt_file", "", "?? Ground Truth File (optional)")
dbutils.widgets.text("new_metric_gt_column", "", "?? Ground Truth Column (optional)")

# Get values
new_name = dbutils.widgets.get("new_metric_name").strip()

if new_name:
    # User filled in the form - add metric
    new_metric = {
        "name": new_name,
        "type": dbutils.widgets.get("new_metric_type"),
        "description": dbutils.widgets.get("new_metric_description"),
        "grading_rubric": dbutils.widgets.get("new_metric_rubric"),
        "threshold": dbutils.widgets.get("new_metric_threshold"),
        "ground_truth_file_path": dbutils.widgets.get("new_metric_gt_file"),
        "ground_truth_column": dbutils.widgets.get("new_metric_gt_column")
    }
    
    # Add to dataframe
    METRICS_CONFIG_DATA = pd.concat([METRICS_CONFIG_DATA, pd.DataFrame([new_metric])], ignore_index=True)
    
    # Update the CSV widget
    dbutils.widgets.remove("metrics_csv")
    dbutils.widgets.text("metrics_csv", METRICS_CONFIG_DATA.to_csv(index=False), "?? Edit Metrics Here (CSV format)")
    
    print("? New metric added!")
    print(f"   Name: {new_name}")
    print(f"   Type: {new_metric['type']}")
    print(f"\n?? Re-run the previous cell to see the updated table")
    print("?? Clear the form widgets to add another metric")
else:
    print("?? Fill in the form widgets above to add a new metric")
    print("?? At minimum, provide a Metric Name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ??? Delete Metric
# MAGIC 
# MAGIC Enter the metric name to delete

# COMMAND ----------

dbutils.widgets.text("delete_metric_name", "", "??? Metric Name to Delete")

delete_name = dbutils.widgets.get("delete_metric_name").strip()

if delete_name:
    if delete_name in METRICS_CONFIG_DATA['name'].values:
        METRICS_CONFIG_DATA = METRICS_CONFIG_DATA[METRICS_CONFIG_DATA['name'] != delete_name]
        
        # Update the CSV widget
        dbutils.widgets.remove("metrics_csv")
        dbutils.widgets.text("metrics_csv", METRICS_CONFIG_DATA.to_csv(index=False), "?? Edit Metrics Here (CSV format)")
        
        print(f"? Deleted metric: {delete_name}")
        print(f"\n?? Re-run Cell 2 to see the updated table")
    else:
        print(f"? Metric '{delete_name}' not found")
        print(f"\nAvailable metrics: {', '.join(METRICS_CONFIG_DATA['name'].values)}")
else:
    print("?? Enter a metric name in the widget above to delete it")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ? Metrics Ready!
# MAGIC 
# MAGIC Your metrics are now stored in `METRICS_CONFIG_DATA` DataFrame.
# MAGIC 
# MAGIC Continue with the rest of your notebook (LLM configuration, evaluation, etc.)
