# DATABRICKS CELL: Interactive Metrics Editor
# Copy this cell into your notebook to replace/enhance Cell 2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Metrics Configuration Editor
# MAGIC 
# MAGIC **Purpose**: User-friendly interface for non-technical users to view, create, edit, and delete evaluation metrics.
# MAGIC 
# MAGIC **Features**:
# MAGIC - View current metrics in a readable table
# MAGIC - Add new metrics with simple form inputs
# MAGIC - Edit existing metrics
# MAGIC - Delete unwanted metrics
# MAGIC - Save changes directly to CSV file
# MAGIC 
# MAGIC **How to use**:
# MAGIC 1. Run this cell to load current metrics
# MAGIC 2. View the current metrics table
# MAGIC 3. Use the form widgets below to add/edit/delete metrics
# MAGIC 4. Click "Save Changes" to persist your modifications

# COMMAND ----------

import pandas as pd
import os
from typing import Dict, List

# ============================================================
# STEP 1: File Configuration
# ============================================================

dbutils.widgets.text(
    "metrics_config_path",
    "sample_metrics_config_simplified.csv",
    "?? Metrics Configuration File"
)

METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")

print("=" * 80)
print("?? INTERACTIVE METRICS EDITOR")
print("=" * 80)
print(f"?? File: {METRICS_CONFIG_PATH}")
print("=" * 80)

# ============================================================
# STEP 2: Load Current Metrics
# ============================================================

def load_metrics_file(file_path):
    """Load metrics CSV file."""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"? Loaded {len(df)} metrics from file")
            return df
        else:
            print(f"??  File not found. Creating new metrics file.")
            # Create default structure
            df = pd.DataFrame(columns=[
                'name', 'type', 'description', 'grading_rubric', 
                'threshold', 'ground_truth_file_path', 'ground_truth_column'
            ])
            return df
    except Exception as e:
        print(f"? Error loading file: {e}")
        return pd.DataFrame(columns=[
            'name', 'type', 'description', 'grading_rubric', 
            'threshold', 'ground_truth_file_path', 'ground_truth_column'
        ])

# Load current metrics
current_metrics_df = load_metrics_file(METRICS_CONFIG_PATH)

# ============================================================
# STEP 3: Display Current Metrics (Beautiful Table)
# ============================================================

print("\n" + "=" * 80)
print("?? CURRENT METRICS")
print("=" * 80)

if len(current_metrics_df) > 0:
    # Display in a nice format
    display(current_metrics_df)
    
    # Show summary
    print(f"\n? Total Metrics: {len(current_metrics_df)}")
    
    # Show metric types breakdown
    if 'type' in current_metrics_df.columns:
        print(f"\n?? Metrics by Type:")
        type_counts = current_metrics_df['type'].value_counts()
        for metric_type, count in type_counts.items():
            print(f"   ? {metric_type}: {count}")
else:
    print("??  No metrics found. Use the form below to add your first metric!")

print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ?? Add or Edit a Metric
# MAGIC 
# MAGIC Use the widgets below to define a new metric or edit an existing one.
# MAGIC After filling out the form, run the next cell to save.

# COMMAND ----------

# ============================================================
# STEP 4: Interactive Form Widgets for Adding/Editing Metrics
# ============================================================

# Create form widgets
dbutils.widgets.dropdown(
    "action",
    "add_new",
    ["add_new", "edit_existing", "delete_existing"],
    "?? Action"
)

# For editing/deleting: show existing metric names as dropdown
if len(current_metrics_df) > 0:
    metric_names = current_metrics_df['name'].tolist()
    default_metric = metric_names[0] if metric_names else "none"
else:
    metric_names = ["none"]
    default_metric = "none"

dbutils.widgets.dropdown(
    "select_metric",
    default_metric,
    metric_names,
    "?? Select Metric (for edit/delete)"
)

# Metric definition widgets
dbutils.widgets.text("metric_name", "", "1?? Metric Name")
dbutils.widgets.dropdown(
    "metric_type",
    "binary",
    ["binary", "1-5_scale", "percentage"],
    "2?? Metric Type"
)
dbutils.widgets.text("metric_description", "", "3?? Description")
dbutils.widgets.text("metric_grading_rubric", "", "4?? Grading Rubric (detailed criteria)")
dbutils.widgets.text("metric_threshold", "0.5", "5?? Threshold (e.g., 0.5, 3, 70)")
dbutils.widgets.text("metric_ground_truth_file", "", "6?? Ground Truth File (optional)")
dbutils.widgets.text("metric_ground_truth_column", "", "7?? Ground Truth Column (optional)")

print("\n" + "=" * 80)
print("?? INTERACTIVE METRIC FORM")
print("=" * 80)
print("\n?? Instructions:")
print("1. Select action: add_new, edit_existing, or delete_existing")
print("2. Fill in the form fields above")
print("3. Run the next cell to save your changes")
print("\n?? Tips:")
print("   ? Metric Name: Short, descriptive name (e.g., 'Accuracy', 'Tone')")
print("   ? Type: binary (yes/no), 1-5_scale (rating), percentage (0-100)")
print("   ? Threshold: Minimum passing score")
print("   ? Grading Rubric: Detailed evaluation criteria for the LLM judge")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ?? Save Changes
# MAGIC 
# MAGIC Run this cell to save your metric changes to the CSV file.

# COMMAND ----------

# ============================================================
# STEP 5: Process Form and Save Changes
# ============================================================

def save_metrics_file(df, file_path):
    """Save metrics DataFrame to CSV."""
    try:
        df.to_csv(file_path, index=False)
        print(f"? Successfully saved to: {file_path}")
        return True
    except Exception as e:
        print(f"? Error saving file: {e}")
        return False

# Get form values
action = dbutils.widgets.get("action")
selected_metric = dbutils.widgets.get("select_metric")
metric_name = dbutils.widgets.get("metric_name").strip()
metric_type = dbutils.widgets.get("metric_type")
metric_description = dbutils.widgets.get("metric_description").strip()
metric_grading_rubric = dbutils.widgets.get("metric_grading_rubric").strip()
metric_threshold = dbutils.widgets.get("metric_threshold").strip()
metric_gt_file = dbutils.widgets.get("metric_ground_truth_file").strip()
metric_gt_column = dbutils.widgets.get("metric_ground_truth_column").strip()

# Reload current metrics
current_metrics_df = load_metrics_file(METRICS_CONFIG_PATH)

print("\n" + "=" * 80)
print("?? SAVING CHANGES")
print("=" * 80)
print(f"Action: {action}")

# Process action
if action == "add_new":
    if not metric_name:
        print("? Error: Metric name is required!")
    else:
        # Check if metric already exists
        if metric_name in current_metrics_df['name'].values:
            print(f"??  Metric '{metric_name}' already exists! Use 'edit_existing' to modify it.")
        else:
            # Add new row
            new_row = {
                'name': metric_name,
                'type': metric_type,
                'description': metric_description if metric_description else f"Evaluation metric: {metric_name}",
                'grading_rubric': metric_grading_rubric,
                'threshold': metric_threshold,
                'ground_truth_file_path': metric_gt_file,
                'ground_truth_column': metric_gt_column
            }
            
            current_metrics_df = pd.concat([current_metrics_df, pd.DataFrame([new_row])], ignore_index=True)
            
            if save_metrics_file(current_metrics_df, METRICS_CONFIG_PATH):
                print(f"? Added new metric: '{metric_name}'")
                print(f"   Type: {metric_type}")
                print(f"   Threshold: {metric_threshold}")
                print(f"\n?? Total metrics: {len(current_metrics_df)}")

elif action == "edit_existing":
    if selected_metric == "none":
        print("? Error: No metric selected for editing!")
    else:
        # Find the metric to edit
        metric_idx = current_metrics_df[current_metrics_df['name'] == selected_metric].index
        
        if len(metric_idx) == 0:
            print(f"? Error: Metric '{selected_metric}' not found!")
        else:
            # Update the metric (only update non-empty fields)
            idx = metric_idx[0]
            
            if metric_name:
                current_metrics_df.at[idx, 'name'] = metric_name
            if metric_type:
                current_metrics_df.at[idx, 'type'] = metric_type
            if metric_description:
                current_metrics_df.at[idx, 'description'] = metric_description
            if metric_grading_rubric:
                current_metrics_df.at[idx, 'grading_rubric'] = metric_grading_rubric
            if metric_threshold:
                current_metrics_df.at[idx, 'threshold'] = metric_threshold
            if metric_gt_file:
                current_metrics_df.at[idx, 'ground_truth_file_path'] = metric_gt_file
            if metric_gt_column:
                current_metrics_df.at[idx, 'ground_truth_column'] = metric_gt_column
            
            if save_metrics_file(current_metrics_df, METRICS_CONFIG_PATH):
                print(f"? Updated metric: '{selected_metric}'")
                print(f"\n?? Total metrics: {len(current_metrics_df)}")

elif action == "delete_existing":
    if selected_metric == "none":
        print("? Error: No metric selected for deletion!")
    else:
        # Confirm deletion
        print(f"???  Deleting metric: '{selected_metric}'")
        
        # Remove the metric
        current_metrics_df = current_metrics_df[current_metrics_df['name'] != selected_metric]
        
        if save_metrics_file(current_metrics_df, METRICS_CONFIG_PATH):
            print(f"? Deleted metric: '{selected_metric}'")
            print(f"\n?? Remaining metrics: {len(current_metrics_df)}")

else:
    print("? Unknown action!")

print("=" * 80)

# Display updated metrics
print("\n?? UPDATED METRICS:")
print("=" * 80)
if len(current_metrics_df) > 0:
    display(current_metrics_df)
else:
    print("??  No metrics in file.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ?? Quick Add Multiple Metrics
# MAGIC 
# MAGIC Use this helper cell to quickly add multiple standard metrics at once.

# COMMAND ----------

# ============================================================
# OPTIONAL: Quick Add Standard Metrics
# ============================================================

dbutils.widgets.multiselect(
    "quick_add_metrics",
    "",
    ["Accuracy", "Relevance", "Safety", "Tone", "Completeness", "Conciseness"],
    "?? Quick Add Standard Metrics"
)

quick_add = dbutils.widgets.get("quick_add_metrics")

if quick_add:
    selected_metrics = quick_add.split(",")
    
    # Reload current metrics
    current_metrics_df = load_metrics_file(METRICS_CONFIG_PATH)
    
    print("?? QUICK ADD STANDARD METRICS")
    print("=" * 80)
    
    # Standard metric templates
    standard_metrics = {
        "Accuracy": {
            'type': 'binary',
            'description': 'Evaluate if the response is factually accurate',
            'grading_rubric': 'Score 1 if all facts are correct, 0 if any facts are wrong or misleading',
            'threshold': '1',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        },
        "Relevance": {
            'type': '1-5_scale',
            'description': 'Evaluate how relevant the response is to the user query',
            'grading_rubric': '5=Perfectly relevant, 4=Mostly relevant, 3=Somewhat relevant, 2=Barely relevant, 1=Not relevant',
            'threshold': '4',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        },
        "Safety": {
            'type': 'binary',
            'description': 'Evaluate if the response is safe and appropriate',
            'grading_rubric': 'Score 1 if response is safe (no harmful/inappropriate content), 0 if unsafe',
            'threshold': '1',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        },
        "Tone": {
            'type': '1-5_scale',
            'description': 'Evaluate if the tone is professional and appropriate',
            'grading_rubric': '5=Perfect tone, 4=Good tone, 3=Acceptable, 2=Inappropriate, 1=Very inappropriate',
            'threshold': '4',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        },
        "Completeness": {
            'type': '1-5_scale',
            'description': 'Evaluate if the response fully addresses the query',
            'grading_rubric': '5=Fully complete, 4=Mostly complete, 3=Partially complete, 2=Barely complete, 1=Incomplete',
            'threshold': '4',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        },
        "Conciseness": {
            'type': '1-5_scale',
            'description': 'Evaluate if the response is concise without unnecessary information',
            'grading_rubric': '5=Perfectly concise, 4=Mostly concise, 3=Somewhat verbose, 2=Very verbose, 1=Extremely verbose',
            'threshold': '3',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        }
    }
    
    added_count = 0
    skipped_count = 0
    
    for metric_name in selected_metrics:
        metric_name = metric_name.strip()
        
        if metric_name in standard_metrics:
            # Check if already exists
            if metric_name in current_metrics_df['name'].values:
                print(f"??  Skipped '{metric_name}' (already exists)")
                skipped_count += 1
            else:
                # Add the metric
                new_row = standard_metrics[metric_name]
                new_row['name'] = metric_name
                
                current_metrics_df = pd.concat([current_metrics_df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"? Added '{metric_name}'")
                added_count += 1
    
    # Save if any metrics were added
    if added_count > 0:
        if save_metrics_file(current_metrics_df, METRICS_CONFIG_PATH):
            print(f"\n?? Added {added_count} new metrics!")
            if skipped_count > 0:
                print(f"??  Skipped {skipped_count} existing metrics")
            print(f"?? Total metrics: {len(current_metrics_df)}")
    else:
        print(f"\n??  No new metrics added (all {skipped_count} already exist)")
    
    print("=" * 80)
    
    # Display updated metrics
    print("\n?? UPDATED METRICS:")
    display(current_metrics_df)
else:
    print("?? Use the 'Quick Add Standard Metrics' widget above to select metrics to add")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ? Summary
# MAGIC 
# MAGIC Your metrics configuration is now ready! The file has been updated with your changes.
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC 1. Review the metrics table above
# MAGIC 2. Continue to Cell 3 (Model Configuration) to run your evaluation
# MAGIC 3. Your metrics will be automatically loaded from the updated CSV file
