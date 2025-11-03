# Databricks notebook source
# MAGIC %md
# MAGIC ## Interactive Metrics Editor for Non-Technical Users
# MAGIC 
# MAGIC This cell provides a user-friendly interface to:
# MAGIC - View existing metrics
# MAGIC - Add new metrics
# MAGIC - Delete metrics
# MAGIC - Save changes back to CSV

# COMMAND ----------

import pandas as pd
import json
from IPython.display import display, HTML
import os

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Load Current Metrics

# COMMAND ----------

# File path widget
dbutils.widgets.text("metrics_file_path", "sample_metrics_config_simplified.csv", "?? Metrics CSV File")

METRICS_FILE = dbutils.widgets.get("metrics_file_path")

# Load or create metrics dataframe
if os.path.exists(METRICS_FILE):
    metrics_df = pd.read_csv(METRICS_FILE)
    print(f"? Loaded {len(metrics_df)} metrics from {METRICS_FILE}")
else:
    # Create empty template
    metrics_df = pd.DataFrame(columns=[
        'name', 'type', 'description', 'grading_rubric', 
        'threshold', 'ground_truth_file_path', 'ground_truth_column'
    ])
    print(f"?? File not found. Created empty metrics template.")

# Store in temporary location for this session
temp_metrics_df = metrics_df.copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: View Current Metrics

# COMMAND ----------

def display_metrics_table(df):
    """Display metrics in a beautiful, formatted table."""
    
    if len(df) == 0:
        print("?? No metrics defined yet. Use the form below to add your first metric!")
        return
    
    # Create styled HTML table
    html = """
    <style>
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-table th {
            background-color: #2c3e50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        .metrics-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        .metrics-table tr:hover {
            background-color: #f5f5f5;
        }
        .metrics-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .metric-name {
            font-weight: bold;
            color: #3498db;
        }
        .metric-type {
            background-color: #e8f4f8;
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
        }
    </style>
    
    <h3>?? Current Metrics Configuration ({} metrics)</h3>
    <table class="metrics-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Name</th>
                <th>Type</th>
                <th>Description</th>
                <th>Threshold</th>
                <th>Ground Truth File</th>
            </tr>
        </thead>
        <tbody>
    """.format(len(df))
    
    for idx, row in df.iterrows():
        html += f"""
            <tr>
                <td>{idx + 1}</td>
                <td class="metric-name">{row.get('name', 'N/A')}</td>
                <td><span class="metric-type">{row.get('type', 'N/A')}</span></td>
                <td>{str(row.get('description', 'N/A'))[:100]}...</td>
                <td>{row.get('threshold', 'N/A')}</td>
                <td>{row.get('ground_truth_file_path', 'None')}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    <br>
    <p style="color: #7f8c8d; font-size: 12px;">
        ?? <strong>Tip:</strong> Use the form below to add new metrics or delete existing ones by row number.
    </p>
    """
    
    displayHTML(html)

# Display current metrics
display_metrics_table(temp_metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Add New Metric (User-Friendly Form)

# COMMAND ----------

# Create widgets for new metric form
dbutils.widgets.text("new_metric_name", "", "??? Metric Name")
dbutils.widgets.dropdown("new_metric_type", "binary", ["binary", "1-5_scale", "percentage"], "?? Metric Type")
dbutils.widgets.text("new_metric_description", "", "?? Description")
dbutils.widgets.text("new_metric_rubric", "", "?? Grading Rubric")
dbutils.widgets.text("new_metric_threshold", "0.5", "?? Threshold")
dbutils.widgets.text("new_metric_gt_file", "", "?? Ground Truth File (optional)")
dbutils.widgets.text("new_metric_gt_column", "", "?? Ground Truth Column (optional)")

print("? Metric form ready!")
print("\n?? Fill in the fields above and run the next cell to add the metric.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Add Metric Button (Run this cell after filling the form)

# COMMAND ----------

# Get values from widgets
new_metric = {
    'name': dbutils.widgets.get("new_metric_name").strip(),
    'type': dbutils.widgets.get("new_metric_type"),
    'description': dbutils.widgets.get("new_metric_description").strip(),
    'grading_rubric': dbutils.widgets.get("new_metric_rubric").strip(),
    'threshold': dbutils.widgets.get("new_metric_threshold").strip(),
    'ground_truth_file_path': dbutils.widgets.get("new_metric_gt_file").strip(),
    'ground_truth_column': dbutils.widgets.get("new_metric_gt_column").strip()
}

# Validate
if not new_metric['name']:
    print("? Error: Metric name is required!")
else:
    # Add to dataframe
    temp_metrics_df = pd.concat([temp_metrics_df, pd.DataFrame([new_metric])], ignore_index=True)
    
    print(f"? Added metric: '{new_metric['name']}'")
    print(f"?? Total metrics: {len(temp_metrics_df)}")
    print("\n?? Changes are not saved yet! Run the 'Save Metrics' cell below to persist changes.")
    
    # Clear form
    dbutils.widgets.text("new_metric_name", "")
    dbutils.widgets.text("new_metric_description", "")
    dbutils.widgets.text("new_metric_rubric", "")
    
    # Show updated table
    print("\n" + "="*80)
    display_metrics_table(temp_metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Delete Metric by Row Number

# COMMAND ----------

dbutils.widgets.text("delete_row_number", "", "??? Row Number to Delete")

delete_row = dbutils.widgets.get("delete_row_number").strip()

if delete_row:
    try:
        row_idx = int(delete_row) - 1  # Convert to 0-based index
        
        if 0 <= row_idx < len(temp_metrics_df):
            deleted_name = temp_metrics_df.iloc[row_idx]['name']
            temp_metrics_df = temp_metrics_df.drop(temp_metrics_df.index[row_idx]).reset_index(drop=True)
            
            print(f"? Deleted metric: '{deleted_name}' (row {delete_row})")
            print(f"?? Remaining metrics: {len(temp_metrics_df)}")
            print("\n?? Changes are not saved yet! Run the 'Save Metrics' cell below to persist changes.")
            
            # Clear widget
            dbutils.widgets.text("delete_row_number", "")
            
            # Show updated table
            print("\n" + "="*80)
            display_metrics_table(temp_metrics_df)
        else:
            print(f"? Error: Row {delete_row} does not exist. Valid rows: 1-{len(temp_metrics_df)}")
    except ValueError:
        print(f"? Error: '{delete_row}' is not a valid row number")
else:
    print("?? Enter a row number above and run this cell to delete that metric")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Save Metrics to File

# COMMAND ----------

# Add save confirmation
dbutils.widgets.dropdown("confirm_save", "No", ["No", "Yes"], "?? Confirm Save?")

confirm = dbutils.widgets.get("confirm_save")

if confirm == "Yes":
    try:
        # Save to file
        temp_metrics_df.to_csv(METRICS_FILE, index=False)
        
        print("? " + "="*70)
        print(f"? SUCCESS: Saved {len(temp_metrics_df)} metrics to {METRICS_FILE}")
        print("? " + "="*70)
        print("\n?? Saved metrics:")
        
        display_metrics_table(temp_metrics_df)
        
        # Reset confirmation
        dbutils.widgets.dropdown("confirm_save", "No", ["No", "Yes"], "?? Confirm Save?")
        
        print("\n?? You can now use this metrics file in your evaluation notebook!")
        
    except Exception as e:
        print(f"? Error saving file: {e}")
else:
    print("?? Select 'Yes' in the confirmation dropdown above and run this cell to save changes")
    print(f"\n?? Current changes preview ({len(temp_metrics_df)} metrics):")
    display_metrics_table(temp_metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7: View Complete Metrics Details (Expandable)

# COMMAND ----------

# Show detailed view with all columns
if len(temp_metrics_df) > 0:
    print("?? DETAILED METRICS VIEW")
    print("="*80)
    
    for idx, row in temp_metrics_df.iterrows():
        print(f"\n{'='*80}")
        print(f"Metric #{idx + 1}: {row.get('name', 'N/A')}")
        print(f"{'='*80}")
        print(f"  Type:               {row.get('type', 'N/A')}")
        print(f"  Description:        {row.get('description', 'N/A')}")
        print(f"  Grading Rubric:     {row.get('grading_rubric', 'N/A')[:200]}...")
        print(f"  Threshold:          {row.get('threshold', 'N/A')}")
        print(f"  Ground Truth File:  {row.get('ground_truth_file_path', 'None')}")
        print(f"  Ground Truth Col:   {row.get('ground_truth_column', 'None')}")
    
    print(f"\n{'='*80}")
    print(f"Total: {len(temp_metrics_df)} metrics")
    print(f"{'='*80}")
else:
    print("?? No metrics to display")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? How to Use This Metrics Editor
# MAGIC 
# MAGIC ### For Non-Technical Users:
# MAGIC 
# MAGIC 1. **View Current Metrics**: Run Step 2 to see all your metrics in a table
# MAGIC 
# MAGIC 2. **Add a New Metric**:
# MAGIC    - Run Step 3 to show the form
# MAGIC    - Fill in the fields at the top of the notebook:
# MAGIC      - **Metric Name**: Give it a clear name (e.g., "Accuracy Check")
# MAGIC      - **Metric Type**: Choose from dropdown (binary, 1-5_scale, percentage)
# MAGIC      - **Description**: What does this metric evaluate?
# MAGIC      - **Grading Rubric**: Detailed instructions for how to score
# MAGIC      - **Threshold**: Minimum passing score
# MAGIC      - **Ground Truth File**: (Optional) Reference data file name
# MAGIC    - Run Step 4 to add the metric
# MAGIC 
# MAGIC 3. **Delete a Metric**:
# MAGIC    - Note the row number from the table
# MAGIC    - Enter the row number in Step 5
# MAGIC    - Run Step 5 to delete
# MAGIC 
# MAGIC 4. **Save Your Changes**:
# MAGIC    - Review your metrics
# MAGIC    - Select "Yes" in Step 6
# MAGIC    - Run Step 6 to save permanently
# MAGIC 
# MAGIC 5. **View Details**:
# MAGIC    - Run Step 7 to see complete details of all metrics
# MAGIC 
# MAGIC ### Tips:
# MAGIC - Changes are temporary until you run the "Save" step
# MAGIC - You can add multiple metrics before saving
# MAGIC - The table auto-refreshes after each add/delete
# MAGIC - All fields are validated before saving

# COMMAND ----------
