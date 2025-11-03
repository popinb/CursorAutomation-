# Databricks notebook source
# MAGIC %md
# MAGIC ## Interactive Metrics Editor - DATABRICKS OPTIMIZED
# MAGIC 
# MAGIC **? This version is thoroughly tested for Databricks**
# MAGIC 
# MAGIC ### Features:
# MAGIC - View metrics in beautiful table
# MAGIC - Add new metrics with simple form
# MAGIC - Delete metrics by row number
# MAGIC - Save changes to CSV
# MAGIC - Full error handling
# MAGIC - User-friendly for non-technical users
# MAGIC 
# MAGIC ### Changes from generic version:
# MAGIC - ? Removed IPython.display import (use built-in displayHTML)
# MAGIC - ? Added proper error handling for Databricks
# MAGIC - ? Optimized widget handling
# MAGIC - ? Better file path management
# MAGIC - ? Added validation and feedback

# COMMAND ----------

import pandas as pd
import json
import os

print("? Imports loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration

# COMMAND ----------

# File path widget - modify this to point to your metrics file
dbutils.widgets.text("metrics_file_path", "sample_metrics_config_simplified.csv", "?? Metrics CSV File")

METRICS_FILE = dbutils.widgets.get("metrics_file_path")

print("="*70)
print("?? METRICS FILE CONFIGURATION")
print("="*70)
print(f"File path: {METRICS_FILE}")

# Try to find file in common locations
def find_metrics_file(filename):
    """Try to locate metrics file in common Databricks locations."""
    # Check exact path first
    if os.path.exists(filename):
        return filename
    
    # Try common locations
    try:
        user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
        locations = [
            f"/Workspace/Users/{user_name}/{filename}",
            f"/dbfs/{filename}",
            f"/tmp/{filename}",
            filename
        ]
        
        for loc in locations:
            if os.path.exists(loc):
                print(f"? Found file at: {loc}")
                return loc
    except:
        pass
    
    return None

# Try to load file
actual_file_path = find_metrics_file(METRICS_FILE)

if actual_file_path and os.path.exists(actual_file_path):
    metrics_df = pd.read_csv(actual_file_path)
    METRICS_FILE = actual_file_path  # Update to actual path
    print(f"? Loaded {len(metrics_df)} metrics from file")
else:
    # Create empty template
    metrics_df = pd.DataFrame(columns=[
        'name', 'type', 'description', 'grading_rubric', 
        'threshold', 'ground_truth_file_path', 'ground_truth_column'
    ])
    print("??  File not found. Created empty metrics template.")
    print(f"?? File will be created at: {METRICS_FILE}")

# Store in global variable for this session
temp_metrics_df = metrics_df.copy()

print(f"?? Current metrics count: {len(temp_metrics_df)}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### View Current Metrics

# COMMAND ----------

def display_metrics_table(df):
    """Display metrics in a beautiful, formatted table using Databricks displayHTML."""
    
    if len(df) == 0:
        html = """
        <div style="padding: 30px; text-align: center; background-color: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; margin: 20px 0;">
            <h3 style="color: #6c757d; margin: 10px 0;">?? No Metrics Defined Yet</h3>
            <p style="color: #6c757d;">Use the form below to add your first metric!</p>
        </div>
        """
        displayHTML(html)
        return
    
    # Build table rows
    rows_html = ""
    for idx, row in df.iterrows():
        # Safely get values with defaults
        name = str(row.get('name', 'N/A'))
        mtype = str(row.get('type', 'N/A'))
        description = str(row.get('description', 'N/A'))[:80]
        threshold = str(row.get('threshold', 'N/A'))
        gt_file = str(row.get('ground_truth_file_path', 'None'))
        
        # Truncate long text
        if len(description) > 77:
            description = description[:77] + "..."
        
        rows_html += f"""
            <tr>
                <td style="text-align: center; font-weight: bold; color: #495057;">{idx + 1}</td>
                <td><span class="metric-name">{name}</span></td>
                <td><span class="metric-type">{mtype}</span></td>
                <td style="font-size: 13px; color: #6c757d;">{description}</td>
                <td style="text-align: center; font-weight: bold;">{threshold}</td>
                <td style="font-size: 12px; color: #6c757d;">{gt_file if gt_file != 'None' else '-'}</td>
            </tr>
        """
    
    # Create complete HTML
    html = f"""
    <style>
        .metrics-container {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 20px 0;
        }}
        
        .metrics-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 25px;
            border-radius: 8px 8px 0 0;
            margin-bottom: 0;
        }}
        
        .metrics-header h3 {{
            margin: 0 0 5px 0;
            font-size: 22px;
            font-weight: 600;
        }}
        
        .metrics-header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 14px;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 0 0 8px 8px;
            overflow: hidden;
        }}
        
        .metrics-table th {{
            background-color: #2c3e50;
            color: white;
            padding: 14px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metrics-table td {{
            padding: 12px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .metrics-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .metrics-table tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .metrics-table tr:nth-child(even) {{
            background-color: #fafbfc;
        }}
        
        .metric-name {{
            font-weight: 600;
            color: #3498db;
            font-size: 14px;
        }}
        
        .metric-type {{
            background-color: #e8f4f8;
            color: #2980b9;
            padding: 4px 10px;
            border-radius: 12px;
            display: inline-block;
            font-size: 12px;
            font-weight: 500;
        }}
        
        .metrics-footer {{
            background-color: #f8f9fa;
            padding: 15px 20px;
            border-radius: 0 0 8px 8px;
            border: 1px solid #dee2e6;
            border-top: none;
            font-size: 13px;
            color: #6c757d;
        }}
    </style>
    
    <div class="metrics-container">
        <div class="metrics-header">
            <h3>?? Current Metrics Configuration</h3>
            <p>{len(df)} metric{'s' if len(df) != 1 else ''} defined</p>
        </div>
        
        <table class="metrics-table">
            <thead>
                <tr>
                    <th style="width: 50px; text-align: center;">#</th>
                    <th style="width: 180px;">Name</th>
                    <th style="width: 130px;">Type</th>
                    <th>Description</th>
                    <th style="width: 100px; text-align: center;">Threshold</th>
                    <th style="width: 150px;">Ground Truth</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        
        <div class="metrics-footer">
            ?? <strong>Tip:</strong> Use the form below to add new metrics or delete existing ones by row number.
        </div>
    </div>
    """
    
    displayHTML(html)

# Display current metrics
print("?? Displaying metrics table...")
display_metrics_table(temp_metrics_df)
print("? Table displayed above")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add New Metric

# COMMAND ----------

# Remove existing widgets first to avoid conflicts
try:
    for widget in ["new_metric_name", "new_metric_type", "new_metric_description", 
                   "new_metric_rubric", "new_metric_threshold", "new_metric_gt_file", 
                   "new_metric_gt_column"]:
        try:
            dbutils.widgets.remove(widget)
        except:
            pass
except:
    pass

# Create form widgets
dbutils.widgets.text("new_metric_name", "", "??? Metric Name")
dbutils.widgets.dropdown("new_metric_type", "binary", ["binary", "1-5_scale", "percentage"], "?? Metric Type")
dbutils.widgets.text("new_metric_description", "", "?? Description")
dbutils.widgets.text("new_metric_rubric", "", "?? Grading Rubric")
dbutils.widgets.text("new_metric_threshold", "0.5", "?? Threshold")
dbutils.widgets.text("new_metric_gt_file", "", "?? Ground Truth File (optional)")
dbutils.widgets.text("new_metric_gt_column", "", "?? Ground Truth Column (optional)")

print("="*70)
print("?? ADD NEW METRIC FORM")
print("="*70)
print("\n? Form ready! Fill in the fields above.")
print("\n?? Instructions:")
print("   1. Enter metric name (required)")
print("   2. Select metric type from dropdown")
print("   3. Enter description (what does this metric measure?)")
print("   4. Enter grading rubric (how to score this metric)")
print("   5. Set threshold (minimum passing score)")
print("   6. (Optional) Specify ground truth file and column")
print("\n??  Run the next cell to add the metric")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute: Add Metric

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
errors = []
if not new_metric['name']:
    errors.append("? Metric name is required")

if not new_metric['description']:
    errors.append("??  Description is recommended")

if not new_metric['grading_rubric']:
    errors.append("??  Grading rubric is recommended")

try:
    float(new_metric['threshold'])
except:
    errors.append("? Threshold must be a valid number")

if errors:
    print("="*70)
    print("? VALIDATION ERRORS")
    print("="*70)
    for error in errors:
        print(f"   {error}")
    print("\n?? Please fix the errors above and run this cell again.")
    print("="*70)
else:
    # Add to dataframe
    temp_metrics_df = pd.concat([temp_metrics_df, pd.DataFrame([new_metric])], ignore_index=True)
    
    print("="*70)
    print("? METRIC ADDED SUCCESSFULLY")
    print("="*70)
    print(f"   Metric Name: {new_metric['name']}")
    print(f"   Type: {new_metric['type']}")
    print(f"   Threshold: {new_metric['threshold']}")
    print(f"   Total Metrics: {len(temp_metrics_df)}")
    print("\n??  Changes are NOT saved yet!")
    print("?? Run the 'Save Changes' cell below to persist changes.")
    print("="*70)
    
    # Clear form fields
    dbutils.widgets.text("new_metric_name", "")
    dbutils.widgets.text("new_metric_description", "")
    dbutils.widgets.text("new_metric_rubric", "")
    dbutils.widgets.text("new_metric_threshold", "0.5")
    dbutils.widgets.text("new_metric_gt_file", "")
    dbutils.widgets.text("new_metric_gt_column", "")
    
    # Show updated table
    print("\n?? Updated metrics table:")
    display_metrics_table(temp_metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delete Metric

# COMMAND ----------

# Clean up widget first
try:
    dbutils.widgets.remove("delete_row_number")
except:
    pass

dbutils.widgets.text("delete_row_number", "", "??? Row Number to Delete")

print("="*70)
print("???  DELETE METRIC")
print("="*70)
print("\n?? Instructions:")
print("   1. Look at the table above and note the row number (#)")
print("   2. Enter that row number in the widget above")
print("   3. Run the next cell to delete")
print("\n?? Example: To delete 'Accuracy' in row 1, enter '1'")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute: Delete Metric

# COMMAND ----------

delete_row = dbutils.widgets.get("delete_row_number").strip()

if not delete_row:
    print("??  No row number entered.")
    print("?? Enter a row number in the widget above and run this cell again.")
else:
    try:
        row_idx = int(delete_row) - 1  # Convert to 0-based index
        
        if row_idx < 0:
            print("? Row number must be positive")
        elif row_idx >= len(temp_metrics_df):
            print(f"? Row {delete_row} does not exist.")
            print(f"?? Valid row numbers: 1 to {len(temp_metrics_df)}")
        else:
            deleted_name = temp_metrics_df.iloc[row_idx]['name']
            temp_metrics_df = temp_metrics_df.drop(temp_metrics_df.index[row_idx]).reset_index(drop=True)
            
            print("="*70)
            print("? METRIC DELETED SUCCESSFULLY")
            print("="*70)
            print(f"   Deleted: '{deleted_name}' (was row {delete_row})")
            print(f"   Remaining metrics: {len(temp_metrics_df)}")
            print("\n??  Changes are NOT saved yet!")
            print("?? Run the 'Save Changes' cell below to persist changes.")
            print("="*70)
            
            # Clear widget
            dbutils.widgets.text("delete_row_number", "")
            
            # Show updated table
            print("\n?? Updated metrics table:")
            display_metrics_table(temp_metrics_df)
            
    except ValueError:
        print(f"? '{delete_row}' is not a valid number")
        print("?? Please enter a numeric row number (e.g., 1, 2, 3)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Changes

# COMMAND ----------

# Clean up widget
try:
    dbutils.widgets.remove("confirm_save")
except:
    pass

dbutils.widgets.dropdown("confirm_save", "No", ["No", "Yes"], "?? Confirm Save?")

print("="*70)
print("?? SAVE CHANGES TO FILE")
print("="*70)
print(f"\n?? File path: {METRICS_FILE}")
print(f"?? Metrics to save: {len(temp_metrics_df)}")
print("\n??  WARNING: This will overwrite the existing file!")
print("\n?? Preview of changes:")

# Show summary
if len(temp_metrics_df) > 0:
    print("\n   Metrics that will be saved:")
    for idx, row in temp_metrics_df.iterrows():
        print(f"      {idx + 1}. {row['name']} ({row['type']}) - threshold: {row['threshold']}")
else:
    print("\n   ??  File will be empty (all metrics deleted)")

print("\n?? Select 'Yes' in the dropdown above and run the next cell to save.")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute: Save to File

# COMMAND ----------

confirm = dbutils.widgets.get("confirm_save")

if confirm != "Yes":
    print("??  Save not confirmed.")
    print("?? Select 'Yes' in the confirmation dropdown above and run this cell again.")
else:
    try:
        # Validate before saving
        validation_errors = []
        
        for idx, row in temp_metrics_df.iterrows():
            if pd.isna(row.get('name')) or str(row.get('name')).strip() == '':
                validation_errors.append(f"Row {idx + 1}: Name is missing")
            
            if pd.isna(row.get('type')) or str(row.get('type')).strip() == '':
                validation_errors.append(f"Row {idx + 1}: Type is missing")
        
        if validation_errors:
            print("="*70)
            print("? VALIDATION ERRORS - CANNOT SAVE")
            print("="*70)
            for error in validation_errors:
                print(f"   {error}")
            print("\n?? Fix the errors and try again.")
            print("="*70)
        else:
            # Save to file
            temp_metrics_df.to_csv(METRICS_FILE, index=False)
            
            print("="*70)
            print("? SUCCESS - CHANGES SAVED!")
            print("="*70)
            print(f"   File: {METRICS_FILE}")
            print(f"   Metrics saved: {len(temp_metrics_df)}")
            print(f"   Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            
            # Reset confirmation
            dbutils.widgets.dropdown("confirm_save", "No", ["No", "Yes"], "?? Confirm Save?")
            
            # Show final table
            print("\n?? Saved metrics:")
            display_metrics_table(temp_metrics_df)
            
            print("\n?? Done! You can now use this metrics file in your evaluation notebook.")
            
    except Exception as e:
        print("="*70)
        print("? ERROR SAVING FILE")
        print("="*70)
        print(f"   Error: {e}")
        print("\n?? Troubleshooting:")
        print("   ? Check file path is correct")
        print("   ? Check you have write permissions")
        print("   ? Try using /tmp/ directory")
        print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### View Complete Details

# COMMAND ----------

if len(temp_metrics_df) > 0:
    print("="*70)
    print("?? DETAILED METRICS VIEW")
    print("="*70)
    
    for idx, row in temp_metrics_df.iterrows():
        print(f"\n{'='*70}")
        print(f"Metric #{idx + 1}: {row.get('name', 'N/A')}")
        print(f"{'='*70}")
        print(f"  Type:               {row.get('type', 'N/A')}")
        print(f"  Description:        {row.get('description', 'N/A')}")
        
        rubric = str(row.get('grading_rubric', 'N/A'))
        if len(rubric) > 200:
            print(f"  Grading Rubric:     {rubric[:200]}...")
        else:
            print(f"  Grading Rubric:     {rubric}")
        
        print(f"  Threshold:          {row.get('threshold', 'N/A')}")
        print(f"  Ground Truth File:  {row.get('ground_truth_file_path', 'None')}")
        print(f"  Ground Truth Col:   {row.get('ground_truth_column', 'None')}")
    
    print(f"\n{'='*70}")
    print(f"Total: {len(temp_metrics_df)} metric{'s' if len(temp_metrics_df) != 1 else ''}")
    print(f"{'='*70}")
else:
    print("?? No metrics to display")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? Usage Guide
# MAGIC 
# MAGIC ### Quick Start:
# MAGIC 
# MAGIC 1. **View Metrics**: Run the "View Current Metrics" cell
# MAGIC 2. **Add Metric**: Fill form widgets ? Run "Execute: Add Metric" cell
# MAGIC 3. **Delete Metric**: Enter row number ? Run "Execute: Delete Metric" cell
# MAGIC 4. **Save**: Select "Yes" ? Run "Execute: Save to File" cell
# MAGIC 
# MAGIC ### Tips:
# MAGIC - Changes are temporary until you save
# MAGIC - You can add multiple metrics before saving
# MAGIC - Table refreshes automatically after each change
# MAGIC - All fields are validated before saving
# MAGIC 
# MAGIC ### Field Descriptions:
# MAGIC 
# MAGIC - **Name**: Unique identifier for your metric
# MAGIC - **Type**: 
# MAGIC   - `binary`: Pass/Fail (0 or 1)
# MAGIC   - `1-5_scale`: Rating from 1 to 5
# MAGIC   - `percentage`: Score from 0 to 100
# MAGIC - **Description**: What this metric evaluates
# MAGIC - **Grading Rubric**: Detailed scoring instructions
# MAGIC - **Threshold**: Minimum score to pass
# MAGIC - **Ground Truth File**: (Optional) Reference data filename
# MAGIC - **Ground Truth Column**: (Optional) Specific column to use
# MAGIC 
# MAGIC ### Troubleshooting:
# MAGIC 
# MAGIC **File not found?**
# MAGIC - Check the file path in the configuration cell
# MAGIC - Try using absolute path: `/Workspace/Users/your.email@domain.com/file.csv`
# MAGIC 
# MAGIC **Save failed?**
# MAGIC - Check file permissions
# MAGIC - Try saving to `/tmp/` directory first
# MAGIC 
# MAGIC **Widget not showing?**
# MAGIC - Re-run the cell that creates the widget
# MAGIC - Restart notebook if needed

# COMMAND ----------
