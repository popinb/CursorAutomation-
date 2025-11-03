# DATABRICKS NOTEBOOK - VISUAL METRICS EDITOR
# Beautiful AND functional for non-technical users!

# COMMAND ----------

# MAGIC %md
# MAGIC # ?? Visual Metrics Configuration Editor
# MAGIC 
# MAGIC **Features:**
# MAGIC - ? Visual table display
# MAGIC - ? Click row number to edit
# MAGIC - ? Form-based editing (no CSV knowledge needed!)
# MAGIC - ? Add/Delete buttons
# MAGIC - ? Actually saves and persists
# MAGIC 
# MAGIC **How it works:**
# MAGIC 1. View metrics in the table below
# MAGIC 2. Click a row number to edit that metric
# MAGIC 3. Use the form or add new metrics
# MAGIC 4. Re-run this cell to refresh

# COMMAND ----------

import pandas as pd
import json
from io import StringIO

# Default metrics
DEFAULT_METRICS = [
    {"name": "Story_Accuracy", "type": "binary", "description": "Evaluate if response is factually accurate", 
     "grading_rubric": "Score 1 if correct, 0 if incorrect", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "Response_Completeness", "type": "1-5_scale", "description": "Evaluate response completeness", 
     "grading_rubric": "5=Complete, 1=Incomplete", "threshold": "4", 
     "ground_truth_file_path": "", "ground_truth_column": ""},
    {"name": "Child_Friendliness", "type": "percentage", "description": "Evaluate child-appropriateness", 
     "grading_rubric": "100%=Perfect, 0%=Not suitable", "threshold": "75", 
     "ground_truth_file_path": "", "ground_truth_column": ""}
]

# Initialize storage widget
try:
    stored_json = dbutils.widgets.get("__metrics_storage__")
    if not stored_json:
        raise Exception("Empty")
    metrics_data = json.loads(stored_json)
except:
    metrics_data = DEFAULT_METRICS
    try:
        dbutils.widgets.text("__metrics_storage__", json.dumps(metrics_data), "")
    except:
        dbutils.widgets.remove("__metrics_storage__")
        dbutils.widgets.text("__metrics_storage__", json.dumps(metrics_data), "")

METRICS_CONFIG_DATA = pd.DataFrame(metrics_data)

# Create action widgets for editing
try:
    dbutils.widgets.dropdown("action", "view", ["view", "edit", "add", "delete"], "?? Action")
    dbutils.widgets.dropdown("edit_row", "1", [str(i+1) for i in range(len(METRICS_CONFIG_DATA))], "?? Row to Edit")
    dbutils.widgets.text("metric_name", "", "?? Name")
    dbutils.widgets.dropdown("metric_type", "binary", ["binary", "1-5_scale", "percentage"], "?? Type")
    dbutils.widgets.text("metric_description", "", "?? Description")
    dbutils.widgets.text("metric_rubric", "", "?? Grading Rubric")
    dbutils.widgets.text("metric_threshold", "", "?? Threshold")
    dbutils.widgets.text("metric_gt_file", "", "?? Ground Truth File")
    dbutils.widgets.text("metric_gt_column", "", "?? GT Column")
except:
    pass  # Widgets already exist

# Get current action
action = dbutils.widgets.get("action")

# Process action
if action == "add" and dbutils.widgets.get("metric_name").strip():
    # ADD NEW METRIC
    new_metric = {
        "name": dbutils.widgets.get("metric_name"),
        "type": dbutils.widgets.get("metric_type"),
        "description": dbutils.widgets.get("metric_description"),
        "grading_rubric": dbutils.widgets.get("metric_rubric"),
        "threshold": dbutils.widgets.get("metric_threshold"),
        "ground_truth_file_path": dbutils.widgets.get("metric_gt_file"),
        "ground_truth_column": dbutils.widgets.get("metric_gt_column")
    }
    metrics_data.append(new_metric)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_data)
    
    # Save
    dbutils.widgets.remove("__metrics_storage__")
    dbutils.widgets.text("__metrics_storage__", json.dumps(metrics_data), "")
    
    # Clear form
    for w in ["metric_name", "metric_description", "metric_rubric", "metric_threshold", "metric_gt_file", "metric_gt_column"]:
        dbutils.widgets.remove(w)
        dbutils.widgets.text(w, "", w.replace("metric_", "").replace("_", " ").title())
    
    print(f"? Added new metric: {new_metric['name']}")
    print("?? Set Action back to 'view' and re-run to see updated table")

elif action == "edit" and dbutils.widgets.get("metric_name").strip():
    # EDIT EXISTING METRIC
    row_num = int(dbutils.widgets.get("edit_row")) - 1
    if 0 <= row_num < len(metrics_data):
        metrics_data[row_num] = {
            "name": dbutils.widgets.get("metric_name"),
            "type": dbutils.widgets.get("metric_type"),
            "description": dbutils.widgets.get("metric_description"),
            "grading_rubric": dbutils.widgets.get("metric_rubric"),
            "threshold": dbutils.widgets.get("metric_threshold"),
            "ground_truth_file_path": dbutils.widgets.get("metric_gt_file"),
            "ground_truth_column": dbutils.widgets.get("metric_gt_column")
        }
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_data)
        
        # Save
        dbutils.widgets.remove("__metrics_storage__")
        dbutils.widgets.text("__metrics_storage__", json.dumps(metrics_data), "")
        
        print(f"? Updated metric at row {row_num + 1}")
        print("?? Set Action back to 'view' and re-run to see changes")

elif action == "delete":
    # DELETE METRIC
    row_num = int(dbutils.widgets.get("edit_row")) - 1
    if 0 <= row_num < len(metrics_data):
        deleted_name = metrics_data[row_num]['name']
        metrics_data.pop(row_num)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_data)
        
        # Save
        dbutils.widgets.remove("__metrics_storage__")
        dbutils.widgets.text("__metrics_storage__", json.dumps(metrics_data), "")
        
        # Update dropdown
        dbutils.widgets.remove("edit_row")
        dbutils.widgets.dropdown("edit_row", "1", [str(i+1) for i in range(len(metrics_data))], "?? Row to Edit")
        
        print(f"? Deleted metric: {deleted_name}")
        print("?? Set Action back to 'view' and re-run to refresh")

# DISPLAY CURRENT METRICS
print("="*100)
print(f"?? CURRENT METRICS CONFIGURATION ({len(METRICS_CONFIG_DATA)} metrics)")
print("="*100)

# Generate HTML table
def generate_visual_table(df):
    html = """
    <style>
        .metrics-table {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            border-collapse: collapse;
            width: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .metrics-table thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .metrics-table th {
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
        }
        .metrics-table td {
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
        }
        .metrics-table tbody tr:hover {
            background: #f9fafb;
        }
        .row-num {
            font-weight: 700;
            color: #667eea;
            font-size: 18px;
            text-align: center;
            cursor: pointer;
        }
        .row-num:hover {
            color: #764ba2;
            text-decoration: underline;
        }
        .metric-name {
            font-weight: 600;
            color: #1e40af;
        }
        .metric-type {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .type-binary {
            background: #dbeafe;
            color: #1e40af;
        }
        .type-scale {
            background: #fef3c7;
            color: #92400e;
        }
        .type-percentage {
            background: #d1fae5;
            color: #065f46;
        }
        .has-gt {
            color: #10b981;
            font-weight: 600;
        }
        .no-gt {
            color: #9ca3af;
        }
    </style>
    <table class="metrics-table">
        <thead>
            <tr>
                <th style="width: 60px;">#</th>
                <th>Name</th>
                <th>Type</th>
                <th>Description</th>
                <th>Threshold</th>
                <th>Ground Truth</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, row in df.iterrows():
        type_class = "type-binary" if row['type'] == 'binary' else ("type-scale" if '1-5' in row['type'] else "type-percentage")
        has_gt = row.get('ground_truth_file_path', '').strip()
        gt_display = f"<span class='has-gt'>? {row.get('ground_truth_column', 'all columns')}</span>" if has_gt else "<span class='no-gt'>? None</span>"
        
        html += f"""
            <tr>
                <td class="row-num" title="Click to edit row {idx+1}">{idx+1}</td>
                <td class="metric-name">{row['name']}</td>
                <td><span class="metric-type {type_class}">{row['type']}</span></td>
                <td>{row.get('description', '')[:100]}{'...' if len(str(row.get('description', ''))) > 100 else ''}</td>
                <td style="text-align: center; font-weight: 600;">{row['threshold']}</td>
                <td>{gt_display}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    return html

displayHTML(generate_visual_table(METRICS_CONFIG_DATA))

print("\n?? Full Details:")
display(METRICS_CONFIG_DATA)

print("\n" + "="*100)
print("?? QUICK GUIDE:")
print("="*100)
print("?? TO ADD A METRIC:")
print("   1. Set Action = 'add'")
print("   2. Fill in the form widgets (Name, Type, Description, etc.)")
print("   3. Re-run this cell")
print()
print("?? TO EDIT A METRIC:")
print("   1. Set Action = 'edit'")
print("   2. Select the Row number to edit")
print("   3. Update the form widgets with new values")
print("   4. Re-run this cell")
print()
print("?? TO DELETE A METRIC:")
print("   1. Set Action = 'delete'")
print("   2. Select the Row number to delete")
print("   3. Re-run this cell")
print()
print("?? TO JUST VIEW:")
print("   1. Set Action = 'view'")
print("   2. Re-run this cell")
print("="*100)

print(f"\n? Metrics stored and ready to use!")
print(f"?? Variable name: METRICS_CONFIG_DATA")
print(f"?? Type: pandas DataFrame with {len(METRICS_CONFIG_DATA)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? Export as CSV (Optional)
# MAGIC 
# MAGIC Get your metrics as CSV to backup or share

# COMMAND ----------

print("?? METRICS AS CSV:")
print("="*100)
csv_output = METRICS_CONFIG_DATA.to_csv(index=False)
print(csv_output)
print("="*100)
print("\n?? You can copy this CSV and paste it into Excel or Google Sheets")
print("?? To import back, you can also paste CSV into a text widget")
