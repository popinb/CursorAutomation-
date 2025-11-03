# Databricks notebook source
# MAGIC %md
# MAGIC # ?? Metrics Editor Demo - Compare Both Approaches
# MAGIC 
# MAGIC This notebook lets you test both implementation options side-by-side
# MAGIC 
# MAGIC **Run this notebook to see:**
# MAGIC - Option 1: Form-Based Editor
# MAGIC - Option 2: Interactive HTML Table
# MAGIC 
# MAGIC **Goal:** Help you decide which approach to use in your workshop

# COMMAND ----------

import pandas as pd
import json
from IPython.display import display, HTML
import os

# Create sample metrics for demo
demo_metrics = pd.DataFrame({
    'name': ['Accuracy', 'Relevance', 'Safety'],
    'type': ['binary', '1-5_scale', 'percentage'],
    'description': [
        'Check if the response is factually accurate',
        'Evaluate how relevant the response is to the query',
        'Ensure the response is safe and appropriate'
    ],
    'grading_rubric': [
        'Score 1 if accurate, 0 if inaccurate',
        'Rate from 1 (not relevant) to 5 (highly relevant)',
        'Score 0-100 based on safety concerns'
    ],
    'threshold': ['1.0', '3.0', '80.0'],
    'ground_truth_file_path': ['', '', ''],
    'ground_truth_column': ['', '', '']
})

# Save to temp file
DEMO_FILE = "/tmp/demo_metrics.csv"
demo_metrics.to_csv(DEMO_FILE, index=False)

print("? Demo setup complete!")
print(f"?? Sample metrics file created: {DEMO_FILE}")
print(f"?? Sample contains {len(demo_metrics)} metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ??? OPTION 1: Form-Based Editor
# MAGIC 
# MAGIC **Pros:**
# MAGIC - Simple to use
# MAGIC - No complex JavaScript
# MAGIC - Easy to maintain
# MAGIC 
# MAGIC **Difficulty:** ?? Easy-Medium

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 1: View Current Metrics

# COMMAND ----------

# Load metrics
metrics_option1 = pd.read_csv(DEMO_FILE)

# Display in styled table
def display_styled_table(df):
    html = f"""
    <style>
        .demo-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: Arial;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .demo-table th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .demo-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .demo-table tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
    <h3>?? Current Metrics ({len(df)} total)</h3>
    <table class="demo-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Name</th>
                <th>Type</th>
                <th>Description</th>
                <th>Threshold</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, row in df.iterrows():
        html += f"""
            <tr>
                <td>{idx + 1}</td>
                <td><strong>{row['name']}</strong></td>
                <td><span style="background: #e8f4f8; padding: 4px 8px; border-radius: 4px;">{row['type']}</span></td>
                <td>{row['description'][:80]}...</td>
                <td>{row['threshold']}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    <p style="color: #666; font-size: 13px; margin-top: 10px;">
        ?? Use the form below to add new metrics or delete by row number
    </p>
    """
    
    displayHTML(html)

display_styled_table(metrics_option1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 1: Add New Metric Form

# COMMAND ----------

# Create form widgets
dbutils.widgets.text("opt1_name", "", "??? Metric Name")
dbutils.widgets.dropdown("opt1_type", "binary", ["binary", "1-5_scale", "percentage"], "?? Type")
dbutils.widgets.text("opt1_description", "", "?? Description")
dbutils.widgets.text("opt1_rubric", "", "?? Grading Rubric")
dbutils.widgets.text("opt1_threshold", "0.5", "?? Threshold")

print("? Form ready! Fill in the fields above and run the next cell to add.")

# COMMAND ----------

# Add metric
new_name = dbutils.widgets.get("opt1_name").strip()

if new_name:
    new_metric = {
        'name': new_name,
        'type': dbutils.widgets.get("opt1_type"),
        'description': dbutils.widgets.get("opt1_description"),
        'grading_rubric': dbutils.widgets.get("opt1_rubric"),
        'threshold': dbutils.widgets.get("opt1_threshold"),
        'ground_truth_file_path': '',
        'ground_truth_column': ''
    }
    
    metrics_option1 = pd.concat([metrics_option1, pd.DataFrame([new_metric])], ignore_index=True)
    metrics_option1.to_csv(DEMO_FILE, index=False)
    
    print(f"? Added: '{new_name}'")
    print(f"?? Total metrics: {len(metrics_option1)}")
    
    # Clear form
    dbutils.widgets.text("opt1_name", "")
    
    display_styled_table(metrics_option1)
else:
    print("?? Enter a metric name above to add")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 1: Delete Metric

# COMMAND ----------

dbutils.widgets.text("opt1_delete", "", "??? Row Number to Delete")

delete_row = dbutils.widgets.get("opt1_delete").strip()

if delete_row:
    try:
        idx = int(delete_row) - 1
        if 0 <= idx < len(metrics_option1):
            deleted = metrics_option1.iloc[idx]['name']
            metrics_option1 = metrics_option1.drop(metrics_option1.index[idx]).reset_index(drop=True)
            metrics_option1.to_csv(DEMO_FILE, index=False)
            
            print(f"? Deleted: '{deleted}'")
            display_styled_table(metrics_option1)
        else:
            print(f"? Invalid row number")
    except:
        print(f"? Invalid input")
else:
    print("?? Enter a row number above to delete")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ??? OPTION 2: Interactive HTML Table
# MAGIC 
# MAGIC **Pros:**
# MAGIC - Spreadsheet-like experience
# MAGIC - Edit cells directly
# MAGIC - More intuitive for Excel users
# MAGIC 
# MAGIC **Difficulty:** ??? Medium-Hard

# COMMAND ----------

# Reload metrics for Option 2
metrics_option2 = pd.read_csv(DEMO_FILE)

# Create interactive HTML table
metrics_json = json.dumps(metrics_option2.to_dict('records'))

html = f"""
<style>
    .interactive-editor {{
        font-family: Arial, sans-serif;
        margin: 20px 0;
    }}
    
    .editor-header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 8px 8px 0 0;
    }}
    
    .editor-controls {{
        background: #f8f9fa;
        padding: 15px;
        border: 1px solid #dee2e6;
        display: flex;
        gap: 10px;
    }}
    
    .btn {{
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s;
    }}
    
    .btn-primary {{
        background-color: #28a745;
        color: white;
    }}
    
    .btn-primary:hover {{
        background-color: #218838;
        transform: translateY(-2px);
    }}
    
    .btn-secondary {{
        background-color: #007bff;
        color: white;
    }}
    
    .btn-danger {{
        background-color: #dc3545;
        color: white;
        padding: 5px 10px;
        font-size: 12px;
    }}
    
    .metrics-table {{
        width: 100%;
        border-collapse: collapse;
        background: white;
    }}
    
    .metrics-table th {{
        background-color: #343a40;
        color: white;
        padding: 12px;
        text-align: left;
        position: sticky;
        top: 0;
    }}
    
    .metrics-table td {{
        padding: 8px;
        border-bottom: 1px solid #dee2e6;
    }}
    
    .metrics-table tr:hover {{
        background-color: #f8f9fa;
    }}
    
    .metrics-table input,
    .metrics-table select,
    .metrics-table textarea {{
        width: 100%;
        padding: 6px;
        border: 1px solid #ced4da;
        border-radius: 4px;
        box-sizing: border-box;
    }}
    
    .metrics-table textarea {{
        min-height: 50px;
        resize: vertical;
    }}
    
    .status {{
        margin-left: auto;
        padding: 8px 15px;
        border-radius: 5px;
        font-weight: bold;
    }}
    
    .status-success {{
        background-color: #d4edda;
        color: #155724;
    }}
    
    .status-info {{
        background-color: #d1ecf1;
        color: #0c5460;
    }}
</style>

<div class="interactive-editor">
    <div class="editor-header">
        <h2 style="margin: 0;">?? Interactive Metrics Editor - Option 2</h2>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Click any cell to edit. Changes saved in browser memory.</p>
    </div>
    
    <div class="editor-controls">
        <button class="btn btn-primary" onclick="addRow()">? Add New Metric</button>
        <button class="btn btn-secondary" onclick="showSaveInstructions()">?? Save Changes</button>
        <span id="status" class="status status-info">Ready to edit</span>
    </div>
    
    <div style="overflow-x: auto; border: 1px solid #dee2e6;">
        <table class="metrics-table" id="metricsTable">
            <thead>
                <tr>
                    <th style="width: 40px;">#</th>
                    <th style="width: 150px;">Name</th>
                    <th style="width: 120px;">Type</th>
                    <th style="min-width: 200px;">Description</th>
                    <th style="min-width: 200px;">Grading Rubric</th>
                    <th style="width: 100px;">Threshold</th>
                    <th style="width: 80px;">Actions</th>
                </tr>
            </thead>
            <tbody id="tableBody">
            </tbody>
        </table>
    </div>
    
    <div style="margin-top: 10px; font-size: 12px; color: #666;">
        ?? <strong>Tips:</strong> Click any field to edit ? Use "Add New Metric" for new rows ? Click "Delete" to remove ? Click "Save" to persist
    </div>
</div>

<div id="saveInstructions" style="display: none; margin-top: 20px; padding: 20px; background: #e7f3ff; border: 2px solid #2196F3; border-radius: 8px;">
    <h3 style="margin-top: 0;">?? Save Instructions</h3>
    <p><strong>Copy this JSON and paste it in the next cell to save:</strong></p>
    <textarea id="jsonOutput" style="width: 100%; height: 150px; font-family: monospace; padding: 10px; border: 2px solid #2196F3; border-radius: 4px;"></textarea>
    <button class="btn btn-secondary" onclick="copyJSON()" style="margin-top: 10px;">?? Copy to Clipboard</button>
</div>

<script>
let metrics = {metrics_json};

function render() {{
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    
    if (metrics.length === 0) {{
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 40px; color: #999;">No metrics. Click "Add New Metric"!</td></tr>';
        return;
    }}
    
    metrics.forEach((m, i) => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="text-align: center; font-weight: bold;">${{i + 1}}</td>
            <td><input type="text" value="${{esc(m.name || '')}}" onchange="update(${{i}}, 'name', this.value)"></td>
            <td>
                <select onchange="update(${{i}}, 'type', this.value)">
                    <option value="binary" ${{m.type === 'binary' ? 'selected' : ''}}>Binary</option>
                    <option value="1-5_scale" ${{m.type === '1-5_scale' ? 'selected' : ''}}>1-5 Scale</option>
                    <option value="percentage" ${{m.type === 'percentage' ? 'selected' : ''}}>Percentage</option>
                </select>
            </td>
            <td><textarea onchange="update(${{i}}, 'description', this.value)">${{esc(m.description || '')}}</textarea></td>
            <td><textarea onchange="update(${{i}}, 'grading_rubric', this.value)">${{esc(m.grading_rubric || '')}}</textarea></td>
            <td><input type="text" value="${{m.threshold || ''}}" onchange="update(${{i}}, 'threshold', this.value)"></td>
            <td style="text-align: center;">
                <button class="btn btn-danger" onclick="deleteRow(${{i}})">???</button>
            </td>
        `;
        tbody.appendChild(tr);
    }});
}}

function update(idx, field, value) {{
    metrics[idx][field] = value;
    showStatus('Changes saved to memory', 'success');
}}

function addRow() {{
    metrics.push({{
        name: '',
        type: 'binary',
        description: '',
        grading_rubric: '',
        threshold: '0.5',
        ground_truth_file_path: '',
        ground_truth_column: ''
    }});
    render();
    showStatus('New metric added. Fill in the fields!', 'success');
}}

function deleteRow(idx) {{
    if (confirm(`Delete "${{metrics[idx].name}}"?`)) {{
        metrics.splice(idx, 1);
        render();
        showStatus('Metric deleted', 'success');
    }}
}}

function showSaveInstructions() {{
    const errors = metrics.filter(m => !m.name).length;
    if (errors > 0) {{
        alert(`Error: ${{errors}} metric(s) missing a name!`);
        return;
    }}
    
    const json = JSON.stringify(metrics, null, 2);
    document.getElementById('jsonOutput').value = json;
    document.getElementById('saveInstructions').style.display = 'block';
    showStatus('Ready to save! Copy JSON below', 'success');
}}

function copyJSON() {{
    const textarea = document.getElementById('jsonOutput');
    textarea.select();
    document.execCommand('copy');
    showStatus('Copied! Paste in next cell', 'success');
}}

function showStatus(msg, type) {{
    const el = document.getElementById('status');
    el.textContent = msg;
    el.className = 'status status-' + type;
}}

function esc(text) {{
    return String(text).replace(/[&<>"']/g, m => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    }})[m]);
}}

render();
</script>
"""

displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 2: Save from Interactive Table

# COMMAND ----------

dbutils.widgets.text("opt2_json", "", "?? Paste JSON from table above")

json_data = dbutils.widgets.get("opt2_json").strip()

if json_data:
    try:
        metrics_list = json.loads(json_data)
        new_df = pd.DataFrame(metrics_list)
        new_df.to_csv(DEMO_FILE, index=False)
        
        print("? " + "="*60)
        print(f"? Saved {len(new_df)} metrics!")
        print("? " + "="*60)
        print("\n?? Saved metrics:")
        print(new_df[['name', 'type', 'threshold']])
        
        dbutils.widgets.text("opt2_json", "")
        
    except Exception as e:
        print(f"? Error: {e}")
else:
    print("?? Instructions:")
    print("1. Click 'Save Changes' in the table above")
    print("2. Copy the JSON that appears")
    print("3. Paste it in the widget above")
    print("4. Run this cell")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ?? Comparison Summary
# MAGIC 
# MAGIC ## Which One Should You Choose?
# MAGIC 
# MAGIC ### Choose **Option 1** (Form-Based) if:
# MAGIC - ? You want simple, reliable code
# MAGIC - ? Users are comfortable with forms
# MAGIC - ? You want easy maintenance
# MAGIC - ? **Recommended for most teams**
# MAGIC 
# MAGIC ### Choose **Option 2** (Interactive HTML) if:
# MAGIC - ? Users demand Excel-like editing
# MAGIC - ? You have time for complex implementation
# MAGIC - ? Inline editing is critical
# MAGIC - ? You're comfortable with JavaScript
# MAGIC 
# MAGIC ## Difficulty Rating
# MAGIC - **Option 1:** ?? (Easy-Medium) - 1-2 hours
# MAGIC - **Option 2:** ??? (Medium-Hard) - 3-5 hours
# MAGIC 
# MAGIC ## My Recommendation
# MAGIC **Start with Option 1**. You can always upgrade later if needed!
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Next Steps
# MAGIC 1. Test both options in this demo
# MAGIC 2. Choose the one you prefer
# MAGIC 3. Copy the code to your main workshop notebook
# MAGIC 4. Customize for your needs
# MAGIC 5. Train your users!
# MAGIC 
# MAGIC **Questions?** Check `METRICS_EDITOR_IMPLEMENTATION_GUIDE.md`

# COMMAND ----------
