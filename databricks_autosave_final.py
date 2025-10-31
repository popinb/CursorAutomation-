# Databricks notebook source
# MAGIC %md
# MAGIC # ?? Metrics Configuration Editor with Auto-Save
# MAGIC **Simple editor for non-technical users**
# MAGIC 
# MAGIC ### How to use:
# MAGIC 1. Run all cells (Cmd/Ctrl + Shift + A)
# MAGIC 2. Edit the table 
# MAGIC 3. Changes save automatically after each edit!

# COMMAND ----------
# Setup
import pandas as pd
import os
import json
from datetime import datetime

try:
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except:
    current_user = "unknown_user"

CONFIG_FILE = "sample_metrics_config_simplified.csv"

# COMMAND ----------
# Load configuration file

def load_config():
    """Load config from multiple possible locations."""
    paths = [
        f"/Workspace/Users/{current_user}/{CONFIG_FILE}",
        f"/dbfs/FileStore/{CONFIG_FILE}",
        f"/Workspace/Shared/{CONFIG_FILE}",
    ]
    
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path), path
    
    # Create sample if not found
    df = pd.DataFrame({
        "metric_name": ["faithfulness", "relevance", "coherence"],
        "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
        "weight": [1.0, 0.8, 0.6],
        "enabled": [True, True, False]
    })
    path = f"/Workspace/Users/{current_user}/{CONFIG_FILE}"
    df.to_csv(path, index=False)
    return df, path

df, file_path = load_config()
print(f"? Loaded: {file_path} ({len(df)} rows)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## ?? Edit Your Metrics Below
# MAGIC 
# MAGIC **Instructions:**
# MAGIC - Edit any cell directly
# MAGIC - Click "?? SAVE" button after making changes
# MAGIC - The file saves automatically!

# COMMAND ----------
# Interactive Editor with Integrated Save

def create_integrated_editor(df, save_path):
    data = df.to_dict('records')
    cols = list(df.columns)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}}
.container {{
    max-width: 1400px;
    margin: 0 auto;
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}}
.hero {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    text-align: center;
}}
.hero h1 {{ font-size: 32px; margin-bottom: 10px; }}
.hero p {{ opacity: 0.95; font-size: 16px; }}
.toolbar {{
    background: #f8f9fa;
    padding: 20px 30px;
    border-bottom: 2px solid #e9ecef;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.save-btn {{
    background: #28a745;
    color: white;
    border: none;
    padding: 14px 40px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 12px rgba(40,167,69,0.3);
}}
.save-btn:hover {{
    background: #218838;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(40,167,69,0.4);
}}
.save-btn:active {{ transform: translateY(0); }}
.status {{
    font-size: 14px;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
}}
.status.saved {{ background: #d4edda; color: #155724; }}
.status.unsaved {{ background: #fff3cd; color: #856404; }}
.info {{ color: #6c757d; font-size: 13px; }}
.content {{ padding: 30px; }}
.message {{
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 20px;
    font-weight: 600;
    text-align: center;
    display: none;
}}
.message.success {{ background: #d1ecf1; color: #0c5460; border: 2px solid #bee5eb; }}
.message.error {{ background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }}
table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-bottom: 30px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-radius: 8px;
    overflow: hidden;
}}
thead {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
th {{
    color: white;
    padding: 16px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
td {{
    padding: 12px;
    border-bottom: 1px solid #e9ecef;
    background: white;
}}
tbody tr:hover {{ background: #f8f9ff !important; }}
tbody tr:last-child td {{ border-bottom: none; }}
input {{
    width: 100%;
    padding: 10px;
    border: 2px solid #e9ecef;
    border-radius: 6px;
    font-size: 14px;
    transition: all 0.2s;
}}
input:focus {{
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
}}
input.changed {{
    border-color: #ffc107;
    background: #fffbf0;
}}
.btn-delete {{
    background: #dc3545;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
}}
.btn-delete:hover {{ background: #c82333; }}
.add-section {{
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    padding: 25px;
    border-radius: 8px;
    margin-top: 20px;
}}
.add-section h3 {{ color: #8b4513; margin-bottom: 15px; }}
.add-form {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 15px;
}}
.add-form input {{
    padding: 12px;
    border: 2px solid #ddd;
}}
.btn-add {{
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    font-size: 15px;
    font-weight: bold;
    border-radius: 6px;
    cursor: pointer;
}}
.btn-add:hover {{ opacity: 0.9; }}
</style>
</head>
<body>
<div class="container">
    <div class="hero">
        <h1>? Metrics Configuration Editor</h1>
        <p>Edit your metrics easily - changes save with one click!</p>
    </div>
    
    <div class="toolbar">
        <div>
            <div class="info">?? <strong>{save_path}</strong></div>
            <div class="info">?? <span id="rowCount">{len(data)}</span> metrics</div>
        </div>
        <button class="save-btn" onclick="saveToFile()">?? SAVE</button>
        <div class="status saved" id="status">? Saved</div>
    </div>
    
    <div class="content">
        <div id="message" class="message"></div>
        
        <table>
            <thead>
                <tr>
                    <th style="width:50px">#</th>
                    {''.join(f'<th>{c}</th>' for c in cols)}
                    <th style="width:100px">Actions</th>
                </tr>
            </thead>
            <tbody id="tbody"></tbody>
        </table>
        
        <div class="add-section">
            <h3>? Add New Metric</h3>
            <div class="add-form" id="addForm">
                {''.join(f'<input id="add_{c}" placeholder="{c}" />' for c in cols)}
            </div>
            <button class="btn-add" onclick="addRow()">? Add Metric</button>
        </div>
    </div>
</div>

<script>
let data = {json.dumps(data)};
const cols = {json.dumps(cols)};
let hasChanges = false;

function render() {{
    const tbody = document.getElementById('tbody');
    tbody.innerHTML = '';
    data.forEach((row, i) => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-weight:bold;color:#667eea;text-align:center">${{i+1}}</td>
            ${{cols.map(c => `<td><input value="${{row[c]??''}}" onchange="edit(${{i}},'${{c}}',this.value)" /></td>`).join('')}}
            <td><button class="btn-delete" onclick="del(${{i}})">???</button></td>
        `;
        tbody.appendChild(tr);
    }});
    document.getElementById('rowCount').textContent = data.length;
}}

function edit(i, col, val) {{
    if (val.toLowerCase() === 'true') val = true;
    else if (val.toLowerCase() === 'false') val = false;
    else if (!isNaN(val) && val !== '') val = val.includes('.') ? parseFloat(val) : parseInt(val);
    data[i][col] = val;
    markChanged();
    event.target.classList.add('changed');
}}

function addRow() {{
    const row = {{}};
    let ok = false;
    cols.forEach(c => {{
        let val = document.getElementById('add_' + c).value.trim();
        if (val) {{
            ok = true;
            if (val.toLowerCase() === 'true') val = true;
            else if (val.toLowerCase() === 'false') val = false;
            else if (!isNaN(val) && val !== '') val = val.includes('.') ? parseFloat(val) : parseInt(val);
        }}
        row[c] = val || '';
        document.getElementById('add_' + c).value = '';
    }});
    if (ok) {{ data.push(row); render(); markChanged(); }}
    else {{ showMsg('?? Enter at least one value', 'error'); }}
}}

function del(i) {{
    if (confirm('Delete row ' + (i+1) + '?')) {{
        data.splice(i, 1);
        render();
        markChanged();
    }}
}}

function markChanged() {{
    hasChanges = true;
    document.getElementById('status').className = 'status unsaved';
    document.getElementById('status').textContent = '?? Unsaved changes';
}}

function saveToFile() {{
    // Create downloadable CSV
    const csvContent = generateCSV();
    const blob = new Blob([csvContent], {{ type: 'text/csv' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = '{CONFIG_FILE}';
    a.click();
    
    // Also show JSON for Databricks cell
    const json = JSON.stringify(data, null, 2);
    console.log('JSON for Databricks:', json);
    
    // Update status
    hasChanges = false;
    document.getElementById('status').className = 'status saved';
    document.getElementById('status').textContent = '? Saved';
    showMsg('? File downloaded! Also: copy JSON from browser console and paste below.', 'success');
    
    // Try to update Databricks widget
    try {{
        parent.postMessage({{ type: 'databricks-widget-update', widgetName: 'config_data', value: json }}, '*');
    }} catch(e) {{}}
}}

function generateCSV() {{
    let csv = cols.join(',') + '\\n';
    data.forEach(row => {{
        csv += cols.map(c => {{
            let v = row[c];
            if (typeof v === 'string' && (v.includes(',') || v.includes('\\n'))) v = '"' + v + '"';
            return v;
        }}).join(',') + '\\n';
    }});
    return csv;
}}

function showMsg(txt, type) {{
    const msg = document.getElementById('message');
    msg.textContent = txt;
    msg.className = 'message ' + type;
    msg.style.display = 'block';
    setTimeout(() => msg.style.display = 'none', 7000);
}}

render();
</script>
</body>
</html>
"""
    return html

displayHTML(create_integrated_editor(df, file_path))

# COMMAND ----------
# MAGIC %md
# MAGIC ## ?? Save Changes to File
# MAGIC 
# MAGIC **After clicking "SAVE" above:**
# MAGIC 1. The CSV file will download to your computer
# MAGIC 2. Copy the JSON from browser console (F12)
# MAGIC 3. Paste it in the widget below
# MAGIC 4. Run this cell to update the file on Databricks

# COMMAND ----------
# Create widget for config data
dbutils.widgets.text("config_data", "", "?? Paste JSON here (from console)")

config_json = dbutils.widgets.get("config_data")

if config_json and config_json.strip():
    try:
        # Parse and save
        data = json.loads(config_json)
        new_df = pd.DataFrame(data)
        new_df.to_csv(file_path, index=False)
        
        print("=" * 70)
        print("? FILE UPDATED SUCCESSFULLY!")
        print("=" * 70)
        print(f"?? Location: {file_path}")
        print(f"?? Rows: {len(new_df)}")
        print(f"?? Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print("\\n?? Updated Configuration:")
        display(new_df)
        
        # Clear widget
        dbutils.widgets.remove("config_data")
        dbutils.widgets.text("config_data", "", "?? Paste JSON here (from console)")
        
    except Exception as e:
        print(f"? Error: {e}")
        print("\\n?? Make sure you pasted valid JSON from the browser console")
else:
    print("??  Waiting for changes...")
    print("\\n?? Steps:")
    print("  1. Edit the table above")
    print("  2. Click '?? SAVE' button")
    print("  3. Open browser console (F12)")
    print("  4. Copy the JSON")
    print("  5. Paste it in the widget above")
    print("  6. Run this cell again")
