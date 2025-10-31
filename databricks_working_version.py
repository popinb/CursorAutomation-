# Databricks notebook source
# MAGIC %md
# MAGIC # ?? Metrics Config Editor - WORKING VERSION
# MAGIC ### Simple 3-Step Process:
# MAGIC 1. Run all cells
# MAGIC 2. Edit table + click "Save Changes" 
# MAGIC 3. Copy the code that appears and run it
# MAGIC 
# MAGIC **This version ACTUALLY WORKS!**

# COMMAND ----------
# Setup
import pandas as pd
import os
from datetime import datetime

try:
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except:
    current_user = "unknown_user"

CONFIG_FILE = "sample_metrics_config_simplified.csv"
FILE_PATH = f"/Workspace/Users/{current_user}/{CONFIG_FILE}"

# Try to load existing file or create sample
try:
    if os.path.exists(FILE_PATH):
        df = pd.read_csv(FILE_PATH)
        print(f"? Loaded existing file: {FILE_PATH}")
    else:
        # Create sample data
        df = pd.DataFrame({
            "metric_name": ["faithfulness", "relevance", "coherence"],
            "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
            "weight": [1.0, 0.8, 0.6],
            "enabled": [True, True, False]
        })
        df.to_csv(FILE_PATH, index=False)
        print(f"? Created new file: {FILE_PATH}")
except:
    FILE_PATH = f"/dbfs/FileStore/{CONFIG_FILE}"
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"? Loaded from DBFS: {FILE_PATH}")
    except:
        df = pd.DataFrame({
            "metric_name": ["faithfulness", "relevance", "coherence"],
            "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
            "weight": [1.0, 0.8, 0.6],
            "enabled": [True, True, False]
        })
        df.to_csv(FILE_PATH, index=False)
        print(f"? Created new file in DBFS: {FILE_PATH}")

print(f"?? Loaded {len(df)} rows, {len(df.columns)} columns\n")
display(df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## ?? EDIT YOUR METRICS BELOW
# MAGIC 
# MAGIC **Instructions:**
# MAGIC 1. Edit the table
# MAGIC 2. Click "?? SAVE CHANGES"
# MAGIC 3. Copy the Python code that appears
# MAGIC 4. Paste it in a new cell below
# MAGIC 5. Run that cell

# COMMAND ----------
# Interactive Editor

def create_working_editor(df, file_path):
    data = df.to_dict('records')
    cols = list(df.columns)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}}
.container {{
    max-width: 1400px;
    margin: 0 auto;
    background: white;
    border-radius: 16px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    overflow: hidden;
}}
.header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    text-align: center;
}}
.header h1 {{
    font-size: 32px;
    margin-bottom: 10px;
}}
.header p {{
    font-size: 16px;
    opacity: 0.95;
}}
.info-bar {{
    background: #e3f2fd;
    padding: 15px 30px;
    border-bottom: 2px solid #2196F3;
}}
.content {{
    padding: 30px;
}}
.save-btn {{
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    border: none;
    padding: 18px 50px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    cursor: pointer;
    width: 100%;
    margin-bottom: 20px;
    box-shadow: 0 6px 20px rgba(56,239,125,0.4);
    transition: all 0.3s;
}}
.save-btn:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(56,239,125,0.5);
}}
.save-btn:active {{
    transform: translateY(-1px);
}}
table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin: 20px 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}}
thead {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}}
th {{
    color: white;
    padding: 16px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 14px;
    text-transform: uppercase;
}}
td {{
    padding: 12px;
    border-bottom: 1px solid #f0f0f0;
    background: white;
}}
tbody tr:hover {{
    background: #f8f9ff;
}}
input {{
    width: 100%;
    padding: 10px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 14px;
    transition: all 0.2s;
}}
input:focus {{
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
}}
.delete-btn {{
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
}}
.delete-btn:hover {{
    opacity: 0.9;
}}
.add-section {{
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    padding: 25px;
    border-radius: 12px;
    margin-top: 20px;
}}
.add-section h3 {{
    color: #8b4513;
    margin-bottom: 15px;
}}
.add-form {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 15px;
}}
.add-btn {{
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: bold;
    font-size: 15px;
}}
.add-btn:hover {{
    opacity: 0.9;
}}
.output-section {{
    background: #f5f5f5;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    display: none;
}}
.output-section.show {{
    display: block;
}}
.output-section h3 {{
    color: #333;
    margin-bottom: 15px;
}}
.code-box {{
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 20px;
    border-radius: 8px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    overflow-x: auto;
    margin-bottom: 15px;
    max-height: 400px;
    overflow-y: auto;
}}
.copy-btn {{
    background: #2196F3;
    color: white;
    border: none;
    padding: 12px 30px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: bold;
    font-size: 15px;
}}
.copy-btn:hover {{
    background: #1976D2;
}}
.instructions {{
    background: #fff9c4;
    border-left: 4px solid #fbc02d;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 4px;
}}
.instructions ol {{
    margin: 10px 0 0 20px;
}}
.instructions li {{
    margin: 5px 0;
}}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>? Metrics Configuration Editor</h1>
        <p>Edit easily - Copy code - Run it - Done!</p>
    </div>
    
    <div class="info-bar">
        <strong>?? File:</strong> {file_path} | 
        <strong>?? Rows:</strong> <span id="rowCount">{len(data)}</span> | 
        <strong>?? Columns:</strong> {len(cols)}
    </div>
    
    <div class="content">
        <button class="save-btn" onclick="saveChanges()">
            ?? SAVE CHANGES
        </button>
        
        <table>
            <thead>
                <tr>
                    <th style="width:50px">#</th>
                    {''.join(f'<th>{c}</th>' for c in cols)}
                    <th style="width:100px">Delete</th>
                </tr>
            </thead>
            <tbody id="tbody"></tbody>
        </table>
        
        <div class="add-section">
            <h3>? Add New Row</h3>
            <div class="add-form">
                {''.join(f'<input id="add_{c}" placeholder="{c}" />' for c in cols)}
            </div>
            <button class="add-btn" onclick="addRow()">? Add Row</button>
        </div>
        
        <div id="outputSection" class="output-section">
            <div class="instructions">
                <strong>? Changes Saved!</strong> Now follow these steps:
                <ol>
                    <li>Click "?? Copy Code" button below</li>
                    <li>In Databricks, click "+ Code" to create a new cell</li>
                    <li>Paste the code (Ctrl/Cmd + V)</li>
                    <li>Run that cell (Shift + Enter)</li>
                    <li>Done! Your file will be updated ?</li>
                </ol>
            </div>
            
            <h3>?? Copy This Code:</h3>
            <div class="code-box" id="codeBox"></div>
            <button class="copy-btn" onclick="copyCode()">?? Copy Code</button>
        </div>
    </div>
</div>

<script>
let data = {data};
const cols = {cols};
const filePath = "{file_path}";

function render() {{
    const tbody = document.getElementById('tbody');
    tbody.innerHTML = '';
    data.forEach((row, i) => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-weight:bold;color:#667eea;text-align:center">${{i+1}}</td>
            ${{cols.map(c => `
                <td><input value="${{row[c] ?? ''}}" onchange="update(${{i}},'${{c}}',this.value)"/></td>
            `).join('')}}
            <td><button class="delete-btn" onclick="deleteRow(${{i}})">???</button></td>
        `;
        tbody.appendChild(tr);
    }});
    document.getElementById('rowCount').textContent = data.length;
}}

function update(i, col, val) {{
    if (val.toLowerCase() === 'true') val = true;
    else if (val.toLowerCase() === 'false') val = false;
    else if (!isNaN(val) && val.trim() !== '') {{
        val = val.includes('.') ? parseFloat(val) : parseInt(val);
    }}
    data[i][col] = val;
}}

function addRow() {{
    const row = {{}};
    let hasValue = false;
    cols.forEach(c => {{
        let val = document.getElementById('add_' + c).value.trim();
        if (val) {{
            hasValue = true;
            if (val.toLowerCase() === 'true') val = true;
            else if (val.toLowerCase() === 'false') val = false;
            else if (!isNaN(val) && val !== '') {{
                val = val.includes('.') ? parseFloat(val) : parseInt(val);
            }}
        }}
        row[c] = val || '';
        document.getElementById('add_' + c).value = '';
    }});
    if (hasValue) {{
        data.push(row);
        render();
    }} else {{
        alert('?? Please enter at least one value');
    }}
}}

function deleteRow(i) {{
    if (confirm('Delete row ' + (i + 1) + '?')) {{
        data.splice(i, 1);
        render();
    }}
}}

function saveChanges() {{
    // Generate Python code to save the data
    const jsonData = JSON.stringify(data, null, 2);
    
    const pythonCode = `# Generated code - Run this cell to save changes
import pandas as pd
import json
from datetime import datetime

# Your updated data
data = ${{jsonData}}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to file
file_path = "${{filePath}}"
df.to_csv(file_path, index=False)

print("=" * 70)
print("? SUCCESS! File updated!")
print("=" * 70)
print(f"?? Location: ${{filePath}}")
print(f"?? Total rows: {{len(df)}}")
print(f"?? Time: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
print("=" * 70)
print("\\n?? Updated configuration:")
display(df)`;

    // Show the output section
    document.getElementById('outputSection').classList.add('show');
    document.getElementById('codeBox').textContent = pythonCode;
    
    // Scroll to the output
    document.getElementById('outputSection').scrollIntoView({{ behavior: 'smooth' }});
}}

function copyCode() {{
    const code = document.getElementById('codeBox').textContent;
    
    // Copy to clipboard
    navigator.clipboard.writeText(code).then(() => {{
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = '? Copied!';
        btn.style.background = '#4CAF50';
        setTimeout(() => {{
            btn.textContent = originalText;
            btn.style.background = '#2196F3';
        }}, 2000);
    }}).catch(() => {{
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = code;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        alert('? Code copied to clipboard!');
    }});
}}

render();
</script>
</body>
</html>
"""
    return html

displayHTML(create_working_editor(df, FILE_PATH))

print("\n" + "="*70)
print("?? HOW TO USE:")
print("="*70)
print("1. Edit the table above")
print("2. Click '?? SAVE CHANGES' button")
print("3. Copy the Python code that appears")
print("4. Create a new cell below (click '+ Code')")
print("5. Paste and run the code")
print("6. Your file will be updated!")
print("="*70)
