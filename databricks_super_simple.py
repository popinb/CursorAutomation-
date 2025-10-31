# Databricks notebook source
# MAGIC %md
# MAGIC # ?? Simple Metrics Editor
# MAGIC ### For Non-Technical Users
# MAGIC 
# MAGIC **How to use:**
# MAGIC 1. Click "Run All" (top of notebook)
# MAGIC 2. Edit the table that appears
# MAGIC 3. Click "Save Changes" button
# MAGIC 4. Done! File is automatically updated.

# COMMAND ----------
import pandas as pd
import os
from datetime import datetime

# Get current user
try:
    USER = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except:
    USER = "default_user"

# Configuration
CONFIG_FILE = "sample_metrics_config_simplified.csv"
FILE_PATH = f"/Workspace/Users/{USER}/{CONFIG_FILE}"

# Try to load existing file
try:
    if os.path.exists(FILE_PATH):
        df = pd.read_csv(FILE_PATH)
    else:
        # Create sample data
        df = pd.DataFrame({
            "metric_name": ["faithfulness", "relevance", "coherence"],
            "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
            "weight": [1.0, 0.8, 0.6],
            "enabled": [True, True, False]
        })
        df.to_csv(FILE_PATH, index=False)
except:
    # Fallback
    FILE_PATH = f"/dbfs/FileStore/{CONFIG_FILE}"
    df = pd.read_csv(FILE_PATH) if os.path.exists(FILE_PATH) else pd.DataFrame({
        "metric_name": ["faithfulness"], "metric_type": ["llm_judge"], 
        "weight": [1.0], "enabled": [True]
    })

print(f"? Loaded: {FILE_PATH}")
print(f"?? {len(df)} rows, {len(df.columns)} columns\\n")

# COMMAND ----------
# MAGIC %md
# MAGIC ## ?? EDIT TABLE BELOW

# COMMAND ----------
# Display editable table

def show_editor():
    cols = list(df.columns)
    rows = df.to_dict('records')
    
    html = f"""
    <style>
        .edit-container {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 20px auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header {{
            background: #0066cc;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 8px 8px 0 0;
            margin: -20px -20px 20px -20px;
        }}
        .save-button {{
            background: #28a745;
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            margin: 20px 0;
            width: 100%;
            transition: background 0.3s;
        }}
        .save-button:hover {{ background: #218838; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: #f0f0f0;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            border: 1px solid #ddd;
        }}
        td {{
            padding: 10px;
            border: 1px solid #ddd;
        }}
        input {{
            width: 100%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
        }}
        input:focus {{
            outline: 2px solid #0066cc;
        }}
        .delete-btn {{
            background: #dc3545;
            color: white;
            border: none;
            padding: 6px 12px;
            cursor: pointer;
            border-radius: 4px;
        }}
        .add-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .add-inputs {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 10px;
        }}
        .add-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 4px;
            font-weight: bold;
        }}
        .status {{
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
            display: none;
        }}
        .status.success {{
            background: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }}
    </style>
    
    <div class="edit-container">
        <div class="header">
            <h1>?? Metrics Configuration</h1>
            <p>{FILE_PATH}</p>
        </div>
        
        <div id="status" class="status"></div>
        
        <button class="save-button" onclick="save()">?? SAVE CHANGES</button>
        
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    {''.join(f'<th>{c}</th>' for c in cols)}
                    <th>Delete</th>
                </tr>
            </thead>
            <tbody id="table-body"></tbody>
        </table>
        
        <div class="add-section">
            <h3>? Add New Row</h3>
            <div class="add-inputs">
                {''.join(f'<input id="new_{c}" placeholder="{c}"/>' for c in cols)}
            </div>
            <button class="add-btn" onclick="addRow()">Add Row</button>
        </div>
    </div>
    
    <script>
        let data = {rows};
        const cols = {cols};
        
        function render() {{
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            data.forEach((row, i) => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight:bold">${{i+1}}</td>
                    ${{cols.map(c => `
                        <td>
                            <input value="${{row[c] ?? ''}}" 
                                   onchange="update(${{i}}, '${{c}}', this.value)"/>
                        </td>
                    `).join('')}}
                    <td>
                        <button class="delete-btn" onclick="deleteRow(${{i}})">???</button>
                    </td>
                `;
                tbody.appendChild(tr);
            }});
        }}
        
        function update(i, col, val) {{
            // Auto-convert types
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
                let val = document.getElementById('new_' + c).value.trim();
                if (val) {{
                    hasValue = true;
                    if (val.toLowerCase() === 'true') val = true;
                    else if (val.toLowerCase() === 'false') val = false;
                    else if (!isNaN(val) && val !== '') {{
                        val = val.includes('.') ? parseFloat(val) : parseInt(val);
                    }}
                }}
                row[c] = val || '';
                document.getElementById('new_' + c).value = '';
            }});
            if (hasValue) {{
                data.push(row);
                render();
            }} else {{
                alert('Please enter at least one value');
            }}
        }}
        
        function deleteRow(i) {{
            if (confirm('Delete row ' + (i + 1) + '?')) {{
                data.splice(i, 1);
                render();
            }}
        }}
        
        function save() {{
            const json = JSON.stringify(data);
            
            // Update Databricks widget
            try {{
                const event = new CustomEvent('databricks-update', {{ 
                    detail: {{ widget: 'saved_config', value: json }}
                }});
                window.dispatchEvent(event);
                
                // Also try postMessage
                if (window.parent) {{
                    window.parent.postMessage({{
                        type: 'databricks-widget-update',
                        widgetName: 'saved_config',
                        value: json
                    }}, '*');
                }}
            }} catch (e) {{ console.error(e); }}
            
            // Show success message
            const status = document.getElementById('status');
            status.className = 'status success';
            status.textContent = '? SAVED! Run the next cell to apply changes.';
            status.style.display = 'block';
            
            // Also log for manual copy if needed
            console.log('COPY THIS IF NEEDED:', json);
        }}
        
        render();
    </script>
    """
    return html

displayHTML(show_editor())

# COMMAND ----------
# MAGIC %md
# MAGIC ## ?? Apply Changes
# MAGIC **Run this cell after clicking "Save Changes" above**

# COMMAND ----------
# Widget to receive saved data
try:
    dbutils.widgets.remove("saved_config")
except:
    pass
dbutils.widgets.text("saved_config", "", "")

# Get saved data
saved = dbutils.widgets.get("saved_config")

if saved and saved.strip():
    try:
        import json
        data = json.loads(saved)
        new_df = pd.DataFrame(data)
        
        # Save to file
        new_df.to_csv(FILE_PATH, index=False)
        
        print("="*70)
        print("? SUCCESS! Changes saved!")
        print("="*70)
        print(f"?? File: {FILE_PATH}")
        print(f"?? Rows: {len(new_df)}")
        print(f"?? {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print("\\nUpdated configuration:")
        display(new_df)
        
        # Clear widget
        dbutils.widgets.remove("saved_config")
        dbutils.widgets.text("saved_config", "", "")
        
    except Exception as e:
        print(f"? Error: {e}")
        print("\\n?? If save button didn't work:")
        print("  1. Open browser console (press F12)")
        print("  2. Look for the JSON output")
        print("  3. Copy it and paste below")
        
        # Fallback widget
        dbutils.widgets.text("manual_json", "", "?? Paste JSON here if needed")
        manual = dbutils.widgets.get("manual_json")
        if manual.strip():
            data = json.loads(manual)
            new_df = pd.DataFrame(data)
            new_df.to_csv(FILE_PATH, index=False)
            print("? Saved from manual input!")
            display(new_df)
else:
    print("??  No changes yet")
    print("\\n?? Steps:")
    print("  1. Edit table above")
    print("  2. Click '?? SAVE CHANGES'")
    print("  3. Run this cell again")
    print("\\n?? Current configuration:")
    display(df)
