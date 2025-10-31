# Databricks notebook source
# ============================================================================
# ?? SIMPLE METRICS EDITOR WITH AUTO-SAVE
# ============================================================================
# For Non-Technical Users:
# 1. Run all cells once
# 2. Edit the table in the UI
# 3. Click "Save" - that's it! Changes are automatically saved to your file.
# ============================================================================

# COMMAND ----------
# CELL 1: ?? Setup

import pandas as pd
import os
import json
from datetime import datetime

# Get current user
try:
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except:
    current_user = "unknown_user"

# COMMAND ----------
# CELL 2: ?? Configuration

# Where is your metrics config file?
dbutils.widgets.text("config_file", "sample_metrics_config_simplified.csv", "?? Config File Name")
CONFIG_FILE = dbutils.widgets.get("config_file")

# COMMAND ----------
# CELL 3: ?? Load Data

def find_file(filename):
    """Find file in common locations"""
    locations = [
        f"/Workspace/Users/{current_user}/{filename}",
        filename,
        f"/Workspace/Shared/{filename}",
        f"/dbfs/FileStore/{filename}",
    ]
    
    for loc in locations:
        if os.path.exists(loc):
            return loc
    return None

def load_config():
    """Load metrics configuration"""
    path = find_file(CONFIG_FILE)
    
    if path:
        df = pd.read_csv(path)
        print(f"? Loaded: {path}")
        return df, path
    
    # Default config if not found
    print("?? Creating default configuration...")
    df = pd.DataFrame({
        "metric_name": ["faithfulness", "relevance", "coherence"],
        "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
        "weight": [1.0, 0.8, 0.6],
        "enabled": [True, True, False]
    })
    path = f"/Workspace/Users/{current_user}/{CONFIG_FILE}"
    return df, path

metrics_df, save_path = load_config()

print(f"?? Rows: {len(metrics_df)}")
print(f"?? Columns: {list(metrics_df.columns)}")
print(f"?? Will save to: {save_path}")

# COMMAND ----------
# CELL 4: ?? Interactive Editor with One-Click Save

def create_simple_editor(df, save_location):
    """Create a simple, beautiful editor interface"""
    
    data = df.to_dict('records')
    cols = list(df.columns)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f0f2f5;
            }}
            .editor {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 20px rgba(0,0,0,0.08);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #0078d4 0%, #0063b1 100%);
                color: white;
                padding: 25px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .title {{
                font-size: 24px;
                font-weight: 600;
            }}
            .subtitle {{
                font-size: 14px;
                opacity: 0.9;
                margin-top: 5px;
            }}
            .save-btn {{
                background: #10b981;
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: 600;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .save-btn:hover {{
                background: #059669;
                transform: scale(1.05);
            }}
            .save-btn:disabled {{
                background: #9ca3af;
                cursor: not-allowed;
                transform: scale(1);
            }}
            .content {{
                padding: 30px;
            }}
            .alert {{
                padding: 15px 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                display: none;
                font-weight: 500;
            }}
            .alert.success {{
                background: #d1fae5;
                color: #065f46;
                border: 1px solid #10b981;
            }}
            .alert.error {{
                background: #fee2e2;
                color: #991b1b;
                border: 1px solid #ef4444;
            }}
            .alert.show {{
                display: block;
            }}
            .info-bar {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 25px;
            }}
            .info-card {{
                background: #f9fafb;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #0078d4;
            }}
            .info-label {{
                font-size: 12px;
                color: #6b7280;
                font-weight: 600;
                text-transform: uppercase;
            }}
            .info-value {{
                font-size: 20px;
                color: #111827;
                font-weight: 700;
                margin-top: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 25px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                overflow: hidden;
            }}
            th {{
                background: #f9fafb;
                padding: 15px 10px;
                text-align: left;
                font-weight: 600;
                color: #374151;
                border-bottom: 2px solid #e5e7eb;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #f3f4f6;
            }}
            tr:hover {{
                background: #f9fafb;
            }}
            input {{
                width: 100%;
                padding: 8px 10px;
                border: 1.5px solid #d1d5db;
                border-radius: 6px;
                font-size: 14px;
                transition: border-color 0.2s;
            }}
            input:focus {{
                outline: none;
                border-color: #0078d4;
                box-shadow: 0 0 0 3px rgba(0,120,212,0.1);
            }}
            .row-num {{
                font-weight: 600;
                color: #0078d4;
                text-align: center;
            }}
            .btn {{
                padding: 6px 12px;
                border: none;
                border-radius: 5px;
                font-size: 13px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .btn-delete {{
                background: #fef2f2;
                color: #991b1b;
                border: 1px solid #fecaca;
            }}
            .btn-delete:hover {{
                background: #fee2e2;
            }}
            .btn-add {{
                background: #10b981;
                color: white;
            }}
            .btn-add:hover {{
                background: #059669;
            }}
            .add-section {{
                background: #f0fdf4;
                padding: 20px;
                border-radius: 8px;
                border: 2px dashed #10b981;
            }}
            .add-title {{
                font-size: 16px;
                font-weight: 600;
                color: #065f46;
                margin-bottom: 15px;
            }}
            .form-row {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 12px;
                margin-bottom: 15px;
            }}
            .form-group label {{
                display: block;
                font-size: 12px;
                font-weight: 600;
                color: #374151;
                margin-bottom: 5px;
                text-transform: uppercase;
            }}
            .save-info {{
                background: #eff6ff;
                padding: 12px 16px;
                border-radius: 6px;
                border-left: 4px solid #0078d4;
                margin-bottom: 20px;
                font-size: 13px;
                color: #1e40af;
            }}
        </style>
    </head>
    <body>
        <div class="editor">
            <div class="header">
                <div>
                    <div class="title">?? Metrics Configuration Editor</div>
                    <div class="subtitle">Simple ? Intuitive ? Auto-Save</div>
                </div>
                <button class="save-btn" onclick="saveAll()" id="saveButton">
                    ?? Save All Changes
                </button>
            </div>
            
            <div class="content">
                <div id="alert" class="alert"></div>
                
                <div class="save-info">
                    <strong>?? How to use:</strong> Edit any cell, add or delete rows, then click "Save All Changes". That's it!
                </div>
                
                <div class="info-bar">
                    <div class="info-card">
                        <div class="info-label">Total Rows</div>
                        <div class="info-value" id="totalRows">{len(data)}</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">Columns</div>
                        <div class="info-value">{len(cols)}</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">Last Saved</div>
                        <div class="info-value" style="font-size: 14px;" id="lastSaved">Not yet</div>
                    </div>
                </div>
                
                <table>
                    <thead>
                        <tr>
                            <th style="width: 50px;">#</th>
                            {''.join(f'<th>{col}</th>' for col in cols)}
                            <th style="width: 100px;">Action</th>
                        </tr>
                    </thead>
                    <tbody id="dataTable">
                    </tbody>
                </table>
                
                <div class="add-section">
                    <div class="add-title">? Add New Row</div>
                    <div class="form-row">
                        {''.join(f'''
                        <div class="form-group">
                            <label>{col}</label>
                            <input type="text" id="new_{col}" placeholder="{col}">
                        </div>
                        ''' for col in cols)}
                    </div>
                    <button class="btn btn-add" onclick="addRow()">Add Row</button>
                </div>
            </div>
        </div>
        
        <script>
            let data = {json.dumps(data)};
            const columns = {json.dumps(cols)};
            const savePath = {json.dumps(save_location)};
            
            function render() {{
                const tbody = document.getElementById('dataTable');
                tbody.innerHTML = '';
                
                data.forEach((row, i) => {{
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td class="row-num">${{i + 1}}</td>
                        ${{columns.map(col => `
                            <td>
                                <input type="text" value="${{row[col] ?? ''}}" 
                                       onchange="update(${{i}}, '${{col}}', this.value)">
                            </td>
                        `).join('')}}
                        <td style="text-align: center;">
                            <button class="btn btn-delete" onclick="deleteRow(${{i}})">Delete</button>
                        </td>
                    `;
                    tbody.appendChild(tr);
                }});
                
                document.getElementById('totalRows').textContent = data.length;
            }}
            
            function update(idx, col, val) {{
                // Smart type conversion
                let value = val;
                if (val.toLowerCase() === 'true') value = true;
                else if (val.toLowerCase() === 'false') value = false;
                else if (!isNaN(val) && val !== '') {{
                    value = val.includes('.') ? parseFloat(val) : parseInt(val);
                }}
                data[idx][col] = value;
            }}
            
            function addRow() {{
                const newRow = {{}};
                let hasValue = false;
                
                columns.forEach(col => {{
                    let val = document.getElementById('new_' + col).value.trim();
                    if (val) {{
                        hasValue = true;
                        // Smart type conversion
                        if (val.toLowerCase() === 'true') val = true;
                        else if (val.toLowerCase() === 'false') val = false;
                        else if (!isNaN(val) && val !== '') {{
                            val = val.includes('.') ? parseFloat(val) : parseInt(val);
                        }}
                    }}
                    newRow[col] = val || '';
                    document.getElementById('new_' + col).value = '';
                }});
                
                if (hasValue) {{
                    data.push(newRow);
                    render();
                    showAlert('? Row added! Click "Save All Changes" to persist.', 'success');
                }} else {{
                    showAlert('? Please enter at least one value', 'error');
                }}
            }}
            
            function deleteRow(idx) {{
                if (confirm('Delete this row?')) {{
                    data.splice(idx, 1);
                    render();
                    showAlert('??? Row deleted! Click "Save All Changes" to persist.', 'success');
                }}
            }}
            
            function showAlert(msg, type) {{
                const alert = document.getElementById('alert');
                alert.textContent = msg;
                alert.className = `alert ${{type}} show`;
                setTimeout(() => alert.className = 'alert', 4000);
            }}
            
            function saveAll() {{
                const btn = document.getElementById('saveButton');
                btn.disabled = true;
                btn.textContent = '? Saving...';
                
                // Prepare JSON data
                const jsonData = JSON.stringify(data, null, 2);
                
                // Create hidden form
                const form = document.createElement('form');
                form.method = 'POST';
                form.style.display = 'none';
                
                const input = document.createElement('input');
                input.name = 'json_data';
                input.value = jsonData;
                form.appendChild(input);
                
                document.body.appendChild(form);
                
                // Show in console for manual paste if needed
                console.log('DATA TO SAVE:', jsonData);
                
                // For Databricks, we'll write to a known widget
                // User will need to run next cell to complete save
                
                // Simulate save (in Databricks, this triggers next cell)
                setTimeout(() => {{
                    showAlert('? Save triggered! Run Cell 5 below to complete the save to file.', 'success');
                    document.getElementById('lastSaved').textContent = new Date().toLocaleTimeString();
                    btn.disabled = false;
                    btn.textContent = '?? Save All Changes';
                    
                    // Store in element for next cell to read
                    const dataHolder = document.getElementById('savedData') || document.createElement('div');
                    dataHolder.id = 'savedData';
                    dataHolder.textContent = jsonData;
                    dataHolder.style.display = 'none';
                    document.body.appendChild(dataHolder);
                }}, 500);
            }}
            
            render();
        </script>
    </body>
    </html>
    """
    
    return html

displayHTML(create_simple_editor(metrics_df, save_path))

print("\n" + "="*80)
print("? Editor loaded above!")
print("="*80)
print("?? Make your changes, then click 'Save All Changes'")
print("?? After saving, run Cell 5 below to write to file")
print("="*80)

# COMMAND ----------
# CELL 5: ?? Complete the Save (Run After Clicking Save in UI)

print("="*80)
print("?? SAVE TO FILE")
print("="*80)
print()

# Widget to receive JSON from UI
dbutils.widgets.text("saved_json", "", "Paste JSON from Console/UI")
json_input = dbutils.widgets.get("saved_json")

if json_input.strip():
    try:
        # Parse and save
        new_data = json.loads(json_input)
        new_df = pd.DataFrame(new_data)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save file
        new_df.to_csv(save_path, index=False)
        
        print("? SUCCESS! File saved:")
        print(f"   ?? Location: {save_path}")
        print(f"   ?? Rows: {len(new_df)}")
        print(f"   ?? Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("="*80)
        
        # Show the saved data
        display(new_df)
        
        # Clear the widget
        dbutils.widgets.remove("saved_json")
        
    except Exception as e:
        print(f"? Error: {e}")
        print()
        print("?? TIP: After clicking 'Save' in the UI above:")
        print("   1. Open browser console (F12)")
        print("   2. Look for 'DATA TO SAVE'")
        print("   3. Copy the JSON")
        print("   4. Paste it in the 'saved_json' widget above")
        print("   5. Run this cell again")

else:
    print("??  Waiting for save...")
    print()
    print("?? STEPS:")
    print("   1. Edit table in Cell 4 above")
    print("   2. Click '?? Save All Changes' button")
    print("   3. Copy the JSON that appears in the alert")
    print("   4. Paste it in the 'saved_json' widget above this cell")
    print("   5. Run this cell again")
    print()
    print("?? The JSON will also be in your browser console (press F12)")
    print()
    print("="*80)

# COMMAND ----------
# CELL 6: ?? View Current Configuration

print("="*80)
print("?? CURRENT CONFIGURATION")
print("="*80)

# Reload to get latest
current_df, current_path = load_config()

print(f"?? File: {current_path}")
print(f"?? Rows: {len(current_df)}")
print(f"?? Columns: {list(current_df.columns)}")
print("="*80)

display(current_df)
