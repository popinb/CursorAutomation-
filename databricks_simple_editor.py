# Databricks notebook source
# COMMAND ----------
# CELL 1: ?? Simple Metrics Editor - Setup

import pandas as pd
import os
import json
from datetime import datetime

# Get current user
try:
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except:
    current_user = "unknown_user"

# Configuration file path
CONFIG_FILENAME = "sample_metrics_config_simplified.csv"

# Widget for passing data between UI and Python
dbutils.widgets.text("updated_metrics", "", "")

# COMMAND ----------
# CELL 2: ?? Find and Load Configuration File

def find_and_load_config():
    """Find configuration file and load it."""
    
    # Try multiple locations
    possible_paths = [
        f"/Workspace/Users/{current_user}/{CONFIG_FILENAME}",
        f"/dbfs/FileStore/{CONFIG_FILENAME}",
        f"/Workspace/Shared/{CONFIG_FILENAME}",
        CONFIG_FILENAME
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df, path
            except:
                pass
    
    # If not found, create sample data
    print(f"?? File '{CONFIG_FILENAME}' not found. Creating sample data...")
    df = pd.DataFrame({
        "metric_name": ["faithfulness", "relevance", "coherence"],
        "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
        "weight": [1.0, 0.8, 0.6],
        "enabled": [True, True, False]
    })
    
    # Save it
    save_path = f"/Workspace/Users/{current_user}/{CONFIG_FILENAME}"
    df.to_csv(save_path, index=False)
    return df, save_path

# Load configuration
metrics_df, config_path = find_and_load_config()

print("? Configuration loaded!")
print(f"?? Location: {config_path}")
print(f"?? Rows: {len(metrics_df)}")

# COMMAND ----------
# CELL 3: ? Interactive Editor

def create_simple_editor(df, file_path):
    """Create a simple, user-friendly editor."""
    
    data = df.to_dict('records')
    columns = list(df.columns)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f0f2f5;
            }}
            .editor-container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 24px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}
            .header p {{
                margin: 8px 0 0 0;
                opacity: 0.9;
            }}
            .file-info {{
                background: #e3f2fd;
                padding: 16px 24px;
                border-bottom: 1px solid #ddd;
            }}
            .file-info strong {{
                color: #1976d2;
            }}
            .content {{
                padding: 24px;
            }}
            .save-button {{
                background: #4caf50;
                color: white;
                border: none;
                padding: 14px 32px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
                cursor: pointer;
                width: 100%;
                margin-bottom: 20px;
                transition: all 0.3s;
                box-shadow: 0 2px 5px rgba(76,175,80,0.3);
            }}
            .save-button:hover {{
                background: #45a049;
                box-shadow: 0 4px 8px rgba(76,175,80,0.4);
                transform: translateY(-2px);
            }}
            .save-button:active {{
                transform: translateY(0);
            }}
            .message {{
                padding: 12px;
                border-radius: 4px;
                margin-bottom: 16px;
                display: none;
                text-align: center;
                font-weight: 600;
            }}
            .message.success {{
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
                display: block;
            }}
            .message.error {{
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
                display: block;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th {{
                background: #f5f5f5;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid #ddd;
                color: #333;
            }}
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #eee;
            }}
            tr:hover {{
                background: #f9f9f9;
            }}
            input {{
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }}
            input:focus {{
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 2px rgba(102,126,234,0.1);
            }}
            .delete-btn {{
                background: #f44336;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
            }}
            .delete-btn:hover {{
                background: #d32f2f;
            }}
            .add-section {{
                background: #fff3e0;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
            }}
            .add-section h3 {{
                margin-top: 0;
                color: #e65100;
            }}
            .add-row-form {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
                margin-bottom: 12px;
            }}
            .add-row-form input {{
                padding: 10px;
            }}
            .add-btn {{
                background: #ff9800;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
            }}
            .add-btn:hover {{
                background: #f57c00;
            }}
        </style>
    </head>
    <body>
        <div class="editor-container">
            <div class="header">
                <h1>?? Metrics Configuration Editor</h1>
                <p>Edit your metrics and click "Save Changes" - it's that simple!</p>
            </div>
            
            <div class="file-info">
                <strong>?? File:</strong> {file_path}
            </div>
            
            <div class="content">
                <button class="save-button" onclick="saveChanges()">
                    ?? SAVE CHANGES
                </button>
                
                <div id="message" class="message"></div>
                
                <table>
                    <thead>
                        <tr>
                            <th style="width: 40px;">#</th>
                            {''.join([f'<th>{col}</th>' for col in columns])}
                            <th style="width: 80px;">Delete</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody"></tbody>
                </table>
                
                <div class="add-section">
                    <h3>? Add New Row</h3>
                    <div class="add-row-form">
                        {''.join([f'<input type="text" id="new_{col}" placeholder="{col}">' for col in columns])}
                    </div>
                    <button class="add-btn" onclick="addRow()">? Add Row</button>
                </div>
            </div>
        </div>
        
        <script>
            let data = {json.dumps(data)};
            const columns = {json.dumps(columns)};
            
            function renderTable() {{
                const tbody = document.getElementById('tableBody');
                tbody.innerHTML = '';
                
                data.forEach((row, idx) => {{
                    const tr = document.createElement('tr');
                    let html = `<td style="font-weight: bold; color: #666;">${{idx + 1}}</td>`;
                    
                    columns.forEach(col => {{
                        html += `<td><input type="text" value="${{row[col] ?? ''}}" 
                                 onchange="updateValue(${{idx}}, '${{col}}', this.value)"></td>`;
                    }});
                    
                    html += `<td><button class="delete-btn" onclick="deleteRow(${{idx}})">???</button></td>`;
                    tr.innerHTML = html;
                    tbody.appendChild(tr);
                }});
            }}
            
            function updateValue(rowIdx, column, value) {{
                // Auto-convert types
                if (value.toLowerCase() === 'true') value = true;
                else if (value.toLowerCase() === 'false') value = false;
                else if (!isNaN(value) && value !== '') {{
                    value = value.includes('.') ? parseFloat(value) : parseInt(value);
                }}
                data[rowIdx][column] = value;
            }}
            
            function addRow() {{
                const newRow = {{}};
                let hasValue = false;
                
                columns.forEach(col => {{
                    let value = document.getElementById('new_' + col).value.trim();
                    if (value) {{
                        hasValue = true;
                        // Auto-convert types
                        if (value.toLowerCase() === 'true') value = true;
                        else if (value.toLowerCase() === 'false') value = false;
                        else if (!isNaN(value) && value !== '') {{
                            value = value.includes('.') ? parseFloat(value) : parseInt(value);
                        }}
                    }}
                    newRow[col] = value || '';
                    document.getElementById('new_' + col).value = '';
                }});
                
                if (hasValue) {{
                    data.push(newRow);
                    renderTable();
                    showMessage('? Row added! Click "Save Changes" to save.', 'success');
                }} else {{
                    showMessage('?? Please enter at least one value', 'error');
                }}
            }}
            
            function deleteRow(idx) {{
                if (confirm('Delete this row?')) {{
                    data.splice(idx, 1);
                    renderTable();
                    showMessage('? Row deleted! Click "Save Changes" to save.', 'success');
                }}
            }}
            
            function saveChanges() {{
                const jsonData = JSON.stringify(data);
                
                // Send data to Databricks via widget update
                try {{
                    parent.postMessage({{
                        type: 'databricks-widget-update',
                        widgetName: 'updated_metrics',
                        value: jsonData
                    }}, '*');
                    
                    showMessage('? Changes saved successfully!', 'success');
                    
                    // Also log to console for debugging
                    console.log('Saved data:', jsonData);
                }} catch (e) {{
                    showMessage('? Error saving: ' + e.message, 'error');
                }}
            }}
            
            function showMessage(text, type) {{
                const msg = document.getElementById('message');
                msg.textContent = text;
                msg.className = 'message ' + type;
                setTimeout(() => msg.style.display = 'none', 5000);
            }}
            
            renderTable();
        </script>
    </body>
    </html>
    """
    return html

# Display editor
displayHTML(create_simple_editor(metrics_df, config_path))

print("\n?? Edit the table above and click 'SAVE CHANGES'")
print("?? Then run the cell below to apply your changes")

# COMMAND ----------
# CELL 4: ?? Apply Changes (Run this after clicking "Save Changes")

updated_data = dbutils.widgets.get("updated_metrics")

if updated_data and updated_data.strip():
    try:
        # Parse updated data
        metrics_list = json.loads(updated_data)
        metrics_df = pd.DataFrame(metrics_list)
        
        # Save to file
        metrics_df.to_csv(config_path, index=False)
        
        print("="*70)
        print("? SUCCESS! Changes saved to file")
        print("="*70)
        print(f"?? File: {config_path}")
        print(f"?? Total rows: {len(metrics_df)}")
        print(f"?? Saved at: {datetime.now().strftime('%H:%M:%S')}")
        print("="*70)
        
        # Show updated data
        display(metrics_df)
        
        # Clear widget
        dbutils.widgets.remove("updated_metrics")
        dbutils.widgets.text("updated_metrics", "", "")
        
    except Exception as e:
        print(f"? Error: {e}")
else:
    print("??  No changes detected")
    print("?? Make edits in the table above, click 'SAVE CHANGES', then run this cell")
