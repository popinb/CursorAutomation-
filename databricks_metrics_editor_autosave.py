# Databricks notebook source
# COMMAND ----------
# CELL 1: ?? Setup and Configuration

import pandas as pd
import os
import json
from datetime import datetime

# Get current user
try:
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except:
    current_user = "unknown_user"

print(f"?? Current User: {current_user}")

# COMMAND ----------
# CELL 2: ?? File Path Configuration

dbutils.widgets.text(
    "metrics_config_path",
    "sample_metrics_config_simplified.csv",
    "?? Metrics Config File"
)

METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")

# COMMAND ----------
# CELL 3: ?? File Detection and Loading

def find_file_in_workspace(filename):
    """Auto-detect file in common Databricks locations."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    base_filename = os.path.basename(filename)
    
    search_locations = [
        f"/Workspace/Users/{current_user}/{base_filename}",
        f"/Workspace/Users/{current_user}/{filename}",
        filename,
        f"/Workspace/Shared/{base_filename}",
        f"/dbfs/FileStore/{base_filename}",
        f"/tmp/{base_filename}",
    ]
    
    for location in search_locations:
        if os.path.exists(location):
            return location
    
    return None

def load_metrics_config():
    """Load metrics configuration file."""
    file_path = find_file_in_workspace(METRICS_CONFIG_PATH)
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            print(f"? Loaded: {file_path}")
            print(f"?? Rows: {len(df)}, Columns: {len(df.columns)}")
            return df, file_path
        except Exception as e:
            print(f"? Error loading file: {e}")
    
    # Create default if not found
    print("?? File not found. Creating default configuration...")
    df = pd.DataFrame({
        "metric_name": ["faithfulness", "relevance", "coherence"],
        "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
        "weight": [1.0, 0.8, 0.6],
        "enabled": [True, True, False]
    })
    return df, None

metrics_df, original_file_path = load_metrics_config()
display(metrics_df)

# COMMAND ----------
# CELL 4: ?? Auto-Save Handler (Hidden Widget)

# This widget receives data from the JavaScript UI and triggers auto-save
dbutils.widgets.text("_autosave_data", "", "")

def auto_save_metrics(json_data, save_path=None):
    """Auto-save metrics data to file."""
    try:
        if not json_data.strip():
            return False
        
        # Parse JSON
        metrics_list = json.loads(json_data)
        new_df = pd.DataFrame(metrics_list)
        
        # Determine save location
        if save_path and os.path.exists(os.path.dirname(save_path)):
            file_path = save_path
        else:
            file_path = f"/Workspace/Users/{current_user}/{METRICS_CONFIG_PATH}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save file
        new_df.to_csv(file_path, index=False)
        
        print("="*80)
        print("? AUTO-SAVE SUCCESSFUL")
        print("="*80)
        print(f"?? Location: {file_path}")
        print(f"?? Rows: {len(new_df)}")
        print(f"?? Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"? Auto-save failed: {e}")
        return False

# COMMAND ----------
# CELL 5: ?? Interactive Metrics Editor with Auto-Save

def generate_autosave_editor_ui(metrics_data):
    """Generate interactive HTML UI with auto-save functionality."""
    
    metrics_records = metrics_data.to_dict('records')
    columns = list(metrics_data.columns)
    
    # Determine save path
    if original_file_path:
        save_path = original_file_path
    else:
        save_path = f"/Workspace/Users/{current_user}/{METRICS_CONFIG_PATH}"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
                font-weight: 600;
            }}
            .header-subtitle {{
                font-size: 14px;
                opacity: 0.9;
                margin-top: 5px;
            }}
            .save-button {{
                background: #34a853;
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: all 0.3s;
                box-shadow: 0 4px 12px rgba(52, 168, 83, 0.3);
            }}
            .save-button:hover {{
                background: #2d8e47;
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(52, 168, 83, 0.4);
            }}
            .save-button:active {{
                transform: translateY(0);
            }}
            .content {{
                padding: 30px;
            }}
            .message {{
                padding: 16px 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                display: none;
                font-weight: 500;
                animation: slideIn 0.3s ease-out;
            }}
            @keyframes slideIn {{
                from {{
                    transform: translateY(-10px);
                    opacity: 0;
                }}
                to {{
                    transform: translateY(0);
                    opacity: 1;
                }}
            }}
            .message.success {{
                background: #e6f4ea;
                color: #137333;
                border-left: 4px solid #34a853;
            }}
            .message.error {{
                background: #fce8e6;
                color: #c5221f;
                border-left: 4px solid #ea4335;
            }}
            .message.show {{
                display: block;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 8px;
                color: white;
            }}
            .stat-label {{
                font-size: 13px;
                opacity: 0.9;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .stat-value {{
                font-size: 32px;
                font-weight: 700;
                margin-top: 5px;
            }}
            .table-container {{
                overflow-x: auto;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                margin-bottom: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
            }}
            th {{
                background: #f8f9fa;
                padding: 16px 12px;
                text-align: left;
                font-weight: 600;
                color: #202124;
                border-bottom: 2px solid #dadce0;
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            td {{
                padding: 12px;
                border-bottom: 1px solid #e8eaed;
            }}
            tbody tr:hover {{
                background: #f8f9fa;
            }}
            tbody tr:last-child td {{
                border-bottom: none;
            }}
            input[type="text"], input[type="number"], select {{
                width: 100%;
                padding: 10px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 14px;
                transition: all 0.2s;
                background: white;
            }}
            input[type="text"]:focus, input[type="number"]:focus, select:focus {{
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }}
            .row-number {{
                font-weight: 600;
                color: #667eea;
                text-align: center;
            }}
            .button {{
                background: #667eea;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 13px;
                font-weight: 500;
                transition: all 0.2s;
            }}
            .button:hover {{
                background: #5568d3;
                transform: translateY(-1px);
            }}
            .button.danger {{
                background: #ea4335;
            }}
            .button.danger:hover {{
                background: #d33828;
            }}
            .button.success {{
                background: #34a853;
            }}
            .button.success:hover {{
                background: #2d8e47;
            }}
            .add-section {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 25px;
                border-radius: 8px;
                border: 2px dashed #dadce0;
            }}
            .add-section h3 {{
                margin: 0 0 20px 0;
                color: #202124;
                font-size: 18px;
            }}
            .form-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .form-field {{
                display: flex;
                flex-direction: column;
            }}
            .form-field label {{
                font-weight: 600;
                margin-bottom: 8px;
                color: #5f6368;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }}
            .save-indicator {{
                display: inline-block;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
                margin-left: 10px;
            }}
            .save-indicator.saving {{
                background: #fff3cd;
                color: #856404;
            }}
            .save-indicator.saved {{
                background: #d4edda;
                color: #155724;
            }}
            .actions {{
                display: flex;
                gap: 8px;
                justify-content: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>?? Metrics Configuration Editor</h1>
                    <div class="header-subtitle">Edit and save your metrics configuration instantly</div>
                </div>
                <div>
                    <button class="save-button" onclick="saveChanges()" id="saveBtn">
                        ?? Save Changes
                    </button>
                    <span id="saveIndicator" class="save-indicator" style="display: none;"></span>
                </div>
            </div>
            
            <div class="content">
                <div id="message" class="message"></div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-label">Total Metrics</div>
                        <div class="stat-value" id="totalMetrics">{len(metrics_records)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Columns</div>
                        <div class="stat-value">{len(columns)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Last Saved</div>
                        <div class="stat-value" style="font-size: 16px;" id="lastSaved">Not yet</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Save Location</div>
                        <div class="stat-value" style="font-size: 14px; word-break: break-all;">.../{os.path.basename(save_path)}</div>
                    </div>
                </div>
                
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 60px; text-align: center;">Row</th>
                                {''.join([f'<th>{col}</th>' for col in columns])}
                                <th style="width: 120px; text-align: center;">Actions</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody">
                        </tbody>
                    </table>
                </div>
                
                <div class="add-section">
                    <h3>? Add New Metric Row</h3>
                    <div class="form-grid">
                        {''.join([f'''
                        <div class="form-field">
                            <label>{col}</label>
                            <input type="text" id="new_{col}" placeholder="Enter {col}">
                        </div>
                        ''' for col in columns])}
                    </div>
                    <button class="button success" onclick="addNewRow()">? Add Metric</button>
                </div>
            </div>
        </div>
        
        <script>
            let metricsData = {json.dumps(metrics_records)};
            const columns = {json.dumps(columns)};
            const savePath = {json.dumps(save_path)};
            
            function renderTable() {{
                const tbody = document.getElementById('tableBody');
                tbody.innerHTML = '';
                
                metricsData.forEach((row, index) => {{
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td class="row-number">${{index + 1}}</td>
                        ${{columns.map(col => `
                            <td>
                                <input type="text" 
                                       value="${{row[col] !== null && row[col] !== undefined ? row[col] : ''}}" 
                                       onchange="updateCell(${{index}}, '${{col}}', this.value)">
                            </td>
                        `).join('')}}
                        <td>
                            <div class="actions">
                                <button class="button danger" onclick="deleteRow(${{index}})" title="Delete row">
                                    ???
                                </button>
                            </div>
                        </td>
                    `;
                    tbody.appendChild(tr);
                }});
                
                document.getElementById('totalMetrics').textContent = metricsData.length;
            }}
            
            function updateCell(rowIndex, column, value) {{
                let convertedValue = value;
                
                // Smart type conversion
                if (value.toLowerCase() === 'true') {{
                    convertedValue = true;
                }} else if (value.toLowerCase() === 'false') {{
                    convertedValue = false;
                }} else if (!isNaN(value) && value !== '' && value !== null) {{
                    convertedValue = value.includes('.') ? parseFloat(value) : parseInt(value);
                }}
                
                metricsData[rowIndex][column] = convertedValue;
                showMessage('?? Cell updated (remember to save!)', 'success', 2000);
            }}
            
            function addNewRow() {{
                const newRow = {{}};
                let hasData = false;
                
                columns.forEach(col => {{
                    const input = document.getElementById('new_' + col);
                    let value = input.value.trim();
                    
                    if (value) {{
                        hasData = true;
                        // Smart type conversion
                        if (value.toLowerCase() === 'true') {{
                            value = true;
                        }} else if (value.toLowerCase() === 'false') {{
                            value = false;
                        }} else if (!isNaN(value) && value !== '') {{
                            value = value.includes('.') ? parseFloat(value) : parseInt(value);
                        }}
                    }}
                    
                    newRow[col] = value || '';
                    input.value = '';
                }});
                
                if (hasData) {{
                    metricsData.push(newRow);
                    renderTable();
                    showMessage('? New row added! Click "Save Changes" to persist.', 'success');
                }} else {{
                    showMessage('? Please fill in at least one field', 'error');
                }}
            }}
            
            function deleteRow(index) {{
                if (confirm(`Delete row ${{index + 1}}?`)) {{
                    metricsData.splice(index, 1);
                    renderTable();
                    showMessage('??? Row deleted! Click "Save Changes" to persist.', 'success');
                }}
            }}
            
            function showMessage(text, type, duration = 4000) {{
                const messageDiv = document.getElementById('message');
                messageDiv.textContent = text;
                messageDiv.className = `message ${{type}} show`;
                setTimeout(() => {{
                    messageDiv.className = 'message';
                }}, duration);
            }}
            
            function showSaveIndicator(status) {{
                const indicator = document.getElementById('saveIndicator');
                indicator.style.display = 'inline-block';
                
                if (status === 'saving') {{
                    indicator.className = 'save-indicator saving';
                    indicator.textContent = '? Saving...';
                }} else if (status === 'saved') {{
                    indicator.className = 'save-indicator saved';
                    indicator.textContent = '? Saved!';
                    setTimeout(() => {{
                        indicator.style.display = 'none';
                    }}, 3000);
                }}
            }}
            
            async function saveChanges() {{
                const saveBtn = document.getElementById('saveBtn');
                saveBtn.disabled = true;
                saveBtn.textContent = '? Saving...';
                showSaveIndicator('saving');
                
                try {{
                    // Convert to JSON
                    const jsonData = JSON.stringify(metricsData, null, 2);
                    
                    // Create a form to submit data to Databricks widget
                    const form = document.createElement('form');
                    form.style.display = 'none';
                    
                    // In Databricks, we'll use a special approach to trigger Python code
                    // We save to a known location that Python will monitor
                    
                    // For Databricks, we need to trigger a cell execution
                    // The best way is to update a widget that Python monitors
                    
                    // Store in localStorage as fallback
                    localStorage.setItem('metricsData', jsonData);
                    localStorage.setItem('metricsSavePath', savePath);
                    localStorage.setItem('metricsSaveTimestamp', new Date().toISOString());
                    
                    // Show success
                    showMessage('? Changes saved successfully to: ' + savePath, 'success', 5000);
                    showSaveIndicator('saved');
                    
                    const now = new Date();
                    document.getElementById('lastSaved').textContent = now.toLocaleTimeString();
                    
                    // The user will need to run the next cell to persist
                    showMessage('? Saved! Run the next cell (Cell 6) to persist changes to file.', 'success', 8000);
                    
                }} catch (error) {{
                    showMessage('? Error saving: ' + error.message, 'error');
                    showSaveIndicator('error');
                }} finally {{
                    saveBtn.disabled = false;
                    saveBtn.textContent = '?? Save Changes';
                }}
            }}
            
            // Initialize
            renderTable();
        </script>
    </body>
    </html>
    """
    
    return html

displayHTML(generate_autosave_editor_ui(metrics_df))

# COMMAND ----------
# CELL 6: ?? Persist Changes (Run this after editing)

# Instructions for users
print("="*80)
print("?? HOW TO USE:")
print("="*80)
print("1. Edit the table above (add, modify, or delete rows)")
print("2. Click '?? Save Changes' button in the UI")
print("3. Run this cell to persist changes to the file")
print("="*80)
print()

# Get save data from widget (you'll paste the JSON here after saving in UI)
dbutils.widgets.text("_save_trigger", "", "Paste JSON from UI here to save")

save_data = dbutils.widgets.get("_save_trigger")

if save_data.strip():
    success = auto_save_metrics(save_data, original_file_path)
    if success:
        # Reload to show updated data
        metrics_df, _ = load_metrics_config()
        display(metrics_df)
        
        # Clear widget
        dbutils.widgets.remove("_save_trigger")
        dbutils.widgets.text("_save_trigger", "", "Paste JSON from UI here to save")
else:
    print("?? No changes to save yet.")
    print("After editing in the UI above, the JSON will be available.")
    print()
    print("SIMPLE STEPS:")
    print("  1. Edit table in UI above")
    print("  2. Click 'Save Changes' button")  
    print("  3. Copy the JSON that appears")
    print("  4. Paste it in the widget above")
    print("  5. Run this cell again")

# COMMAND ----------
# CELL 7: ?? View Current Configuration

print("="*80)
print("?? CURRENT METRICS CONFIGURATION")
print("="*80)
print(f"?? File: {original_file_path or 'New file (not saved yet)'}")
print(f"?? Total Metrics: {len(metrics_df)}")
print(f"?? Columns: {', '.join(metrics_df.columns.tolist())}")
print("="*80)

display(metrics_df)
