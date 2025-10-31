# Databricks notebook source
# COMMAND ----------
# CELL 1: Setup and Configuration

import pandas as pd
import os
import json
from datetime import datetime

# Get current user
try:
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except:
    current_user = "unknown_user"

# Create widgets for file paths
dbutils.widgets.text(
    "metrics_config_path",
    "sample_metrics_config_simplified.csv",
    "?? Metrics Configuration File"
)

METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")

# Hidden widget for auto-save functionality
dbutils.widgets.text("_autosave_data", "", "")

# COMMAND ----------
# CELL 2: Helper Functions

def find_file_in_workspace(filename):
    """Auto-detect file in common Databricks locations."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    base_filename = os.path.basename(filename)
    
    try:
        user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    except:
        user_name = None
    
    search_locations = []
    
    if user_name:
        search_locations.extend([
            f"/Workspace/Users/{user_name}/{base_filename}",
            f"/Workspace/Users/{user_name}/{filename}",
        ])
    
    search_locations.extend([
        filename,
        base_filename,
        f"./{base_filename}",
        f"/Workspace/Shared/{base_filename}",
        f"/dbfs/FileStore/{base_filename}",
        f"/tmp/{base_filename}",
        f"/Workspace/{base_filename}",
    ])
    
    for location in search_locations:
        if os.path.exists(location):
            return location
    
    return None

def load_metrics_config():
    """Load metrics configuration from CSV."""
    try:
        file_path = find_file_in_workspace(METRICS_CONFIG_PATH)
        
        if file_path:
            df = pd.read_csv(file_path)
            return df, file_path
        else:
            # Return sample data if file not found
            df = pd.DataFrame({
                "metric_name": ["faithfulness", "relevance", "coherence"],
                "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
                "weight": [1.0, 0.8, 0.6],
                "enabled": [True, True, False]
            })
            return df, None
            
    except Exception as e:
        print(f"?? Error loading file: {e}")
        df = pd.DataFrame({
            "metric_name": ["faithfulness", "relevance", "coherence"],
            "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
            "weight": [1.0, 0.8, 0.6],
            "enabled": [True, True, False]
        })
        return df, None

def save_metrics_config(df, file_path=None):
    """Save metrics configuration to CSV."""
    try:
        if file_path is None:
            file_path = find_file_in_workspace(METRICS_CONFIG_PATH)
        
        if file_path is None:
            # Create new file in user's workspace
            file_path = f"/Workspace/Users/{current_user}/{METRICS_CONFIG_PATH}"
        
        df.to_csv(file_path, index=False)
        return True, file_path
    except Exception as e:
        try:
            # Try alternative location
            alt_path = f"/dbfs/FileStore/{os.path.basename(METRICS_CONFIG_PATH)}"
            df.to_csv(alt_path, index=False)
            return True, alt_path
        except:
            return False, str(e)

# Load initial data
METRICS_CONFIG_DATA, CONFIG_FILE_PATH = load_metrics_config()

print("? Configuration loaded successfully!")
print(f"?? File path: {CONFIG_FILE_PATH or 'Using sample data'}")
print(f"?? Total metrics: {len(METRICS_CONFIG_DATA)}")

# COMMAND ----------
# CELL 3: ?? Interactive Metrics Editor with Auto-Save

def generate_autosave_editor_ui(metrics_df, file_path):
    """Generate interactive HTML UI with auto-save functionality."""
    
    metrics_data = metrics_df.to_dict('records')
    columns = list(metrics_df.columns)
    
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
                padding: 30px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 3px solid #667eea;
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                color: #667eea;
                font-size: 32px;
                font-weight: 700;
            }}
            .save-indicator {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
                margin-top: 10px;
                transition: all 0.3s;
            }}
            .save-indicator.saved {{
                background: #d4edda;
                color: #155724;
            }}
            .save-indicator.saving {{
                background: #fff3cd;
                color: #856404;
            }}
            .spinner {{
                width: 14px;
                height: 14px;
                border: 2px solid #856404;
                border-top-color: transparent;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }}
            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}
            .info-bar {{
                background: #e7f3ff;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 4px solid #2196F3;
            }}
            .info-bar strong {{
                color: #1976D2;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 14px 12px;
                text-align: left;
                font-weight: 600;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            th:first-child {{
                border-radius: 8px 0 0 0;
            }}
            th:last-child {{
                border-radius: 0 8px 0 0;
            }}
            td {{
                padding: 12px;
                border-bottom: 1px solid #e8eaed;
            }}
            tr:hover {{
                background: #f8f9ff;
            }}
            tr:last-child td:first-child {{
                border-radius: 0 0 0 8px;
            }}
            tr:last-child td:last-child {{
                border-radius: 0 0 8px 0;
            }}
            input[type="text"], input[type="number"], select {{
                width: 100%;
                padding: 8px 12px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 14px;
                transition: all 0.2s;
            }}
            input[type="text"]:focus, input[type="number"]:focus, select:focus {{
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }}
            .edited {{
                border-color: #ffc107 !important;
                background: #fffbf0;
            }}
            .button {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: all 0.2s;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
            }}
            .button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }}
            .button:active {{
                transform: translateY(0);
            }}
            .button.danger {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }}
            .button.success {{
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            }}
            .button.small {{
                padding: 6px 12px;
                font-size: 12px;
            }}
            .add-row-section {{
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                padding: 20px;
                border-radius: 12px;
                margin-top: 20px;
            }}
            .add-row-section h3 {{
                margin-top: 0;
                color: #8b4513;
            }}
            .form-row {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }}
            .form-field {{
                display: flex;
                flex-direction: column;
            }}
            .form-field label {{
                font-weight: 600;
                margin-bottom: 5px;
                color: #8b4513;
                font-size: 13px;
            }}
            .stats {{
                display: flex;
                gap: 15px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }}
            .stat-card {{
                flex: 1;
                min-width: 150px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }}
            .stat-label {{
                font-size: 12px;
                opacity: 0.9;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .stat-value {{
                font-size: 28px;
                font-weight: 700;
                margin-top: 5px;
            }}
            .actions {{
                display: flex;
                gap: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>? Metrics Configuration Editor</h1>
                <div class="save-indicator saved" id="saveIndicator">
                    <span>?</span>
                    <span>All changes saved</span>
                </div>
            </div>
            
            <div class="info-bar">
                <strong>?? File:</strong> {file_path or 'New file (will be created on first save)'}<br>
                <strong>?? How to use:</strong> Edit any cell directly in the table below. Changes are saved automatically!
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-label">Total Metrics</div>
                    <div class="stat-value" id="totalMetrics">{len(metrics_data)}</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <div class="stat-label">Columns</div>
                    <div class="stat-value">{len(columns)}</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                    <div class="stat-label">Last Saved</div>
                    <div class="stat-value" style="font-size: 16px;" id="lastSaved">Just now</div>
                </div>
            </div>
            
            <table id="metricsTable">
                <thead>
                    <tr>
                        <th style="width: 50px;">#</th>
                        {''.join([f'<th>{col}</th>' for col in columns])}
                        <th style="width: 120px;">Actions</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                </tbody>
            </table>
            
            <div class="add-row-section">
                <h3>? Add New Metric Row</h3>
                <div class="form-row">
                    {''.join([f'''
                    <div class="form-field">
                        <label>{col}</label>
                        <input type="text" id="new_{col}" placeholder="Enter {col}">
                    </div>
                    ''' for col in columns])}
                </div>
                <button class="button success" onclick="addNewRow()">? Add Row</button>
            </div>
        </div>
        
        <script>
            let metricsData = {json.dumps(metrics_data)};
            const columns = {json.dumps(columns)};
            let saveTimeout = null;
            let pendingChanges = false;
            
            function renderTable() {{
                const tbody = document.getElementById('tableBody');
                tbody.innerHTML = '';
                
                metricsData.forEach((row, index) => {{
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="font-weight: 700; color: #667eea; text-align: center;">${{index + 1}}</td>
                        ${{columns.map(col => `
                            <td>
                                <input type="text" 
                                       value="${{row[col] !== null && row[col] !== undefined ? row[col] : ''}}" 
                                       onchange="updateCell(${{index}}, '${{col}}', this.value)"
                                       onfocus="this.classList.add('edited')">
                            </td>
                        `).join('')}}
                        <td>
                            <div class="actions">
                                <button class="button danger small" onclick="deleteRow(${{index}})">???</button>
                            </div>
                        </td>
                    `;
                    tbody.appendChild(tr);
                }});
                
                document.getElementById('totalMetrics').textContent = metricsData.length;
            }}
            
            function updateCell(rowIndex, column, value) {{
                let convertedValue = value;
                
                // Auto-convert data types
                if (value.toLowerCase() === 'true') {{
                    convertedValue = true;
                }} else if (value.toLowerCase() === 'false') {{
                    convertedValue = false;
                }} else if (!isNaN(value) && value !== '' && value.trim() !== '') {{
                    convertedValue = value.includes('.') ? parseFloat(value) : parseInt(value);
                }}
                
                metricsData[rowIndex][column] = convertedValue;
                scheduleAutoSave();
            }}
            
            function addNewRow() {{
                const newRow = {{}};
                let hasData = false;
                
                columns.forEach(col => {{
                    const input = document.getElementById('new_' + col);
                    let value = input.value.trim();
                    
                    if (value) {{
                        hasData = true;
                        // Auto-convert data types
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
                    scheduleAutoSave();
                }} else {{
                    alert('?? Please fill in at least one field');
                }}
            }}
            
            function deleteRow(index) {{
                if (confirm(`Delete row ${{index + 1}}?`)) {{
                    metricsData.splice(index, 1);
                    renderTable();
                    scheduleAutoSave();
                }}
            }}
            
            function scheduleAutoSave() {{
                pendingChanges = true;
                showSaving();
                
                // Clear existing timeout
                if (saveTimeout) {{
                    clearTimeout(saveTimeout);
                }}
                
                // Schedule save after 1 second of no changes
                saveTimeout = setTimeout(() => {{
                    autoSave();
                }}, 1000);
            }}
            
            function showSaving() {{
                const indicator = document.getElementById('saveIndicator');
                indicator.className = 'save-indicator saving';
                indicator.innerHTML = '<div class="spinner"></div><span>Saving changes...</span>';
            }}
            
            function showSaved() {{
                const indicator = document.getElementById('saveIndicator');
                indicator.className = 'save-indicator saved';
                indicator.innerHTML = '<span>?</span><span>All changes saved</span>';
                document.getElementById('lastSaved').textContent = new Date().toLocaleTimeString();
            }}
            
            function autoSave() {{
                if (!pendingChanges) return;
                
                // Update hidden widget with JSON data
                const jsonStr = JSON.stringify(metricsData);
                
                // Use Databricks widget API to update the hidden widget
                try {{
                    // This will trigger the Python code to save
                    parent.postMessage({{
                        type: 'databricks-widget-update',
                        widgetName: '_autosave_data',
                        value: jsonStr
                    }}, '*');
                    
                    pendingChanges = false;
                    showSaved();
                }} catch (e) {{
                    console.error('Save error:', e);
                }}
            }}
            
            // Initial render
            renderTable();
            
            // Save on page unload
            window.addEventListener('beforeunload', (e) => {{
                if (pendingChanges) {{
                    autoSave();
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html

# Display the interactive editor
displayHTML(generate_autosave_editor_ui(METRICS_CONFIG_DATA, CONFIG_FILE_PATH))

print("\n" + "="*80)
print("? INTERACTIVE EDITOR LOADED")
print("="*80)
print("?? Edit the table above - changes save automatically!")
print("? Changes are saved 1 second after you stop typing")
print("?? Run the cell below to apply your changes")
print("="*80)

# COMMAND ----------
# CELL 4: ?? Auto-Save Handler (Run this cell to apply changes)

# Check if there's data from the UI
autosave_data = dbutils.widgets.get("_autosave_data")

if autosave_data and autosave_data.strip():
    try:
        # Parse the JSON data from the UI
        metrics_list = json.loads(autosave_data)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
        
        # Save to file
        success, save_path = save_metrics_config(METRICS_CONFIG_DATA, CONFIG_FILE_PATH)
        
        if success:
            print("="*80)
            print("? CHANGES SAVED SUCCESSFULLY")
            print("="*80)
            print(f"?? File: {save_path}")
            print(f"?? Total metrics: {len(METRICS_CONFIG_DATA)}")
            print(f"?? Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # Display updated data
            display(METRICS_CONFIG_DATA)
            
            # Clear the widget to prevent re-saving
            dbutils.widgets.remove("_autosave_data")
            dbutils.widgets.text("_autosave_data", "", "")
        else:
            print(f"? Save failed: {save_path}")
            
    except Exception as e:
        print(f"? Error processing changes: {e}")
else:
    print("?? No changes detected. Edit the table above and run this cell again to save.")

# COMMAND ----------
# CELL 5: ?? View Current Configuration

print("="*80)
print("?? CURRENT METRICS CONFIGURATION")
print("="*80)
print(f"Total Metrics: {len(METRICS_CONFIG_DATA)}")
print(f"Columns: {', '.join(METRICS_CONFIG_DATA.columns.tolist())}")
print("="*80)

display(METRICS_CONFIG_DATA)
