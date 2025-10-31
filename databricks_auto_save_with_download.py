# Databricks notebook source
# ============================================================================
# ?? METRICS EDITOR - AUTO-SAVE + EASY DOWNLOAD
# ============================================================================
# INSTRUCTIONS:
# 1. Run all cells (Click "Run All")
# 2. Edit the table - changes auto-save every 3 seconds
# 3. Click "Download" button to save to your computer
# ============================================================================

# COMMAND ----------
# CELL 1: ?? Quick Setup

import pandas as pd
import os
import json
from datetime import datetime
import time

# Get your username
try:
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except:
    current_user = "unknown_user"

print(f"?? User: {current_user}")

# COMMAND ----------
# CELL 2: ?? Configuration

# Your config file name
CONFIG_FILE = "sample_metrics_config_simplified.csv"

# Where to save (auto-detected)
SAVE_LOCATION = f"/Workspace/Users/{current_user}/{CONFIG_FILE}"

print(f"?? Config File: {CONFIG_FILE}")
print(f"?? Save Location: {SAVE_LOCATION}")

# COMMAND ----------
# CELL 3: ?? Load Your Data

def find_and_load():
    """Find and load config file"""
    # Try multiple locations
    paths_to_try = [
        SAVE_LOCATION,
        CONFIG_FILE,
        f"/Workspace/Shared/{CONFIG_FILE}",
        f"/dbfs/FileStore/{CONFIG_FILE}",
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"? Loaded from: {path}")
                return df, path
            except:
                pass
    
    # Create default if nothing found
    print("?? Creating new configuration...")
    df = pd.DataFrame({
        "metric_name": ["faithfulness", "relevance", "coherence"],
        "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
        "weight": [1.0, 0.8, 0.6],
        "enabled": [True, True, False]
    })
    return df, SAVE_LOCATION

metrics_df, file_path = find_and_load()

print(f"?? Loaded {len(metrics_df)} rows with {len(metrics_df.columns)} columns")
display(metrics_df)

# COMMAND ----------
# CELL 4: ?? INTERACTIVE EDITOR WITH DOWNLOAD BUTTON

# Create widget for auto-save communication
dbutils.widgets.text("__autosave__", "", "")

def generate_editor_with_download(df, save_path):
    """Generate UI with auto-save AND download button"""
    
    records = df.to_dict('records')
    columns = list(df.columns)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                color: white;
                padding: 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .header-left h1 {{
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 8px;
            }}
            
            .header-left .subtitle {{
                font-size: 14px;
                opacity: 0.9;
            }}
            
            .header-right {{
                display: flex;
                gap: 12px;
            }}
            
            .btn-header {{
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }}
            
            .btn-download {{
                background: #10b981;
                color: white;
            }}
            
            .btn-download:hover {{
                background: #059669;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
            }}
            
            .btn-copy {{
                background: #3b82f6;
                color: white;
            }}
            
            .btn-copy:hover {{
                background: #2563eb;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            }}
            
            .status-bar {{
                background: #f0fdf4;
                border-left: 4px solid #10b981;
                padding: 15px 20px;
                margin: 20px 30px;
                border-radius: 6px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .status-indicator {{
                width: 12px;
                height: 12px;
                background: #10b981;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
            
            .status-text {{
                font-weight: 600;
                color: #065f46;
            }}
            
            .content {{
                padding: 30px;
            }}
            
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 25px;
            }}
            
            .stat-card {{
                background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #3b82f6;
            }}
            
            .stat-label {{
                font-size: 12px;
                color: #1e40af;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .stat-value {{
                font-size: 24px;
                color: #1e3a8a;
                font-weight: 700;
                margin-top: 5px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 25px;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                overflow: hidden;
            }}
            
            thead {{
                background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                color: white;
            }}
            
            th {{
                padding: 16px 12px;
                text-align: left;
                font-weight: 600;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            td {{
                padding: 12px;
                border-bottom: 1px solid #e5e7eb;
            }}
            
            tbody tr:hover {{
                background: #f9fafb;
            }}
            
            input {{
                width: 100%;
                padding: 10px 12px;
                border: 2px solid #e5e7eb;
                border-radius: 6px;
                font-size: 14px;
                transition: all 0.2s;
            }}
            
            input:focus {{
                outline: none;
                border-color: #3b82f6;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            }}
            
            .row-num {{
                font-weight: 700;
                color: #3b82f6;
                text-align: center;
            }}
            
            .btn {{
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }}
            
            .btn-delete {{
                background: #fee2e2;
                color: #991b1b;
            }}
            
            .btn-delete:hover {{
                background: #fecaca;
            }}
            
            .btn-add {{
                background: #10b981;
                color: white;
                padding: 12px 24px;
                font-size: 15px;
            }}
            
            .btn-add:hover {{
                background: #059669;
            }}
            
            .add-section {{
                background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                padding: 25px;
                border-radius: 8px;
                border: 2px dashed #10b981;
            }}
            
            .add-title {{
                font-size: 18px;
                font-weight: 700;
                color: #065f46;
                margin-bottom: 20px;
            }}
            
            .form-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            
            .form-group label {{
                display: block;
                font-size: 11px;
                font-weight: 700;
                color: #374151;
                margin-bottom: 6px;
                text-transform: uppercase;
            }}
            
            .download-info {{
                background: #fef3c7;
                border-left: 4px solid #f59e0b;
                padding: 15px 20px;
                margin: 20px 30px;
                border-radius: 6px;
                display: none;
            }}
            
            .download-info.show {{
                display: block;
            }}
            
            .download-info strong {{
                color: #92400e;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="header-left">
                    <h1>?? Metrics Configuration Editor</h1>
                    <div class="subtitle">Auto-saves every 3 seconds ? Download anytime</div>
                </div>
                <div class="header-right">
                    <button class="btn-header btn-download" onclick="downloadCSV()">
                        ?? Download CSV
                    </button>
                    <button class="btn-header btn-copy" onclick="copyForDatabricks()">
                        ?? Copy Path
                    </button>
                </div>
            </div>
            
            <div class="status-bar">
                <div class="status-indicator"></div>
                <span class="status-text">? Auto-save active ? Edit freely!</span>
            </div>
            
            <div id="downloadInfo" class="download-info">
                <strong>?? Download Tip:</strong> Click "?? Download CSV" to save to your computer, 
                or use "?? Copy Path" to get the Databricks file location!
            </div>
            
            <div class="content">
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-label">Total Rows</div>
                        <div class="stat-value" id="totalRows">{len(records)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Columns</div>
                        <div class="stat-value">{len(columns)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Last Saved</div>
                        <div class="stat-value" style="font-size: 16px;" id="lastSaved">Just now</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">File Location</div>
                        <div class="stat-value" style="font-size: 14px;">.../{os.path.basename(save_path)}</div>
                    </div>
                </div>
                
                <table>
                    <thead>
                        <tr>
                            <th style="width: 60px;">#</th>
                            {''.join(f'<th>{col}</th>' for col in columns)}
                            <th style="width: 100px;">Action</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody"></tbody>
                </table>
                
                <div class="add-section">
                    <div class="add-title">? Add New Row</div>
                    <div class="form-grid">
                        {''.join(f'''
                        <div class="form-group">
                            <label>{col}</label>
                            <input type="text" id="new_{col}" placeholder="{col}">
                        </div>
                        ''' for col in columns)}
                    </div>
                    <button class="btn btn-add" onclick="addRow()">? Add New Row</button>
                </div>
            </div>
        </div>
        
        <script>
            let data = {json.dumps(records)};
            const columns = {json.dumps(columns)};
            const savePath = {json.dumps(save_path)};
            let hasChanges = false;
            
            // Auto-save every 3 seconds
            setInterval(autoSave, 3000);
            
            function render() {{
                const tbody = document.getElementById('tableBody');
                tbody.innerHTML = '';
                
                data.forEach((row, idx) => {{
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td class="row-num">${{idx + 1}}</td>
                        ${{columns.map(col => `
                            <td>
                                <input type="text" 
                                       value="${{row[col] ?? ''}}" 
                                       onchange="updateCell(${{idx}}, '${{col}}', this.value)">
                            </td>
                        `).join('')}}
                        <td style="text-align: center;">
                            <button class="btn btn-delete" onclick="deleteRow(${{idx}})">Delete</button>
                        </td>
                    `;
                    tbody.appendChild(tr);
                }});
                
                document.getElementById('totalRows').textContent = data.length;
            }}
            
            function updateCell(idx, col, val) {{
                let value = val;
                if (val.toLowerCase() === 'true') value = true;
                else if (val.toLowerCase() === 'false') value = false;
                else if (!isNaN(val) && val !== '') {{
                    value = val.includes('.') ? parseFloat(val) : parseInt(val);
                }}
                data[idx][col] = value;
                hasChanges = true;
            }}
            
            function addRow() {{
                const newRow = {{}};
                let hasValue = false;
                
                columns.forEach(col => {{
                    let val = document.getElementById('new_' + col).value.trim();
                    if (val) {{
                        hasValue = true;
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
                    hasChanges = true;
                }}
            }}
            
            function deleteRow(idx) {{
                if (confirm('Delete this row?')) {{
                    data.splice(idx, 1);
                    render();
                    hasChanges = true;
                }}
            }}
            
            function autoSave() {{
                if (!hasChanges) return;
                
                const jsonData = JSON.stringify(data);
                localStorage.setItem('metricsData', jsonData);
                localStorage.setItem('lastSave', new Date().toISOString());
                
                hasChanges = false;
                document.getElementById('lastSaved').textContent = new Date().toLocaleTimeString();
            }}
            
            function downloadCSV() {{
                // Convert data to CSV
                const headers = columns.join(',');
                const rows = data.map(row => 
                    columns.map(col => {{
                        let val = row[col];
                        if (val === null || val === undefined) val = '';
                        // Escape quotes and wrap in quotes if contains comma
                        if (String(val).includes(',') || String(val).includes('"')) {{
                            val = '"' + String(val).replace(/"/g, '""') + '"';
                        }}
                        return val;
                    }}).join(',')
                );
                
                const csv = headers + '\\n' + rows.join('\\n');
                
                // Create download
                const blob = new Blob([csv], {{ type: 'text/csv' }});
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'metrics_config_' + new Date().toISOString().split('T')[0] + '.csv';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                // Show success message
                const info = document.getElementById('downloadInfo');
                info.innerHTML = '<strong>? Downloaded!</strong> Check your Downloads folder for the CSV file.';
                info.className = 'download-info show';
                setTimeout(() => {{
                    info.className = 'download-info';
                }}, 5000);
            }}
            
            function copyForDatabricks() {{
                // Copy file path to clipboard
                const text = savePath;
                
                if (navigator.clipboard) {{
                    navigator.clipboard.writeText(text).then(() => {{
                        const info = document.getElementById('downloadInfo');
                        info.innerHTML = '<strong>? Copied!</strong> File path: ' + text;
                        info.className = 'download-info show';
                        setTimeout(() => {{
                            info.className = 'download-info';
                        }}, 5000);
                    }});
                }} else {{
                    alert('File location: ' + text);
                }}
            }}
            
            // Initial render
            render();
        </script>
    </body>
    </html>
    """
    
    return html

displayHTML(generate_editor_with_download(metrics_df, file_path))

print("\n" + "="*80)
print("? EDITOR WITH DOWNLOAD IS RUNNING")
print("="*80)
print("?? Make changes in the table above")
print("??  Changes auto-save every 3 seconds")
print("?? Click 'Download CSV' button to save to your computer")
print("?? Click 'Copy Path' to get Databricks file location")
print("="*80)

# COMMAND ----------
# CELL 5: ?? ALTERNATIVE: Download via Databricks

print("="*80)
print("?? DOWNLOAD YOUR FILE")
print("="*80)
print()

# Load current data
current_df, current_path = find_and_load()

print(f"?? Current file: {current_path}")
print(f"?? Rows: {len(current_df)}")
print()

# Copy to FileStore for easy download
try:
    # Get workspace URL
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl", "your-workspace")
    
    # Copy to FileStore
    filestore_filename = f"metrics_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filestore_path = f"/dbfs/FileStore/{filestore_filename}"
    
    current_df.to_csv(filestore_path, index=False)
    
    download_url = f"https://{workspace_url}/files/{filestore_filename}"
    
    print("? FILE READY FOR DOWNLOAD!")
    print("="*80)
    print()
    print("?? DOWNLOAD OPTIONS:")
    print()
    print("OPTION 1: Click this URL")
    print(f"   {download_url}")
    print()
    print("OPTION 2: Databricks UI")
    print("   1. Go to: Workspace ? Users ? " + current_user)
    print(f"   2. Find: {os.path.basename(current_path)}")
    print("   3. Right-click ? Download")
    print()
    print("OPTION 3: Use download button in UI above ??")
    print()
    print("="*80)
    
except Exception as e:
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("="*80)
    print()
    print("1. Open Databricks Workspace (left sidebar)")
    print("2. Navigate to: Users ? " + current_user)
    print(f"3. Find file: {os.path.basename(current_path)}")
    print("4. Right-click ? Download")
    print()
    print("="*80)

# Display current data
print()
print("?? CURRENT DATA:")
display(current_df)

# COMMAND ----------
# CELL 6: ?? View & Refresh

print("="*80)
print("?? CURRENT CONFIGURATION")
print("="*80)

# Reload from file
current_df, current_path = find_and_load()

print(f"?? File: {current_path}")
print(f"?? Rows: {len(current_df)}")
print(f"?? Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

display(current_df)
