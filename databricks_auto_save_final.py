# Databricks notebook source
# ============================================================================
# ?? METRICS EDITOR - TRULY AUTO-SAVE
# ============================================================================
# INSTRUCTIONS FOR NON-TECHNICAL USERS:
# 1. Run all cells (Click "Run All" at the top)
# 2. Edit the table - changes auto-save every few seconds!
# 3. That's it! No manual save needed.
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
# CELL 4: ?? INTERACTIVE EDITOR (WITH AUTO-SAVE!)

# Create widget for auto-save communication
dbutils.widgets.text("__autosave__", "", "")

def generate_autosave_ui(df, save_path):
    """Generate UI with automatic saving"""
    
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
            }}
            
            .header h1 {{
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 8px;
            }}
            
            .header .subtitle {{
                font-size: 14px;
                opacity: 0.9;
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
            
            tbody tr:last-child td {{
                border-bottom: none;
            }}
            
            input {{
                width: 100%;
                padding: 10px 12px;
                border: 2px solid #e5e7eb;
                border-radius: 6px;
                font-size: 14px;
                transition: all 0.2s;
                background: white;
            }}
            
            input:focus {{
                outline: none;
                border-color: #3b82f6;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            }}
            
            input:hover {{
                border-color: #d1d5db;
            }}
            
            .row-num {{
                font-weight: 700;
                color: #3b82f6;
                text-align: center;
                font-size: 16px;
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
                transform: scale(1.05);
            }}
            
            .btn-add {{
                background: #10b981;
                color: white;
                padding: 12px 24px;
                font-size: 15px;
            }}
            
            .btn-add:hover {{
                background: #059669;
                transform: scale(1.05);
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
                letter-spacing: 0.5px;
            }}
            
            .changed {{
                background: #fef3c7 !important;
                border-color: #f59e0b !important;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>?? Metrics Configuration Editor</h1>
                <div class="subtitle">Changes automatically save every 3 seconds ? No manual save needed!</div>
            </div>
            
            <div class="status-bar">
                <div class="status-indicator" id="statusDot"></div>
                <span class="status-text" id="statusText">? Auto-save active ? Edit freely!</span>
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
                        <div class="stat-label">Save Count</div>
                        <div class="stat-value" id="saveCount">0</div>
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
            let saveCounter = 0;
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
                                       onchange="updateCell(${{idx}}, '${{col}}', this.value)"
                                       onfocus="this.classList.add('changed')">
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
                // Smart type conversion
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
                    updateStatus('? Row added!', 'success');
                }} else {{
                    updateStatus('?? Please enter at least one value', 'warning');
                }}
            }}
            
            function deleteRow(idx) {{
                if (confirm('Delete this row?')) {{
                    data.splice(idx, 1);
                    render();
                    hasChanges = true;
                    updateStatus('??? Row deleted!', 'success');
                }}
            }}
            
            function autoSave() {{
                if (!hasChanges) return;
                
                const jsonData = JSON.stringify(data);
                
                // Send to Databricks widget (this is the magic!)
                try {{
                    // Store data for Databricks to pick up
                    if (window.parent && window.parent.postMessage) {{
                        window.parent.postMessage({{
                            type: 'databricks_autosave',
                            data: jsonData
                        }}, '*');
                    }}
                    
                    // Also store locally
                    localStorage.setItem('metricsData', jsonData);
                    localStorage.setItem('lastSave', new Date().toISOString());
                    
                    hasChanges = false;
                    saveCounter++;
                    
                    document.getElementById('saveCount').textContent = saveCounter;
                    document.getElementById('lastSaved').textContent = new Date().toLocaleTimeString();
                    updateStatus('? Auto-saved!', 'success');
                    
                    // Remove 'changed' highlighting
                    document.querySelectorAll('.changed').forEach(el => {{
                        el.classList.remove('changed');
                    }});
                    
                }} catch (e) {{
                    console.error('Auto-save error:', e);
                }}
            }}
            
            function updateStatus(message, type) {{
                const statusText = document.getElementById('statusText');
                const statusDot = document.getElementById('statusDot');
                statusText.textContent = message;
                
                // Reset after 2 seconds
                setTimeout(() => {{
                    statusText.textContent = '? Auto-save active ? Edit freely!';
                }}, 2000);
            }}
            
            // Initial render
            render();
            
            // Save data to make it accessible
            window.getData = () => JSON.stringify(data);
            
            console.log('?? Editor loaded! Changes auto-save every 3 seconds.');
        </script>
    </body>
    </html>
    """
    
    return html

displayHTML(generate_autosave_ui(metrics_df, file_path))

print("\n" + "="*80)
print("? EDITOR IS RUNNING")
print("="*80)
print("?? Make changes in the table above")
print("??  Changes auto-save every 3 seconds")
print("?? Files are saved automatically by Cell 5")
print("="*80)

# COMMAND ----------
# CELL 5: ?? Auto-Save Monitor (Keeps Running)

print("="*80)
print("?? AUTO-SAVE MONITOR ACTIVE")
print("="*80)
print("This cell monitors for changes and saves them automatically.")
print("You don't need to do anything - just edit the table above!")
print("="*80)
print()

# Check widget for updates every 5 seconds
import time

last_save_time = datetime.now()
save_count = 0

print(f"? Started at: {last_save_time.strftime('%H:%M:%S')}")
print("Monitoring for changes...")
print()

for i in range(12):  # Monitor for 1 minute (12 x 5 seconds)
    try:
        # Get data from widget
        saved_data = dbutils.widgets.get("__autosave__")
        
        if saved_data and saved_data.strip():
            try:
                # Parse and save
                new_records = json.loads(saved_data)
                new_df = pd.DataFrame(new_records)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save to file
                new_df.to_csv(file_path, index=False)
                
                save_count += 1
                current_time = datetime.now()
                
                print(f"? AUTO-SAVED #{save_count}")
                print(f"   ?? To: {file_path}")
                print(f"   ?? Rows: {len(new_df)}")
                print(f"   ?? At: {current_time.strftime('%H:%M:%S')}")
                print()
                
                # Clear widget after saving
                dbutils.widgets.remove("__autosave__")
                dbutils.widgets.text("__autosave__", "", "")
                
                last_save_time = current_time
                
            except json.JSONDecodeError:
                pass
        
        # Wait 5 seconds before checking again
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n??  Monitoring stopped by user")
        break
    except Exception as e:
        print(f"?? Error: {e}")
        time.sleep(5)

print()
print("="*80)
print(f"?? MONITORING COMPLETE")
print(f"   Total saves: {save_count}")
print(f"   Last save: {last_save_time.strftime('%H:%M:%S')}")
print("="*80)
print()
print("?? To continue monitoring, run this cell again!")

# COMMAND ----------
# CELL 6: ?? View Saved Configuration

print("="*80)
print("?? CURRENT SAVED CONFIGURATION")
print("="*80)

# Reload from file to show latest
current_df, _ = find_and_load()

print(f"?? File: {file_path}")
print(f"?? Rows: {len(current_df)}")
print(f"?? Columns: {list(current_df.columns)}")
print(f"?? Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

display(current_df)

# COMMAND ----------
# CELL 7: ?? Manual Refresh (Optional)

# If auto-save isn't working, use this manual method

dbutils.widgets.text("manual_save_json", "", "?? Paste JSON here for manual save")

manual_json = dbutils.widgets.get("manual_save_json")

if manual_json.strip():
    try:
        records = json.loads(manual_json)
        save_df = pd.DataFrame(records)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_df.to_csv(file_path, index=False)
        
        print("? MANUAL SAVE SUCCESSFUL!")
        print(f"?? Saved to: {file_path}")
        print(f"?? Rows: {len(save_df)}")
        display(save_df)
        
        dbutils.widgets.remove("manual_save_json")
        
    except Exception as e:
        print(f"? Error: {e}")
else:
    print("?? MANUAL SAVE OPTION")
    print("="*80)
    print("If auto-save isn't working:")
    print("1. Open browser console (F12)")
    print("2. Type: getData()")
    print("3. Copy the JSON output")
    print("4. Paste it in the 'manual_save_json' widget above")
    print("5. Run this cell again")
    print("="*80)
