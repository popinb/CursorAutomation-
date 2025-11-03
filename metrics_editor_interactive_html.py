# Databricks notebook source
# MAGIC %md
# MAGIC ## Interactive Metrics Editor with Full HTML Table
# MAGIC 
# MAGIC This provides a fully interactive table where users can:
# MAGIC - Edit metrics inline
# MAGIC - Add new rows with a button
# MAGIC - Delete rows with a button
# MAGIC - Save changes back to CSV
# MAGIC 
# MAGIC **Best for:** Non-technical users who want a spreadsheet-like experience

# COMMAND ----------

import pandas as pd
import json
from IPython.display import display, HTML
import os

# COMMAND ----------

# File path widget
dbutils.widgets.text("metrics_file_path", "sample_metrics_config_simplified.csv", "?? Metrics CSV File")

METRICS_FILE = dbutils.widgets.get("metrics_file_path")

# Load or create metrics dataframe
if os.path.exists(METRICS_FILE):
    metrics_df = pd.read_csv(METRICS_FILE)
    print(f"? Loaded {len(metrics_df)} metrics from {METRICS_FILE}")
else:
    # Create empty template
    metrics_df = pd.DataFrame(columns=[
        'name', 'type', 'description', 'grading_rubric', 
        'threshold', 'ground_truth_file_path', 'ground_truth_column'
    ])
    print(f"?? File not found. Created empty metrics template.")

# COMMAND ----------

def create_interactive_editor(df):
    """Create a fully interactive HTML table editor."""
    
    # Convert dataframe to list of dicts for easier JavaScript handling
    metrics_list = df.to_dict('records')
    metrics_json = json.dumps(metrics_list)
    
    html = f"""
    <style>
        .metrics-editor {{
            font-family: Arial, sans-serif;
            max-width: 100%;
            margin: 20px 0;
        }}
        
        .editor-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px 8px 0 0;
            margin-bottom: 0;
        }}
        
        .editor-header h2 {{
            margin: 0;
            font-size: 24px;
        }}
        
        .editor-controls {{
            background-color: #f8f9fa;
            padding: 15px;
            border: 1px solid #dee2e6;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
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
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .btn-secondary {{
            background-color: #007bff;
            color: white;
        }}
        
        .btn-secondary:hover {{
            background-color: #0056b3;
            transform: translateY(-2px);
        }}
        
        .btn-danger {{
            background-color: #dc3545;
            color: white;
        }}
        
        .btn-danger:hover {{
            background-color: #c82333;
        }}
        
        .metrics-table-container {{
            overflow-x: auto;
            border: 1px solid #dee2e6;
            border-top: none;
            border-radius: 0 0 8px 8px;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
        }}
        
        .metrics-table th {{
            background-color: #343a40;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 10;
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
            font-size: 14px;
            box-sizing: border-box;
        }}
        
        .metrics-table textarea {{
            min-height: 60px;
            resize: vertical;
        }}
        
        .metrics-table input:focus,
        .metrics-table select:focus,
        .metrics-table textarea:focus {{
            outline: none;
            border-color: #80bdff;
            box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
        }}
        
        .status-message {{
            margin: 15px 0;
            padding: 12px 20px;
            border-radius: 5px;
            font-weight: bold;
        }}
        
        .status-success {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        
        .status-error {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        
        .status-info {{
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }}
        
        .help-text {{
            font-size: 12px;
            color: #6c757d;
            margin-top: 10px;
            font-style: italic;
        }}
    </style>
    
    <div class="metrics-editor">
        <div class="editor-header">
            <h2>?? Interactive Metrics Editor</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">Edit metrics directly in the table. All changes are saved automatically.</p>
        </div>
        
        <div class="editor-controls">
            <button class="btn btn-primary" onclick="addNewRow()">? Add New Metric</button>
            <button class="btn btn-secondary" onclick="saveMetrics()">?? Save to File</button>
            <button class="btn btn-secondary" onclick="exportJSON()">?? Export as JSON</button>
            <span id="statusMessage" style="margin-left: auto; align-self: center;"></span>
        </div>
        
        <div class="metrics-table-container">
            <table class="metrics-table" id="metricsTable">
                <thead>
                    <tr>
                        <th style="width: 40px;">#</th>
                        <th style="width: 150px;">Name</th>
                        <th style="width: 120px;">Type</th>
                        <th style="min-width: 200px;">Description</th>
                        <th style="min-width: 250px;">Grading Rubric</th>
                        <th style="width: 100px;">Threshold</th>
                        <th style="width: 150px;">GT File</th>
                        <th style="width: 120px;">GT Column</th>
                        <th style="width: 80px;">Actions</th>
                    </tr>
                </thead>
                <tbody id="metricsTableBody">
                    <!-- Rows will be generated by JavaScript -->
                </tbody>
            </table>
        </div>
        
        <div class="help-text">
            ?? <strong>Tips:</strong> 
            ? Click any field to edit inline 
            ? Use "Add New Metric" to add a row 
            ? Click "Delete" to remove a metric 
            ? Click "Save to File" to persist changes
            ? Changes are stored in browser memory until you save
        </div>
    </div>
    
    <div id="outputArea"></div>
    
    <script>
        // Initialize metrics data
        let metricsData = {metrics_json};
        
        // Render table on load
        renderTable();
        
        function renderTable() {{
            const tbody = document.getElementById('metricsTableBody');
            tbody.innerHTML = '';
            
            if (metricsData.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; padding: 40px; color: #6c757d;">No metrics defined. Click "Add New Metric" to get started!</td></tr>';
                return;
            }}
            
            metricsData.forEach((metric, index) => {{
                const row = createMetricRow(metric, index);
                tbody.appendChild(row);
            }});
        }}
        
        function createMetricRow(metric, index) {{
            const tr = document.createElement('tr');
            
            tr.innerHTML = `
                <td style="text-align: center; font-weight: bold;">${{index + 1}}</td>
                <td><input type="text" value="${{escapeHtml(metric.name || '')}}" onchange="updateMetric(${{index}}, 'name', this.value)"></td>
                <td>
                    <select onchange="updateMetric(${{index}}, 'type', this.value)">
                        <option value="binary" ${{metric.type === 'binary' ? 'selected' : ''}}>Binary</option>
                        <option value="1-5_scale" ${{metric.type === '1-5_scale' ? 'selected' : ''}}>1-5 Scale</option>
                        <option value="percentage" ${{metric.type === 'percentage' ? 'selected' : ''}}>Percentage</option>
                    </select>
                </td>
                <td><textarea onchange="updateMetric(${{index}}, 'description', this.value)">${{escapeHtml(metric.description || '')}}</textarea></td>
                <td><textarea onchange="updateMetric(${{index}}, 'grading_rubric', this.value)">${{escapeHtml(metric.grading_rubric || '')}}</textarea></td>
                <td><input type="text" value="${{metric.threshold || ''}}" onchange="updateMetric(${{index}}, 'threshold', this.value)"></td>
                <td><input type="text" value="${{escapeHtml(metric.ground_truth_file_path || '')}}" onchange="updateMetric(${{index}}, 'ground_truth_file_path', this.value)"></td>
                <td><input type="text" value="${{escapeHtml(metric.ground_truth_column || '')}}" onchange="updateMetric(${{index}}, 'ground_truth_column', this.value)"></td>
                <td style="text-align: center;">
                    <button class="btn btn-danger" style="padding: 5px 10px; font-size: 12px;" onclick="deleteMetric(${{index}})">??? Delete</button>
                </td>
            `;
            
            return tr;
        }}
        
        function updateMetric(index, field, value) {{
            metricsData[index][field] = value;
            showStatus('Changes saved to memory. Click "Save to File" to persist.', 'info');
        }}
        
        function addNewRow() {{
            const newMetric = {{
                name: '',
                type: 'binary',
                description: '',
                grading_rubric: '',
                threshold: '0.5',
                ground_truth_file_path: '',
                ground_truth_column: ''
            }};
            
            metricsData.push(newMetric);
            renderTable();
            showStatus('? New metric added. Fill in the fields and click "Save to File".', 'success');
        }}
        
        function deleteMetric(index) {{
            if (confirm(`Are you sure you want to delete metric "${{metricsData[index].name}}"?`)) {{
                metricsData.splice(index, 1);
                renderTable();
                showStatus('? Metric deleted. Click "Save to File" to persist changes.', 'success');
            }}
        }}
        
        function saveMetrics() {{
            // Validate metrics
            const errors = [];
            metricsData.forEach((metric, idx) => {{
                if (!metric.name || metric.name.trim() === '') {{
                    errors.push(`Row ${{idx + 1}}: Name is required`);
                }}
            }});
            
            if (errors.length > 0) {{
                showStatus('? Validation errors: ' + errors.join(', '), 'error');
                return;
            }}
            
            // Prepare data for Python
            const metricsJson = JSON.stringify(metricsData, null, 2);
            
            // Create output area with instructions
            const outputArea = document.getElementById('outputArea');
            outputArea.innerHTML = `
                <div class="status-message status-info" style="margin-top: 20px;">
                    <strong>?? Ready to Save!</strong><br>
                    Copy the JSON data below and run the next cell to save to file.
                </div>
                <textarea id="metricsJsonOutput" style="width: 100%; height: 200px; font-family: monospace; padding: 10px; margin-top: 10px; border: 2px solid #007bff; border-radius: 5px;">${{metricsJson}}</textarea>
                <button class="btn btn-primary" style="margin-top: 10px;" onclick="copyToClipboard()">?? Copy JSON</button>
            `;
            
            // Also trigger Python save via hidden element (Databricks specific)
            window.metricsDataForSave = metricsData;
            
            showStatus('? Metrics ready to save. See instructions below.', 'success');
        }}
        
        function exportJSON() {{
            const dataStr = JSON.stringify(metricsData, null, 2);
            const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
            
            const exportFileDefaultName = 'metrics_export.json';
            
            const linkElement = document.createElement('a');
            linkElement.setAttribute('href', dataUri);
            linkElement.setAttribute('download', exportFileDefaultName);
            linkElement.click();
            
            showStatus('? Exported as JSON file', 'success');
        }}
        
        function copyToClipboard() {{
            const textarea = document.getElementById('metricsJsonOutput');
            textarea.select();
            document.execCommand('copy');
            showStatus('? Copied to clipboard!', 'success');
        }}
        
        function showStatus(message, type) {{
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = 'status-message status-' + type;
            
            setTimeout(() => {{
                statusEl.textContent = '';
                statusEl.className = '';
            }}, 5000);
        }}
        
        function escapeHtml(text) {{
            const map = {{
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            }};
            return String(text).replace(/[&<>"']/g, m => map[m]);
        }}
        
        // Initialize
        showStatus('?? Metrics loaded. Make your changes and click "Save to File" when done.', 'info');
    </script>
    """
    
    displayHTML(html)

# Display the interactive editor
create_interactive_editor(metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Metrics from Interactive Editor
# MAGIC 
# MAGIC After editing in the interactive table above:
# MAGIC 1. Click "Save to File" in the table
# MAGIC 2. Copy the JSON that appears
# MAGIC 3. Paste it into the text widget below
# MAGIC 4. Run this cell to save to CSV

# COMMAND ----------

dbutils.widgets.text("metrics_json_data", "", "?? Paste JSON data here")

metrics_json_str = dbutils.widgets.get("metrics_json_data")

if metrics_json_str and metrics_json_str.strip():
    try:
        # Parse JSON
        metrics_list = json.loads(metrics_json_str)
        
        # Convert to DataFrame
        new_metrics_df = pd.DataFrame(metrics_list)
        
        # Validate
        if 'name' not in new_metrics_df.columns or new_metrics_df['name'].isna().any():
            print("? Error: All metrics must have a name")
        else:
            # Save to file
            new_metrics_df.to_csv(METRICS_FILE, index=False)
            
            print("? " + "="*70)
            print(f"? SUCCESS: Saved {len(new_metrics_df)} metrics to {METRICS_FILE}")
            print("? " + "="*70)
            print("\n?? Saved metrics:")
            print(new_metrics_df[['name', 'type', 'threshold']].to_string())
            
            # Clear widget
            dbutils.widgets.text("metrics_json_data", "")
            
            print("\n?? Reload the editor cell above to see your saved changes!")
            
    except json.JSONDecodeError as e:
        print(f"? Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"? Error saving metrics: {e}")
else:
    print("?? Instructions:")
    print("1. Edit metrics in the interactive table above")
    print("2. Click 'Save to File' button")
    print("3. Copy the JSON that appears")
    print("4. Paste it in the widget above")
    print("5. Run this cell")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? How to Use the Interactive Editor
# MAGIC 
# MAGIC ### For Non-Technical Users:
# MAGIC 
# MAGIC 1. **View Metrics**: The table shows all your current metrics
# MAGIC 
# MAGIC 2. **Edit Any Field**: Click on any cell to edit it directly
# MAGIC    - Type changes will be saved automatically to memory
# MAGIC 
# MAGIC 3. **Add New Metric**: 
# MAGIC    - Click the "? Add New Metric" button
# MAGIC    - A new blank row appears at the bottom
# MAGIC    - Fill in all the fields
# MAGIC 
# MAGIC 4. **Delete a Metric**:
# MAGIC    - Click the "??? Delete" button on any row
# MAGIC    - Confirm the deletion
# MAGIC 
# MAGIC 5. **Save Changes**:
# MAGIC    - Click "?? Save to File" button
# MAGIC    - Copy the JSON that appears below the table
# MAGIC    - Paste it in the widget
# MAGIC    - Run the save cell
# MAGIC 
# MAGIC 6. **Export**: Click "?? Export as JSON" to download a backup
# MAGIC 
# MAGIC ### Field Descriptions:
# MAGIC 
# MAGIC - **Name**: A clear, unique name for your metric
# MAGIC - **Type**: Choose the scoring method
# MAGIC   - `binary`: Pass/Fail (0 or 1)
# MAGIC   - `1-5_scale`: Rating from 1 to 5
# MAGIC   - `percentage`: Score from 0 to 100
# MAGIC - **Description**: What does this metric measure?
# MAGIC - **Grading Rubric**: Detailed instructions for scoring
# MAGIC - **Threshold**: Minimum passing score
# MAGIC - **GT File**: (Optional) Ground truth reference file
# MAGIC - **GT Column**: (Optional) Specific column to use

# COMMAND ----------
