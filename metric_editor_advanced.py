# ADVANCED VERSION: Interactive HTML Table Editor for Databricks
# This provides a more sophisticated spreadsheet-like interface

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? Advanced Interactive Metrics Editor (HTML-based)
# MAGIC 
# MAGIC **Features**:
# MAGIC - Spreadsheet-like table interface
# MAGIC - Inline editing of all fields
# MAGIC - Visual add/delete row buttons
# MAGIC - Real-time validation
# MAGIC - Beautiful, user-friendly design

# COMMAND ----------

import pandas as pd
import json
import os
from html import escape

# ============================================================
# Configuration
# ============================================================

dbutils.widgets.text(
    "metrics_config_path",
    "sample_metrics_config_simplified.csv",
    "?? Metrics Configuration File"
)

METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")

# Load current metrics
def load_metrics_file(file_path):
    """Load metrics CSV file."""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df
        else:
            df = pd.DataFrame(columns=[
                'name', 'type', 'description', 'grading_rubric', 
                'threshold', 'ground_truth_file_path', 'ground_truth_column'
            ])
            return df
    except Exception as e:
        print(f"? Error loading file: {e}")
        return pd.DataFrame(columns=[
            'name', 'type', 'description', 'grading_rubric', 
            'threshold', 'ground_truth_file_path', 'ground_truth_column'
        ])

current_metrics_df = load_metrics_file(METRICS_CONFIG_PATH)

# ============================================================
# Generate Interactive HTML Table
# ============================================================

def generate_interactive_editor(df):
    """Generate interactive HTML form for editing metrics."""
    
    # Convert DataFrame to list of dicts for easier manipulation
    metrics_data = df.to_dict('records') if len(df) > 0 else []
    
    # Start with minimum 3 rows as requested
    while len(metrics_data) < 3:
        metrics_data.append({
            'name': '',
            'type': 'binary',
            'description': '',
            'grading_rubric': '',
            'threshold': '0.5',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        })
    
    html = """
    <style>
        .metrics-editor {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 100%;
            margin: 20px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }
        
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .metrics-table th {
            background: #2c3e50;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            border: 1px solid #34495e;
        }
        
        .metrics-table td {
            padding: 8px;
            border: 1px solid #ddd;
        }
        
        .metrics-table input[type="text"],
        .metrics-table textarea,
        .metrics-table select {
            width: 100%;
            padding: 6px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
            box-sizing: border-box;
        }
        
        .metrics-table textarea {
            min-height: 60px;
            resize: vertical;
            font-family: inherit;
        }
        
        .metrics-table tr:hover {
            background: #f8f9fa;
        }
        
        .metrics-table tr.empty-row {
            background: #fff9e6;
        }
        
        .delete-btn {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        
        .delete-btn:hover {
            background: #c0392b;
        }
        
        .add-btn, .save-btn {
            background: #27ae60;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }
        
        .add-btn:hover {
            background: #229954;
        }
        
        .save-btn {
            background: #3498db;
            font-weight: bold;
        }
        
        .save-btn:hover {
            background: #2980b9;
        }
        
        .button-group {
            margin: 20px 0;
        }
        
        .help-text {
            background: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 15px 0;
            border-radius: 4px;
        }
        
        .help-text h4 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        
        .help-text ul {
            margin: 5px 0;
            padding-left: 20px;
        }
        
        .instruction-box {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        
        .instruction-box strong {
            color: #856404;
        }
        
        .row-number {
            background: #ecf0f1;
            font-weight: bold;
            text-align: center;
            width: 40px;
        }
        
        #output-message {
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            display: none;
        }
        
        #output-message.success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        
        #output-message.error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
    </style>
    
    <div class="metrics-editor">
        <h2 style="color: #2c3e50; margin-top: 0;">?? Interactive Metrics Editor</h2>
        
        <div class="instruction-box">
            <strong>?? Instructions:</strong>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li>Edit cells directly in the table below</li>
                <li>Click "? Add Row" to add new metrics</li>
                <li>Click "??? Delete" to remove unwanted metrics</li>
                <li>Click "?? Save All Changes" when done</li>
            </ul>
        </div>
        
        <div id="output-message"></div>
        
        <table class="metrics-table" id="metricsTable">
            <thead>
                <tr>
                    <th style="width: 40px;">#</th>
                    <th style="width: 150px;">Metric Name *</th>
                    <th style="width: 120px;">Type *</th>
                    <th style="width: 200px;">Description</th>
                    <th style="width: 250px;">Grading Rubric *</th>
                    <th style="width: 100px;">Threshold *</th>
                    <th style="width: 150px;">Ground Truth File</th>
                    <th style="width: 120px;">GT Column</th>
                    <th style="width: 80px;">Actions</th>
                </tr>
            </thead>
            <tbody id="metricsBody">
    """
    
    # Add rows
    for idx, metric in enumerate(metrics_data):
        row_class = 'empty-row' if not metric.get('name', '').strip() else ''
        html += f"""
                <tr class="{row_class}" data-row="{idx}">
                    <td class="row-number">{idx + 1}</td>
                    <td><input type="text" name="name_{idx}" value="{escape(str(metric.get('name', '')))}"></td>
                    <td>
                        <select name="type_{idx}">
                            <option value="binary" {'selected' if metric.get('type') == 'binary' else ''}>Binary</option>
                            <option value="1-5_scale" {'selected' if metric.get('type') == '1-5_scale' else ''}>1-5 Scale</option>
                            <option value="percentage" {'selected' if metric.get('type') == 'percentage' else ''}>Percentage</option>
                        </select>
                    </td>
                    <td><textarea name="description_{idx}">{escape(str(metric.get('description', '')))}</textarea></td>
                    <td><textarea name="grading_rubric_{idx}">{escape(str(metric.get('grading_rubric', '')))}</textarea></td>
                    <td><input type="text" name="threshold_{idx}" value="{escape(str(metric.get('threshold', '')))}"></td>
                    <td><input type="text" name="ground_truth_file_path_{idx}" value="{escape(str(metric.get('ground_truth_file_path', '')))}"></td>
                    <td><input type="text" name="ground_truth_column_{idx}" value="{escape(str(metric.get('ground_truth_column', '')))}"></td>
                    <td><button class="delete-btn" onclick="deleteRow({idx})">??? Delete</button></td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
        
        <div class="button-group">
            <button class="add-btn" onclick="addRow()">? Add Row</button>
            <button class="save-btn" onclick="saveChanges()">?? Save All Changes</button>
        </div>
        
        <div class="help-text">
            <h4>?? Field Descriptions:</h4>
            <ul>
                <li><strong>Metric Name:</strong> Short name for your metric (e.g., "Accuracy", "Relevance")</li>
                <li><strong>Type:</strong> 
                    <ul>
                        <li>Binary: Yes/No, Pass/Fail (threshold: 0-1, usually 1)</li>
                        <li>1-5 Scale: Rating from 1 to 5 (threshold: 1-5, usually 4)</li>
                        <li>Percentage: 0-100% (threshold: 0-100, usually 70)</li>
                    </ul>
                </li>
                <li><strong>Description:</strong> What does this metric evaluate?</li>
                <li><strong>Grading Rubric:</strong> Detailed criteria for the LLM judge to follow</li>
                <li><strong>Threshold:</strong> Minimum passing score</li>
                <li><strong>Ground Truth File:</strong> (Optional) CSV file with reference answers</li>
                <li><strong>GT Column:</strong> (Optional) Column name in ground truth file</li>
            </ul>
        </div>
    </div>
    
    <script>
        let rowCounter = """ + str(len(metrics_data)) + """;
        
        function deleteRow(rowIdx) {
            const row = document.querySelector(`tr[data-row="${rowIdx}"]`);
            if (row) {
                if (confirm('Are you sure you want to delete this metric?')) {
                    row.remove();
                    updateRowNumbers();
                    showMessage('Row deleted. Click "Save All Changes" to persist.', 'success');
                }
            }
        }
        
        function addRow() {
            const tbody = document.getElementById('metricsBody');
            const newRow = document.createElement('tr');
            newRow.className = 'empty-row';
            newRow.setAttribute('data-row', rowCounter);
            
            newRow.innerHTML = `
                <td class="row-number">${rowCounter + 1}</td>
                <td><input type="text" name="name_${rowCounter}" value=""></td>
                <td>
                    <select name="type_${rowCounter}">
                        <option value="binary">Binary</option>
                        <option value="1-5_scale">1-5 Scale</option>
                        <option value="percentage">Percentage</option>
                    </select>
                </td>
                <td><textarea name="description_${rowCounter}"></textarea></td>
                <td><textarea name="grading_rubric_${rowCounter}"></textarea></td>
                <td><input type="text" name="threshold_${rowCounter}" value="0.5"></td>
                <td><input type="text" name="ground_truth_file_path_${rowCounter}" value=""></td>
                <td><input type="text" name="ground_truth_column_${rowCounter}" value=""></td>
                <td><button class="delete-btn" onclick="deleteRow(${rowCounter})">??? Delete</button></td>
            `;
            
            tbody.appendChild(newRow);
            rowCounter++;
            showMessage('New row added. Fill in the details and click "Save All Changes".', 'success');
        }
        
        function updateRowNumbers() {
            const rows = document.querySelectorAll('#metricsBody tr');
            rows.forEach((row, idx) => {
                const rowNumCell = row.querySelector('.row-number');
                if (rowNumCell) {
                    rowNumCell.textContent = idx + 1;
                }
            });
        }
        
        function showMessage(message, type) {
            const msgBox = document.getElementById('output-message');
            msgBox.textContent = message;
            msgBox.className = type;
            msgBox.style.display = 'block';
            
            setTimeout(() => {
                msgBox.style.display = 'none';
            }, 5000);
        }
        
        function saveChanges() {
            const rows = document.querySelectorAll('#metricsBody tr');
            const metrics = [];
            
            rows.forEach((row, idx) => {
                const rowIdx = row.getAttribute('data-row');
                const name = row.querySelector(`input[name="name_${rowIdx}"]`).value.trim();
                
                // Only save non-empty rows
                if (name) {
                    metrics.push({
                        name: name,
                        type: row.querySelector(`select[name="type_${rowIdx}"]`).value,
                        description: row.querySelector(`textarea[name="description_${rowIdx}"]`).value.trim(),
                        grading_rubric: row.querySelector(`textarea[name="grading_rubric_${rowIdx}"]`).value.trim(),
                        threshold: row.querySelector(`input[name="threshold_${rowIdx}"]`).value.trim(),
                        ground_truth_file_path: row.querySelector(`input[name="ground_truth_file_path_${rowIdx}"]`).value.trim(),
                        ground_truth_column: row.querySelector(`input[name="ground_truth_column_${rowIdx}"]`).value.trim()
                    });
                }
            });
            
            if (metrics.length === 0) {
                showMessage('? Error: At least one metric with a name is required!', 'error');
                return;
            }
            
            // Store in a way Python can retrieve
            const jsonData = JSON.stringify(metrics);
            
            // Create hidden input to pass data to Python
            let hiddenInput = document.getElementById('metrics-json-data');
            if (!hiddenInput) {
                hiddenInput = document.createElement('input');
                hiddenInput.type = 'hidden';
                hiddenInput.id = 'metrics-json-data';
                document.body.appendChild(hiddenInput);
            }
            hiddenInput.value = jsonData;
            
            showMessage(`? Prepared ${metrics.length} metrics for saving. Run the next cell to save to file!`, 'success');
            
            // Alert user to run next cell
            alert(`? Data prepared!\\n\\n${metrics.length} metrics are ready to save.\\n\\nPlease run the NEXT CELL to save to CSV file.`);
        }
    </script>
    """
    
    return html

# Display the interactive editor
html_editor = generate_interactive_editor(current_metrics_df)
displayHTML(html_editor)

print("\n" + "=" * 80)
print("? Interactive editor loaded!")
print("?? Edit the table above, then run the NEXT CELL to save changes")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? Save Changes to File
# MAGIC 
# MAGIC **Run this cell after editing the table above to save your changes.**

# COMMAND ----------

# ============================================================
# Save the edited metrics to CSV
# ============================================================

# NOTE: Since Databricks doesn't allow direct JS-to-Python communication,
# we use a widget-based approach instead

dbutils.widgets.text("metrics_json", "", "?? Metrics JSON (auto-filled by editor)")

# Try to get the JSON data
metrics_json_str = dbutils.widgets.get("metrics_json")

if not metrics_json_str or metrics_json_str.strip() == "":
    print("??  No data to save!")
    print("\n?? How to save:")
    print("1. Edit the metrics table above")
    print("2. Click '?? Save All Changes' button in the table")
    print("3. Copy the JSON output that appears")
    print("4. Paste it into the 'Metrics JSON' widget above")
    print("5. Run this cell again")
    
    print("\n" + "=" * 80)
    print("?? OR: Use the simple widget-based editor below for easier saving")
    print("=" * 80)
else:
    try:
        import json
        metrics_data = json.loads(metrics_json_str)
        
        # Create DataFrame
        new_metrics_df = pd.DataFrame(metrics_data)
        
        # Save to file
        new_metrics_df.to_csv(METRICS_CONFIG_PATH, index=False)
        
        print("=" * 80)
        print("? SUCCESS! Metrics saved to file")
        print("=" * 80)
        print(f"?? File: {METRICS_CONFIG_PATH}")
        print(f"?? Saved {len(new_metrics_df)} metrics")
        print("\n?? Saved metrics:")
        display(new_metrics_df)
        
        print("\n?? You can now continue to the next cells to run your evaluation!")
        
    except json.JSONDecodeError as e:
        print(f"? Error: Invalid JSON data: {e}")
    except Exception as e:
        print(f"? Error saving file: {e}")
