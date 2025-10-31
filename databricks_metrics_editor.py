"""
Databricks Metrics Editor Module
=================================
Easy-to-use metrics configuration editor for Databricks notebooks.

Usage:
    from databricks_metrics_editor import MetricsEditor
    
    # Initialize
    editor = MetricsEditor()
    
    # Load data
    editor.load_data(
        evaluation_data_path="evaluation_data.csv",
        metrics_config_path="metrics_config.csv",
        ground_truth_files="gt1.csv;gt2.csv"
    )
    
    # Show interactive editor
    editor.show_editor()
    
    # Apply changes (run after editing)
    editor.apply_changes()
    
    # Save to file
    editor.save_metrics("metrics_edited.csv")
    
    # Access data
    eval_df = editor.evaluation_data
    metrics_df = editor.metrics_config_data
    gt_dict = editor.ground_truth_data
"""

import pandas as pd
import os
import json
from datetime import datetime


class MetricsEditor:
    """Interactive metrics configuration editor for Databricks."""
    
    def __init__(self, dbutils=None):
        """
        Initialize the Metrics Editor.
        
        Args:
            dbutils: Databricks utilities object (auto-detected if not provided)
        """
        self.dbutils = dbutils
        self.current_user = self._get_current_user()
        self.temp_metrics_path = f"/tmp/metrics_edit_{self.current_user.replace('@', '_').replace('.', '_')}.json"
        
        # Data storage
        self.evaluation_data = None
        self.metrics_config_data = None
        self.ground_truth_data = {}
        
    def _get_current_user(self):
        """Get current Databricks user."""
        try:
            if self.dbutils is None:
                # Try to import dbutils
                try:
                    from pyspark.dbutils import DBUtils
                    from pyspark.sql import SparkSession
                    spark = SparkSession.builder.getOrCreate()
                    self.dbutils = DBUtils(spark)
                except:
                    pass
            
            if self.dbutils:
                return self.dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
            return "unknown_user"
        except:
            return "unknown_user"
    
    def _parse_ground_truth_files(self, gt_files_string):
        """Parse semicolon or comma separated ground truth file paths."""
        if not gt_files_string or not str(gt_files_string).strip():
            return []
        
        gt_files_string = str(gt_files_string)
        
        files = []
        for separator in [';', ',']:
            if separator in gt_files_string:
                files = [f.strip() for f in gt_files_string.split(separator) if f.strip()]
                break
        
        if not files:
            files = [gt_files_string.strip()]
        
        return files
    
    def _find_file_in_workspace(self, filename):
        """Auto-detect file in common Databricks locations."""
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        
        base_filename = os.path.basename(filename)
        
        search_locations = []
        
        if self.current_user and self.current_user != "unknown_user":
            search_locations.extend([
                f"/Workspace/Users/{self.current_user}/{base_filename}",
                f"/Workspace/Users/{self.current_user}/{filename}",
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
    
    def _load_csv_file(self, filename):
        """Load CSV file with auto-detection."""
        try:
            file_path = self._find_file_in_workspace(filename)
            
            if not file_path:
                return None
            
            df = pd.read_csv(file_path)
            return df
            
        except Exception as e:
            return None
    
    def load_data(self, evaluation_data_path=None, metrics_config_path=None, ground_truth_files=None):
        """
        Load evaluation data, metrics configuration, and ground truth files.
        
        Args:
            evaluation_data_path (str): Path to evaluation data CSV
            metrics_config_path (str): Path to metrics configuration CSV
            ground_truth_files (str): Semicolon-separated paths to ground truth files
        
        Returns:
            dict: Summary of loaded data
        """
        summary = {}
        
        # Load evaluation data
        if evaluation_data_path:
            self.evaluation_data = self._load_csv_file(evaluation_data_path)
            
            if self.evaluation_data is None:
                # Create sample data
                self.evaluation_data = pd.DataFrame({
                    "sample_id": [1, 2, 3],
                    "prompt": [
                        "What is the capital of France?",
                        "Explain machine learning in simple terms",
                        "How do I bake a chocolate cake?"
                    ],
                    "response": [
                        "The capital of France is Paris.",
                        "Machine learning is AI that learns from data.",
                        "Mix ingredients and bake at 350?F for 30 minutes."
                    ]
                })
            
            summary['evaluation_data'] = {
                'rows': len(self.evaluation_data),
                'columns': list(self.evaluation_data.columns)
            }
        
        # Load metrics configuration
        if metrics_config_path:
            self.metrics_config_data = self._load_csv_file(metrics_config_path)
            
            if self.metrics_config_data is None:
                # Create sample metrics
                self.metrics_config_data = pd.DataFrame({
                    "metric_name": ["faithfulness", "relevance", "coherence"],
                    "metric_type": ["llm_judge", "llm_judge", "llm_judge"],
                    "weight": [1.0, 0.8, 0.6],
                    "enabled": [True, True, False]
                })
            
            # Initialize temp file
            with open(self.temp_metrics_path, 'w') as f:
                json.dump(self.metrics_config_data.to_dict('records'), f)
            
            summary['metrics_config'] = {
                'rows': len(self.metrics_config_data),
                'columns': list(self.metrics_config_data.columns)
            }
        
        # Load ground truth files
        if ground_truth_files:
            gt_files_list = self._parse_ground_truth_files(ground_truth_files)
            
            for file_path in gt_files_list:
                if not file_path:
                    continue
                
                filename = os.path.basename(file_path)
                found_path = self._find_file_in_workspace(file_path)
                
                if found_path:
                    try:
                        df = pd.read_csv(found_path)
                        self.ground_truth_data[filename] = df
                    except:
                        pass
            
            summary['ground_truth_files'] = list(self.ground_truth_data.keys())
        
        return summary
    
    def show_editor(self):
        """
        Display the interactive metrics editor UI.
        
        Note: Must be called from a Databricks notebook.
        """
        if self.metrics_config_data is None:
            raise ValueError("No metrics configuration loaded. Call load_data() first.")
        
        html = self._generate_editor_html()
        
        # Display in Databricks
        try:
            displayHTML(html)
        except NameError:
            # If displayHTML is not available, try IPython
            try:
                from IPython.display import HTML, display
                display(HTML(html))
            except:
                raise RuntimeError("Cannot display HTML. Must be run in Databricks or Jupyter notebook.")
    
    def _generate_editor_html(self):
        """Generate the HTML for the interactive editor."""
        metrics_data = self.metrics_config_data.to_dict('records')
        columns = list(self.metrics_config_data.columns)
        temp_path = self.temp_metrics_path
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    padding: 30px;
                }}
                .header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #e0e0e0;
                }}
                .header h1 {{
                    margin: 0;
                    color: #1a73e8;
                    font-size: 28px;
                }}
                .auto-save-indicator {{
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-size: 14px;
                    font-weight: 500;
                }}
                .auto-save-indicator.saved {{
                    background: #e6f4ea;
                    color: #137333;
                }}
                .auto-save-indicator.saving {{
                    background: #fef7e0;
                    color: #b06000;
                }}
                .button {{
                    background: #1a73e8;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                    transition: background 0.2s;
                }}
                .button:hover {{
                    background: #1557b0;
                }}
                .button.success {{
                    background: #34a853;
                }}
                .button.success:hover {{
                    background: #2d8e47;
                }}
                .button.danger {{
                    background: #ea4335;
                }}
                .button.danger:hover {{
                    background: #c5362c;
                }}
                .button.small {{
                    padding: 6px 12px;
                    font-size: 12px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th {{
                    background: #f1f3f4;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                    color: #202124;
                    border-bottom: 2px solid #dadce0;
                }}
                td {{
                    padding: 12px;
                    border-bottom: 1px solid #e8eaed;
                }}
                tr:hover {{
                    background: #f8f9fa;
                }}
                input[type="text"], input[type="number"], select {{
                    width: 100%;
                    padding: 8px;
                    border: 1px solid #dadce0;
                    border-radius: 4px;
                    font-size: 14px;
                    box-sizing: border-box;
                }}
                input[type="text"]:focus, input[type="number"]:focus, select:focus {{
                    outline: none;
                    border-color: #1a73e8;
                    box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.1);
                }}
                .add-row-section {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin-top: 20px;
                    border: 2px dashed #dadce0;
                }}
                .add-row-section h3 {{
                    margin-top: 0;
                    color: #202124;
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
                    font-weight: 500;
                    margin-bottom: 5px;
                    color: #5f6368;
                    font-size: 13px;
                }}
                .actions {{
                    display: flex;
                    gap: 8px;
                }}
                .message {{
                    padding: 12px 16px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                    display: none;
                }}
                .message.success {{
                    background: #e6f4ea;
                    color: #137333;
                    border: 1px solid #34a853;
                }}
                .message.error {{
                    background: #fce8e6;
                    color: #c5221f;
                    border: 1px solid #ea4335;
                }}
                .message.show {{
                    display: block;
                }}
                .stats {{
                    display: flex;
                    gap: 20px;
                    margin-bottom: 20px;
                    padding: 15px;
                    background: #e8f0fe;
                    border-radius: 4px;
                }}
                .stat-item {{
                    flex: 1;
                }}
                .stat-label {{
                    font-size: 12px;
                    color: #5f6368;
                    font-weight: 500;
                }}
                .stat-value {{
                    font-size: 24px;
                    color: #1a73e8;
                    font-weight: 600;
                }}
                .instruction-banner {{
                    background: #e8f5e9;
                    border-left: 4px solid #34a853;
                    padding: 15px 20px;
                    margin-bottom: 20px;
                    border-radius: 4px;
                }}
                .instruction-banner h3 {{
                    margin: 0 0 10px 0;
                    color: #137333;
                }}
                .instruction-banner p {{
                    margin: 5px 0;
                    color: #1e4620;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>?? Metrics Configuration Editor</h1>
                    <div class="auto-save-indicator saved" id="saveStatus">? Auto-Saved</div>
                </div>
                
                <div class="instruction-banner">
                    <h3>?? How to Use</h3>
                    <p><strong>1.</strong> Edit any cell in the table - changes are saved automatically</p>
                    <p><strong>2.</strong> Add or delete rows as needed</p>
                    <p><strong>3.</strong> Run <code>editor.apply_changes()</code> to load changes</p>
                    <p><strong>4.</strong> Run <code>editor.save_metrics("filename.csv")</code> to save</p>
                </div>
                
                <div id="message" class="message"></div>
                
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-label">TOTAL METRICS</div>
                        <div class="stat-value" id="totalMetrics">{len(metrics_data)}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">COLUMNS</div>
                        <div class="stat-value">{len(columns)}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">LAST UPDATED</div>
                        <div class="stat-value" style="font-size: 16px;" id="lastUpdated">Just now</div>
                    </div>
                </div>
                
                <table id="metricsTable">
                    <thead>
                        <tr>
                            <th style="width: 50px;">Row</th>
                            {''.join([f'<th>{col}</th>' for col in columns])}
                            <th style="width: 150px;">Actions</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                    </tbody>
                </table>
                
                <div class="add-row-section">
                    <h3>? Add New Metric</h3>
                    <div class="form-row">
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
            
            <script>
                let metricsData = {json.dumps(metrics_data)};
                const columns = {json.dumps(columns)};
                const tempPath = '{temp_path}';
                
                function renderTable() {{
                    const tbody = document.getElementById('tableBody');
                    tbody.innerHTML = '';
                    
                    metricsData.forEach((row, index) => {{
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td style="font-weight: 600; color: #5f6368;">${{index}}</td>
                            ${{columns.map(col => `
                                <td>
                                    <input type="text" 
                                           value="${{row[col] !== null && row[col] !== undefined ? row[col] : ''}}" 
                                           onchange="updateCell(${{index}}, '${{col}}', this.value)">
                                </td>
                            `).join('')}}
                            <td>
                                <div class="actions">
                                    <button class="button danger small" onclick="deleteRow(${{index}})">??? Delete</button>
                                </div>
                            </td>
                        `;
                        tbody.appendChild(tr);
                    }});
                    
                    document.getElementById('totalMetrics').textContent = metricsData.length;
                    document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
                }}
                
                function autoSave() {{
                    const saveStatus = document.getElementById('saveStatus');
                    saveStatus.textContent = '?? Saving...';
                    saveStatus.className = 'auto-save-indicator saving';
                    
                    const dataStr = JSON.stringify(metricsData, null, 2);
                    
                    const code = `
import json
with open('` + tempPath + `', 'w') as f:
    f.write('''` + dataStr + `''')
`;
                    
                    try {{
                        IPython.notebook.kernel.execute(code);
                        
                        setTimeout(() => {{
                            saveStatus.textContent = '? Auto-Saved';
                            saveStatus.className = 'auto-save-indicator saved';
                        }}, 500);
                    }} catch(e) {{
                        setTimeout(() => {{
                            saveStatus.textContent = '? Changes Ready';
                            saveStatus.className = 'auto-save-indicator saved';
                        }}, 500);
                    }}
                }}
                
                function updateCell(rowIndex, column, value) {{
                    let convertedValue = value;
                    if (value.toLowerCase() === 'true') {{
                        convertedValue = true;
                    }} else if (value.toLowerCase() === 'false') {{
                        convertedValue = false;
                    }} else if (!isNaN(value) && value !== '') {{
                        convertedValue = value.includes('.') ? parseFloat(value) : parseInt(value);
                    }}
                    
                    metricsData[rowIndex][column] = convertedValue;
                    showMessage('? Cell updated', 'success');
                    document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
                    autoSave();
                }}
                
                function addNewRow() {{
                    const newRow = {{}};
                    let hasData = false;
                    
                    columns.forEach(col => {{
                        const input = document.getElementById('new_' + col);
                        let value = input.value.trim();
                        
                        if (value) {{
                            hasData = true;
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
                        showMessage('? New metric added', 'success');
                        autoSave();
                    }} else {{
                        showMessage('? Please fill in at least one field', 'error');
                    }}
                }}
                
                function deleteRow(index) {{
                    if (confirm(`Are you sure you want to delete row ${{index}}?`)) {{
                        metricsData.splice(index, 1);
                        renderTable();
                        showMessage('? Row deleted', 'success');
                        autoSave();
                    }}
                }}
                
                function showMessage(text, type) {{
                    const messageDiv = document.getElementById('message');
                    messageDiv.textContent = text;
                    messageDiv.className = `message ${{type}} show`;
                    setTimeout(() => {{
                        messageDiv.className = 'message';
                    }}, 2000);
                }}
                
                autoSave();
                renderTable();
            </script>
        </body>
        </html>
        """
        
        return html
    
    def apply_changes(self):
        """
        Apply changes from the interactive editor.
        
        Returns:
            pd.DataFrame: Updated metrics configuration
        """
        try:
            if os.path.exists(self.temp_metrics_path):
                with open(self.temp_metrics_path, 'r') as f:
                    metrics_list = json.load(f)
                
                self.metrics_config_data = pd.DataFrame(metrics_list)
                
                print("="*80)
                print("? METRICS CONFIGURATION UPDATED")
                print("="*80)
                print(f"Total Metrics: {len(self.metrics_config_data)}")
                print(f"Columns: {', '.join(self.metrics_config_data.columns.tolist())}")
                print("="*80)
                
                return self.metrics_config_data
            else:
                print("?? No changes detected. Edit metrics using show_editor() first.")
                return self.metrics_config_data
                
        except Exception as e:
            print(f"? Error loading changes: {e}")
            return None
    
    def save_metrics(self, filename="metrics_config_edited.csv", location=None):
        """
        Save metrics configuration to a CSV file.
        
        Args:
            filename (str): Name of the file to save
            location (str): Optional custom save location
        
        Returns:
            str: Path where file was saved
        """
        if self.metrics_config_data is None:
            raise ValueError("No metrics configuration to save.")
        
        try:
            if location:
                save_path = os.path.join(location, filename)
            else:
                save_path = f"/Workspace/Users/{self.current_user}/{filename}"
            
            self.metrics_config_data.to_csv(save_path, index=False)
            
            print("="*80)
            print("? FILE SAVED")
            print("="*80)
            print(f"Location: {save_path}")
            print(f"Rows: {len(self.metrics_config_data)}")
            print("="*80)
            
            return save_path
            
        except Exception as e:
            try:
                alt_path = f"/dbfs/FileStore/{filename}"
                self.metrics_config_data.to_csv(alt_path, index=False)
                
                print("="*80)
                print("? FILE SAVED")
                print("="*80)
                print(f"Location: {alt_path}")
                print(f"Download: /FileStore/{filename}")
                print("="*80)
                
                return alt_path
            except Exception as e2:
                print(f"? Failed to save: {e2}")
                return None
    
    def get_summary(self):
        """
        Get a summary of all loaded data.
        
        Returns:
            dict: Summary information
        """
        summary = {
            'user': self.current_user,
            'evaluation_data': {
                'loaded': self.evaluation_data is not None,
                'rows': len(self.evaluation_data) if self.evaluation_data is not None else 0,
                'columns': list(self.evaluation_data.columns) if self.evaluation_data is not None else []
            },
            'metrics_config': {
                'loaded': self.metrics_config_data is not None,
                'rows': len(self.metrics_config_data) if self.metrics_config_data is not None else 0,
                'columns': list(self.metrics_config_data.columns) if self.metrics_config_data is not None else []
            },
            'ground_truth': {
                'loaded': len(self.ground_truth_data) > 0,
                'files': list(self.ground_truth_data.keys())
            }
        }
        return summary
    
    def display_summary(self):
        """Display a formatted summary of all loaded data."""
        print("="*80)
        print("?? DATA SUMMARY")
        print("="*80)
        print(f"?? User: {self.current_user}")
        
        if self.evaluation_data is not None:
            print(f"\n1?? EVALUATION DATA:")
            print(f"   Rows: {len(self.evaluation_data)}")
            print(f"   Columns: {', '.join(self.evaluation_data.columns.tolist())}")
        
        if self.metrics_config_data is not None:
            print(f"\n2?? METRICS CONFIG:")
            print(f"   Rows: {len(self.metrics_config_data)}")
            print(f"   Columns: {', '.join(self.metrics_config_data.columns.tolist())}")
        
        if self.ground_truth_data:
            print(f"\n3?? GROUND TRUTH FILES:")
            for filename, df in self.ground_truth_data.items():
                print(f"   ? {filename}: {len(df)} rows")
        
        print("="*80)
    
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_metrics_path):
            os.remove(self.temp_metrics_path)
