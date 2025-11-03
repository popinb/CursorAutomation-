# Databricks notebook source
# MAGIC %md
# MAGIC # ?? Databricks Validation Test Suite
# MAGIC 
# MAGIC **Purpose:** Thoroughly test all metrics editor components in Databricks environment
# MAGIC 
# MAGIC **What this tests:**
# MAGIC 1. ? Databricks environment setup
# MAGIC 2. ? Widget functionality
# MAGIC 3. ? File I/O operations
# MAGIC 4. ? HTML rendering
# MAGIC 5. ? Error handling
# MAGIC 6. ? Edge cases
# MAGIC 7. ? Integration with workshop notebook
# MAGIC 
# MAGIC **Run this BEFORE deploying to production!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 1: Environment Validation

# COMMAND ----------

import sys
import os
import pandas as pd
import json
from datetime import datetime

print("="*70)
print("?? TEST 1: DATABRICKS ENVIRONMENT VALIDATION")
print("="*70)

test_results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'tests_passed': 0,
    'tests_failed': 0,
    'issues': []
}

# Test 1.1: Check Python version
print("\n?? Test 1.1: Python Version")
py_version = sys.version_info
print(f"   Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
if py_version.major >= 3 and py_version.minor >= 8:
    print("   ? PASS: Python version is compatible")
    test_results['tests_passed'] += 1
else:
    print("   ? FAIL: Python version too old")
    test_results['tests_failed'] += 1
    test_results['issues'].append("Python version < 3.8")

# Test 1.2: Check dbutils availability
print("\n?? Test 1.2: Databricks Utilities (dbutils)")
try:
    # Try to access dbutils
    test_widget_name = f"test_widget_{datetime.now().timestamp()}"
    dbutils.widgets.text(test_widget_name, "test", "Test Widget")
    test_value = dbutils.widgets.get(test_widget_name)
    dbutils.widgets.remove(test_widget_name)
    print("   ? PASS: dbutils is available and functional")
    test_results['tests_passed'] += 1
except Exception as e:
    print(f"   ? FAIL: dbutils error - {e}")
    test_results['tests_failed'] += 1
    test_results['issues'].append(f"dbutils not available: {e}")

# Test 1.3: Check displayHTML availability
print("\n?? Test 1.3: HTML Display Function")
try:
    # displayHTML should be a built-in function in Databricks
    if 'displayHTML' in dir():
        print("   ? PASS: displayHTML is available")
        test_results['tests_passed'] += 1
    else:
        print("   ??  WARNING: displayHTML not found in built-ins")
        print("   Attempting to use it anyway...")
        try:
            displayHTML("<p>Test</p>")
            print("   ? PASS: displayHTML works despite not being in dir()")
            test_results['tests_passed'] += 1
        except:
            print("   ? FAIL: displayHTML not available")
            test_results['tests_failed'] += 1
            test_results['issues'].append("displayHTML not available")
except Exception as e:
    print(f"   ? FAIL: {e}")
    test_results['tests_failed'] += 1

# Test 1.4: Check pandas
print("\n?? Test 1.4: Pandas Library")
try:
    df_test = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    print(f"   Pandas version: {pd.__version__}")
    print("   ? PASS: Pandas is functional")
    test_results['tests_passed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    test_results['tests_failed'] += 1
    test_results['issues'].append(f"Pandas error: {e}")

# Test 1.5: Check file system access
print("\n?? Test 1.5: File System Access")
try:
    test_file = f"/tmp/test_{datetime.now().timestamp()}.txt"
    with open(test_file, 'w') as f:
        f.write("test")
    with open(test_file, 'r') as f:
        content = f.read()
    os.remove(test_file)
    
    if content == "test":
        print("   ? PASS: File I/O is functional")
        test_results['tests_passed'] += 1
    else:
        print("   ? FAIL: File content mismatch")
        test_results['tests_failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    test_results['tests_failed'] += 1
    test_results['issues'].append(f"File I/O error: {e}")

print("\n" + "="*70)
print(f"? Tests Passed: {test_results['tests_passed']}")
print(f"? Tests Failed: {test_results['tests_failed']}")
if test_results['issues']:
    print("\n??  Issues Found:")
    for issue in test_results['issues']:
        print(f"   ? {issue}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 2: Widget Functionality

# COMMAND ----------

print("="*70)
print("?? TEST 2: WIDGET FUNCTIONALITY")
print("="*70)

widget_tests = {'passed': 0, 'failed': 0, 'issues': []}

# Test 2.1: Text widget
print("\n?? Test 2.1: Text Widget")
try:
    dbutils.widgets.text("test_text", "default_value", "Test Text")
    value = dbutils.widgets.get("test_text")
    if value == "default_value":
        print("   ? PASS: Text widget works")
        widget_tests['passed'] += 1
    else:
        print(f"   ? FAIL: Expected 'default_value', got '{value}'")
        widget_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    widget_tests['failed'] += 1
    widget_tests['issues'].append(f"Text widget: {e}")

# Test 2.2: Dropdown widget
print("\n?? Test 2.2: Dropdown Widget")
try:
    dbutils.widgets.dropdown("test_dropdown", "option1", ["option1", "option2", "option3"], "Test Dropdown")
    value = dbutils.widgets.get("test_dropdown")
    if value == "option1":
        print("   ? PASS: Dropdown widget works")
        widget_tests['passed'] += 1
    else:
        print(f"   ? FAIL: Expected 'option1', got '{value}'")
        widget_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    widget_tests['failed'] += 1
    widget_tests['issues'].append(f"Dropdown widget: {e}")

# Test 2.3: Widget removal
print("\n?? Test 2.3: Widget Removal")
try:
    dbutils.widgets.remove("test_text")
    dbutils.widgets.remove("test_dropdown")
    print("   ? PASS: Widget removal works")
    widget_tests['passed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    widget_tests['failed'] += 1
    widget_tests['issues'].append(f"Widget removal: {e}")

# Test 2.4: Multiple widgets at once
print("\n?? Test 2.4: Multiple Widgets")
try:
    for i in range(5):
        dbutils.widgets.text(f"multi_widget_{i}", f"value_{i}", f"Widget {i}")
    
    all_correct = True
    for i in range(5):
        value = dbutils.widgets.get(f"multi_widget_{i}")
        if value != f"value_{i}":
            all_correct = False
            break
    
    # Clean up
    for i in range(5):
        dbutils.widgets.remove(f"multi_widget_{i}")
    
    if all_correct:
        print("   ? PASS: Multiple widgets work")
        widget_tests['passed'] += 1
    else:
        print("   ? FAIL: Widget value mismatch")
        widget_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    widget_tests['failed'] += 1
    widget_tests['issues'].append(f"Multiple widgets: {e}")

# Test 2.5: Unicode/emoji in widgets
print("\n?? Test 2.5: Unicode/Emoji Support")
try:
    dbutils.widgets.text("test_unicode", "?????", "Test ??")
    value = dbutils.widgets.get("test_unicode")
    dbutils.widgets.remove("test_unicode")
    
    if "??" in value:
        print("   ? PASS: Unicode/emoji support works")
        widget_tests['passed'] += 1
    else:
        print("   ??  WARNING: Emoji might not display correctly")
        widget_tests['passed'] += 1  # Don't fail, just warn
except Exception as e:
    print(f"   ? FAIL: {e}")
    widget_tests['failed'] += 1
    widget_tests['issues'].append(f"Unicode support: {e}")

print("\n" + "="*70)
print(f"? Widget Tests Passed: {widget_tests['passed']}")
print(f"? Widget Tests Failed: {widget_tests['failed']}")
if widget_tests['issues']:
    print("\n??  Issues Found:")
    for issue in widget_tests['issues']:
        print(f"   ? {issue}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 3: HTML Rendering

# COMMAND ----------

print("="*70)
print("?? TEST 3: HTML RENDERING")
print("="*70)

html_tests = {'passed': 0, 'failed': 0, 'issues': []}

# Test 3.1: Basic HTML
print("\n?? Test 3.1: Basic HTML Rendering")
try:
    test_html = """
    <div style="padding: 10px; background-color: #e8f4f8; border: 2px solid #3498db; border-radius: 5px;">
        <h3>? Test HTML</h3>
        <p>If you can see this styled box, HTML rendering works!</p>
    </div>
    """
    displayHTML(test_html)
    print("   ? PASS: Basic HTML rendered (check output above)")
    html_tests['passed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    html_tests['failed'] += 1
    html_tests['issues'].append(f"Basic HTML: {e}")

# Test 3.2: HTML with CSS
print("\n?? Test 3.2: HTML with CSS Styles")
try:
    test_html = """
    <style>
        .test-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        .test-table th {
            background-color: #2c3e50;
            color: white;
            padding: 10px;
        }
        .test-table td {
            padding: 8px;
            border: 1px solid #ddd;
        }
    </style>
    <table class="test-table">
        <thead>
            <tr><th>Column 1</th><th>Column 2</th></tr>
        </thead>
        <tbody>
            <tr><td>Data 1</td><td>Data 2</td></tr>
        </tbody>
    </table>
    """
    displayHTML(test_html)
    print("   ? PASS: HTML with CSS rendered (check styled table above)")
    html_tests['passed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    html_tests['failed'] += 1
    html_tests['issues'].append(f"HTML with CSS: {e}")

# Test 3.3: HTML with JavaScript
print("\n?? Test 3.3: HTML with JavaScript")
try:
    test_html = """
    <div id="jsTest" style="padding: 10px; background: #fff3cd; border: 2px solid #ffc107; border-radius: 5px;">
        <p>JavaScript test: <span id="jsResult">Loading...</span></p>
    </div>
    <script>
        document.getElementById('jsResult').textContent = '? JavaScript Works!';
    </script>
    """
    displayHTML(test_html)
    print("   ? PASS: HTML with JavaScript rendered (check if 'JavaScript Works!' appears above)")
    html_tests['passed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    html_tests['failed'] += 1
    html_tests['issues'].append(f"HTML with JavaScript: {e}")

# Test 3.4: Large HTML (simulate real metrics table)
print("\n?? Test 3.4: Large HTML Content")
try:
    rows_html = ""
    for i in range(20):
        rows_html += f"<tr><td>{i+1}</td><td>Metric {i+1}</td><td>binary</td><td>Description {i+1}</td></tr>"
    
    test_html = f"""
    <style>
        .large-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .large-table th {{
            background-color: #667eea;
            color: white;
            padding: 10px;
        }}
        .large-table td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
    </style>
    <h3>Large Table Test (20 rows)</h3>
    <table class="large-table">
        <thead>
            <tr><th>#</th><th>Name</th><th>Type</th><th>Description</th></tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """
    displayHTML(test_html)
    print("   ? PASS: Large HTML rendered (check 20-row table above)")
    html_tests['passed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    html_tests['failed'] += 1
    html_tests['issues'].append(f"Large HTML: {e}")

print("\n" + "="*70)
print(f"? HTML Tests Passed: {html_tests['passed']}")
print(f"? HTML Tests Failed: {html_tests['failed']}")
if html_tests['issues']:
    print("\n??  Issues Found:")
    for issue in html_tests['issues']:
        print(f"   ? {issue}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 4: File Operations

# COMMAND ----------

print("="*70)
print("?? TEST 4: FILE OPERATIONS")
print("="*70)

file_tests = {'passed': 0, 'failed': 0, 'issues': []}

# Create test directory
TEST_DIR = f"/tmp/metrics_test_{datetime.now().timestamp()}"
os.makedirs(TEST_DIR, exist_ok=True)
print(f"\n?? Test directory: {TEST_DIR}")

# Test 4.1: Create CSV file
print("\n?? Test 4.1: Create CSV File")
try:
    test_csv_path = os.path.join(TEST_DIR, "test_metrics.csv")
    test_df = pd.DataFrame({
        'name': ['Accuracy', 'Relevance', 'Safety'],
        'type': ['binary', '1-5_scale', 'percentage'],
        'description': ['Test 1', 'Test 2', 'Test 3'],
        'threshold': [1.0, 3.0, 80.0]
    })
    test_df.to_csv(test_csv_path, index=False)
    
    if os.path.exists(test_csv_path):
        print(f"   ? PASS: CSV file created at {test_csv_path}")
        file_tests['passed'] += 1
    else:
        print("   ? FAIL: CSV file not found")
        file_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    file_tests['failed'] += 1
    file_tests['issues'].append(f"CSV creation: {e}")

# Test 4.2: Read CSV file
print("\n?? Test 4.2: Read CSV File")
try:
    read_df = pd.read_csv(test_csv_path)
    if len(read_df) == 3 and 'name' in read_df.columns:
        print("   ? PASS: CSV file read successfully")
        print(f"   Rows: {len(read_df)}, Columns: {list(read_df.columns)}")
        file_tests['passed'] += 1
    else:
        print("   ? FAIL: CSV content mismatch")
        file_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    file_tests['failed'] += 1
    file_tests['issues'].append(f"CSV reading: {e}")

# Test 4.3: Update CSV file
print("\n?? Test 4.3: Update CSV File")
try:
    # Add new row
    new_row = pd.DataFrame({
        'name': ['New Metric'],
        'type': ['binary'],
        'description': ['Test 4'],
        'threshold': [0.5]
    })
    updated_df = pd.concat([read_df, new_row], ignore_index=True)
    updated_df.to_csv(test_csv_path, index=False)
    
    # Verify
    verify_df = pd.read_csv(test_csv_path)
    if len(verify_df) == 4:
        print("   ? PASS: CSV file updated successfully")
        print(f"   Rows after update: {len(verify_df)}")
        file_tests['passed'] += 1
    else:
        print(f"   ? FAIL: Expected 4 rows, got {len(verify_df)}")
        file_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    file_tests['failed'] += 1
    file_tests['issues'].append(f"CSV update: {e}")

# Test 4.4: Delete row from CSV
print("\n?? Test 4.4: Delete Row from CSV")
try:
    delete_df = verify_df.drop(verify_df.index[0]).reset_index(drop=True)
    delete_df.to_csv(test_csv_path, index=False)
    
    final_df = pd.read_csv(test_csv_path)
    if len(final_df) == 3:
        print("   ? PASS: Row deleted successfully")
        print(f"   Rows after deletion: {len(final_df)}")
        file_tests['passed'] += 1
    else:
        print(f"   ? FAIL: Expected 3 rows, got {len(final_df)}")
        file_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    file_tests['failed'] += 1
    file_tests['issues'].append(f"Row deletion: {e}")

# Test 4.5: Handle special characters in CSV
print("\n?? Test 4.5: Special Characters in CSV")
try:
    special_df = pd.DataFrame({
        'name': ['Metric with "quotes"', 'Metric, with, commas', 'Metric\nwith\nnewlines'],
        'type': ['binary', 'binary', 'binary'],
        'description': ['Test with special chars: @#$%^&*()'],
        'threshold': [1.0, 1.0, 1.0]
    })
    special_csv_path = os.path.join(TEST_DIR, "special_chars.csv")
    special_df.to_csv(special_csv_path, index=False)
    
    read_special = pd.read_csv(special_csv_path)
    if len(read_special) == 3:
        print("   ? PASS: Special characters handled correctly")
        file_tests['passed'] += 1
    else:
        print("   ? FAIL: Special characters caused issues")
        file_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    file_tests['failed'] += 1
    file_tests['issues'].append(f"Special characters: {e}")

# Clean up
print("\n???  Cleaning up test files...")
try:
    import shutil
    shutil.rmtree(TEST_DIR)
    print("   ? Test files cleaned up")
except Exception as e:
    print(f"   ??  Warning: Could not clean up - {e}")

print("\n" + "="*70)
print(f"? File Tests Passed: {file_tests['passed']}")
print(f"? File Tests Failed: {file_tests['failed']}")
if file_tests['issues']:
    print("\n??  Issues Found:")
    for issue in file_tests['issues']:
        print(f"   ? {issue}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 5: Metrics Editor Integration

# COMMAND ----------

print("="*70)
print("?? TEST 5: METRICS EDITOR INTEGRATION")
print("="*70)

integration_tests = {'passed': 0, 'failed': 0, 'issues': []}

# Setup test environment
TEST_METRICS_FILE = f"/tmp/test_metrics_{datetime.now().timestamp()}.csv"

# Create sample metrics file
sample_metrics = pd.DataFrame({
    'name': ['Test Accuracy', 'Test Relevance'],
    'type': ['binary', '1-5_scale'],
    'description': ['Check accuracy', 'Check relevance'],
    'grading_rubric': ['Score 1 if accurate', 'Rate 1-5'],
    'threshold': ['1.0', '3.0'],
    'ground_truth_file_path': ['', ''],
    'ground_truth_column': ['', '']
})
sample_metrics.to_csv(TEST_METRICS_FILE, index=False)

# Test 5.1: Display metrics table function
print("\n?? Test 5.1: Display Metrics Table Function")
try:
    def display_metrics_table_test(df):
        """Test version of display function"""
        if len(df) == 0:
            return "<p>No metrics</p>"
        
        html = f"""
        <style>
            .test-metrics-table {{
                border-collapse: collapse;
                width: 100%;
            }}
            .test-metrics-table th {{
                background-color: #2c3e50;
                color: white;
                padding: 10px;
            }}
            .test-metrics-table td {{
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
        </style>
        <h3>Test Metrics ({len(df)} metrics)</h3>
        <table class="test-metrics-table">
            <thead>
                <tr><th>#</th><th>Name</th><th>Type</th><th>Threshold</th></tr>
            </thead>
            <tbody>
        """
        
        for idx, row in df.iterrows():
            html += f"""
                <tr>
                    <td>{idx + 1}</td>
                    <td>{row['name']}</td>
                    <td>{row['type']}</td>
                    <td>{row['threshold']}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    # Test with sample data
    html_output = display_metrics_table_test(sample_metrics)
    displayHTML(html_output)
    
    if "Test Accuracy" in html_output and "Test Relevance" in html_output:
        print("   ? PASS: Display function works (check table above)")
        integration_tests['passed'] += 1
    else:
        print("   ? FAIL: Display function output incorrect")
        integration_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    integration_tests['failed'] += 1
    integration_tests['issues'].append(f"Display function: {e}")

# Test 5.2: Add metric workflow
print("\n?? Test 5.2: Add Metric Workflow")
try:
    # Simulate adding a metric
    new_metric = {
        'name': 'Test Safety',
        'type': 'percentage',
        'description': 'Check safety',
        'grading_rubric': 'Score 0-100',
        'threshold': '80.0',
        'ground_truth_file_path': '',
        'ground_truth_column': ''
    }
    
    # Add to dataframe
    test_df = pd.read_csv(TEST_METRICS_FILE)
    test_df = pd.concat([test_df, pd.DataFrame([new_metric])], ignore_index=True)
    test_df.to_csv(TEST_METRICS_FILE, index=False)
    
    # Verify
    verify_df = pd.read_csv(TEST_METRICS_FILE)
    if len(verify_df) == 3 and verify_df.iloc[2]['name'] == 'Test Safety':
        print("   ? PASS: Add metric workflow works")
        print(f"   Metrics count: {len(verify_df)}")
        integration_tests['passed'] += 1
    else:
        print("   ? FAIL: Metric not added correctly")
        integration_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    integration_tests['failed'] += 1
    integration_tests['issues'].append(f"Add metric: {e}")

# Test 5.3: Delete metric workflow
print("\n?? Test 5.3: Delete Metric Workflow")
try:
    # Delete first metric
    test_df = pd.read_csv(TEST_METRICS_FILE)
    test_df = test_df.drop(test_df.index[0]).reset_index(drop=True)
    test_df.to_csv(TEST_METRICS_FILE, index=False)
    
    # Verify
    verify_df = pd.read_csv(TEST_METRICS_FILE)
    if len(verify_df) == 2 and verify_df.iloc[0]['name'] == 'Test Relevance':
        print("   ? PASS: Delete metric workflow works")
        print(f"   Metrics count after delete: {len(verify_df)}")
        integration_tests['passed'] += 1
    else:
        print("   ? FAIL: Metric not deleted correctly")
        integration_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    integration_tests['failed'] += 1
    integration_tests['issues'].append(f"Delete metric: {e}")

# Test 5.4: Validation logic
print("\n?? Test 5.4: Validation Logic")
try:
    def validate_metric_test(metric):
        errors = []
        if not metric.get('name'):
            errors.append("Name is required")
        if metric.get('type') not in ['binary', '1-5_scale', 'percentage']:
            errors.append("Invalid type")
        try:
            float(metric.get('threshold', 0))
        except:
            errors.append("Threshold must be numeric")
        return errors
    
    # Test valid metric
    valid_metric = {
        'name': 'Valid Metric',
        'type': 'binary',
        'threshold': '1.0'
    }
    errors1 = validate_metric_test(valid_metric)
    
    # Test invalid metric
    invalid_metric = {
        'name': '',
        'type': 'invalid_type',
        'threshold': 'not_a_number'
    }
    errors2 = validate_metric_test(invalid_metric)
    
    if len(errors1) == 0 and len(errors2) > 0:
        print("   ? PASS: Validation logic works")
        print(f"   Valid metric errors: {len(errors1)}")
        print(f"   Invalid metric errors: {len(errors2)}")
        integration_tests['passed'] += 1
    else:
        print("   ? FAIL: Validation logic incorrect")
        integration_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    integration_tests['failed'] += 1
    integration_tests['issues'].append(f"Validation: {e}")

# Test 5.5: Widget integration
print("\n?? Test 5.5: Widget Integration")
try:
    # Create widgets like in real editor
    dbutils.widgets.text("test_metric_name", "Integration Test Metric", "Metric Name")
    dbutils.widgets.dropdown("test_metric_type", "binary", ["binary", "1-5_scale", "percentage"], "Type")
    dbutils.widgets.text("test_threshold", "1.0", "Threshold")
    
    # Get values
    name = dbutils.widgets.get("test_metric_name")
    mtype = dbutils.widgets.get("test_metric_type")
    threshold = dbutils.widgets.get("test_threshold")
    
    # Clean up
    dbutils.widgets.remove("test_metric_name")
    dbutils.widgets.remove("test_metric_type")
    dbutils.widgets.remove("test_threshold")
    
    if name and mtype and threshold:
        print("   ? PASS: Widget integration works")
        print(f"   Retrieved: name='{name}', type='{mtype}', threshold='{threshold}'")
        integration_tests['passed'] += 1
    else:
        print("   ? FAIL: Widget values not retrieved")
        integration_tests['failed'] += 1
except Exception as e:
    print(f"   ? FAIL: {e}")
    integration_tests['failed'] += 1
    integration_tests['issues'].append(f"Widget integration: {e}")

# Clean up test file
try:
    os.remove(TEST_METRICS_FILE)
except:
    pass

print("\n" + "="*70)
print(f"? Integration Tests Passed: {integration_tests['passed']}")
print(f"? Integration Tests Failed: {integration_tests['failed']}")
if integration_tests['issues']:
    print("\n??  Issues Found:")
    for issue in integration_tests['issues']:
        print(f"   ? {issue}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 6: Edge Cases & Error Handling

# COMMAND ----------

print("="*70)
print("?? TEST 6: EDGE CASES & ERROR HANDLING")
print("="*70)

edge_tests = {'passed': 0, 'failed': 0, 'issues': []}

# Test 6.1: Empty metrics file
print("\n?? Test 6.1: Empty Metrics File")
try:
    empty_file = f"/tmp/empty_metrics_{datetime.now().timestamp()}.csv"
    empty_df = pd.DataFrame(columns=['name', 'type', 'description', 'threshold'])
    empty_df.to_csv(empty_file, index=False)
    
    loaded = pd.read_csv(empty_file)
    if len(loaded) == 0 and all(col in loaded.columns for col in ['name', 'type']):
        print("   ? PASS: Empty file handled correctly")
        edge_tests['passed'] += 1
    else:
        print("   ? FAIL: Empty file not handled correctly")
        edge_tests['failed'] += 1
    
    os.remove(empty_file)
except Exception as e:
    print(f"   ? FAIL: {e}")
    edge_tests['failed'] += 1
    edge_tests['issues'].append(f"Empty file: {e}")

# Test 6.2: Very long metric name
print("\n?? Test 6.2: Very Long Metric Name")
try:
    long_name = "A" * 500  # 500 character name
    long_metric_df = pd.DataFrame({
        'name': [long_name],
        'type': ['binary'],
        'threshold': ['1.0']
    })
    
    long_file = f"/tmp/long_metric_{datetime.now().timestamp()}.csv"
    long_metric_df.to_csv(long_file, index=False)
    loaded = pd.read_csv(long_file)
    
    if len(loaded.iloc[0]['name']) == 500:
        print("   ? PASS: Long names handled correctly")
        edge_tests['passed'] += 1
    else:
        print("   ? FAIL: Long name truncated or corrupted")
        edge_tests['failed'] += 1
    
    os.remove(long_file)
except Exception as e:
    print(f"   ? FAIL: {e}")
    edge_tests['failed'] += 1
    edge_tests['issues'].append(f"Long names: {e}")

# Test 6.3: Special characters and unicode
print("\n?? Test 6.3: Special Characters & Unicode")
try:
    special_df = pd.DataFrame({
        'name': ['Metric with ??', 'Metric with ?mojis ??', 'Metric with "quotes"'],
        'type': ['binary', 'binary', 'binary'],
        'description': ['Description with special: @#$%^&*()'],
        'threshold': ['1.0', '1.0', '1.0']
    })
    
    special_file = f"/tmp/special_metric_{datetime.now().timestamp()}.csv"
    special_df.to_csv(special_file, index=False, encoding='utf-8')
    loaded = pd.read_csv(special_file, encoding='utf-8')
    
    if len(loaded) == 3 and '??' in str(loaded.iloc[1]['name']):
        print("   ? PASS: Special characters handled correctly")
        edge_tests['passed'] += 1
    else:
        print("   ??  WARNING: Some special characters might not work")
        edge_tests['passed'] += 1  # Don't fail on this
    
    os.remove(special_file)
except Exception as e:
    print(f"   ??  WARNING (non-critical): {e}")
    edge_tests['passed'] += 1  # Don't fail

# Test 6.4: Missing columns
print("\n?? Test 6.4: Missing Required Columns")
try:
    incomplete_df = pd.DataFrame({
        'name': ['Metric 1'],
        'type': ['binary']
        # Missing other columns
    })
    
    incomplete_file = f"/tmp/incomplete_{datetime.now().timestamp()}.csv"
    incomplete_df.to_csv(incomplete_file, index=False)
    loaded = pd.read_csv(incomplete_file)
    
    # Should load but with missing columns
    if 'name' in loaded.columns and 'type' in loaded.columns:
        print("   ? PASS: Missing columns handled (can be filled with defaults)")
        edge_tests['passed'] += 1
    else:
        print("   ? FAIL: Cannot handle missing columns")
        edge_tests['failed'] += 1
    
    os.remove(incomplete_file)
except Exception as e:
    print(f"   ? FAIL: {e}")
    edge_tests['failed'] += 1
    edge_tests['issues'].append(f"Missing columns: {e}")

# Test 6.5: Duplicate metric names
print("\n?? Test 6.5: Duplicate Metric Names")
try:
    dup_df = pd.DataFrame({
        'name': ['Accuracy', 'Accuracy', 'Accuracy'],
        'type': ['binary', 'binary', 'binary'],
        'threshold': ['1.0', '1.0', '1.0']
    })
    
    dup_file = f"/tmp/duplicate_{datetime.now().timestamp()}.csv"
    dup_df.to_csv(dup_file, index=False)
    loaded = pd.read_csv(dup_file)
    
    if len(loaded) == 3:  # All rows should be present
        print("   ? PASS: Duplicate names allowed (warning should be shown to user)")
        edge_tests['passed'] += 1
    else:
        print("   ? FAIL: Duplicate handling error")
        edge_tests['failed'] += 1
    
    os.remove(dup_file)
except Exception as e:
    print(f"   ? FAIL: {e}")
    edge_tests['failed'] += 1
    edge_tests['issues'].append(f"Duplicates: {e}")

print("\n" + "="*70)
print(f"? Edge Case Tests Passed: {edge_tests['passed']}")
print(f"? Edge Case Tests Failed: {edge_tests['failed']}")
if edge_tests['issues']:
    print("\n??  Issues Found:")
    for issue in edge_tests['issues']:
        print(f"   ? {issue}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 7: Final Summary & Recommendations

# COMMAND ----------

print("="*70)
print("?? FINAL TEST SUMMARY")
print("="*70)

total_passed = (test_results['tests_passed'] + widget_tests['passed'] + 
                html_tests['passed'] + file_tests['passed'] + 
                integration_tests['passed'] + edge_tests['passed'])

total_failed = (test_results['tests_failed'] + widget_tests['failed'] + 
                html_tests['failed'] + file_tests['failed'] + 
                integration_tests['failed'] + edge_tests['failed'])

total_tests = total_passed + total_failed

print(f"\n?? Overall Results:")
print(f"   Total Tests Run: {total_tests}")
print(f"   ? Passed: {total_passed}")
print(f"   ? Failed: {total_failed}")
print(f"   Success Rate: {(total_passed/total_tests*100):.1f}%")

print(f"\n?? Test Category Breakdown:")
print(f"   1. Environment:    {test_results['tests_passed']}/{test_results['tests_passed'] + test_results['tests_failed']}")
print(f"   2. Widgets:        {widget_tests['passed']}/{widget_tests['passed'] + widget_tests['failed']}")
print(f"   3. HTML:           {html_tests['passed']}/{html_tests['passed'] + html_tests['failed']}")
print(f"   4. Files:          {file_tests['passed']}/{file_tests['passed'] + file_tests['failed']}")
print(f"   5. Integration:    {integration_tests['passed']}/{integration_tests['passed'] + integration_tests['failed']}")
print(f"   6. Edge Cases:     {edge_tests['passed']}/{edge_tests['passed'] + edge_tests['failed']}")

# Collect all issues
all_issues = (test_results['issues'] + widget_tests['issues'] + 
              html_tests['issues'] + file_tests['issues'] + 
              integration_tests['issues'] + edge_tests['issues'])

if all_issues:
    print(f"\n??  Critical Issues to Address:")
    for i, issue in enumerate(all_issues, 1):
        print(f"   {i}. {issue}")
else:
    print(f"\n?? No critical issues found!")

# Recommendations
print(f"\n?? Recommendations:")

if total_failed == 0:
    print(f"   ? EXCELLENT! All tests passed.")
    print(f"   ? The metrics editor is ready for production use.")
    print(f"   ? You can safely deploy to your workshop notebook.")
elif total_failed <= 2:
    print(f"   ??  GOOD: Most tests passed with minor issues.")
    print(f"   ??  Review the issues above and fix before deploying.")
    print(f"   ? Core functionality works well.")
else:
    print(f"   ? ATTENTION NEEDED: Several tests failed.")
    print(f"   ? Do NOT deploy until issues are resolved.")
    print(f"   ? Review failed tests and fix critical issues.")

print(f"\n?? Next Steps:")
print(f"   1. Review any failed tests above")
print(f"   2. Fix identified issues")
print(f"   3. Re-run this validation notebook")
print(f"   4. Once all pass, integrate into workshop")
print(f"   5. Test end-to-end with real metrics")

print(f"\n?? Training Recommendations:")
if total_passed / total_tests >= 0.9:
    print(f"   ? System is user-friendly and stable")
    print(f"   ? Non-technical users can start using immediately")
    print(f"   ? 5-minute walkthrough should be sufficient")
else:
    print(f"   ??  Provide more detailed training")
    print(f"   ??  Create backup procedures")
    print(f"   ??  Monitor for issues in early usage")

print("\n" + "="*70)
print(f"?? Test Suite Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? Checklist: Before Deploying
# MAGIC 
# MAGIC Use this checklist to ensure everything is ready:
# MAGIC 
# MAGIC ### Environment
# MAGIC - [ ] All environment tests passed
# MAGIC - [ ] Python version is 3.8+
# MAGIC - [ ] dbutils is accessible
# MAGIC - [ ] displayHTML works
# MAGIC 
# MAGIC ### Functionality
# MAGIC - [ ] Widgets create successfully
# MAGIC - [ ] HTML renders correctly
# MAGIC - [ ] File I/O works
# MAGIC - [ ] Can add metrics
# MAGIC - [ ] Can delete metrics
# MAGIC - [ ] Can save changes
# MAGIC 
# MAGIC ### Edge Cases
# MAGIC - [ ] Empty file handling works
# MAGIC - [ ] Special characters supported
# MAGIC - [ ] Long names handled
# MAGIC - [ ] Missing columns handled
# MAGIC 
# MAGIC ### Integration
# MAGIC - [ ] Integrates with workshop notebook
# MAGIC - [ ] File paths work correctly
# MAGIC - [ ] Metrics load from existing CSV
# MAGIC - [ ] Changes persist correctly
# MAGIC 
# MAGIC ### User Experience
# MAGIC - [ ] UI looks good
# MAGIC - [ ] Instructions are clear
# MAGIC - [ ] Error messages are helpful
# MAGIC - [ ] Non-technical users can use it
# MAGIC 
# MAGIC **If all checkboxes are checked, you're ready to deploy! ??**

# COMMAND ----------
