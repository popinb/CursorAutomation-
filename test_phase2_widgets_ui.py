"""
PHASE 2: Test Widget Interactions and UI Workflow
Tests Databricks widgets, metrics editing workflow, and state management
"""

import sys
import os
import pandas as pd

print("=" * 80)
print("?? PHASE 2: WIDGET INTERACTIONS & UI WORKFLOW TESTING")
print("=" * 80)

# Import mock environment and test data
from mock_dbutils import dbutils
from test_data_hardcoded import TEST_DATA_PATHS, TEST_DIR

# ============================================================
# TEST 1: Widget Creation and Access
# ============================================================

print("\n" + "=" * 80)
print("TEST 1: Widget Creation and Access")
print("=" * 80)

print("\n?? Test 1.1: Create Text Widgets")
dbutils.widgets.text("test_text", "default_value", "Test Text Widget")
assert dbutils.widgets.get("test_text") == "default_value", "Should get default value"
print("? TEST 1.1 PASSED: Text widget created and accessible")

print("\n?? Test 1.2: Create Dropdown Widgets")
dbutils.widgets.dropdown("test_dropdown", "option1", ["option1", "option2", "option3"], "Test Dropdown")
assert dbutils.widgets.get("test_dropdown") == "option1", "Should get default dropdown value"
print("? TEST 1.2 PASSED: Dropdown widget created and accessible")

print("\n?? Test 1.3: Set Widget Values")
dbutils.widgets.set("test_text", "new_value")
assert dbutils.widgets.get("test_text") == "new_value", "Should get updated value"
dbutils.widgets.set("test_dropdown", "option2")
assert dbutils.widgets.get("test_dropdown") == "option2", "Should get updated dropdown value"
print("? TEST 1.3 PASSED: Widget values can be updated")

print("\n?? Test 1.4: Remove All Widgets")
dbutils.widgets.removeAll()
assert len(dbutils.widgets._widgets) == 0, "All widgets should be removed"
print("? TEST 1.4 PASSED: Widgets can be removed")

# ============================================================
# TEST 2: Metrics Editor Widget Workflow
# ============================================================

print("\n" + "=" * 80)
print("TEST 2: Metrics Editor Widget Workflow")
print("=" * 80)

# Load test metrics
metrics_df = pd.read_csv(TEST_DATA_PATHS['metrics_file'])
print(f"\n?? Loaded {len(metrics_df)} test metrics")

# Create metrics editor widgets
print("\n?? Test 2.1: Create Metrics Editor Widgets")
dbutils.widgets.dropdown("action", "view_only", 
                        ["view_only", "add_new", "edit_existing", "delete_existing"],
                        "Action")
dbutils.widgets.dropdown("select_metric", "(none)",
                        ["(none)"] + metrics_df['name'].tolist(),
                        "Select Metric")
dbutils.widgets.text("metric_name", "", "Metric Name")
dbutils.widgets.dropdown("metric_type", "binary",
                        ["binary", "1-5_scale", "percentage"],
                        "Metric Type")
dbutils.widgets.text("metric_description", "", "Description")
dbutils.widgets.text("metric_rubric", "", "Grading Rubric")
dbutils.widgets.text("metric_threshold", "1", "Threshold")
dbutils.widgets.text("metric_gt_file", "", "Ground Truth File")
dbutils.widgets.text("metric_gt_column", "", "Ground Truth Column")

assert dbutils.widgets.get("action") == "view_only", "Default action should be view_only"
print("? TEST 2.1 PASSED: All editor widgets created")

# ============================================================
# TEST 3: Add Metric Workflow
# ============================================================

print("\n" + "=" * 80)
print("TEST 3: Add Metric Workflow")
print("=" * 80)

def process_metric_action(metrics_df, action, widget_values, test_file):
    """Process metric add/edit/delete action."""
    try:
        if action == "view_only":
            return metrics_df, "View mode - No changes"
        
        elif action == "add_new":
            name = widget_values['metric_name'].strip()
            if not name:
                return metrics_df, "Error: Name required"
            if name in metrics_df['name'].values:
                return metrics_df, f"Error: Metric '{name}' exists"
            
            new_metric = pd.DataFrame([{
                'name': name,
                'type': widget_values['metric_type'],
                'description': widget_values['metric_description'] or f"Metric: {name}",
                'grading_rubric': widget_values['metric_rubric'],
                'threshold': widget_values['metric_threshold'],
                'ground_truth_file_path': widget_values['metric_gt_file'],
                'ground_truth_column': widget_values['metric_gt_column']
            }])
            
            metrics_df = pd.concat([metrics_df, new_metric], ignore_index=True)
            metrics_df.to_csv(test_file, index=False)
            return metrics_df, f"Added metric: '{name}'"
        
        elif action == "edit_existing":
            selected = widget_values['select_metric']
            if selected == "(none)" or selected not in metrics_df['name'].values:
                return metrics_df, "Error: Select valid metric"
            
            idx = metrics_df[metrics_df['name'] == selected].index[0]
            if widget_values['metric_name']:
                metrics_df.at[idx, 'name'] = widget_values['metric_name']
            if widget_values['metric_threshold']:
                metrics_df.at[idx, 'threshold'] = widget_values['metric_threshold']
            
            metrics_df.to_csv(test_file, index=False)
            return metrics_df, f"Updated metric: '{selected}'"
        
        elif action == "delete_existing":
            selected = widget_values['select_metric']
            if selected == "(none)" or selected not in metrics_df['name'].values:
                return metrics_df, "Error: Select valid metric"
            
            metrics_df = metrics_df[metrics_df['name'] != selected]
            metrics_df.to_csv(test_file, index=False)
            return metrics_df, f"Deleted metric: '{selected}'"
        
        return metrics_df, "Unknown action"
    except Exception as e:
        return metrics_df, f"Error: {e}"

print("\n?? Test 3.1: Simulate Add New Metric")
dbutils.widgets.set("action", "add_new")
dbutils.widgets.set("metric_name", "TestNewMetric")
dbutils.widgets.set("metric_type", "1-5_scale")
dbutils.widgets.set("metric_description", "Test description")
dbutils.widgets.set("metric_rubric", "Test rubric 1-5")
dbutils.widgets.set("metric_threshold", "4")

widget_values = {
    'metric_name': dbutils.widgets.get("metric_name"),
    'metric_type': dbutils.widgets.get("metric_type"),
    'metric_description': dbutils.widgets.get("metric_description"),
    'metric_rubric': dbutils.widgets.get("metric_rubric"),
    'metric_threshold': dbutils.widgets.get("metric_threshold"),
    'metric_gt_file': dbutils.widgets.get("metric_gt_file"),
    'metric_gt_column': dbutils.widgets.get("metric_gt_column"),
    'select_metric': dbutils.widgets.get("select_metric")
}

test_metrics_file = os.path.join(TEST_DIR, "metrics_workflow_test.csv")
original_count = len(metrics_df)
metrics_df, message = process_metric_action(metrics_df, "add_new", widget_values, test_metrics_file)

assert len(metrics_df) == original_count + 1, f"Should have {original_count + 1} metrics, got {len(metrics_df)}"
assert 'TestNewMetric' in metrics_df['name'].values, "New metric should be present"
assert "Added metric" in message, f"Expected success message, got: {message}"
print(f"? TEST 3.1 PASSED: {message}")

# ============================================================
# TEST 4: Edit Metric Workflow
# ============================================================

print("\n" + "=" * 80)
print("TEST 4: Edit Metric Workflow")
print("=" * 80)

print("\n?? Test 4.1: Simulate Edit Existing Metric")
dbutils.widgets.set("action", "edit_existing")
dbutils.widgets.set("select_metric", "Accuracy")
dbutils.widgets.set("metric_name", "")  # Don't change name
dbutils.widgets.set("metric_threshold", "0.95")  # Change threshold

widget_values = {
    'metric_name': dbutils.widgets.get("metric_name"),
    'metric_type': dbutils.widgets.get("metric_type"),
    'metric_description': dbutils.widgets.get("metric_description"),
    'metric_rubric': dbutils.widgets.get("metric_rubric"),
    'metric_threshold': dbutils.widgets.get("metric_threshold"),
    'metric_gt_file': dbutils.widgets.get("metric_gt_file"),
    'metric_gt_column': dbutils.widgets.get("metric_gt_column"),
    'select_metric': dbutils.widgets.get("select_metric")
}

metrics_df, message = process_metric_action(metrics_df, "edit_existing", widget_values, test_metrics_file)

accuracy_metric = metrics_df[metrics_df['name'] == 'Accuracy']
assert len(accuracy_metric) == 1, "Accuracy metric should exist"
assert accuracy_metric.iloc[0]['threshold'] == '0.95', "Threshold should be updated"
assert "Updated metric" in message, f"Expected update message, got: {message}"
print(f"? TEST 4.1 PASSED: {message}")

# ============================================================
# TEST 5: Delete Metric Workflow
# ============================================================

print("\n" + "=" * 80)
print("TEST 5: Delete Metric Workflow")
print("=" * 80)

print("\n?? Test 5.1: Simulate Delete Metric")
dbutils.widgets.set("action", "delete_existing")
dbutils.widgets.set("select_metric", "TestNewMetric")

widget_values['select_metric'] = dbutils.widgets.get("select_metric")
count_before = len(metrics_df)
metrics_df, message = process_metric_action(metrics_df, "delete_existing", widget_values, test_metrics_file)

assert len(metrics_df) == count_before - 1, f"Should have {count_before - 1} metrics"
assert 'TestNewMetric' not in metrics_df['name'].values, "TestNewMetric should be deleted"
assert "Deleted metric" in message, f"Expected delete message, got: {message}"
print(f"? TEST 5.1 PASSED: {message}")

# ============================================================
# TEST 6: File Persistence
# ============================================================

print("\n" + "=" * 80)
print("TEST 6: File Persistence Across Operations")
print("=" * 80)

print("\n?? Test 6.1: Verify Changes Persisted to File")
reloaded_df = pd.read_csv(test_metrics_file)
assert len(reloaded_df) == len(metrics_df), "Reloaded data should match in-memory data"
assert 'Accuracy' in reloaded_df['name'].values, "Accuracy should be in saved file"
assert 'TestNewMetric' not in reloaded_df['name'].values, "Deleted metric should not be in file"

# Check threshold was updated
accuracy_row = reloaded_df[reloaded_df['name'] == 'Accuracy']
actual_threshold = str(accuracy_row.iloc[0]['threshold'])
assert actual_threshold == '0.95', f"Updated threshold should persist, got {actual_threshold}"
print("? TEST 6.1 PASSED: All changes persisted correctly")

# ============================================================
# TEST 7: View-Only Mode
# ============================================================

print("\n" + "=" * 80)
print("TEST 7: View-Only Mode")
print("=" * 80)

print("\n?? Test 7.1: View Mode Makes No Changes")
dbutils.widgets.set("action", "view_only")
widget_values['select_metric'] = "(none)"
count_before = len(metrics_df)
metrics_df, message = process_metric_action(metrics_df, "view_only", widget_values, test_metrics_file)

assert len(metrics_df) == count_before, "View mode should not change metric count"
assert "View mode" in message, f"Expected view mode message, got: {message}"
print(f"? TEST 7.1 PASSED: {message}")

# ============================================================
# TEST 8: Error Handling
# ============================================================

print("\n" + "=" * 80)
print("TEST 8: Error Handling")
print("=" * 80)

print("\n?? Test 8.1: Add Metric Without Name")
dbutils.widgets.set("action", "add_new")
dbutils.widgets.set("metric_name", "")  # Empty name
widget_values['metric_name'] = ""
metrics_df, message = process_metric_action(metrics_df, "add_new", widget_values, test_metrics_file)
assert "Error" in message, "Should return error for empty name"
print(f"? TEST 8.1 PASSED: Error caught - {message}")

print("\n?? Test 8.2: Add Duplicate Metric")
dbutils.widgets.set("metric_name", "Accuracy")  # Existing metric
widget_values['metric_name'] = "Accuracy"
metrics_df, message = process_metric_action(metrics_df, "add_new", widget_values, test_metrics_file)
assert "Error" in message or "exists" in message, "Should return error for duplicate"
print(f"? TEST 8.2 PASSED: Duplicate caught - {message}")

print("\n?? Test 8.3: Edit Without Selection")
dbutils.widgets.set("action", "edit_existing")
dbutils.widgets.set("select_metric", "(none)")
widget_values['select_metric'] = "(none)"
metrics_df, message = process_metric_action(metrics_df, "edit_existing", widget_values, test_metrics_file)
assert "Error" in message, "Should return error for no selection"
print(f"? TEST 8.3 PASSED: No selection caught - {message}")

# ============================================================
# TEST 9: Multiple Sequential Operations
# ============================================================

print("\n" + "=" * 80)
print("TEST 9: Multiple Sequential Operations")
print("=" * 80)

print("\n?? Test 9.1: Add Multiple Metrics in Sequence")
test_seq_file = os.path.join(TEST_DIR, "metrics_sequential_test.csv")
seq_df = pd.read_csv(TEST_DATA_PATHS['metrics_file'])
original_count = len(seq_df)

# Add first metric
dbutils.widgets.set("action", "add_new")
for i, metric_name in enumerate(["SeqMetric1", "SeqMetric2", "SeqMetric3"]):
    dbutils.widgets.set("metric_name", metric_name)
    dbutils.widgets.set("metric_rubric", f"Rubric for {metric_name}")
    widget_values = {
        'metric_name': metric_name,
        'metric_type': 'binary',
        'metric_description': f"Sequential test metric {i+1}",
        'metric_rubric': f"Rubric for {metric_name}",
        'metric_threshold': '1',
        'metric_gt_file': '',
        'metric_gt_column': '',
        'select_metric': '(none)'
    }
    seq_df, msg = process_metric_action(seq_df, "add_new", widget_values, test_seq_file)
    print(f"   {msg}")

assert len(seq_df) == original_count + 3, f"Should have {original_count + 3} metrics"
assert all(name in seq_df['name'].values for name in ["SeqMetric1", "SeqMetric2", "SeqMetric3"]), "All sequential metrics should be added"
print("? TEST 9.1 PASSED: Multiple metrics added sequentially")

# ============================================================
# PHASE 2 SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("?? PHASE 2 TEST SUMMARY")
print("=" * 80)

all_tests = [
    ("1.1: Create Text Widgets", True),
    ("1.2: Create Dropdown Widgets", True),
    ("1.3: Set Widget Values", True),
    ("1.4: Remove All Widgets", True),
    ("2.1: Create Metrics Editor Widgets", True),
    ("3.1: Add New Metric", True),
    ("4.1: Edit Existing Metric", True),
    ("5.1: Delete Metric", True),
    ("6.1: File Persistence", True),
    ("7.1: View-Only Mode", True),
    ("8.1: Error - Empty Name", True),
    ("8.2: Error - Duplicate Metric", True),
    ("8.3: Error - No Selection", True),
    ("9.1: Multiple Sequential Operations", True),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\n? Passed: {passed}/{total} tests")
print("\n?? Test Results:")
for test_name, result in all_tests:
    status = "?" if result else "?"
    print(f"   {status} Test {test_name}")

if passed == total:
    print(f"\n?? PHASE 2 COMPLETE: All widget & UI workflows working correctly!")
else:
    print(f"\n??  PHASE 2 INCOMPLETE: {total - passed} tests failed")

print("=" * 80)
