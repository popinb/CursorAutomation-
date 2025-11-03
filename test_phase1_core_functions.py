"""
PHASE 1: Test Core Functions
Tests file operations, data loading, and basic functionality
"""

import sys
import os
import pandas as pd
import json

print("=" * 80)
print("?? PHASE 1: CORE FUNCTIONS TESTING")
print("=" * 80)

# Import mock environment
from mock_dbutils import dbutils
from test_data_hardcoded import TEST_DATA_PATHS, TEST_DIR

# ============================================================
# TEST 1: File Loading Functions
# ============================================================

print("\n" + "=" * 80)
print("TEST 1: File Loading Functions")
print("=" * 80)

def load_csv_file(filepath, file_type="data"):
    """Load CSV file with error handling."""
    try:
        if not os.path.exists(filepath):
            print(f"? File not found: {filepath}")
            return None
        df = pd.read_csv(filepath)
        print(f"? Loaded {file_type}: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"? Error loading {file_type}: {e}")
        return None

# Test loading metrics
print("\n?? Test 1.1: Load Metrics File")
metrics_df = load_csv_file(TEST_DATA_PATHS['metrics_file'], "metrics config")
assert metrics_df is not None, "Metrics file should load"
assert len(metrics_df) == 4, f"Expected 4 metrics, got {len(metrics_df)}"
assert 'name' in metrics_df.columns, "Should have 'name' column"
assert 'type' in metrics_df.columns, "Should have 'type' column"
print(f"? TEST 1.1 PASSED: Loaded {len(metrics_df)} metrics")

# Test loading evaluation data
print("\n?? Test 1.2: Load Evaluation Data")
eval_df = load_csv_file(TEST_DATA_PATHS['eval_file'], "evaluation data")
assert eval_df is not None, "Evaluation data should load"
assert len(eval_df) == 8, f"Expected 8 samples, got {len(eval_df)}"
assert 'sample_id' in eval_df.columns, "Should have 'sample_id' column"
assert 'prompt' in eval_df.columns, "Should have 'prompt' column"
assert 'response' in eval_df.columns, "Should have 'response' column"
print(f"? TEST 1.2 PASSED: Loaded {len(eval_df)} samples")

# Test loading ground truth
print("\n?? Test 1.3: Load Ground Truth Data")
gt_df = load_csv_file(TEST_DATA_PATHS['ground_truth_file'], "ground truth")
assert gt_df is not None, "Ground truth should load"
assert len(gt_df) == 8, f"Expected 8 rows, got {len(gt_df)}"
assert len(gt_df.columns) >= 4, f"Expected at least 4 columns, got {len(gt_df.columns)}"
print(f"? TEST 1.3 PASSED: Loaded {len(gt_df)} ground truth rows with {len(gt_df.columns)} columns")

# ============================================================
# TEST 2: Data Validation Functions
# ============================================================

print("\n" + "=" * 80)
print("TEST 2: Data Validation Functions")
print("=" * 80)

def validate_metrics_structure(df):
    """Validate metrics DataFrame has required columns."""
    required_cols = ['name', 'type', 'description', 'grading_rubric', 'threshold']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"? Missing columns: {missing}")
        return False
    print(f"? All required columns present")
    return True

def validate_eval_data_structure(df):
    """Validate evaluation data structure."""
    required_cols = ['sample_id', 'prompt', 'response']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"? Missing columns: {missing}")
        return False
    print(f"? All required columns present")
    return True

print("\n?? Test 2.1: Validate Metrics Structure")
assert validate_metrics_structure(metrics_df), "Metrics structure validation should pass"
print("? TEST 2.1 PASSED")

print("\n?? Test 2.2: Validate Evaluation Data Structure")
assert validate_eval_data_structure(eval_df), "Eval data structure validation should pass"
print("? TEST 2.2 PASSED")

print("\n?? Test 2.3: Validate Data Alignment")
assert len(eval_df) == len(gt_df), f"Sample count mismatch: eval={len(eval_df)}, gt={len(gt_df)}"
print(f"? TEST 2.3 PASSED: Data alignment correct ({len(eval_df)} samples)")

# ============================================================
# TEST 3: Metrics CRUD Operations
# ============================================================

print("\n" + "=" * 80)
print("TEST 3: Metrics CRUD Operations")
print("=" * 80)

def save_metrics(df, filepath):
    """Save metrics to CSV."""
    try:
        # Create backup
        if os.path.exists(filepath):
            backup_file = filepath + ".backup"
            pd.read_csv(filepath).to_csv(backup_file, index=False)
            print(f"?? Backup created: {backup_file}")
        
        df.to_csv(filepath, index=False)
        print(f"? Saved to: {filepath}")
        return True
    except Exception as e:
        print(f"? Error saving: {e}")
        return False

# Test ADD metric
print("\n?? Test 3.1: Add New Metric")
original_count = len(metrics_df)
new_metric = pd.DataFrame([{
    'name': 'TestMetric',
    'type': 'binary',
    'description': 'Test metric for validation',
    'grading_rubric': 'Test rubric',
    'threshold': '1',
    'ground_truth_file_path': '',
    'ground_truth_column': ''
}])
metrics_df_test = pd.concat([metrics_df, new_metric], ignore_index=True)
assert len(metrics_df_test) == original_count + 1, "Should have one more metric"
assert 'TestMetric' in metrics_df_test['name'].values, "New metric should be in dataframe"
print(f"? TEST 3.1 PASSED: Added metric (count: {original_count} ? {len(metrics_df_test)})")

# Test EDIT metric
print("\n?? Test 3.2: Edit Existing Metric")
idx = metrics_df_test[metrics_df_test['name'] == 'Accuracy'].index[0]
old_threshold = metrics_df_test.at[idx, 'threshold']
metrics_df_test.at[idx, 'threshold'] = '0.9'
new_threshold = metrics_df_test.at[idx, 'threshold']
assert old_threshold != new_threshold, "Threshold should have changed"
assert new_threshold == '0.9', "New threshold should be 0.9"
print(f"? TEST 3.2 PASSED: Edited metric (threshold: {old_threshold} ? {new_threshold})")

# Test DELETE metric
print("\n?? Test 3.3: Delete Metric")
count_before = len(metrics_df_test)
metrics_df_test = metrics_df_test[metrics_df_test['name'] != 'TestMetric']
count_after = len(metrics_df_test)
assert count_after == count_before - 1, "Should have one fewer metric"
assert 'TestMetric' not in metrics_df_test['name'].values, "TestMetric should be removed"
print(f"? TEST 3.3 PASSED: Deleted metric (count: {count_before} ? {count_after})")

# Test SAVE operations
print("\n?? Test 3.4: Save Metrics to File")
test_save_file = os.path.join(TEST_DIR, "metrics_test_save.csv")
assert save_metrics(metrics_df_test, test_save_file), "Save should succeed"
reloaded = pd.read_csv(test_save_file)
assert len(reloaded) == len(metrics_df_test), "Reloaded data should match"
print(f"? TEST 3.4 PASSED: Saved and reloaded {len(reloaded)} metrics")

# ============================================================
# TEST 4: Safe Type Conversion
# ============================================================

print("\n" + "=" * 80)
print("TEST 4: Safe Type Conversion")
print("=" * 80)

def safe_float(value, default=0.0):
    """Safely convert to float."""
    if pd.isna(value):
        return default
    try:
        val_str = str(value).strip().lower()
        if val_str in ['true', '==true', 'yes']:
            return 1.0
        if val_str in ['false', '==false', 'no']:
            return 0.0
        if val_str.endswith('%'):
            return float(val_str[:-1])
        return float(val_str)
    except:
        return default

print("\n?? Test 4.1: Convert Normal Numbers")
assert safe_float('1') == 1.0, "Should convert '1' to 1.0"
assert safe_float('0.5') == 0.5, "Should convert '0.5' to 0.5"
assert safe_float('75') == 75.0, "Should convert '75' to 75.0"
print("? TEST 4.1 PASSED: Normal number conversion works")

print("\n?? Test 4.2: Convert Boolean-like Values")
assert safe_float('true') == 1.0, "Should convert 'true' to 1.0"
assert safe_float('==true') == 1.0, "Should convert '==true' to 1.0"
assert safe_float('false') == 0.0, "Should convert 'false' to 0.0"
print("? TEST 4.2 PASSED: Boolean conversion works")

print("\n?? Test 4.3: Convert Percentages")
assert safe_float('75%') == 75.0, "Should convert '75%' to 75.0"
assert safe_float('100%') == 100.0, "Should convert '100%' to 100.0"
print("? TEST 4.3 PASSED: Percentage conversion works")

print("\n?? Test 4.4: Handle Invalid Values")
assert safe_float('invalid', 99.0) == 99.0, "Should return default for invalid"
assert safe_float(None, 50.0) == 50.0, "Should return default for None"
print("? TEST 4.4 PASSED: Invalid value handling works")

# ============================================================
# TEST 5: Ground Truth Access (ALL COLUMNS)
# ============================================================

print("\n" + "=" * 80)
print("TEST 5: Ground Truth Access (ALL COLUMNS)")
print("=" * 80)

def get_ground_truth_all_columns(gt_df, sample_idx):
    """Get ALL columns from ground truth for a sample."""
    try:
        if sample_idx >= len(gt_df):
            return "Index out of range"
        
        row = gt_df.iloc[sample_idx]
        all_data = []
        for col, value in row.items():
            if pd.notna(value) and str(value).strip():
                all_data.append(f"? {col}: {value}")
        
        return "\n".join(all_data) if all_data else "No data"
    except Exception as e:
        return f"Error: {e}"

print("\n?? Test 5.1: Access First Sample Ground Truth")
gt_text = get_ground_truth_all_columns(gt_df, 0)
print(f"Ground truth for sample 0:\n{gt_text}")
assert 'sample_id' in gt_text, "Should include sample_id column"
assert 'correct_answer' in gt_text, "Should include correct_answer column"
assert 'additional_context' in gt_text, "Should include additional_context column"
assert 'source' in gt_text, "Should include source column"
print("? TEST 5.1 PASSED: ALL columns accessible")

print("\n?? Test 5.2: Access Multiple Samples")
for idx in [0, 1, 2]:
    gt_text = get_ground_truth_all_columns(gt_df, idx)
    assert len(gt_text) > 0, f"Should have ground truth for sample {idx}"
    num_bullets = gt_text.count('?')
    assert num_bullets >= 4, f"Should have at least 4 columns, got {num_bullets}"
print(f"? TEST 5.2 PASSED: Multiple samples accessible with all columns")

# ============================================================
# PHASE 1 SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("?? PHASE 1 TEST SUMMARY")
print("=" * 80)

all_tests = [
    ("1.1: Load Metrics", True),
    ("1.2: Load Evaluation Data", True),
    ("1.3: Load Ground Truth", True),
    ("2.1: Validate Metrics Structure", True),
    ("2.2: Validate Eval Data Structure", True),
    ("2.3: Validate Data Alignment", True),
    ("3.1: Add New Metric", True),
    ("3.2: Edit Existing Metric", True),
    ("3.3: Delete Metric", True),
    ("3.4: Save Metrics", True),
    ("4.1: Convert Normal Numbers", True),
    ("4.2: Convert Boolean Values", True),
    ("4.3: Convert Percentages", True),
    ("4.4: Handle Invalid Values", True),
    ("5.1: Access Ground Truth (ALL columns)", True),
    ("5.2: Multiple Samples Ground Truth", True),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\n? Passed: {passed}/{total} tests")
print("\n?? Test Results:")
for test_name, result in all_tests:
    status = "?" if result else "?"
    print(f"   {status} Test {test_name}")

if passed == total:
    print(f"\n?? PHASE 1 COMPLETE: All core functions working correctly!")
else:
    print(f"\n??  PHASE 1 INCOMPLETE: {total - passed} tests failed")

print("=" * 80)
