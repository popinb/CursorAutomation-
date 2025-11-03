#!/usr/bin/env python3
"""
Standalone Test Script for Metrics Editor
Tests the metrics file editor functionality without Databricks dependencies
"""

import pandas as pd
import os
import json
from pathlib import Path

# Test data paths
METRICS_FILE = "test_metrics_config.csv"
EVAL_DATA_FILE = "test_evaluation_data.csv"
GROUND_TRUTH_FILE = "ground_truth_accuracy.csv"

print("=" * 80)
print("?? METRICS EDITOR TEST SUITE")
print("=" * 80)

# ============================================================
# TEST 1: Load Existing Metrics File
# ============================================================

print("\n" + "=" * 80)
print("TEST 1: Load Existing Metrics File")
print("=" * 80)

def test_load_metrics():
    """Test loading the metrics CSV file."""
    try:
        if os.path.exists(METRICS_FILE):
            df = pd.read_csv(METRICS_FILE)
            print(f"? Successfully loaded metrics file")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {', '.join(df.columns.tolist())}")
            
            print(f"\n?? Current Metrics:")
            print(df.to_string(index=False))
            
            return df
        else:
            print(f"? File not found: {METRICS_FILE}")
            return None
    except Exception as e:
        print(f"? Error loading file: {e}")
        return None

metrics_df = test_load_metrics()

# ============================================================
# TEST 2: Validate Metrics Structure
# ============================================================

print("\n" + "=" * 80)
print("TEST 2: Validate Metrics Structure")
print("=" * 80)

def test_validate_structure(df):
    """Validate that metrics have required columns."""
    required_columns = ['name', 'type', 'description', 'grading_rubric', 'threshold']
    
    if df is None:
        print("? No dataframe to validate")
        return False
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"? Missing required columns: {', '.join(missing_cols)}")
        return False
    else:
        print(f"? All required columns present")
        
        # Check for empty values
        for col in required_columns:
            empty_count = df[col].isna().sum() + (df[col] == '').sum()
            if empty_count > 0:
                print(f"   ??  Column '{col}' has {empty_count} empty values")
            else:
                print(f"   ? Column '{col}' is complete")
        
        return True

structure_valid = test_validate_structure(metrics_df)

# ============================================================
# TEST 3: Add New Metric
# ============================================================

print("\n" + "=" * 80)
print("TEST 3: Add New Metric")
print("=" * 80)

def test_add_metric(df, metric_name="TestMetric"):
    """Test adding a new metric."""
    try:
        # Check if metric already exists
        if metric_name in df['name'].values:
            print(f"??  Metric '{metric_name}' already exists, skipping add test")
            return df
        
        new_row = {
            'name': metric_name,
            'type': '1-5_scale',
            'description': 'Test metric for validation',
            'grading_rubric': 'Rate from 1 to 5 based on test criteria',
            'threshold': '4',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"? Successfully added metric: '{metric_name}'")
        print(f"   Total metrics: {len(df)}")
        
        return df
    except Exception as e:
        print(f"? Error adding metric: {e}")
        return df

if metrics_df is not None:
    metrics_df = test_add_metric(metrics_df, "TestMetric_Temp")

# ============================================================
# TEST 4: Edit Existing Metric
# ============================================================

print("\n" + "=" * 80)
print("TEST 4: Edit Existing Metric")
print("=" * 80)

def test_edit_metric(df, metric_name, new_threshold):
    """Test editing an existing metric."""
    try:
        if metric_name not in df['name'].values:
            print(f"? Metric '{metric_name}' not found")
            return df
        
        idx = df[df['name'] == metric_name].index[0]
        old_threshold = df.at[idx, 'threshold']
        
        df.at[idx, 'threshold'] = new_threshold
        
        print(f"? Successfully edited metric: '{metric_name}'")
        print(f"   Threshold changed: {old_threshold} ? {new_threshold}")
        
        return df
    except Exception as e:
        print(f"? Error editing metric: {e}")
        return df

if metrics_df is not None and 'Accuracy' in metrics_df['name'].values:
    # Edit and then restore
    original_threshold = metrics_df[metrics_df['name'] == 'Accuracy']['threshold'].values[0]
    metrics_df = test_edit_metric(metrics_df, 'Accuracy', '0.9')
    metrics_df = test_edit_metric(metrics_df, 'Accuracy', original_threshold)  # Restore

# ============================================================
# TEST 5: Delete Metric
# ============================================================

print("\n" + "=" * 80)
print("TEST 5: Delete Metric")
print("=" * 80)

def test_delete_metric(df, metric_name):
    """Test deleting a metric."""
    try:
        if metric_name not in df['name'].values:
            print(f"??  Metric '{metric_name}' not found, skipping delete test")
            return df
        
        original_count = len(df)
        df = df[df['name'] != metric_name]
        new_count = len(df)
        
        print(f"? Successfully deleted metric: '{metric_name}'")
        print(f"   Metrics count: {original_count} ? {new_count}")
        
        return df
    except Exception as e:
        print(f"? Error deleting metric: {e}")
        return df

if metrics_df is not None:
    metrics_df = test_delete_metric(metrics_df, 'TestMetric_Temp')

# ============================================================
# TEST 6: Save Metrics File
# ============================================================

print("\n" + "=" * 80)
print("TEST 6: Save Metrics File")
print("=" * 80)

def test_save_metrics(df, filepath):
    """Test saving metrics to CSV."""
    try:
        # Create backup first
        backup_path = filepath + ".backup"
        if os.path.exists(filepath):
            pd.read_csv(filepath).to_csv(backup_path, index=False)
            print(f"?? Created backup: {backup_path}")
        
        df.to_csv(filepath, index=False)
        print(f"? Successfully saved metrics to: {filepath}")
        print(f"   Saved {len(df)} metrics")
        
        # Verify by reloading
        verify_df = pd.read_csv(filepath)
        if len(verify_df) == len(df):
            print(f"? Verification passed: File reloaded successfully")
        else:
            print(f"??  Verification warning: Row count mismatch")
        
        return True
    except Exception as e:
        print(f"? Error saving file: {e}")
        return False

if metrics_df is not None:
    test_save_metrics(metrics_df, METRICS_FILE)

# ============================================================
# TEST 7: Load Evaluation Data
# ============================================================

print("\n" + "=" * 80)
print("TEST 7: Load Evaluation Data")
print("=" * 80)

def test_load_eval_data(filepath):
    """Test loading evaluation data."""
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"? Successfully loaded evaluation data")
            print(f"   Samples: {len(df)}")
            print(f"   Columns: {', '.join(df.columns.tolist())}")
            
            # Check for required columns
            required = ['sample_id', 'prompt', 'response']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                print(f"   ??  Missing columns: {', '.join(missing)}")
            else:
                print(f"   ? All required columns present")
            
            print(f"\n?? Sample Data (first 2 rows):")
            print(df.head(2).to_string(index=False))
            
            return df
        else:
            print(f"? File not found: {filepath}")
            return None
    except Exception as e:
        print(f"? Error loading file: {e}")
        return None

eval_df = test_load_eval_data(EVAL_DATA_FILE)

# ============================================================
# TEST 8: Load Ground Truth Data
# ============================================================

print("\n" + "=" * 80)
print("TEST 8: Load Ground Truth Data")
print("=" * 80)

def test_load_ground_truth(filepath):
    """Test loading ground truth data."""
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"? Successfully loaded ground truth data")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {', '.join(df.columns.tolist())}")
            
            print(f"\n?? Ground Truth Sample (first 2 rows):")
            print(df.head(2).to_string(index=False))
            
            return df
        else:
            print(f"? File not found: {filepath}")
            return None
    except Exception as e:
        print(f"? Error loading file: {e}")
        return None

gt_df = test_load_ground_truth(GROUND_TRUTH_FILE)

# ============================================================
# TEST 9: Validate Data Alignment
# ============================================================

print("\n" + "=" * 80)
print("TEST 9: Validate Data Alignment")
print("=" * 80)

def test_data_alignment(eval_df, gt_df):
    """Test that evaluation data and ground truth are aligned."""
    try:
        if eval_df is None or gt_df is None:
            print("??  Cannot validate alignment: missing data")
            return False
        
        eval_count = len(eval_df)
        gt_count = len(gt_df)
        
        print(f"?? Data counts:")
        print(f"   Evaluation samples: {eval_count}")
        print(f"   Ground truth rows: {gt_count}")
        
        if eval_count == gt_count:
            print(f"? Data alignment: Perfect match")
            return True
        elif gt_count >= eval_count:
            print(f"? Data alignment: Ground truth covers all samples")
            return True
        else:
            print(f"??  Data alignment: Ground truth has fewer rows than evaluation data")
            print(f"   Some samples won't have ground truth reference")
            return False
    except Exception as e:
        print(f"? Error checking alignment: {e}")
        return False

test_data_alignment(eval_df, gt_df)

# ============================================================
# TEST 10: Metric Type Validation
# ============================================================

print("\n" + "=" * 80)
print("TEST 10: Metric Type Validation")
print("=" * 80)

def test_metric_types(df):
    """Test that metric types are valid."""
    try:
        if df is None or 'type' not in df.columns:
            print("??  Cannot validate metric types")
            return False
        
        valid_types = ['binary', '1-5_scale', 'percentage']
        invalid_count = 0
        
        print(f"?? Metric Type Distribution:")
        type_counts = df['type'].value_counts()
        
        for metric_type, count in type_counts.items():
            if metric_type in valid_types:
                print(f"   ? {metric_type}: {count} metric(s)")
            else:
                print(f"   ? {metric_type}: {count} metric(s) (INVALID)")
                invalid_count += 1
        
        if invalid_count == 0:
            print(f"\n? All metric types are valid")
            return True
        else:
            print(f"\n? Found {invalid_count} invalid metric type(s)")
            return False
    except Exception as e:
        print(f"? Error validating types: {e}")
        return False

test_metric_types(metrics_df)

# ============================================================
# TEST 11: Threshold Validation
# ============================================================

print("\n" + "=" * 80)
print("TEST 11: Threshold Validation")
print("=" * 80)

def test_thresholds(df):
    """Test that thresholds are valid for their metric types."""
    try:
        if df is None:
            print("??  Cannot validate thresholds")
            return False
        
        issues = []
        
        for idx, row in df.iterrows():
            name = row['name']
            metric_type = row['type']
            threshold = row['threshold']
            
            try:
                threshold_val = float(threshold)
                
                if metric_type == 'binary':
                    if threshold_val < 0 or threshold_val > 1:
                        issues.append(f"   ??  {name}: binary threshold {threshold_val} should be 0-1")
                elif metric_type == '1-5_scale':
                    if threshold_val < 1 or threshold_val > 5:
                        issues.append(f"   ??  {name}: 1-5 scale threshold {threshold_val} should be 1-5")
                elif metric_type == 'percentage':
                    if threshold_val < 0 or threshold_val > 100:
                        issues.append(f"   ??  {name}: percentage threshold {threshold_val} should be 0-100")
                
            except (ValueError, TypeError):
                issues.append(f"   ? {name}: threshold '{threshold}' is not a valid number")
        
        if issues:
            print(f"??  Found {len(issues)} threshold issue(s):")
            for issue in issues:
                print(issue)
            return False
        else:
            print(f"? All thresholds are valid for their metric types")
            return True
    except Exception as e:
        print(f"? Error validating thresholds: {e}")
        return False

test_thresholds(metrics_df)

# ============================================================
# TEST SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("?? TEST SUMMARY")
print("=" * 80)

test_results = {
    "Metrics File Loading": metrics_df is not None,
    "Structure Validation": structure_valid,
    "Evaluation Data Loading": eval_df is not None,
    "Ground Truth Loading": gt_df is not None,
}

passed = sum(1 for v in test_results.values() if v)
total = len(test_results)

print(f"\n? Passed: {passed}/{total} tests")
print(f"\n?? Test Results:")
for test_name, result in test_results.items():
    status = "?" if result else "?"
    print(f"   {status} {test_name}")

if passed == total:
    print(f"\n?? All tests passed! System is ready to use.")
else:
    print(f"\n??  Some tests failed. Please review the issues above.")

print("\n" + "=" * 80)
print("? TEST SUITE COMPLETE")
print("=" * 80)

# ============================================================
# FILE VERIFICATION
# ============================================================

print("\n" + "=" * 80)
print("?? FILE VERIFICATION")
print("=" * 80)

files_to_check = [
    METRICS_FILE,
    EVAL_DATA_FILE,
    GROUND_TRUTH_FILE
]

print("\n?? Checking required files:")
for filepath in files_to_check:
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"   ? {filepath} ({size} bytes)")
    else:
        print(f"   ? {filepath} (NOT FOUND)")

print("\n" + "=" * 80)
