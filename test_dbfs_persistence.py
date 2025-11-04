"""
Test TRUE PERSISTENCE with DBFS file storage
Simulates complete notebook restart
"""
import json
import pandas as pd
import os
import tempfile

print("="*100)
print("TEST: TRUE PERSISTENCE WITH DBFS FILE STORAGE")
print("="*100)

# Use temp file to simulate DBFS
METRICS_STORAGE_PATH = tempfile.mktemp(suffix=".json")

print(f"Using storage file: {METRICS_STORAGE_PATH}")

# Default metrics
METRICS_CONFIG_JSON = [
    {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "M2", "type": "1-5_scale", "description": "Test2", "grading_rubric": "R2", "threshold": "4", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
]

def run_cell_2():
    """Simulate Cell 2 with DBFS persistence"""
    global METRICS_CONFIG_DATA
    
    print("\n" + "="*80)
    print("RUNNING CELL 2 (with DBFS check)")
    print("="*80)
    
    # Check DBFS file first
    if os.path.exists(METRICS_STORAGE_PATH):
        with open(METRICS_STORAGE_PATH, 'r') as f:
            metrics_list = json.load(f)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
        print(f"??  LOADED FROM DBFS FILE: {len(METRICS_CONFIG_DATA)} metrics")
        print(f"   Metrics: {list(METRICS_CONFIG_DATA['name'])}")
        return
    
    # No file, use defaults and save
    METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
    
    # Normalize GT
    for idx in range(len(METRICS_CONFIG_DATA)):
        if pd.isna(METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path']) or METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path'] == '':
            METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path'] = 'ground_truth.csv'
        if pd.isna(METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column']) or METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column'] == '':
            METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column'] = 'correct_answer'
    
    # Save to file
    os.makedirs(os.path.dirname(METRICS_STORAGE_PATH), exist_ok=True)
    with open(METRICS_STORAGE_PATH, 'w') as f:
        json.dump(METRICS_CONFIG_DATA.to_dict('records'), f, indent=2)
    
    print(f"? LOADED DEFAULTS: {len(METRICS_CONFIG_DATA)} metrics")
    print(f"   Metrics: {list(METRICS_CONFIG_DATA['name'])}")
    print(f"   Saved to: {METRICS_STORAGE_PATH}")

def run_cell_25_add(metric_name):
    """Simulate adding a metric with DBFS save"""
    global METRICS_CONFIG_DATA
    
    print("\n" + "="*80)
    print(f"RUNNING CELL 2.5: ADD '{metric_name}'")
    print("="*80)
    
    # Load from file
    with open(METRICS_STORAGE_PATH, 'r') as f:
        metrics_list = json.load(f)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
    print(f"   Loaded from file: {list(METRICS_CONFIG_DATA['name'])}")
    
    # Add new metric
    new_metric = {
        "name": metric_name,
        "type": "binary",
        "description": "Test",
        "grading_rubric": "Test",
        "threshold": "1",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    }
    metrics_list.append(new_metric)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
    
    # SAVE TO FILE
    with open(METRICS_STORAGE_PATH, 'w') as f:
        json.dump(metrics_list, f, indent=2)
    
    print(f"   ? ADDED: {metric_name}")
    print(f"   ?? Saved to file")
    print(f"   Now have: {list(METRICS_CONFIG_DATA['name'])}")

# SCENARIO: Full notebook lifecycle
print("\n" + "="*100)
print("SCENARIO: Complete notebook lifecycle")
print("="*100)

# Session 1: First run
print("\n?? SESSION 1: Fresh notebook start")
METRICS_CONFIG_DATA = None
run_cell_2()
step1_metrics = list(METRICS_CONFIG_DATA['name'])

# User adds a metric
run_cell_25_add("NewMetric")
step2_metrics = list(METRICS_CONFIG_DATA['name'])

print(f"\n?? End of Session 1:")
print(f"   Memory: {list(METRICS_CONFIG_DATA['name'])}")
print(f"   File: {[m['name'] for m in json.load(open(METRICS_STORAGE_PATH))]}")

# Simulate notebook restart (clear memory!)
print("\n" + "="*100)
print("?? SIMULATING NOTEBOOK RESTART (all variables cleared!)")
print("="*100)
METRICS_CONFIG_DATA = None  # All memory lost!

# Session 2: After restart
print("\n?? SESSION 2: After notebook restart")
run_cell_2()  # Should load from file!
step3_metrics = list(METRICS_CONFIG_DATA['name'])

print(f"\n?? After restart:")
print(f"   Memory: {list(METRICS_CONFIG_DATA['name'])}")
print(f"   File: {[m['name'] for m in json.load(open(METRICS_STORAGE_PATH))]}")

# Add another metric
run_cell_25_add("AnotherMetric")
step4_metrics = list(METRICS_CONFIG_DATA['name'])

# Simulate another restart
print("\n" + "="*100)
print("?? SIMULATING ANOTHER RESTART")
print("="*100)
METRICS_CONFIG_DATA = None

# Session 3: After second restart
print("\n?? SESSION 3: After second restart")
run_cell_2()
step5_metrics = list(METRICS_CONFIG_DATA['name'])

# VERDICT
print("\n" + "="*100)
print("FINAL VERDICT")
print("="*100)

print("Session 1 - Initial:         ", step1_metrics)
print("Session 1 - After ADD:       ", step2_metrics)
print("Session 2 - After RESTART:   ", step3_metrics)
print("Session 2 - After ADD:       ", step4_metrics)
print("Session 3 - After RESTART:   ", step5_metrics)

if step5_metrics == ["M1", "M2", "NewMetric", "AnotherMetric"]:
    print("\n??? TRUE PERSISTENCE WORKS! ???")
    print("? Metrics survive notebook restarts")
    print("? Data persists in DBFS file")
    print(f"\n?? Storage file: {METRICS_STORAGE_PATH}")
    print(f"   Size: {os.path.getsize(METRICS_STORAGE_PATH)} bytes")
    print("\n?? USER CAN RESTART NOTEBOOK FREELY! ??")
else:
    print(f"\n? FAILED")
    print(f"   Expected: ['M1', 'M2', 'NewMetric', 'AnotherMetric']")
    print(f"   Got: {step5_metrics}")

# Cleanup
os.remove(METRICS_STORAGE_PATH)
print(f"\n?? Cleaned up test file")
print("="*100)
