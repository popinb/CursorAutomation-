"""
Test Cell 2 preservation - metrics should survive Cell 2 re-runs
"""
import json
import pandas as pd

print("="*100)
print("TEST: CELL 2 PRESERVATION - Metrics survive Cell 2 re-runs")
print("="*100)

# Mock dbutils
class MockWidgets:
    def __init__(self):
        self.storage = {}
    
    def dropdown(self, name, default, options, label):
        if name not in self.storage:
            self.storage[name] = default
    
    def text(self, name, default, label):
        if name not in self.storage:
            self.storage[name] = default
    
    def get(self, name):
        return self.storage.get(name, "")
    
    def remove(self, name):
        if name in self.storage:
            del self.storage[name]

class MockDBUtils:
    def __init__(self):
        self.widgets = MockWidgets()

dbutils = MockDBUtils()

# Hardcoded defaults (from Cell 2)
METRICS_CONFIG_JSON = [
    {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "M2", "type": "1-5_scale", "description": "Test2", "grading_rubric": "R2", "threshold": "4", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
]

def run_cell_2():
    """Simulate running Cell 2 with preservation logic"""
    global METRICS_CONFIG_DATA
    
    print("\n" + "="*80)
    print("RUNNING CELL 2")
    print("="*80)
    
    # CRITICAL: Check if metrics already exist in storage
    try:
        existing_metrics = dbutils.widgets.get("__metrics_storage__")
        if existing_metrics and existing_metrics.strip():
            # Load from storage (user has edited metrics)
            metrics_list = json.loads(existing_metrics)
            METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
            print("??  LOADED EXISTING METRICS FROM STORAGE")
            print(f"   Loaded {len(METRICS_CONFIG_DATA)} metrics: {list(METRICS_CONFIG_DATA['name'])}")
            return
    except:
        pass
    
    # No storage exists, use defaults
    METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
    
    # Normalize GT
    for idx in range(len(METRICS_CONFIG_DATA)):
        if pd.isna(METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path']) or METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path'] == '':
            METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path'] = 'ground_truth.csv'
        if pd.isna(METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column']) or METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column'] == '':
            METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column'] = 'correct_answer'
    
    print("? LOADED DEFAULT METRICS (first run)")
    print(f"   Loaded {len(METRICS_CONFIG_DATA)} metrics: {list(METRICS_CONFIG_DATA['name'])}")

def run_cell_25_add(metric_name):
    """Simulate adding a metric in Cell 2.5"""
    global METRICS_CONFIG_DATA
    
    print("\n" + "="*80)
    print(f"RUNNING CELL 2.5: ADD '{metric_name}'")
    print("="*80)
    
    # Load from storage
    stored_metrics = dbutils.widgets.get("__metrics_storage__")
    if stored_metrics:
        metrics_list = json.loads(stored_metrics)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
    else:
        metrics_list = METRICS_CONFIG_DATA.to_dict('records')
    
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
    
    # Save to storage
    dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
    
    print(f"? ADDED: {metric_name}")
    print(f"   Now have {len(METRICS_CONFIG_DATA)} metrics: {list(METRICS_CONFIG_DATA['name'])}")
    print(f"   Storage contains: {len(json.loads(dbutils.widgets.get('__metrics_storage__')))} metrics")

# TEST SCENARIO
print("\n" + "="*100)
print("SCENARIO: User workflow")
print("="*100)

# 1. First run - Cell 2 (defaults)
METRICS_CONFIG_DATA = None
run_cell_2()
assert len(METRICS_CONFIG_DATA) == 2, "Should start with 2 default metrics"
step1_metrics = list(METRICS_CONFIG_DATA['name'])

# 2. User runs Cell 2.5 and adds a metric
run_cell_25_add("NewMetric")
assert len(METRICS_CONFIG_DATA) == 3, "Should have 3 metrics after adding"
step2_metrics = list(METRICS_CONFIG_DATA['name'])

# 3. User accidentally re-runs Cell 2 (THIS IS THE CRITICAL TEST!)
run_cell_2()
step3_metrics = list(METRICS_CONFIG_DATA['name'])
print(f"\n?? After re-running Cell 2: {step3_metrics}")

# 4. Check storage is still intact
storage_metrics = json.loads(dbutils.widgets.get("__metrics_storage__"))
print(f"?? Storage still has: {[m['name'] for m in storage_metrics]}")

# VERDICT
print("\n" + "="*100)
print("VERDICT")
print("="*100)

print(f"Step 1 (Cell 2 first run):  {step1_metrics}")
print(f"Step 2 (Cell 2.5 add):      {step2_metrics}")
print(f"Step 3 (Cell 2 re-run):     {step3_metrics}")
print(f"Storage contents:           {[m['name'] for m in storage_metrics]}")

if len(METRICS_CONFIG_DATA) == 3 and "NewMetric" in step3_metrics:
    print("\n??? PRESERVATION WORKS! ???")
    print("? Added metric survives Cell 2 re-runs")
    print("? Storage is preserved")
    print("\n?? USER CAN FREELY RE-RUN CELLS WITHOUT LOSING WORK! ??")
else:
    print("\n? FAILED: Metrics were lost when Cell 2 was re-run")
    print(f"? Expected 3 metrics with NewMetric, got {len(METRICS_CONFIG_DATA)}: {step3_metrics}")

print("="*100)
