"""
Test the VIEW mode bug - metric disappears after switching from ADD to VIEW
"""
import json
import pandas as pd

print("="*100)
print("TEST: VIEW MODE BUG - Metric Disappears")
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

# Initial data (from Cell 2)
METRICS_CONFIG_JSON = [
    {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "M2", "type": "1-5_scale", "description": "Test2", "grading_rubric": "R2", "threshold": "4", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
]

METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)

# Normalize GT (happens in Cell 2)
for idx in range(len(METRICS_CONFIG_DATA)):
    if pd.isna(METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path']) or METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path'] == '':
        METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path'] = 'ground_truth.csv'
    if pd.isna(METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column']) or METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column'] == '':
        METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column'] = 'correct_answer'

print(f"\n? Initial load (Cell 2): {len(METRICS_CONFIG_DATA)} metrics")

# Initialize storage (Cell 2.5 start)
metrics_list = METRICS_CONFIG_DATA.to_dict('records')
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)

# SCENARIO 1: User runs Cell 2.5 in ADD mode
print("\n" + "="*100)
print("SCENARIO 1: User runs Cell 2.5 in ADD mode")
print("="*100)

dbutils.widgets.dropdown("action", "add", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.text("m_name", "NewMetric", "1. Name")
dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "2. Type")
dbutils.widgets.text("m_desc", "New desc", "3. Description")
dbutils.widgets.text("m_rubric", "New rubric", "4. Grading Rubric")
dbutils.widgets.text("m_threshold", "1", "5. Threshold")
dbutils.widgets.dropdown("save_action", "yes", ["no", "yes"], "6. ? Save Changes?")

action = dbutils.widgets.get("action")
save_action = dbutils.widgets.get("save_action")

# Load from storage (BEFORE PROCESSING)
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)  # CRITICAL FIX!
else:
    metrics_list = METRICS_CONFIG_DATA.to_dict('records')

print(f"Before ADD: METRICS_CONFIG_DATA has {len(METRICS_CONFIG_DATA)} metrics")
print(f"Before ADD: metrics_list has {len(metrics_list)} metrics")

# Process ADD
if action == "add" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
    new_metric = {
        "name": dbutils.widgets.get("m_name"),
        "type": dbutils.widgets.get("m_type"),
        "description": dbutils.widgets.get("m_desc"),
        "grading_rubric": dbutils.widgets.get("m_rubric"),
        "threshold": dbutils.widgets.get("m_threshold"),
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    }
    metrics_list.append(new_metric)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
    dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
    print(f"? ADDED: {new_metric['name']}")

print(f"After ADD: METRICS_CONFIG_DATA has {len(METRICS_CONFIG_DATA)} metrics")
print(f"After ADD: metrics_list has {len(metrics_list)} metrics")
print(f"After ADD: __metrics_storage__ has {len(json.loads(dbutils.widgets.get('__metrics_storage__')))} metrics")

# Display table (what user sees)
print(f"\n?? TABLE SHOWS: {len(METRICS_CONFIG_DATA)} metrics")
for i, row in METRICS_CONFIG_DATA.iterrows():
    print(f"   {i+1}. {row['name']}")

# SCENARIO 2: User switches to VIEW mode and re-runs Cell 2.5
print("\n" + "="*100)
print("SCENARIO 2: User switches to VIEW mode")
print("="*100)

# User changes action to "view"
dbutils.widgets.storage["action"] = "view"
dbutils.widgets.storage["save_action"] = "no"

# Remove form widgets (happens in view mode)
for widget_name in ["row_select", "m_name", "m_type", "m_desc", "m_rubric", "m_threshold", "save_action"]:
    try:
        dbutils.widgets.remove(widget_name)
    except:
        pass

action = dbutils.widgets.get("action")
save_action = "no"

# Load from storage (BEFORE DISPLAY) - THIS IS THE KEY!
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)  # CRITICAL FIX!
else:
    metrics_list = METRICS_CONFIG_DATA.to_dict('records')

print(f"In VIEW mode: METRICS_CONFIG_DATA has {len(METRICS_CONFIG_DATA)} metrics")
print(f"In VIEW mode: metrics_list has {len(metrics_list)} metrics")
print(f"In VIEW mode: __metrics_storage__ has {len(json.loads(dbutils.widgets.get('__metrics_storage__')))} metrics")

# Display table (what user sees)
print(f"\n?? TABLE SHOWS: {len(METRICS_CONFIG_DATA)} metrics")
for i, row in METRICS_CONFIG_DATA.iterrows():
    print(f"   {i+1}. {row['name']}")

# VERDICT
print("\n" + "="*100)
print("VERDICT")
print("="*100)

if len(METRICS_CONFIG_DATA) == 3 and "NewMetric" in METRICS_CONFIG_DATA['name'].values:
    print("??? BUG FIXED! ???")
    print("? Metric persists when switching from ADD to VIEW")
    print(f"? Final count: {len(METRICS_CONFIG_DATA)} metrics (M1, M2, NewMetric)")
else:
    print("? BUG STILL EXISTS")
    print(f"? Expected 3 metrics, got {len(METRICS_CONFIG_DATA)}")
    print(f"? NewMetric in list: {'NewMetric' in METRICS_CONFIG_DATA['name'].values}")

print("="*100)
