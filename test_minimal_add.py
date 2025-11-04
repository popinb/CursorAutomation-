"""
Test MINIMAL add functionality - super simple
"""
import pandas as pd
import json

print("="*100)
print("TEST: MINIMAL ADD FUNCTIONALITY")
print("="*100)

# Mock dbutils
class MockWidgets:
    def __init__(self):
        self.storage = {}
    
    def dropdown(self, name, default, options, label):
        if name not in self.storage:
            self.storage[name] = default
        print(f"    Widget created: {label} = '{default}'")
    
    def text(self, name, default, label):
        if name not in self.storage:
            self.storage[name] = default
        print(f"    Widget created: {label} = '{default}'")
    
    def get(self, name):
        return self.storage.get(name, "")
    
    def remove(self, name):
        if name in self.storage:
            del self.storage[name]

class MockDBUtils:
    def __init__(self):
        self.widgets = MockWidgets()

dbutils = MockDBUtils()

# Initial data
METRICS_CONFIG_JSON = [
    {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "M2", "type": "1-5_scale", "description": "Test2", "grading_rubric": "R2", "threshold": "4", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
]

METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)

print(f"\n? Initial: {len(METRICS_CONFIG_DATA)} metrics - {list(METRICS_CONFIG_DATA['name'])}")

# SCENARIO 1: User runs Cell 2.5 in VIEW mode
print("\n" + "="*80)
print("RUN 1: User runs Cell 2.5 in VIEW mode")
print("="*80)

dbutils.widgets.dropdown("action", "view", ["view", "add"], "Action")
action = dbutils.widgets.get("action")

if action == "view":
    print("  ? VIEW mode, removing form widgets")
    for w in ["m_name", "m_desc", "m_rubric", "m_threshold", "save_btn"]:
        try:
            dbutils.widgets.remove(w)
        except:
            pass

print(f"\n?? Table shows: {list(METRICS_CONFIG_DATA['name'])}")

# SCENARIO 2: User changes to ADD mode and re-runs
print("\n" + "="*80)
print("RUN 2: User changes Action to 'add' and re-runs Cell 2.5")
print("="*80)

dbutils.widgets.storage["action"] = "add"
action = dbutils.widgets.get("action")

if action == "add":
    print("  ? ADD mode, creating form widgets:")
    dbutils.widgets.text("m_name", "", "1. Name")
    dbutils.widgets.text("m_desc", "", "2. Description")
    dbutils.widgets.text("m_rubric", "", "3. Grading Rubric")
    dbutils.widgets.text("m_threshold", "", "4. Threshold")
    dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "5. Save?")

print(f"\n?? Table shows: {list(METRICS_CONFIG_DATA['name'])}")
print("?? User sees form, Save button is 'no'")

# SCENARIO 3: User fills form and clicks Save=yes
print("\n" + "="*80)
print("RUN 3: User fills form and sets Save='yes', then re-runs")
print("="*80)

# User fills form
dbutils.widgets.storage["m_name"] = "NewMetric"
dbutils.widgets.storage["m_desc"] = "New description"
dbutils.widgets.storage["m_rubric"] = "New rubric"
dbutils.widgets.storage["m_threshold"] = "1"
dbutils.widgets.storage["save_btn"] = "yes"

action = dbutils.widgets.get("action")

# Process add (same widgets already exist)
if action == "add":
    save_btn = dbutils.widgets.get("save_btn")
    m_name = dbutils.widgets.get("m_name").strip()
    
    if save_btn == "yes" and m_name:
        print(f"  ? Processing: Add '{m_name}'")
        
        # Get current metrics
        current_metrics = METRICS_CONFIG_DATA.to_dict('records')
        print(f"     Current metrics list: {[m['name'] for m in current_metrics]}")
        
        # Add new metric
        new_metric = {
            "name": m_name,
            "type": "binary",
            "description": dbutils.widgets.get("m_desc"),
            "grading_rubric": dbutils.widgets.get("m_rubric"),
            "threshold": dbutils.widgets.get("m_threshold"),
            "ground_truth_file_path": "ground_truth.csv",
            "ground_truth_column": "correct_answer"
        }
        
        current_metrics.append(new_metric)
        print(f"     After append: {[m['name'] for m in current_metrics]}")
        
        # Update DataFrame
        METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
        print(f"     DataFrame updated: {list(METRICS_CONFIG_DATA['name'])}")
        
        # Clear form
        dbutils.widgets.remove("m_name")
        dbutils.widgets.text("m_name", "", "1. Name")
        dbutils.widgets.remove("save_btn")
        dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "5. Save?")
        
        print(f"  ? ADDED: {m_name}")

print(f"\n?? Table shows: {list(METRICS_CONFIG_DATA['name'])}")

# SCENARIO 4: User switches to VIEW
print("\n" + "="*80)
print("RUN 4: User changes Action to 'view' and re-runs")
print("="*80)

dbutils.widgets.storage["action"] = "view"
action = dbutils.widgets.get("action")

if action == "view":
    print("  ? VIEW mode")
    for w in ["m_name", "m_desc", "m_rubric", "m_threshold", "save_btn"]:
        try:
            dbutils.widgets.remove(w)
        except:
            pass

print(f"\n?? Table shows: {list(METRICS_CONFIG_DATA['name'])}")

# VERDICT
print("\n" + "="*80)
print("VERDICT")
print("="*80)

if len(METRICS_CONFIG_DATA) == 3 and "NewMetric" in list(METRICS_CONFIG_DATA['name']):
    print("??? MINIMAL VERSION WORKS! ???")
    print(f"? Final: {list(METRICS_CONFIG_DATA['name'])}")
    print("? Metric persists when switching to VIEW")
else:
    print("? FAILED")
    print(f"   Expected: ['M1', 'M2', 'NewMetric']")
    print(f"   Got: {list(METRICS_CONFIG_DATA['name'])}")

print("="*80)
