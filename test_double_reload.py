"""
Test with DOUBLE reload - once at start, once before display
"""
import json
import pandas as pd

print("="*100)
print("TEST: DOUBLE RELOAD - Once at start, once before display")
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

# Initial data
METRICS_CONFIG_JSON = [
    {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "M2", "type": "1-5_scale", "description": "Test2", "grading_rubric": "R2", "threshold": "4", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
]

METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
metrics_list = METRICS_CONFIG_DATA.to_dict('records')
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)

print(f"? Initial: {len(METRICS_CONFIG_DATA)} metrics")

# SCENARIO 1: User adds a metric
print("\n" + "="*100)
print("SCENARIO 1: User runs Cell 2.5 in ADD mode")
print("="*100)

dbutils.widgets.dropdown("action", "add", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.text("m_name", "NewMetric", "1. Name")
dbutils.widgets.text("m_desc", "New", "3. Description")
dbutils.widgets.text("m_rubric", "New", "4. Grading Rubric")
dbutils.widgets.text("m_threshold", "1", "5. Threshold")
dbutils.widgets.dropdown("save_action", "yes", ["no", "yes"], "6. ? Save Changes?")

action = dbutils.widgets.get("action")

# FIRST RELOAD (at start of cell)
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)

print(f"After FIRST reload: {len(METRICS_CONFIG_DATA)} metrics")

# Process ADD
save_action = dbutils.widgets.get("save_action")
if action == "add" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
    new_metric = {
        "name": dbutils.widgets.get("m_name"),
        "type": "binary",
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

print(f"After ADD action: {len(METRICS_CONFIG_DATA)} metrics")

# SECOND RELOAD (right before display)
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)

print(f"After SECOND reload (before display): {len(METRICS_CONFIG_DATA)} metrics")
print(f"?? TABLE SHOWS: {list(METRICS_CONFIG_DATA['name'])}")

# SCENARIO 2: User switches to VIEW
print("\n" + "="*100)
print("SCENARIO 2: User switches to VIEW mode")
print("="*100)

dbutils.widgets.storage["action"] = "view"
action = dbutils.widgets.get("action")

# Remove widgets
for widget_name in ["row_select", "m_name", "m_type", "m_desc", "m_rubric", "m_threshold", "save_action"]:
    try:
        dbutils.widgets.remove(widget_name)
    except:
        pass

# FIRST RELOAD (at start of cell)
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)

print(f"After FIRST reload: {len(METRICS_CONFIG_DATA)} metrics")

# No action processing in VIEW mode
save_action = "no"

# SECOND RELOAD (right before display)
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)

print(f"After SECOND reload (before display): {len(METRICS_CONFIG_DATA)} metrics")
print(f"?? TABLE SHOWS: {list(METRICS_CONFIG_DATA['name'])}")

# VERDICT
print("\n" + "="*100)
print("VERDICT")
print("="*100)

if len(METRICS_CONFIG_DATA) == 3 and "NewMetric" in list(METRICS_CONFIG_DATA['name']):
    print("??? DOUBLE RELOAD WORKS! ???")
    print("? Metric persists when switching to VIEW")
    print(f"? Final: {list(METRICS_CONFIG_DATA['name'])}")
else:
    print("? Still broken")
    print(f"? Expected ['M1', 'M2', 'NewMetric'], got {list(METRICS_CONFIG_DATA['name'])}")

print("="*100)
