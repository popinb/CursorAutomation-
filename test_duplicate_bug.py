"""
Test duplicate bug - when toggling from add to view
"""
import pandas as pd

print("="*100)
print("TEST: DUPLICATE BUG WHEN TOGGLING ADD ? VIEW")
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
        val = self.storage.get(name, "")
        return val
    
    def remove(self, name):
        if name in self.storage:
            del self.storage[name]

class MockDBUtils:
    def __init__(self):
        self.widgets = MockWidgets()

dbutils = MockDBUtils()

METRICS_CONFIG_JSON = [
    {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
]

METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)

print(f"\n? Initial: {list(METRICS_CONFIG_DATA['name'])}")

# RUN 1: Add a metric
print("\n" + "="*80)
print("RUN 1: ADD mode, fill form, save")
print("="*80)

dbutils.widgets.dropdown("action", "add", ["view", "add"], "Action")
dbutils.widgets.text("m_name", "NewMetric", "1. Name")
dbutils.widgets.text("m_desc", "Test", "2. Description")
dbutils.widgets.text("m_rubric", "Test", "3. Grading Rubric")
dbutils.widgets.text("m_threshold", "1", "4. Threshold")
dbutils.widgets.dropdown("save_btn", "yes", ["no", "yes"], "5. Save?")

action = dbutils.widgets.get("action")

# OLD BUGGY CODE:
if action == "view":
    for w in ["m_name", "m_desc", "m_rubric", "m_threshold", "save_btn"]:
        try:
            dbutils.widgets.remove(w)
        except:
            pass
elif action == "add":
    pass  # widgets already created

# Process add
if action == "add":
    save_btn = dbutils.widgets.get("save_btn")
    m_name = dbutils.widgets.get("m_name").strip()
    
    print(f"  action={action}, save_btn={save_btn}, m_name={m_name}")
    
    if save_btn == "yes" and m_name:
        current_metrics = METRICS_CONFIG_DATA.to_dict('records')
        new_metric = {
            "name": m_name, "type": "binary", "description": dbutils.widgets.get("m_desc"),
            "grading_rubric": dbutils.widgets.get("m_rubric"), "threshold": dbutils.widgets.get("m_threshold"),
            "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"
        }
        current_metrics.append(new_metric)
        METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
        print(f"  ? ADDED")

print(f"?? After RUN 1: {list(METRICS_CONFIG_DATA['name'])}")

# RUN 2: Toggle to VIEW (THIS IS WHERE BUG MIGHT HAPPEN!)
print("\n" + "="*80)
print("RUN 2: Toggle to VIEW mode")
print("="*80)

# User changes dropdown
dbutils.widgets.storage["action"] = "view"
action = dbutils.widgets.get("action")

print(f"  Widget values BEFORE removing:")
print(f"    action = {dbutils.widgets.get('action')}")
print(f"    save_btn exists = {'save_btn' in dbutils.widgets.storage}")
if 'save_btn' in dbutils.widgets.storage:
    print(f"    save_btn = {dbutils.widgets.get('save_btn')}")
if 'm_name' in dbutils.widgets.storage:
    print(f"    m_name = {dbutils.widgets.get('m_name')}")

# OLD BUGGY CODE: Remove widgets
if action == "view":
    for w in ["m_name", "m_desc", "m_rubric", "m_threshold", "save_btn"]:
        try:
            dbutils.widgets.remove(w)
        except:
            pass

# Process add (this should NOT run because action == "view")
if action == "add":
    save_btn = dbutils.widgets.get("save_btn")
    m_name = dbutils.widgets.get("m_name").strip()
    
    print(f"  ? BUG! Processing add even though action={action}")
    print(f"     save_btn={save_btn}, m_name={m_name}")
    
    if save_btn == "yes" and m_name:
        current_metrics = METRICS_CONFIG_DATA.to_dict('records')
        new_metric = {
            "name": m_name, "type": "binary", "description": dbutils.widgets.get("m_desc"),
            "grading_rubric": dbutils.widgets.get("m_rubric"), "threshold": dbutils.widgets.get("m_threshold"),
            "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"
        }
        current_metrics.append(new_metric)
        METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
        print(f"  ? DUPLICATE ADDED!")
else:
    print(f"  ? Correctly skipped add processing (action={action})")

print(f"?? After RUN 2: {list(METRICS_CONFIG_DATA['name'])}")

# VERDICT
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if len(METRICS_CONFIG_DATA) == 2 and list(METRICS_CONFIG_DATA['name']) == ['M1', 'NewMetric']:
    print("? NO DUPLICATE - Code is correct!")
elif len(METRICS_CONFIG_DATA) == 3 and list(METRICS_CONFIG_DATA['name']) == ['M1', 'NewMetric', 'NewMetric']:
    print("? DUPLICATE FOUND!")
    print("   This means the add logic ran even in VIEW mode")
else:
    print(f"??  Unexpected result: {list(METRICS_CONFIG_DATA['name'])}")

print("="*80)
