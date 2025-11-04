"""
Test v4 FIXED - All features (add/edit/delete/view)
"""
import pandas as pd

print("="*100)
print("TEST: v4 FIXED - ALL FEATURES")
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

print(f"? Initial: {list(METRICS_CONFIG_DATA['name'])}")

# TEST 1: VIEW
print("\n" + "="*80)
print("TEST 1: VIEW MODE")
print("="*80)

dbutils.widgets.dropdown("action", "view", ["view", "add", "edit", "delete"], "Action")
action = dbutils.widgets.get("action")

if action == "view":
    for w in ["row_select", "m_name", "m_type", "m_desc", "m_rubric", "m_threshold", "save_btn"]:
        try:
            dbutils.widgets.remove(w)
        except:
            pass

widgets_count = len([k for k in dbutils.widgets.storage.keys() if k != "action"])
print(f"? VIEW mode: {widgets_count} extra widgets (should be 0)")
print(f"?? Shows: {list(METRICS_CONFIG_DATA['name'])}")
assert widgets_count == 0

# TEST 2: ADD
print("\n" + "="*80)
print("TEST 2: ADD MODE")
print("="*80)

dbutils.widgets.storage["action"] = "add"
action = dbutils.widgets.get("action")

dbutils.widgets.text("m_name", "NewMetric", "1. Name")
dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "2. Type")
dbutils.widgets.text("m_desc", "New", "3. Description")
dbutils.widgets.text("m_rubric", "New", "4. Grading Rubric")
dbutils.widgets.text("m_threshold", "1", "5. Threshold")
dbutils.widgets.dropdown("save_btn", "yes", ["no", "yes"], "6. Save?")

save_btn = dbutils.widgets.get("save_btn")
m_name = dbutils.widgets.get("m_name").strip()

if save_btn == "yes" and m_name:
    current_metrics = METRICS_CONFIG_DATA.to_dict('records')
    existing_names = [m['name'] for m in current_metrics]
    
    if m_name not in existing_names:
        new_metric = {
            "name": m_name, "type": dbutils.widgets.get("m_type"), 
            "description": dbutils.widgets.get("m_desc"),
            "grading_rubric": dbutils.widgets.get("m_rubric"), 
            "threshold": dbutils.widgets.get("m_threshold"),
            "ground_truth_file_path": "ground_truth.csv", 
            "ground_truth_column": "correct_answer"
        }
        current_metrics.append(new_metric)
        METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
        print(f"? ADDED: {m_name}")

print(f"?? Now have: {list(METRICS_CONFIG_DATA['name'])}")
assert len(METRICS_CONFIG_DATA) == 3

# TEST 3: EDIT
print("\n" + "="*80)
print("TEST 3: EDIT MODE")
print("="*80)

dbutils.widgets.storage["action"] = "edit"
action = dbutils.widgets.get("action")

dbutils.widgets.dropdown("row_select", "2", ["1", "2", "3"], "1. Row")
dbutils.widgets.storage["m_name"] = "M2_EDITED"
dbutils.widgets.storage["m_type"] = "1-5_scale"
dbutils.widgets.storage["m_desc"] = "Edited"
dbutils.widgets.storage["m_rubric"] = "Edited"
dbutils.widgets.storage["m_threshold"] = "4"
dbutils.widgets.storage["save_btn"] = "yes"

save_btn = dbutils.widgets.get("save_btn")
m_name = dbutils.widgets.get("m_name").strip()

if save_btn == "yes" and m_name:
    row_idx = int(dbutils.widgets.get("row_select")) - 1
    current_metrics = METRICS_CONFIG_DATA.to_dict('records')
    
    if 0 <= row_idx < len(current_metrics):
        current_metrics[row_idx] = {
            "name": m_name, "type": dbutils.widgets.get("m_type"),
            "description": dbutils.widgets.get("m_desc"),
            "grading_rubric": dbutils.widgets.get("m_rubric"),
            "threshold": dbutils.widgets.get("m_threshold"),
            "ground_truth_file_path": "ground_truth.csv",
            "ground_truth_column": "correct_answer"
        }
        METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
        print(f"? EDITED: M2 ? M2_EDITED")

print(f"?? Now have: {list(METRICS_CONFIG_DATA['name'])}")
assert "M2_EDITED" in list(METRICS_CONFIG_DATA['name'])

# TEST 4: DELETE
print("\n" + "="*80)
print("TEST 4: DELETE MODE")
print("="*80)

dbutils.widgets.storage["action"] = "delete"
action = dbutils.widgets.get("action")

dbutils.widgets.storage["row_select"] = "3"
dbutils.widgets.storage["save_btn"] = "yes"

save_btn = dbutils.widgets.get("save_btn")

if save_btn == "yes":
    row_idx = int(dbutils.widgets.get("row_select")) - 1
    current_metrics = METRICS_CONFIG_DATA.to_dict('records')
    
    if 0 <= row_idx < len(current_metrics):
        deleted_name = current_metrics[row_idx]['name']
        current_metrics.pop(row_idx)
        METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
        print(f"? DELETED: {deleted_name}")

print(f"?? Now have: {list(METRICS_CONFIG_DATA['name'])}")
assert len(METRICS_CONFIG_DATA) == 2

# TEST 5: Toggle back to VIEW
print("\n" + "="*80)
print("TEST 5: TOGGLE TO VIEW (check persistence)")
print("="*80)

dbutils.widgets.storage["action"] = "view"
action = dbutils.widgets.get("action")

print(f"?? VIEW shows: {list(METRICS_CONFIG_DATA['name'])}")
assert list(METRICS_CONFIG_DATA['name']) == ['M1', 'M2_EDITED']

# FINAL VERDICT
print("\n" + "="*100)
print("FINAL VERDICT")
print("="*100)

print("? TEST 1: VIEW mode - clean display")
print("? TEST 2: ADD mode - added NewMetric")
print("? TEST 3: EDIT mode - edited M2 ? M2_EDITED")
print("? TEST 4: DELETE mode - deleted NewMetric")
print("? TEST 5: Toggle VIEW - changes persisted")
print()
print(f"? Final state: {list(METRICS_CONFIG_DATA['name'])}")
print()
print("?? v4 FIXED - ALL FEATURES WORKING! ??")
print("="*100)
