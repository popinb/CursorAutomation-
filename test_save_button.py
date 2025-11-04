"""
Test Save/Confirm button workflow
"""
import json
import pandas as pd

print("="*100)
print("TEST: SAVE/CONFIRM BUTTON WORKFLOW")
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

# TEST 1: ADD with Save Button
print("\n" + "="*100)
print("TEST 1: ADD WITH SAVE BUTTON")
print("="*100)

dbutils.widgets.dropdown("action", "add", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.text("m_name", "", "1. Name")
dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "2. Type")
dbutils.widgets.text("m_desc", "", "3. Description")
dbutils.widgets.text("m_rubric", "", "4. Grading Rubric")
dbutils.widgets.text("m_threshold", "", "5. Threshold")
dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "6. ? Save Changes?")

# User fills form but doesn't save yet
dbutils.widgets.storage["m_name"] = "NewMetric"
dbutils.widgets.storage["m_type"] = "binary"
dbutils.widgets.storage["m_desc"] = "New desc"
dbutils.widgets.storage["m_rubric"] = "New rubric"
dbutils.widgets.storage["m_threshold"] = "1"
dbutils.widgets.storage["save_action"] = "no"

# Process (should NOT add yet)
action = dbutils.widgets.get("action")
save_action = dbutils.widgets.get("save_action")

stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)

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
    dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
    print("? ADDED")
else:
    print("??  Form filled but not saved yet (save_action=no)")

assert len(metrics_list) == 2, f"Should still have 2 metrics, got {len(metrics_list)}"

# Now user clicks Save
print("\n?? User clicks Save button...")
dbutils.widgets.storage["save_action"] = "yes"
save_action = dbutils.widgets.get("save_action")

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
    # Reset save button
    dbutils.widgets.storage["save_action"] = "no"
    print("? ADDED: NewMetric")

assert len(metrics_list) == 3, f"Should have 3 metrics after save, got {len(metrics_list)}"

# TEST 2: EDIT with Save Button
print("\n" + "="*100)
print("TEST 2: EDIT WITH SAVE BUTTON")
print("="*100)

dbutils.widgets.storage.clear()
dbutils.widgets.dropdown("action", "edit", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
dbutils.widgets.dropdown("row_select", "1", ["1", "2", "3"], "1. Row to Edit")
dbutils.widgets.text("m_name", "", "2. Name")
dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "3. Type")
dbutils.widgets.text("m_desc", "", "4. Description")
dbutils.widgets.text("m_rubric", "", "5. Grading Rubric")
dbutils.widgets.text("m_threshold", "", "6. Threshold")
dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "7. ? Save Changes?")

# User edits but doesn't save
dbutils.widgets.storage["row_select"] = "1"
dbutils.widgets.storage["m_name"] = "M1_EDITED"
dbutils.widgets.storage["m_type"] = "binary"
dbutils.widgets.storage["m_desc"] = "Edited"
dbutils.widgets.storage["m_rubric"] = "Edited"
dbutils.widgets.storage["m_threshold"] = "1"
dbutils.widgets.storage["save_action"] = "no"

action = dbutils.widgets.get("action")
save_action = dbutils.widgets.get("save_action")

stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)

if action == "edit" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
    row_idx = int(dbutils.widgets.get("row_select")) - 1
    if 0 <= row_idx < len(metrics_list):
        metrics_list[row_idx]["name"] = dbutils.widgets.get("m_name")
        print("? EDITED")
else:
    print("??  Form filled but not saved yet (save_action=no)")

assert metrics_list[0]['name'] == "M1", f"M1 should not be edited yet, got {metrics_list[0]['name']}"

# User clicks Save
print("\n?? User clicks Save button...")
dbutils.widgets.storage["save_action"] = "yes"
save_action = dbutils.widgets.get("save_action")

if action == "edit" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
    row_idx = int(dbutils.widgets.get("row_select")) - 1
    if 0 <= row_idx < len(metrics_list):
        metrics_list[row_idx] = {
            "name": dbutils.widgets.get("m_name"),
            "type": dbutils.widgets.get("m_type"),
            "description": dbutils.widgets.get("m_desc"),
            "grading_rubric": dbutils.widgets.get("m_rubric"),
            "threshold": dbutils.widgets.get("m_threshold"),
            "ground_truth_file_path": "ground_truth.csv",
            "ground_truth_column": "correct_answer"
        }
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
        dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
        dbutils.widgets.storage["save_action"] = "no"
        print("? EDITED: M1 ? M1_EDITED")

assert metrics_list[0]['name'] == "M1_EDITED", f"M1 should be edited now, got {metrics_list[0]['name']}"

# TEST 3: DELETE with Save Button
print("\n" + "="*100)
print("TEST 3: DELETE WITH SAVE BUTTON")
print("="*100)

dbutils.widgets.storage.clear()
dbutils.widgets.dropdown("action", "delete", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
dbutils.widgets.dropdown("row_select", "3", ["1", "2", "3"], "1. Row to Delete")
dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "2. ?? Confirm Delete?")

dbutils.widgets.storage["row_select"] = "3"
dbutils.widgets.storage["save_action"] = "no"

action = dbutils.widgets.get("action")
save_action = dbutils.widgets.get("save_action")

stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)

if action == "delete" and save_action == "yes":
    row_idx = int(dbutils.widgets.get("row_select")) - 1
    if 0 <= row_idx < len(metrics_list):
        metrics_list.pop(row_idx)
        print("? DELETED")
else:
    print("??  Delete selected but not confirmed yet (save_action=no)")

assert len(metrics_list) == 3, f"Should still have 3 metrics, got {len(metrics_list)}"

# User confirms delete
print("\n?? User confirms delete...")
dbutils.widgets.storage["save_action"] = "yes"
save_action = dbutils.widgets.get("save_action")

if action == "delete" and save_action == "yes":
    row_idx = int(dbutils.widgets.get("row_select")) - 1
    if 0 <= row_idx < len(metrics_list):
        deleted_name = metrics_list[row_idx]['name']
        metrics_list.pop(row_idx)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
        dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
        dbutils.widgets.storage["save_action"] = "no"
        print(f"? DELETED: {deleted_name}")

assert len(metrics_list) == 2, f"Should have 2 metrics after delete, got {len(metrics_list)}"

# FINAL VERDICT
print("\n" + "="*100)
print("FINAL VERDICT")
print("="*100)
print("? TEST 1: ADD with Save button (only saves when user clicks Yes)")
print("? TEST 2: EDIT with Save button (only saves when user clicks Yes)")
print("? TEST 3: DELETE with Confirm button (only deletes when user confirms)")
print()
print("?? ALL TESTS PASSED! ??")
print("="*100)
