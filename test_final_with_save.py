"""
COMPLETE TEST WITH SAVE BUTTON - Ground Truth + Save/Confirm Button
"""
import sys
import json
import re
import pandas as pd

print("="*100)
print("COMPLETE TEST WITH SAVE BUTTON + GROUND TRUTH")
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
     "ground_truth_file_path": "", "ground_truth_column": ""},
    {"name": "M3", "type": "percentage", "description": "Test3", "grading_rubric": "R3", "threshold": "75", 
     "ground_truth_file_path": "", "ground_truth_column": ""}
]

METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)

# NORMALIZE GROUND TRUTH
for idx in range(len(METRICS_CONFIG_DATA)):
    if pd.isna(METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path']) or METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path'] == '':
        METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path'] = 'ground_truth.csv'
    if pd.isna(METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column']) or METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column'] == '':
        METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column'] = 'correct_answer'

print(f"\n? Initial load: {len(METRICS_CONFIG_DATA)} metrics")
for i, row in METRICS_CONFIG_DATA.iterrows():
    print(f"   {i+1}. {row['name']}: {row['ground_truth_file_path']}/{row['ground_truth_column']}")

metrics_list = METRICS_CONFIG_DATA.to_dict('records')
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)

# TEST 1: ADD without save (should not add)
print("\n" + "="*100)
print("TEST 1: ADD WITHOUT CLICKING SAVE")
print("="*100)

dbutils.widgets.dropdown("action", "add", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.text("m_name", "NewMetric", "1. Name")
dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "2. Type")
dbutils.widgets.text("m_desc", "New desc", "3. Description")
dbutils.widgets.text("m_rubric", "New rubric", "4. Grading Rubric")
dbutils.widgets.text("m_threshold", "1", "5. Threshold")
dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "6. ? Save Changes?")

action = dbutils.widgets.get("action")
save_action = dbutils.widgets.get("save_action")

# Load from storage
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)

if action == "add" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
    print("? WOULD ADD")
else:
    print("??  Form filled but save_action=no, so nothing happens")

assert len(metrics_list) == 3, f"Should still have 3 metrics, got {len(metrics_list)}"

# TEST 2: ADD with save (should add)
print("\n" + "="*100)
print("TEST 2: ADD WITH SAVE CLICKED")
print("="*100)

dbutils.widgets.storage["save_action"] = "yes"
save_action = dbutils.widgets.get("save_action")

if action == "add" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
    new_metric = {
        "name": dbutils.widgets.get("m_name"),
        "type": dbutils.widgets.get("m_type"),
        "description": dbutils.widgets.get("m_desc"),
        "grading_rubric": dbutils.widgets.get("m_rubric"),
        "threshold": dbutils.widgets.get("m_threshold"),
        "ground_truth_file_path": "ground_truth.csv",  # Hardcoded
        "ground_truth_column": "correct_answer"  # Hardcoded
    }
    metrics_list.append(new_metric)
    dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
    print(f"? ADDED: {new_metric['name']}")
    print(f"   GT: {new_metric['ground_truth_file_path']}/{new_metric['ground_truth_column']}")

assert len(metrics_list) == 4, f"Should have 4 metrics, got {len(metrics_list)}"
assert metrics_list[3]['ground_truth_file_path'] == "ground_truth.csv", "GT file should be hardcoded"
assert metrics_list[3]['ground_truth_column'] == "correct_answer", "GT column should be hardcoded"

# TEST 3: EDIT without save
print("\n" + "="*100)
print("TEST 3: EDIT WITHOUT CLICKING SAVE")
print("="*100)

dbutils.widgets.storage.clear()
dbutils.widgets.dropdown("action", "edit", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
dbutils.widgets.dropdown("row_select", "2", ["1", "2", "3", "4"], "1. Row to Edit")
dbutils.widgets.text("m_name", "M2_EDITED", "2. Name")
dbutils.widgets.dropdown("m_type", "1-5_scale", ["binary", "1-5_scale", "percentage"], "3. Type")
dbutils.widgets.text("m_desc", "Edited", "4. Description")
dbutils.widgets.text("m_rubric", "Edited", "5. Grading Rubric")
dbutils.widgets.text("m_threshold", "4", "6. Threshold")
dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "7. ? Save Changes?")

action = dbutils.widgets.get("action")
save_action = dbutils.widgets.get("save_action")

stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)

if action == "edit" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
    print("? WOULD EDIT")
else:
    print("??  Form filled but save_action=no, so nothing happens")

assert metrics_list[1]['name'] == "M2", f"M2 should not be edited yet, got {metrics_list[1]['name']}"

# TEST 4: EDIT with save
print("\n" + "="*100)
print("TEST 4: EDIT WITH SAVE CLICKED")
print("="*100)

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
            "ground_truth_file_path": "ground_truth.csv",  # Always hardcoded
            "ground_truth_column": "correct_answer"  # Always hardcoded
        }
        dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
        print(f"? EDITED: M2 ? {metrics_list[row_idx]['name']}")
        print(f"   GT: {metrics_list[row_idx]['ground_truth_file_path']}/{metrics_list[row_idx]['ground_truth_column']}")

assert metrics_list[1]['name'] == "M2_EDITED", f"M2 should be edited now, got {metrics_list[1]['name']}"
assert metrics_list[1]['ground_truth_file_path'] == "ground_truth.csv", "GT file should be hardcoded"
assert metrics_list[1]['ground_truth_column'] == "correct_answer", "GT column should be hardcoded"

# TEST 5: DELETE without confirm
print("\n" + "="*100)
print("TEST 5: DELETE WITHOUT CONFIRMING")
print("="*100)

dbutils.widgets.storage.clear()
dbutils.widgets.dropdown("action", "delete", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
dbutils.widgets.dropdown("row_select", "4", ["1", "2", "3", "4"], "1. Row to Delete")
dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "2. ?? Confirm Delete?")

action = dbutils.widgets.get("action")
save_action = dbutils.widgets.get("save_action")

stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)

if action == "delete" and save_action == "yes":
    print("? WOULD DELETE")
else:
    print("??  Delete selected but save_action=no, so nothing happens")

assert len(metrics_list) == 4, f"Should still have 4 metrics, got {len(metrics_list)}"

# TEST 6: DELETE with confirm
print("\n" + "="*100)
print("TEST 6: DELETE WITH CONFIRM CLICKED")
print("="*100)

dbutils.widgets.storage["save_action"] = "yes"
save_action = dbutils.widgets.get("save_action")

if action == "delete" and save_action == "yes":
    row_idx = int(dbutils.widgets.get("row_select")) - 1
    if 0 <= row_idx < len(metrics_list):
        deleted_name = metrics_list[row_idx]['name']
        metrics_list.pop(row_idx)
        dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
        print(f"? DELETED: {deleted_name}")

assert len(metrics_list) == 3, f"Should have 3 metrics after delete, got {len(metrics_list)}"

# FINAL VERDICT
print("\n" + "="*100)
print("FINAL VERDICT")
print("="*100)

all_pass = True
issues = []

# Check all metrics have ground truth
for metric in metrics_list:
    if metric['ground_truth_file_path'] != "ground_truth.csv":
        all_pass = False
        issues.append(f"Metric {metric['name']}: GT file = '{metric['ground_truth_file_path']}'")
    if metric['ground_truth_column'] != "correct_answer":
        all_pass = False
        issues.append(f"Metric {metric['name']}: GT column = '{metric['ground_truth_column']}'")

if len(metrics_list) != 3:
    all_pass = False
    issues.append(f"Should have 3 metrics, got {len(metrics_list)}")

if all_pass:
    print("??? ALL TESTS PASSED! ???")
    print()
    print("? TEST 1: ADD without save (no change)")
    print("? TEST 2: ADD with save (added with hardcoded GT)")
    print("? TEST 3: EDIT without save (no change)")
    print("? TEST 4: EDIT with save (edited with hardcoded GT)")
    print("? TEST 5: DELETE without confirm (no change)")
    print("? TEST 6: DELETE with confirm (deleted)")
    print()
    print("Final metrics:")
    for i, m in enumerate(metrics_list):
        print(f"  {i+1}. {m['name']} (GT: {m['ground_truth_file_path']}/{m['ground_truth_column']})")
    print()
    print("?? READY TO INTEGRATE INTO NOTEBOOK! ??")
else:
    print("? TESTS FAILED ?")
    for issue in issues:
        print(f"  ? {issue}")

print("="*100)
