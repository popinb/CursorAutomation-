"""
Test ALL modes (ADD/EDIT/DELETE) with VIEW switching
"""
import json
import pandas as pd

print("="*100)
print("TEST: ALL MODES PERSISTENCE WHEN SWITCHING TO VIEW")
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

def run_cell_25(action_mode, save_yes=False, **kwargs):
    """Simulate running Cell 2.5"""
    global METRICS_CONFIG_DATA
    
    # Set action
    dbutils.widgets.storage["action"] = action_mode
    action = dbutils.widgets.get("action")
    
    # Create widgets based on action
    if action == "view":
        for widget_name in ["row_select", "m_name", "m_type", "m_desc", "m_rubric", "m_threshold", "save_action"]:
            try:
                dbutils.widgets.remove(widget_name)
            except:
                pass
    elif action == "add":
        dbutils.widgets.text("m_name", kwargs.get("m_name", ""), "1. Name")
        dbutils.widgets.dropdown("m_type", kwargs.get("m_type", "binary"), ["binary", "1-5_scale", "percentage"], "2. Type")
        dbutils.widgets.text("m_desc", kwargs.get("m_desc", ""), "3. Description")
        dbutils.widgets.text("m_rubric", kwargs.get("m_rubric", ""), "4. Grading Rubric")
        dbutils.widgets.text("m_threshold", kwargs.get("m_threshold", ""), "5. Threshold")
        dbutils.widgets.dropdown("save_action", "yes" if save_yes else "no", ["no", "yes"], "6. ? Save Changes?")
    elif action == "edit":
        dbutils.widgets.dropdown("row_select", str(kwargs.get("row_select", 1)), ["1", "2", "3", "4"], "1. Row to Edit")
        dbutils.widgets.text("m_name", kwargs.get("m_name", ""), "2. Name")
        dbutils.widgets.dropdown("m_type", kwargs.get("m_type", "binary"), ["binary", "1-5_scale", "percentage"], "3. Type")
        dbutils.widgets.text("m_desc", kwargs.get("m_desc", ""), "4. Description")
        dbutils.widgets.text("m_rubric", kwargs.get("m_rubric", ""), "5. Grading Rubric")
        dbutils.widgets.text("m_threshold", kwargs.get("m_threshold", ""), "6. Threshold")
        dbutils.widgets.dropdown("save_action", "yes" if save_yes else "no", ["no", "yes"], "7. ? Save Changes?")
    elif action == "delete":
        dbutils.widgets.dropdown("row_select", str(kwargs.get("row_select", 1)), ["1", "2", "3", "4"], "1. Row to Delete")
        dbutils.widgets.dropdown("save_action", "yes" if save_yes else "no", ["no", "yes"], "2. ?? Confirm Delete?")
    
    # Load metrics_list from storage (CRITICAL!)
    stored_metrics = dbutils.widgets.get("__metrics_storage__")
    if stored_metrics:
        metrics_list = json.loads(stored_metrics)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)  # UPDATE DataFrame too!
    else:
        metrics_list = METRICS_CONFIG_DATA.to_dict('records')
    
    # Process actions
    save_action = dbutils.widgets.get("save_action") if action in ["add", "edit", "delete"] else "no"
    
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
        return f"ADDED: {new_metric['name']}"
    
    elif action == "edit" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
        row_idx = int(dbutils.widgets.get("row_select")) - 1
        if 0 <= row_idx < len(metrics_list):
            old_name = metrics_list[row_idx]['name']
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
            return f"EDITED: {old_name} ? {metrics_list[row_idx]['name']}"
    
    elif action == "delete" and save_action == "yes":
        row_idx = int(dbutils.widgets.get("row_select")) - 1
        if 0 <= row_idx < len(metrics_list):
            deleted_name = metrics_list[row_idx]['name']
            metrics_list.pop(row_idx)
            METRICS_CONFIG_DATA = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame(columns=["name", "type", "description", "grading_rubric", "threshold", "ground_truth_file_path", "ground_truth_column"])
            dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
            return f"DELETED: {deleted_name}"
    
    return f"Showing {len(METRICS_CONFIG_DATA)} metrics"

# Initial setup
METRICS_CONFIG_JSON = [
    {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "M2", "type": "1-5_scale", "description": "Test2", "grading_rubric": "R2", "threshold": "4", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
]

METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
metrics_list = METRICS_CONFIG_DATA.to_dict('records')
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)

print(f"\n? Initial: {len(METRICS_CONFIG_DATA)} metrics (M1, M2)")

# TEST 1: ADD ? VIEW
print("\n" + "="*100)
print("TEST 1: ADD ? VIEW")
print("="*100)

result = run_cell_25("add", save_yes=True, m_name="NewMetric", m_type="binary", 
                     m_desc="New", m_rubric="New", m_threshold="1")
print(f"After ADD: {result}")
print(f"  METRICS_CONFIG_DATA: {len(METRICS_CONFIG_DATA)} metrics - {list(METRICS_CONFIG_DATA['name'])}")

result = run_cell_25("view")
print(f"After VIEW: {result}")
print(f"  METRICS_CONFIG_DATA: {len(METRICS_CONFIG_DATA)} metrics - {list(METRICS_CONFIG_DATA['name'])}")

assert len(METRICS_CONFIG_DATA) == 3, f"? Expected 3, got {len(METRICS_CONFIG_DATA)}"
assert "NewMetric" in list(METRICS_CONFIG_DATA['name']), "? NewMetric missing"
print("? ADD ? VIEW: PASS")

# TEST 2: EDIT ? VIEW
print("\n" + "="*100)
print("TEST 2: EDIT ? VIEW")
print("="*100)

result = run_cell_25("edit", save_yes=True, row_select=2, m_name="M2_EDITED", 
                     m_type="1-5_scale", m_desc="Edited", m_rubric="Edited", m_threshold="4")
print(f"After EDIT: {result}")
print(f"  METRICS_CONFIG_DATA: {len(METRICS_CONFIG_DATA)} metrics - {list(METRICS_CONFIG_DATA['name'])}")

result = run_cell_25("view")
print(f"After VIEW: {result}")
print(f"  METRICS_CONFIG_DATA: {len(METRICS_CONFIG_DATA)} metrics - {list(METRICS_CONFIG_DATA['name'])}")

assert len(METRICS_CONFIG_DATA) == 3, f"? Expected 3, got {len(METRICS_CONFIG_DATA)}"
assert "M2_EDITED" in list(METRICS_CONFIG_DATA['name']), "? M2_EDITED missing"
assert "M2" not in list(METRICS_CONFIG_DATA['name']), "? Old M2 still present"
print("? EDIT ? VIEW: PASS")

# TEST 3: DELETE ? VIEW
print("\n" + "="*100)
print("TEST 3: DELETE ? VIEW")
print("="*100)

result = run_cell_25("delete", save_yes=True, row_select=3)
print(f"After DELETE: {result}")
print(f"  METRICS_CONFIG_DATA: {len(METRICS_CONFIG_DATA)} metrics - {list(METRICS_CONFIG_DATA['name'])}")

result = run_cell_25("view")
print(f"After VIEW: {result}")
print(f"  METRICS_CONFIG_DATA: {len(METRICS_CONFIG_DATA)} metrics - {list(METRICS_CONFIG_DATA['name'])}")

assert len(METRICS_CONFIG_DATA) == 2, f"? Expected 2, got {len(METRICS_CONFIG_DATA)}"
assert "NewMetric" not in list(METRICS_CONFIG_DATA['name']), "? NewMetric still present"
print("? DELETE ? VIEW: PASS")

# FINAL VERDICT
print("\n" + "="*100)
print("FINAL VERDICT")
print("="*100)

if len(METRICS_CONFIG_DATA) == 2 and list(METRICS_CONFIG_DATA['name']) == ["M1", "M2_EDITED"]:
    print("??? ALL TESTS PASSED! ???")
    print("? ADD ? VIEW: Metric persists")
    print("? EDIT ? VIEW: Changes persist")
    print("? DELETE ? VIEW: Deletion persists")
    print(f"\nFinal state: {list(METRICS_CONFIG_DATA['name'])}")
else:
    print("? TESTS FAILED")
    print(f"? Expected ['M1', 'M2_EDITED'], got {list(METRICS_CONFIG_DATA['name'])}")

print("="*100)
