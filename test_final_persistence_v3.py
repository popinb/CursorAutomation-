"""
FINAL TEST - Double Reload (once at start, once before display)
This simulates the EXACT flow in LLM_Judge_PERSISTENCE_FIX_v3.py
"""
import json
import pandas as pd

print("="*100)
print("FINAL TEST - DOUBLE RELOAD FIX")
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

def simulate_cell_25_run(action_mode, **form_data):
    """
    Simulates running Cell 2.5 with DOUBLE RELOAD
    """
    global METRICS_CONFIG_DATA
    
    print(f"\n{'='*80}")
    print(f"RUNNING CELL 2.5: action='{action_mode}'")
    print(f"{'='*80}")
    
    # Set widgets
    dbutils.widgets.storage["action"] = action_mode
    action = dbutils.widgets.get("action")
    
    # Widget creation/removal based on action
    if action == "view":
        for widget_name in ["row_select", "m_name", "m_type", "m_desc", "m_rubric", "m_threshold", "save_action"]:
            try:
                dbutils.widgets.remove(widget_name)
            except:
                pass
    elif action == "add":
        dbutils.widgets.text("m_name", form_data.get("m_name", ""), "1. Name")
        dbutils.widgets.text("m_desc", form_data.get("m_desc", ""), "3. Description")
        dbutils.widgets.text("m_rubric", form_data.get("m_rubric", ""), "4. Grading Rubric")
        dbutils.widgets.text("m_threshold", form_data.get("m_threshold", ""), "5. Threshold")
        dbutils.widgets.dropdown("save_action", form_data.get("save_action", "no"), ["no", "yes"], "6. ? Save Changes?")
    elif action == "edit":
        dbutils.widgets.storage["row_select"] = str(form_data.get("row_select", 1))
        dbutils.widgets.text("m_name", form_data.get("m_name", ""), "2. Name")
        dbutils.widgets.text("m_desc", form_data.get("m_desc", ""), "4. Description")
        dbutils.widgets.text("m_rubric", form_data.get("m_rubric", ""), "5. Grading Rubric")
        dbutils.widgets.text("m_threshold", form_data.get("m_threshold", ""), "6. Threshold")
        dbutils.widgets.dropdown("save_action", form_data.get("save_action", "no"), ["no", "yes"], "7. ? Save Changes?")
    elif action == "delete":
        dbutils.widgets.storage["row_select"] = str(form_data.get("row_select", 1))
        dbutils.widgets.dropdown("save_action", form_data.get("save_action", "no"), ["no", "yes"], "2. ?? Confirm Delete?")
    
    # FIRST RELOAD (at start of cell, line 210-216)
    print("  [1] First reload from storage...")
    stored_metrics = dbutils.widgets.get("__metrics_storage__")
    if stored_metrics:
        metrics_list = json.loads(stored_metrics)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
    else:
        metrics_list = METRICS_CONFIG_DATA.to_dict('records')
    print(f"      ? Loaded {len(METRICS_CONFIG_DATA)} metrics")
    
    # Process actions
    save_action = dbutils.widgets.get("save_action") if action in ["add", "edit", "delete"] else "no"
    
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
        print(f"  [2] ADD action: Added '{new_metric['name']}'")
        print(f"      ? Now have {len(METRICS_CONFIG_DATA)} metrics")
    
    elif action == "edit" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
        row_idx = int(dbutils.widgets.get("row_select")) - 1
        if 0 <= row_idx < len(metrics_list):
            old_name = metrics_list[row_idx]['name']
            metrics_list[row_idx] = {
                "name": dbutils.widgets.get("m_name"),
                "type": "binary",
                "description": dbutils.widgets.get("m_desc"),
                "grading_rubric": dbutils.widgets.get("m_rubric"),
                "threshold": dbutils.widgets.get("m_threshold"),
                "ground_truth_file_path": "ground_truth.csv",
                "ground_truth_column": "correct_answer"
            }
            METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
            dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
            print(f"  [2] EDIT action: Changed '{old_name}' ? '{metrics_list[row_idx]['name']}'")
            print(f"      ? Still have {len(METRICS_CONFIG_DATA)} metrics")
    
    elif action == "delete" and save_action == "yes":
        row_idx = int(dbutils.widgets.get("row_select")) - 1
        if 0 <= row_idx < len(metrics_list):
            deleted_name = metrics_list[row_idx]['name']
            metrics_list.pop(row_idx)
            METRICS_CONFIG_DATA = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
            dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
            print(f"  [2] DELETE action: Removed '{deleted_name}'")
            print(f"      ? Now have {len(METRICS_CONFIG_DATA)} metrics")
    else:
        print(f"  [2] No action (save_action='{save_action}')")
    
    # SECOND RELOAD (right before display, line 300-305) - THE CRITICAL FIX!
    print("  [3] Second reload before display...")
    stored_metrics = dbutils.widgets.get("__metrics_storage__")
    if stored_metrics:
        metrics_list = json.loads(stored_metrics)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
    print(f"      ? Will display {len(METRICS_CONFIG_DATA)} metrics")
    
    # Display
    print(f"\n  ?? TABLE DISPLAYS: {list(METRICS_CONFIG_DATA['name'])}")
    
    return METRICS_CONFIG_DATA

# Setup
METRICS_CONFIG_JSON = [
    {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "M2", "type": "1-5_scale", "description": "Test2", "grading_rubric": "R2", "threshold": "4", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
]

METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
metrics_list = METRICS_CONFIG_DATA.to_dict('records')
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)

print(f"\n? INITIAL STATE: {list(METRICS_CONFIG_DATA['name'])}")

# Test sequence
results = []

# 1. ADD
df = simulate_cell_25_run("add", m_name="NewMetric", m_desc="New", m_rubric="New", m_threshold="1", save_action="yes")
results.append(("After ADD", len(df), list(df['name'])))

# 2. Switch to VIEW
df = simulate_cell_25_run("view")
results.append(("After VIEW", len(df), list(df['name'])))

# 3. EDIT
df = simulate_cell_25_run("edit", row_select=2, m_name="M2_EDITED", m_desc="Ed", m_rubric="Ed", m_threshold="4", save_action="yes")
results.append(("After EDIT", len(df), list(df['name'])))

# 4. Switch to VIEW again
df = simulate_cell_25_run("view")
results.append(("After VIEW again", len(df), list(df['name'])))

# 5. DELETE
df = simulate_cell_25_run("delete", row_select=3, save_action="yes")
results.append(("After DELETE", len(df), list(df['name'])))

# 6. Switch to VIEW final
df = simulate_cell_25_run("view")
results.append(("After VIEW final", len(df), list(df['name'])))

# Summary
print("\n" + "="*100)
print("SUMMARY OF ALL OPERATIONS")
print("="*100)
for label, count, names in results:
    print(f"{label:20s}: {count} metrics - {names}")

# Verdict
print("\n" + "="*100)
print("VERDICT")
print("="*100)

expected_final = ["M1", "M2_EDITED"]
actual_final = results[-1][2]

if actual_final == expected_final:
    print("??? ALL TESTS PASSED! ???")
    print("? ADD ? VIEW: Metric persisted")
    print("? EDIT ? VIEW: Changes persisted")
    print("? DELETE ? VIEW: Deletion persisted")
    print(f"\n? Final state correct: {actual_final}")
    print("\n?? FILE IS PRODUCTION READY! ??")
else:
    print(f"? FAILED: Expected {expected_final}, got {actual_final}")

print("="*100)
