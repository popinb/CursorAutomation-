"""
Test EXACT user scenario: Cell 2 once, then Cell 2.5 ADD ? VIEW (no Cell 2 re-run)
"""
import json
import pandas as pd

print("="*100)
print("EXACT USER SCENARIO: Cell 2 ? Cell 2.5 ADD ? Cell 2.5 VIEW")
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

def run_cell_2():
    """Simulate Cell 2"""
    global METRICS_CONFIG_DATA
    
    print("\n" + "="*80)
    print("USER RUNS CELL 2 (FIRST TIME)")
    print("="*80)
    
    METRICS_CONFIG_JSON = [
        {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
         "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
        {"name": "M2", "type": "1-5_scale", "description": "Test2", "grading_rubric": "R2", "threshold": "4", 
         "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
    ]
    
    # Check storage first
    try:
        existing_metrics = dbutils.widgets.get("__metrics_storage__")
        if existing_metrics and existing_metrics.strip():
            metrics_list = json.loads(existing_metrics)
            METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
            print("??  LOADED FROM STORAGE")
            return
    except:
        pass
    
    # No storage, use defaults
    METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
    print("? LOADED DEFAULTS")
    print(f"   METRICS_CONFIG_DATA: {list(METRICS_CONFIG_DATA['name'])}")

def run_cell_25(action, **kwargs):
    """Simulate Cell 2.5 with EXACT logic from v4 file"""
    global METRICS_CONFIG_DATA
    
    print("\n" + "="*80)
    print(f"USER RUNS CELL 2.5: action='{action}'")
    print("="*80)
    
    # [START OF CELL 2.5 - Lines 147-162]
    print("  [1] Cell 2.5 starts...")
    try:
        stored_metrics = dbutils.widgets.get("__metrics_storage__")
        if stored_metrics:
            metrics_list = json.loads(stored_metrics)
            METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
            print(f"      ? Loaded from storage: {list(METRICS_CONFIG_DATA['name'])}")
        else:
            raise ValueError("No storage")
    except:
        # First run - store defaults
        metrics_list = METRICS_CONFIG_DATA.to_dict('records')
        try:
            dbutils.widgets.text("__metrics_storage__", json.dumps(metrics_list), "")
            print(f"      ? Created storage with: {list(METRICS_CONFIG_DATA['name'])}")
        except:
            dbutils.widgets.remove("__metrics_storage__")
            dbutils.widgets.text("__metrics_storage__", json.dumps(metrics_list), "")
            print(f"      ? Created storage (after remove) with: {list(METRICS_CONFIG_DATA['name'])}")
    
    # Set action
    dbutils.widgets.storage["action"] = action
    
    # Create widgets based on action
    if action == "view":
        print("  [2] VIEW mode: removing form widgets")
        for widget_name in ["row_select", "m_name", "m_type", "m_desc", "m_rubric", "m_threshold", "save_action"]:
            try:
                dbutils.widgets.remove(widget_name)
            except:
                pass
    elif action == "add":
        print("  [2] ADD mode: creating form widgets")
        dbutils.widgets.text("m_name", kwargs.get("m_name", ""), "1. Name")
        dbutils.widgets.text("m_desc", kwargs.get("m_desc", ""), "3. Description")
        dbutils.widgets.text("m_rubric", kwargs.get("m_rubric", ""), "4. Grading Rubric")
        dbutils.widgets.text("m_threshold", kwargs.get("m_threshold", ""), "5. Threshold")
        dbutils.widgets.dropdown("save_action", kwargs.get("save_action", "no"), ["no", "yes"], "6. ? Save?")
    
    # [SECOND RELOAD - Lines 228-236]
    print("  [3] Second reload from storage...")
    stored_metrics = dbutils.widgets.get("__metrics_storage__")
    if stored_metrics:
        metrics_list = json.loads(stored_metrics)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
        print(f"      ? Reloaded: {list(METRICS_CONFIG_DATA['name'])}")
    else:
        metrics_list = METRICS_CONFIG_DATA.to_dict('records')
        print(f"      ? No storage, using DataFrame: {list(METRICS_CONFIG_DATA['name'])}")
    
    # Process actions
    save_action = dbutils.widgets.get("save_action") if action in ["add", "edit", "delete"] else "no"
    
    if action == "add" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
        print("  [4] Processing ADD action...")
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
        dbutils.widgets.remove("__metrics_storage__")
        dbutils.widgets.text("__metrics_storage__", json.dumps(metrics_list), "")
        print(f"      ? Added '{new_metric['name']}'")
        print(f"      ? Storage now has: {[m['name'] for m in json.loads(dbutils.widgets.get('__metrics_storage__'))]}")
    else:
        print(f"  [4] No action (save_action='{save_action}')")
    
    # [THIRD RELOAD - Line 317]
    print("  [5] Third reload before display...")
    stored_metrics = dbutils.widgets.get("__metrics_storage__")
    if stored_metrics:
        metrics_list = json.loads(stored_metrics)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
        print(f"      ? Final reload: {list(METRICS_CONFIG_DATA['name'])}")
    else:
        print(f"      ? No storage found! Using DataFrame: {list(METRICS_CONFIG_DATA['name'])}")
    
    # Display
    print(f"\n  ?? USER SEES IN TABLE: {list(METRICS_CONFIG_DATA['name'])}")
    print(f"  ?? Storage contains: {[m['name'] for m in json.loads(dbutils.widgets.get('__metrics_storage__'))]}")
    
    return METRICS_CONFIG_DATA

# EXACT USER SCENARIO
print("\nStarting test...")

# Step 1: User runs Cell 2
METRICS_CONFIG_DATA = None
run_cell_2()

# Step 2: User runs Cell 2.5 in ADD mode
df = run_cell_25("add", m_name="TestMetric", m_desc="Test", m_rubric="Test", m_threshold="1", save_action="yes")

# Step 3: User changes to VIEW and re-runs Cell 2.5 (WITHOUT re-running Cell 2!)
df = run_cell_25("view")

# VERDICT
print("\n" + "="*100)
print("VERDICT")
print("="*100)

if len(df) == 3 and "TestMetric" in list(df['name']):
    print("??? IT WORKS! ???")
    print(f"? Final display shows: {list(df['name'])}")
    print("? TestMetric persists when switching to VIEW")
else:
    print("? STILL BROKEN")
    print(f"? Expected 3 metrics with TestMetric, got: {list(df['name'])}")

print("="*100)
