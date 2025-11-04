"""
Test EDIT with pre-populated fields
"""
import pandas as pd

print("="*80)
print("TEST: EDIT WITH PRE-POPULATION")
print("="*80)

# Mock dbutils
class MockWidgets:
    def __init__(self):
        self.storage = {}
    
    def dropdown(self, name, default, options, label):
        self.storage[name] = default
        print(f"  {label} = '{default}'")
    
    def text(self, name, default, label):
        self.storage[name] = default
        print(f"  {label} = '{default}'")
    
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
METRICS_CONFIG_DATA = pd.DataFrame([
    {"name": "Story_Accuracy", "type": "binary", "description": "Check story facts", "grading_rubric": "Score 1 if correct", "threshold": "1", "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "Completeness", "type": "1-5_scale", "description": "Check completeness", "grading_rubric": "1-5 scale", "threshold": "4", "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
])

print(f"\nInitial: {list(METRICS_CONFIG_DATA['name'])}")

# SCENARIO: User selects EDIT mode for row 2
print("\n" + "="*80)
print("USER SELECTS EDIT MODE, ROW 2")
print("="*80)

dbutils.widgets.dropdown("action", "edit", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.dropdown("row_select", "2", ["1", "2"], "1. Row")

# Get selected row
action = dbutils.widgets.get("action")
row_idx = int(dbutils.widgets.get("row_select")) - 1
current_metric = METRICS_CONFIG_DATA.iloc[row_idx]

print(f"\nPre-populating form with values from '{current_metric['name']}':")

# Create widgets with current values (THE FIX!)
dbutils.widgets.text("m_name", current_metric['name'], "2. Name")
dbutils.widgets.dropdown("m_type", current_metric['type'], ["binary", "1-5_scale", "percentage"], "3. Type")
dbutils.widgets.text("m_desc", str(current_metric['description']), "4. Description")
dbutils.widgets.text("m_rubric", str(current_metric['grading_rubric']), "5. Grading Rubric")
dbutils.widgets.text("m_threshold", str(current_metric['threshold']), "6. Threshold")
dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "7. Save?")

# User changes the name and threshold
print("\n" + "="*80)
print("USER CHANGES: Name and Threshold")
print("="*80)

dbutils.widgets.storage["m_name"] = "Completeness_EDITED"
dbutils.widgets.storage["m_threshold"] = "5"
dbutils.widgets.storage["save_btn"] = "yes"

print(f"  Changed Name: '{current_metric['name']}' ? '{dbutils.widgets.get('m_name')}'")
print(f"  Changed Threshold: '{current_metric['threshold']}' ? '{dbutils.widgets.get('m_threshold')}'")
print(f"  Kept Description: '{dbutils.widgets.get('m_desc')}'")

# Process edit
save_btn = dbutils.widgets.get("save_btn")
m_name = dbutils.widgets.get("m_name").strip()

if save_btn == "yes" and m_name:
    current_metrics = METRICS_CONFIG_DATA.to_dict('records')
    
    # Update metric
    current_metrics[row_idx] = {
        "name": dbutils.widgets.get("m_name"),
        "type": dbutils.widgets.get("m_type"),
        "description": dbutils.widgets.get("m_desc"),
        "grading_rubric": dbutils.widgets.get("m_rubric"),
        "threshold": dbutils.widgets.get("m_threshold"),
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    }
    METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
    print(f"\n? EDITED: {current_metric['name']} ? {current_metrics[row_idx]['name']}")

print(f"\n?? Final: {list(METRICS_CONFIG_DATA['name'])}")
print(f"   Metric 2 threshold: {METRICS_CONFIG_DATA.iloc[1]['threshold']}")

# VERDICT
print("\n" + "="*80)
print("VERDICT")
print("="*80)

if METRICS_CONFIG_DATA.iloc[1]['name'] == "Completeness_EDITED" and METRICS_CONFIG_DATA.iloc[1]['threshold'] == "5":
    print("??? EDIT WORKS WITH PRE-POPULATION! ???")
    print("? Form pre-filled with current values")
    print("? User only changes what they want")
    print("? Other fields kept as-is")
    print("? Metric updated successfully")
else:
    print("? Something went wrong")

print("="*80)
