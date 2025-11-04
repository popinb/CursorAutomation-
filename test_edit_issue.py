"""
Test why EDIT might not be working
"""
import pandas as pd

print("="*80)
print("TEST: EDIT FUNCTIONALITY")
print("="*80)

# Mock dbutils
class MockWidgets:
    def __init__(self):
        self.storage = {}
    
    def dropdown(self, name, default, options, label):
        if name not in self.storage:
            self.storage[name] = default
        print(f"  Widget: {label} = '{default}'")
    
    def text(self, name, default, label):
        if name not in self.storage:
            self.storage[name] = default
        print(f"  Widget: {label} = '{default}'")
    
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
    {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "M2", "type": "1-5_scale", "description": "Test2", "grading_rubric": "R2", "threshold": "4", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
])

print(f"\nInitial: {list(METRICS_CONFIG_DATA['name'])}")

# Scenario: User selects EDIT mode
print("\n" + "="*80)
print("USER SELECTS EDIT MODE")
print("="*80)

dbutils.widgets.dropdown("action", "edit", ["view", "add", "edit", "delete"], "Action")
action = dbutils.widgets.get("action")

# CREATE EDIT WIDGETS (CURRENT V5 LOGIC - EMPTY FIELDS!)
print("\nCreating edit widgets with EMPTY fields:")
dbutils.widgets.dropdown("row_select", "1", [str(i+1) for i in range(len(METRICS_CONFIG_DATA))], "1. Row to Edit")
dbutils.widgets.text("m_name", "", "2. Name")  # ? EMPTY!
dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "3. Type")
dbutils.widgets.text("m_desc", "", "4. Description")  # ? EMPTY!
dbutils.widgets.text("m_rubric", "", "5. Grading Rubric")  # ? EMPTY!
dbutils.widgets.text("m_threshold", "", "6. Threshold")  # ? EMPTY!
dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "7. Save?")

print("\n? PROBLEM: All form fields are EMPTY!")
print("   User has to manually re-type everything")
print("   If they don't fill all fields ? edit saves with empty values!")

# User tries to edit without filling form
print("\n" + "="*80)
print("USER CLICKS SAVE WITHOUT FILLING FORM")
print("="*80)

row_idx = int(dbutils.widgets.get("row_select")) - 1
print(f"\nSelected row {row_idx+1}: {METRICS_CONFIG_DATA.iloc[row_idx]['name']}")

dbutils.widgets.storage["save_btn"] = "yes"
save_btn = dbutils.widgets.get("save_btn")
m_name = dbutils.widgets.get("m_name").strip()

print(f"  save_btn = {save_btn}")
print(f"  m_name = '{m_name}' (EMPTY!)")

if save_btn == "yes" and m_name:
    print("  ? Would update metric")
else:
    print("  ? ? EDIT FAILS because m_name is empty!")

# THE FIX: Pre-populate form fields
print("\n" + "="*80)
print("THE FIX: PRE-POPULATE FORM FIELDS")
print("="*80)

# Load current values for selected row
row_idx = int(dbutils.widgets.get("row_select")) - 1
current_metric = METRICS_CONFIG_DATA.iloc[row_idx]

print(f"\nPre-populating with row {row_idx+1} values:")
dbutils.widgets.storage.clear()
dbutils.widgets.dropdown("action", "edit", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.dropdown("row_select", "1", ["1", "2"], "1. Row to Edit")
dbutils.widgets.text("m_name", current_metric['name'], "2. Name")  # PRE-FILLED!
dbutils.widgets.dropdown("m_type", current_metric['type'], ["binary", "1-5_scale", "percentage"], "3. Type")
dbutils.widgets.text("m_desc", current_metric['description'], "4. Description")  # PRE-FILLED!
dbutils.widgets.text("m_rubric", current_metric['grading_rubric'], "5. Grading Rubric")  # PRE-FILLED!
dbutils.widgets.text("m_threshold", str(current_metric['threshold']), "6. Threshold")  # PRE-FILLED!
dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "7. Save?")

print("\n? NOW user sees their current values and can edit them!")

print("\n" + "="*80)
print("SOLUTION")
print("="*80)
print("? CURRENT: Edit widgets created with EMPTY fields")
print("? NEEDED: Edit widgets pre-populated with current metric values")
print("\nUser needs to:")
print("  1. Select row number")
print("  2. See form filled with current values")
print("  3. Change what they want")
print("  4. Click Save=yes")
