# Databricks notebook source
# MAGIC %md
# MAGIC # LLM-as-a-Judge: TRUE PERSISTENCE VERSION
# MAGIC 
# MAGIC **Metrics persist across notebook restarts using DBFS file storage**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Install Packages

# COMMAND ----------

%pip install openai pandas requests --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Load Data with DBFS Persistence

# COMMAND ----------

import pandas as pd
import json
import os

print("="*80)
print("CELL 2: LOADING DATA")
print("="*80)

# DBFS path for storing metrics (persists across sessions!)
METRICS_STORAGE_PATH = "/dbfs/tmp/llm_judge_metrics.json"

# Hardcoded defaults
METRICS_CONFIG_JSON = [
    {
        "name": "Story_Accuracy",
        "type": "binary",
        "description": "Evaluate if the response is factually accurate about the Cinderella story",
        "grading_rubric": "Score 1 if all story facts are correct and align with the classic Cinderella tale. Score 0 if any facts are incorrect, made up, or contradict the original story.",
        "threshold": "1",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    },
    {
        "name": "Response_Completeness",
        "type": "1-5_scale",
        "description": "Evaluate how complete and thorough the response is",
        "grading_rubric": "5=Fully complete, addresses all aspects comprehensively; 4=Mostly complete with minor gaps; 3=Partially complete, missing some details; 2=Barely complete, many gaps; 1=Incomplete or inadequate",
        "threshold": "4",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    },
    {
        "name": "Child_Friendliness",
        "type": "percentage",
        "description": "Evaluate what percentage of the response is appropriate and understandable for children",
        "grading_rubric": "100%=Perfectly child-friendly language and content; 75%=Mostly appropriate with minor complex words; 50%=Somewhat child-friendly; 25%=Barely appropriate for children; 0%=Not suitable for children",
        "threshold": "75",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    }
]

# Evaluation Data
EVALUATION_DATA_JSON = [
    {"sample_id": 1, "prompt": "Who is Cinderella?", "response": "Cinderella is a young girl who lives with her mean stepmother and two stepsisters. With help from her Fairy Godmother, she goes to the prince's ball and loses her glass slipper at midnight."},
    {"sample_id": 2, "prompt": "What did the Fairy Godmother turn into a carriage?", "response": "The Fairy Godmother used her magic wand to turn a big orange pumpkin into a beautiful golden carriage."},
    {"sample_id": 3, "prompt": "What happened at midnight?", "response": "When the clock struck midnight, Cinderella had to run away from the ball. In her hurry, she lost one of her glass slippers on the palace steps."}
]

# Ground Truth
GROUND_TRUTH_JSON = [
    {"sample_id": 1, "correct_answer": "Cinderella is a kind young girl mistreated by her stepmother and stepsisters. With help from her Fairy Godmother, she attends a royal ball, loses her glass slipper at midnight, and is found by the prince.", "story_element": "Main plot"},
    {"sample_id": 2, "correct_answer": "A pumpkin", "story_element": "Magic transformation"},
    {"sample_id": 3, "correct_answer": "Cinderella had to leave the ball because the magic spell would break at midnight. She ran away and lost her glass slipper.", "story_element": "Midnight deadline"}
]

# Convert to DataFrames
EVALUATION_DATA = pd.DataFrame(EVALUATION_DATA_JSON)
GROUND_TRUTH_DATA_DF = pd.DataFrame(GROUND_TRUTH_JSON)
GROUND_TRUTH_DATA = {'ground_truth.csv': GROUND_TRUTH_DATA_DF}

# CRITICAL: Load metrics from DBFS file (TRUE PERSISTENCE!)
if os.path.exists(METRICS_STORAGE_PATH):
    try:
        with open(METRICS_STORAGE_PATH, 'r') as f:
            metrics_list = json.load(f)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
        print(f"\n??  LOADED EXISTING METRICS FROM DBFS FILE")
        print(f"   File: {METRICS_STORAGE_PATH}")
        print(f"   You have {len(METRICS_CONFIG_DATA)} custom metrics")
        print(f"   To reset to defaults, delete: {METRICS_STORAGE_PATH}")
    except Exception as e:
        print(f"\n? Error loading from DBFS: {e}")
        print("   Using defaults instead")
        METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
else:
    # No file exists, use defaults
    METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
    
    # Normalize GT
    for idx in range(len(METRICS_CONFIG_DATA)):
        if pd.isna(METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path']) or METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path'] == '':
            METRICS_CONFIG_DATA.loc[idx, 'ground_truth_file_path'] = 'ground_truth.csv'
        if pd.isna(METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column']) or METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column'] == '':
            METRICS_CONFIG_DATA.loc[idx, 'ground_truth_column'] = 'correct_answer'
    
    # Save defaults to file
    os.makedirs(os.path.dirname(METRICS_STORAGE_PATH), exist_ok=True)
    with open(METRICS_STORAGE_PATH, 'w') as f:
        json.dump(METRICS_CONFIG_DATA.to_dict('records'), f, indent=2)
    
    print(f"\n? LOADED DEFAULT METRICS (first run)")
    print(f"   Saved to: {METRICS_STORAGE_PATH}")

print(f"\nData loaded:")
print(f"  Metrics: {len(METRICS_CONFIG_DATA)}")
print(f"  Samples: {len(EVALUATION_DATA)}")
print(f"  Ground truth: {len(GROUND_TRUTH_DATA_DF)}")
print(f"  All metrics use: ground_truth.csv/correct_answer")
print(f"\n{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2.5: Metrics Editor with DBFS Persistence

# COMMAND ----------

import json
from io import StringIO

print("="*80)
print("CELL 2.5: METRICS EDITOR (DBFS PERSISTENCE)")
print("="*80)

# Reload from DBFS file (always get latest!)
try:
    with open(METRICS_STORAGE_PATH, 'r') as f:
        metrics_list = json.load(f)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
    print(f"? Loaded {len(METRICS_CONFIG_DATA)} metrics from DBFS")
except Exception as e:
    print(f"??  Could not load from DBFS: {e}")
    metrics_list = METRICS_CONFIG_DATA.to_dict('records')

# Create action widget
try:
    dbutils.widgets.dropdown("action", "view", ["view", "add", "edit", "delete"], "Action")
except:
    pass

action = dbutils.widgets.get("action")

# Show/hide widgets based on action
if action == "view":
    # Remove all form widgets
    for widget_name in ["row_select", "m_name", "m_type", "m_desc", "m_rubric", "m_threshold", "save_action"]:
        try:
            dbutils.widgets.remove(widget_name)
        except:
            pass
            
elif action == "add":
    # ADD mode widgets
    try:
        dbutils.widgets.text("m_name", "", "1. Name")
        dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "2. Type")
        dbutils.widgets.text("m_desc", "", "3. Description")
        dbutils.widgets.text("m_rubric", "", "4. Grading Rubric")
        dbutils.widgets.text("m_threshold", "", "5. Threshold")
        dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "6. ? Save Changes?")
    except:
        pass
    try:
        dbutils.widgets.remove("row_select")
    except:
        pass
        
elif action == "edit":
    # EDIT mode widgets
    try:
        dbutils.widgets.dropdown("row_select", "1", [str(i+1) for i in range(max(1, len(METRICS_CONFIG_DATA)))], "1. Row to Edit")
        dbutils.widgets.text("m_name", "", "2. Name")
        dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "3. Type")
        dbutils.widgets.text("m_desc", "", "4. Description")
        dbutils.widgets.text("m_rubric", "", "5. Grading Rubric")
        dbutils.widgets.text("m_threshold", "", "6. Threshold")
        dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "7. ? Save Changes?")
    except:
        pass
        
elif action == "delete":
    # DELETE mode widgets
    try:
        dbutils.widgets.dropdown("row_select", "1", [str(i+1) for i in range(max(1, len(METRICS_CONFIG_DATA)))], "1. Row to Delete")
        dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "2. ?? Confirm Delete?")
    except:
        pass
    for widget_name in ["m_name", "m_type", "m_desc", "m_rubric", "m_threshold"]:
        try:
            dbutils.widgets.remove(widget_name)
        except:
            pass

# Reload from file before processing
try:
    with open(METRICS_STORAGE_PATH, 'r') as f:
        metrics_list = json.load(f)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
except:
    metrics_list = METRICS_CONFIG_DATA.to_dict('records')

# Process actions
save_action = dbutils.widgets.get("save_action") if action in ["add", "edit", "delete"] else "no"

if action == "add" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
    # ADD
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
    
    # SAVE TO DBFS FILE
    with open(METRICS_STORAGE_PATH, 'w') as f:
        json.dump(metrics_list, f, indent=2)
    
    # Reset widgets
    for w in ["m_name", "m_desc", "m_rubric", "m_threshold"]:
        dbutils.widgets.remove(w)
        dbutils.widgets.text(w, "", w)
    dbutils.widgets.remove("save_action")
    dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "6. ? Save Changes?")
    
    print(f"\n? ADDED: {new_metric['name']}")
    print(f"?? Saved to DBFS: {METRICS_STORAGE_PATH}")

elif action == "edit" and save_action == "yes" and dbutils.widgets.get("m_name").strip():
    # EDIT
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
        
        # SAVE TO DBFS FILE
        with open(METRICS_STORAGE_PATH, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        
        dbutils.widgets.remove("save_action")
        dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "7. ? Save Changes?")
        
        print(f"\n? EDITED: {old_name} ? {metrics_list[row_idx]['name']}")
        print(f"?? Saved to DBFS: {METRICS_STORAGE_PATH}")

elif action == "delete" and save_action == "yes":
    # DELETE
    row_idx = int(dbutils.widgets.get("row_select")) - 1
    if 0 <= row_idx < len(metrics_list):
        deleted_name = metrics_list[row_idx]['name']
        metrics_list.pop(row_idx)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame(columns=["name", "type", "description", "grading_rubric", "threshold", "ground_truth_file_path", "ground_truth_column"])
        
        # SAVE TO DBFS FILE
        with open(METRICS_STORAGE_PATH, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        
        dbutils.widgets.remove("save_action")
        dbutils.widgets.dropdown("save_action", "no", ["no", "yes"], "2. ?? Confirm Delete?")
        
        if len(metrics_list) > 0:
            dbutils.widgets.remove("row_select")
            dbutils.widgets.dropdown("row_select", "1", [str(i+1) for i in range(len(metrics_list))], "1. Row to Delete")
        
        print(f"\n??? DELETED: {deleted_name}")
        print(f"?? Saved to DBFS: {METRICS_STORAGE_PATH}")

# FINAL RELOAD before display
try:
    with open(METRICS_STORAGE_PATH, 'r') as f:
        metrics_list = json.load(f)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
except:
    pass

# Display table
print(f"\n?? CURRENT METRICS CONFIGURATION")
print("="*80)
print(f"Total metrics: {len(METRICS_CONFIG_DATA)}")
print("="*80)

# Visual HTML table
def generate_metrics_table(df):
    html = """
    <style>
        .metrics-table { border-collapse: collapse; width: 100%; font-family: Arial; }
        .metrics-table th { background: #1e40af; color: white; padding: 12px; text-align: left; }
        .metrics-table td { border: 1px solid #ddd; padding: 10px; }
        .metrics-table tr:nth-child(even) { background: #f9fafb; }
        .metrics-table tr:hover { background: #e5e7eb; }
    </style>
    <table class="metrics-table">
        <tr>
            <th>#</th>
            <th>Name</th>
            <th>Type</th>
            <th>Description</th>
            <th>Threshold</th>
            <th>Ground Truth</th>
        </tr>
    """
    for idx, row in df.iterrows():
        html += f"""
        <tr>
            <td><strong>{idx+1}</strong></td>
            <td><strong>{row['name']}</strong></td>
            <td>{row['type']}</td>
            <td>{row.get('description', '')[:100]}...</td>
            <td>{row.get('threshold', '')}</td>
            <td>{row.get('ground_truth_file_path', '')}/{row.get('ground_truth_column', '')}</td>
        </tr>
        """
    html += "</table>"
    return html

displayHTML(generate_metrics_table(METRICS_CONFIG_DATA))

print(f"\n?? Storage location: {METRICS_STORAGE_PATH}")
print("="*80)

# COMMAND ----------

# MAGIC %md  
# MAGIC ## Cell 3: Configure LLM (rest of notebook continues same as before...)

# COMMAND ----------

print("? Rest of cells continue as before (Cell 3-7 unchanged)")
print(f"   METRICS_CONFIG_DATA has {len(METRICS_CONFIG_DATA)} metrics")
print("   These will be used for evaluation")
