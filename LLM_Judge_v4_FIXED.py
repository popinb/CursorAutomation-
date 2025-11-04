# Databricks notebook source
# MAGIC %md
# MAGIC # LLM-as-a-Judge: v4 FIXED - Full Features Working
# MAGIC 
# MAGIC **Complete with Add/Edit/Delete/View - No widget storage, no sync issues**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Install Packages

# COMMAND ----------

%pip install openai pandas requests --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Load Data

# COMMAND ----------

import pandas as pd
import json

print("="*80)
print("CELL 2: LOADING DATA")
print("="*80)

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

EVALUATION_DATA_JSON = [
    {"sample_id": 1, "prompt": "Who is Cinderella?", "response": "Cinderella is a young girl who lives with her mean stepmother and two stepsisters. With help from her Fairy Godmother, she goes to the prince's ball and loses her glass slipper at midnight."},
    {"sample_id": 2, "prompt": "What did the Fairy Godmother turn into a carriage?", "response": "The Fairy Godmother used her magic wand to turn a big orange pumpkin into a beautiful golden carriage."},
    {"sample_id": 3, "prompt": "What happened at midnight?", "response": "When the clock struck midnight, Cinderella had to run away from the ball. In her hurry, she lost one of her glass slippers on the palace steps."}
]

GROUND_TRUTH_JSON = [
    {"sample_id": 1, "correct_answer": "Cinderella is a kind young girl mistreated by her stepmother and stepsisters. With help from her Fairy Godmother, she attends a royal ball, loses her glass slipper at midnight, and is found by the prince.", "story_element": "Main plot"},
    {"sample_id": 2, "correct_answer": "A pumpkin", "story_element": "Magic transformation"},
    {"sample_id": 3, "correct_answer": "Cinderella had to leave the ball because the magic spell would break at midnight. She ran away and lost her glass slipper.", "story_element": "Midnight deadline"}
]

# Convert to DataFrames
METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
EVALUATION_DATA = pd.DataFrame(EVALUATION_DATA_JSON)
GROUND_TRUTH_DATA_DF = pd.DataFrame(GROUND_TRUTH_JSON)
GROUND_TRUTH_DATA = {'ground_truth.csv': GROUND_TRUTH_DATA_DF}

print(f"\n? Data loaded:")
print(f"  Metrics: {len(METRICS_CONFIG_DATA)}")
print(f"  Samples: {len(EVALUATION_DATA)}")
print(f"  Ground truth: {len(GROUND_TRUTH_DATA_DF)}")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2.5: Full Metrics Editor (Add/Edit/Delete/View)

# COMMAND ----------

import json

print("="*80)
print("CELL 2.5: METRICS EDITOR - ALL FEATURES")
print("="*80)

# Create action dropdown
try:
    dbutils.widgets.dropdown("action", "view", ["view", "add", "edit", "delete"], "Action")
except:
    pass

action = dbutils.widgets.get("action")

# CREATE/REMOVE WIDGETS BASED ON ACTION
if action == "view":
    print("\n?? VIEW MODE")
    # Remove all form widgets
    for w in ["row_select", "m_name", "m_type", "m_desc", "m_rubric", "m_threshold", "save_btn"]:
        try:
            dbutils.widgets.remove(w)
        except:
            pass

elif action == "add":
    print("\n? ADD MODE")
    try:
        dbutils.widgets.text("m_name", "", "1. Name")
        dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "2. Type")
        dbutils.widgets.text("m_desc", "", "3. Description")
        dbutils.widgets.text("m_rubric", "", "4. Grading Rubric")
        dbutils.widgets.text("m_threshold", "", "5. Threshold (1, 4, or 75)")
        dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "6. ? Save?")
    except:
        pass
    # Remove row selector
    try:
        dbutils.widgets.remove("row_select")
    except:
        pass

elif action == "edit":
    print("\n?? EDIT MODE")
    try:
        dbutils.widgets.dropdown("row_select", "1", [str(i+1) for i in range(max(1, len(METRICS_CONFIG_DATA)))], "1. Row to Edit")
        dbutils.widgets.text("m_name", "", "2. Name")
        dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "3. Type")
        dbutils.widgets.text("m_desc", "", "4. Description")
        dbutils.widgets.text("m_rubric", "", "5. Grading Rubric")
        dbutils.widgets.text("m_threshold", "", "6. Threshold")
        dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "7. ? Save?")
    except:
        pass

elif action == "delete":
    print("\n??? DELETE MODE")
    try:
        dbutils.widgets.dropdown("row_select", "1", [str(i+1) for i in range(max(1, len(METRICS_CONFIG_DATA)))], "1. Row to Delete")
        dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "2. ?? Confirm?")
    except:
        pass
    # Remove form widgets
    for w in ["m_name", "m_type", "m_desc", "m_rubric", "m_threshold"]:
        try:
            dbutils.widgets.remove(w)
        except:
            pass

# PROCESS ACTIONS
if action == "add":
    save_btn = dbutils.widgets.get("save_btn")
    m_name = dbutils.widgets.get("m_name").strip()
    
    if save_btn == "yes" and m_name:
        # Check for duplicates
        current_metrics = METRICS_CONFIG_DATA.to_dict('records')
        existing_names = [m['name'] for m in current_metrics]
        
        if m_name in existing_names:
            print(f"\n? ERROR: Metric '{m_name}' already exists!")
        else:
            # Add new metric
            new_metric = {
                "name": m_name,
                "type": dbutils.widgets.get("m_type"),
                "description": dbutils.widgets.get("m_desc"),
                "grading_rubric": dbutils.widgets.get("m_rubric"),
                "threshold": dbutils.widgets.get("m_threshold"),
                "ground_truth_file_path": "ground_truth.csv",
                "ground_truth_column": "correct_answer"
            }
            current_metrics.append(new_metric)
            METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
            
            print(f"\n? ADDED: {m_name}")
            
            # Auto-reset
            dbutils.widgets.remove("save_btn")
            dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "6. ? Save?")
            dbutils.widgets.remove("m_name")
            dbutils.widgets.text("m_name", "", "1. Name")
    elif save_btn == "yes" and not m_name:
        print("\n??  Please enter a metric name")

elif action == "edit":
    save_btn = dbutils.widgets.get("save_btn")
    m_name = dbutils.widgets.get("m_name").strip()
    
    if save_btn == "yes" and m_name:
        row_idx = int(dbutils.widgets.get("row_select")) - 1
        current_metrics = METRICS_CONFIG_DATA.to_dict('records')
        
        if 0 <= row_idx < len(current_metrics):
            old_name = current_metrics[row_idx]['name']
            
            # Check if new name conflicts with other metrics
            existing_names = [m['name'] for i, m in enumerate(current_metrics) if i != row_idx]
            if m_name in existing_names:
                print(f"\n? ERROR: Metric name '{m_name}' already exists!")
            else:
                # Update metric
                current_metrics[row_idx] = {
                    "name": m_name,
                    "type": dbutils.widgets.get("m_type"),
                    "description": dbutils.widgets.get("m_desc"),
                    "grading_rubric": dbutils.widgets.get("m_rubric"),
                    "threshold": dbutils.widgets.get("m_threshold"),
                    "ground_truth_file_path": "ground_truth.csv",
                    "ground_truth_column": "correct_answer"
                }
                METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
                
                print(f"\n? EDITED: {old_name} ? {m_name}")
                
                # Auto-reset
                dbutils.widgets.remove("save_btn")
                dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "7. ? Save?")
    elif save_btn == "yes" and not m_name:
        print("\n??  Please enter a metric name")

elif action == "delete":
    save_btn = dbutils.widgets.get("save_btn")
    
    if save_btn == "yes":
        row_idx = int(dbutils.widgets.get("row_select")) - 1
        current_metrics = METRICS_CONFIG_DATA.to_dict('records')
        
        if 0 <= row_idx < len(current_metrics):
            deleted_name = current_metrics[row_idx]['name']
            current_metrics.pop(row_idx)
            
            if current_metrics:
                METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
            else:
                METRICS_CONFIG_DATA = pd.DataFrame(columns=["name", "type", "description", "grading_rubric", "threshold", "ground_truth_file_path", "ground_truth_column"])
            
            print(f"\n??? DELETED: {deleted_name}")
            
            # Update row selector if metrics remain
            if len(current_metrics) > 0:
                dbutils.widgets.remove("row_select")
                dbutils.widgets.dropdown("row_select", "1", [str(i+1) for i in range(len(current_metrics))], "1. Row to Delete")
            
            # Auto-reset
            dbutils.widgets.remove("save_btn")
            dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "2. ?? Confirm?")

# DISPLAY TABLE
print(f"\n?? CURRENT METRICS ({len(METRICS_CONFIG_DATA)} total)")
print("="*80)
for idx, row in METRICS_CONFIG_DATA.iterrows():
    gt = f"{row.get('ground_truth_file_path', '')}/{row.get('ground_truth_column', '')}"
    print(f"{idx+1}. {row['name']} ({row['type']}) - Threshold: {row['threshold']} - GT: {gt}")
print("="*80)

# Instructions
if action == "view":
    print("\n?? Select Action (add/edit/delete) to modify metrics")
elif action == "add":
    print("\n?? Fill form, set Save='yes', then re-run cell")
elif action == "edit":
    print("\n?? Select row, fill form, set Save='yes', then re-run cell")
elif action == "delete":
    print("\n?? Select row, set Confirm='yes', then re-run cell")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Configure LLM (continues from here with all features from original v4...)

# COMMAND ----------

print("="*80)
print("CELL 3: CONFIGURE LLM")
print("="*80)

print(f"\n? METRICS_CONFIG_DATA has {len(METRICS_CONFIG_DATA)} metrics:")
for idx, row in METRICS_CONFIG_DATA.iterrows():
    print(f"  {idx+1}. {row['name']}")

print("\n?? These metrics will be used for evaluation in Cell 6")
print("="*80)
