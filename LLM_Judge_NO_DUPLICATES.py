# Databricks notebook source
# MAGIC %md
# MAGIC # LLM-as-a-Judge: NO DUPLICATES VERSION
# MAGIC 
# MAGIC **Prevents duplicate additions with automatic save button reset**

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
    {"name": "Story_Accuracy", "type": "binary", "description": "Evaluate if the response is factually accurate about the Cinderella story", "grading_rubric": "Score 1 if all story facts are correct and align with the classic Cinderella tale. Score 0 if any facts are incorrect, made up, or contradict the original story.", "threshold": "1", "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "Response_Completeness", "type": "1-5_scale", "description": "Evaluate how complete and thorough the response is", "grading_rubric": "5=Fully complete, addresses all aspects comprehensively; 4=Mostly complete with minor gaps; 3=Partially complete, missing some details; 2=Barely complete, many gaps; 1=Incomplete or inadequate", "threshold": "4", "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "Child_Friendliness", "type": "percentage", "description": "Evaluate what percentage of the response is appropriate and understandable for children", "grading_rubric": "100%=Perfectly child-friendly language and content; 75%=Mostly appropriate with minor complex words; 50%=Somewhat child-friendly; 25%=Barely appropriate for children; 0%=Not suitable for children", "threshold": "75", "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"}
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
# MAGIC ## Cell 2.5: Metrics Editor - NO DUPLICATES

# COMMAND ----------

import json

print("="*80)
print("CELL 2.5: METRICS EDITOR (NO DUPLICATES)")
print("="*80)

# Create action dropdown
try:
    dbutils.widgets.dropdown("action", "view", ["view", "add"], "Action")
except:
    pass

action = dbutils.widgets.get("action")

# CREATE WIDGETS BASED ON ACTION
if action == "view":
    print("\n?? VIEW MODE")
    # Remove all form widgets
    for w in ["m_name", "m_desc", "m_rubric", "m_threshold", "save_btn"]:
        try:
            dbutils.widgets.remove(w)
        except:
            pass

elif action == "add":
    print("\n? ADD MODE")
    # Create form widgets
    try:
        dbutils.widgets.text("m_name", "", "1. Name")
        dbutils.widgets.text("m_desc", "", "2. Description")
        dbutils.widgets.text("m_rubric", "", "3. Grading Rubric")
        dbutils.widgets.text("m_threshold", "", "4. Threshold")
        dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "5. Save?")
    except:
        pass

# PROCESS ADD ACTION (ONLY if action is add)
if action == "add":
    save_btn = dbutils.widgets.get("save_btn")
    m_name = dbutils.widgets.get("m_name").strip()
    
    if save_btn == "yes" and m_name:
        print(f"\n?? PROCESSING: Add '{m_name}'")
        
        # Check for duplicate names
        current_metrics = METRICS_CONFIG_DATA.to_dict('records')
        existing_names = [m['name'] for m in current_metrics]
        
        if m_name in existing_names:
            print(f"? ERROR: Metric '{m_name}' already exists!")
            print(f"   Existing metrics: {existing_names}")
            print(f"   Please use a different name.")
        else:
            # Add new metric
            new_metric = {
                "name": m_name,
                "type": "binary",
                "description": dbutils.widgets.get("m_desc"),
                "grading_rubric": dbutils.widgets.get("m_rubric"),
                "threshold": dbutils.widgets.get("m_threshold"),
                "ground_truth_file_path": "ground_truth.csv",
                "ground_truth_column": "correct_answer"
            }
            
            current_metrics.append(new_metric)
            
            # Update global DataFrame
            METRICS_CONFIG_DATA = pd.DataFrame(current_metrics)
            
            print(f"? ADDED: {m_name}")
            print(f"?? Total metrics: {len(METRICS_CONFIG_DATA)}")
            
            # CRITICAL: Auto-reset save button to prevent duplicate if user re-runs
            dbutils.widgets.remove("save_btn")
            dbutils.widgets.dropdown("save_btn", "no", ["no", "yes"], "5. Save?")
            print("?? Save button reset to 'no' (prevents accidental duplicates)")
            
            # Clear name field
            dbutils.widgets.remove("m_name")
            dbutils.widgets.text("m_name", "", "1. Name")
            
    elif save_btn == "yes" and not m_name:
        print("\n??  Please enter a metric name")
    else:
        print(f"\n?? Fill form and set Save='yes' to add metric")

# DISPLAY TABLE
print(f"\n?? CURRENT METRICS ({len(METRICS_CONFIG_DATA)} total)")
print("="*80)
for idx, row in METRICS_CONFIG_DATA.iterrows():
    print(f"{idx+1}. {row['name']} ({row['type']}) - Threshold: {row['threshold']}")
print("="*80)

if action == "add":
    print("\n?? TO ADD A METRIC:")
    print("  1. Fill in: Name, Description, Rubric, Threshold")
    print("  2. Set Save='yes'")
    print("  3. Re-run this cell (Shift+Enter)")
    print("  4. Save button will auto-reset to 'no' after adding")
    print("  5. To see clean list, change Action='view'")
elif action == "view":
    print("\n?? TO ADD A METRIC:")
    print("  - Change Action to 'add' and re-run this cell")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: Check current metrics

# COMMAND ----------

print(f"? METRICS_CONFIG_DATA has {len(METRICS_CONFIG_DATA)} metrics:")
for idx, row in METRICS_CONFIG_DATA.iterrows():
    print(f"  {idx+1}. {row['name']}")
