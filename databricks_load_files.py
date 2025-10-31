# Databricks notebook source
# COMMAND ----------

import pandas as pd
import os

# Get current user dynamically
try:
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except:
    current_user = "unknown_user"

print(f"?? Current User: {current_user}")

# Widgets for file paths with dynamic user paths
dbutils.widgets.text(
    "evaluation_data_path", 
    "evaluation_data.csv", 
    "?? Evaluation Data (CSV)"
)

dbutils.widgets.text(
    "metrics_config_path",
    "sample_metrics_config_simplified.csv",
    "?? Metrics Configuration (CSV)"
)

dbutils.widgets.text(
    "ground_truth_files",
    "ground_truth_accuracy.csv;ground_truth_safety.csv",
    "?? Ground Truth Files (semicolon separated)"
)

# Get settings
EVAL_DATA_PATH = dbutils.widgets.get("evaluation_data_path")
METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")
GROUND_TRUTH_FILES_STRING = dbutils.widgets.get("ground_truth_files")

print("\n?? FILE CONFIGURATION")
print("="*60)
print(f"Evaluation Data: {EVAL_DATA_PATH}")
print(f"Metrics Config: {METRICS_CONFIG_PATH}")
print(f"Ground Truth Files: {GROUND_TRUTH_FILES_STRING}")
print("="*60)

# COMMAND ----------

def parse_ground_truth_files(gt_files_string):
    """Parse semicolon or comma separated ground truth file paths."""
    if not gt_files_string or not str(gt_files_string).strip():
        return []
    
    gt_files_string = str(gt_files_string)
    
    files = []
    for separator in [';', ',']:
        if separator in gt_files_string:
            files = [f.strip() for f in gt_files_string.split(separator) if f.strip()]
            break
    
    if not files:
        files = [gt_files_string.strip()]
    
    return files

def find_file_in_workspace(filename):
    """
    Auto-detect file in common Databricks locations.
    Supports both absolute paths and filenames.
    Works for any user dynamically.
    """
    # If it's already an absolute path and exists, return it
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    # Extract just the filename if a path was provided
    base_filename = os.path.basename(filename)
    
    # Get current user dynamically
    try:
        user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    except:
        user_name = None
    
    # Build search locations
    search_locations = []
    
    # Add user-specific workspace locations
    if user_name:
        search_locations.extend([
            f"/Workspace/Users/{user_name}/{base_filename}",
            f"/Workspace/Users/{user_name}/{filename}",
        ])
    
    # Add common locations
    search_locations.extend([
        filename,  # Original path as-is
        base_filename,  # Just the filename
        f"./{base_filename}",  # Current directory
        f"/Workspace/Shared/{base_filename}",  # Shared workspace
        f"/dbfs/FileStore/{base_filename}",  # DBFS FileStore
        f"/tmp/{base_filename}",  # Temp directory
        f"/Workspace/{base_filename}",  # Root workspace
    ])
    
    # Search for the file
    for location in search_locations:
        if os.path.exists(location):
            print(f"   ?? Found at: {location}")
            return location
    
    return None

def load_csv_file(filename, file_type="data"):
    """Load CSV file with auto-detection."""
    try:
        file_path = find_file_in_workspace(filename)
        
        if not file_path:
            print(f"? {file_type.title()} file not found: {filename}")
            print(f"   Searched locations for user: {current_user}")
            return None
        
        df = pd.read_csv(file_path)
        print(f"? Loaded {file_type}: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"? Error loading {file_type}: {e}")
        return None

# COMMAND ----------

# Load evaluation data
print(f"\n?? Loading evaluation data...")
evaluation_data_df = load_csv_file(EVAL_DATA_PATH, "evaluation data")

if evaluation_data_df is None:
    print("\n?? Creating sample data...")
    evaluation_data_df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "prompt": [
            "What is the capital of France?",
            "Explain machine learning in simple terms",
            "How do I bake a chocolate cake?"
        ],
        "response": [
            "The capital of France is Paris, a beautiful city known for its culture and history.",
            "Machine learning is a type of AI where computers learn patterns from data to make predictions.",
            "To bake a chocolate cake, mix flour, cocoa, eggs, and sugar, then bake at 350?F for 30 minutes."
        ]
    })
    print("? Sample data created")

# COMMAND ----------

# Load metrics configuration
print(f"\n?? Loading metrics configuration...")
metrics_config_df = load_csv_file(METRICS_CONFIG_PATH, "metrics config")

if metrics_config_df is None:
    print("? No metrics configuration loaded!")
else:
    print(f"? Loaded {len(metrics_config_df)} metrics")

# COMMAND ----------

# Load ground truth files
GROUND_TRUTH_FILES_LIST = parse_ground_truth_files(GROUND_TRUTH_FILES_STRING)
ground_truth_data = {}

if GROUND_TRUTH_FILES_LIST:
    print(f"\n?? Loading {len(GROUND_TRUTH_FILES_LIST)} ground truth files...")
    print("?? TIP: Just specify filenames (e.g., 'ground_truth_accuracy.csv')")
    print(f"        Files will be searched in /Workspace/Users/{current_user}/ and other common locations\n")
    
    for file_path in GROUND_TRUTH_FILES_LIST:
        if not file_path:
            continue
            
        filename = os.path.basename(file_path)
        print(f"\n?? Searching for: {file_path}")
        
        found_path = find_file_in_workspace(file_path)
        
        if found_path:
            try:
                df = pd.read_csv(found_path)
                ground_truth_data[filename] = df
                print(f"? {filename} ? {len(df)} rows, columns: {', '.join(df.columns.tolist())}")
            except Exception as e:
                print(f"? Error loading {filename}: {e}")
        else:
            print(f"?? File not found: {file_path}")
            print(f"   ?? Make sure the file exists in /Workspace/Users/{current_user}/")
    
    print(f"\n? Successfully loaded {len(ground_truth_data)} ground truth files")
else:
    print("\n?? No ground truth files specified")

# COMMAND ----------

# Store data globally
EVALUATION_DATA = evaluation_data_df
METRICS_CONFIG_DATA = metrics_config_df
GROUND_TRUTH_DATA = ground_truth_data

print(f"\n" + "="*60)
print(f"?? DATA LOADED SUCCESSFULLY!")
print("="*60)
print(f"   Evaluation samples: {len(EVALUATION_DATA)}")
print(f"   Metrics configured: {len(METRICS_CONFIG_DATA) if METRICS_CONFIG_DATA is not None else 0}")
print(f"   Ground truth files: {len(GROUND_TRUTH_DATA)}")
print(f"   Available ground truth: {', '.join(GROUND_TRUTH_DATA.keys())}")
print("="*60)

# COMMAND ----------

# Display summary of loaded data
print("\n?? LOADED DATASETS SUMMARY:\n")

if EVALUATION_DATA is not None:
    print("1?? EVALUATION DATA:")
    print(f"   Columns: {', '.join(EVALUATION_DATA.columns.tolist())}")
    print(f"   Rows: {len(EVALUATION_DATA)}")
    
if METRICS_CONFIG_DATA is not None:
    print("\n2?? METRICS CONFIGURATION:")
    print(f"   Columns: {', '.join(METRICS_CONFIG_DATA.columns.tolist())}")
    print(f"   Metrics: {len(METRICS_CONFIG_DATA)}")

if GROUND_TRUTH_DATA:
    print("\n3?? GROUND TRUTH FILES:")
    for filename, df in GROUND_TRUTH_DATA.items():
        print(f"   ? {filename}: {len(df)} rows, {len(df.columns)} columns")
