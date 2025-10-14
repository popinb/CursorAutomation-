# =============================================================================
# SIMPLIFIED UI-BASED FILE CONFIGURATION
# =============================================================================

# Simple widgets - just the essentials
dbutils.widgets.text(
    "evaluation_data_path", 
    "evaluation_data.csv", 
    "📊 Evaluation Data (CSV filename or full path)"
)

dbutils.widgets.text(
    "metrics_config_path",
    "sample_metrics_config_simplified.csv",
    "📋 Metrics Configuration File (CSV filename or full path)"
)

# Ground truth file path widgets - users specify complete paths here
dbutils.widgets.text(
    "ground_truth_accuracy_path",
    "/Workspace/Users/popinb@zillowgroup.com/ground_truth_accuracy.csv",
    "📚 Ground Truth Accuracy File (full path)"
)

dbutils.widgets.text(
    "ground_truth_safety_path",
    "/Workspace/Users/popinb@zillowgroup.com/ground_truth_safety.csv",
    "📚 Ground Truth Safety File (full path)"
)

# Get settings
EVAL_DATA_PATH = dbutils.widgets.get("evaluation_data_path")
METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")
GROUND_TRUTH_ACCURACY_PATH = dbutils.widgets.get("ground_truth_accuracy_path")
GROUND_TRUTH_SAFETY_PATH = dbutils.widgets.get("ground_truth_safety_path")

print("📁 SIMPLIFIED FILE CONFIGURATION")
print("="*60)
print(f"Evaluation Data: {EVAL_DATA_PATH}")
print(f"Metrics Config: {METRICS_CONFIG_PATH}")
print(f"Ground Truth Accuracy: {GROUND_TRUTH_ACCURACY_PATH}")
print(f"Ground Truth Safety: {GROUND_TRUTH_SAFETY_PATH}")
print("="*60)

# Create mapping from filename to full path
GROUND_TRUTH_PATH_MAPPING = {
    "ground_truth_accuracy.csv": GROUND_TRUTH_ACCURACY_PATH,
    "ground_truth_safety.csv": GROUND_TRUTH_SAFETY_PATH
}

def find_file_in_workspace(filename):
    """Auto-detect file in common Databricks locations."""
    import os
    import glob
    
    # If it's already a full path, check it first
    if os.path.isabs(filename) or filename.startswith('/'):
        if os.path.exists(filename):
            return filename
    
    # Check if we have a UI-specified path for this file
    if filename in GROUND_TRUTH_PATH_MAPPING:
        ui_path = GROUND_TRUTH_PATH_MAPPING[filename]
        if ui_path and os.path.exists(ui_path):
            return ui_path
    
    # Common locations to search
    search_locations = [
        filename,  # Try the filename as-is first
        f"/Workspace/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/{filename}",
        f"./{filename}",
        f"/tmp/{filename}",
        f"/Workspace/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/**/{filename}"
    ]
    
    for location in search_locations:
        if "**" in location:
            # Use glob for recursive search
            files = glob.glob(location, recursive=True)
            if files:
                return files[0]  # Return first match
        else:
            # Direct path check
            if os.path.exists(location):
                return location
    
    return None

def load_csv_file(filename, file_type="data"):
    """Load CSV file with auto-detection."""
    try:
        # Try to find the file
        file_path = find_file_in_workspace(filename)
        
        if not file_path:
            print(f"❌ {file_type.title()} file not found: {filename}")
            print(f"   Please upload {filename} to your Databricks workspace")
            print(f"   Or provide the full path to the file")
            return None
        
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {file_type}: {len(df)} rows, {len(df.columns)} columns")
        print(f"   File: {file_path}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {file_type}: {e}")
        return None

def load_ground_truth_for_metric(metric_config, eval_df):
    """Load ground truth data for a specific metric."""
    ground_truth_file = metric_config.get("ground_truth_file_path", "")
    ground_truth_column = metric_config.get("ground_truth_column", "ground_truth")
    
    if not ground_truth_file:
        print(f"   No ground truth file specified for metric: {metric_config.get('name', 'unknown')}")
        return eval_df
    
    print(f"   Loading ground truth for {metric_config.get('name', 'unknown')}: {ground_truth_file}")
    
    try:
        # Load the ground truth file
        gt_df = load_csv_file(ground_truth_file, f"ground truth for {metric_config.get('name', 'unknown')}")
        
        if gt_df is None:
            print(f"   ⚠️ Failed to load ground truth file: {ground_truth_file}")
            return eval_df
        
        # Find common columns for merging
        common_cols = set(eval_df.columns) & set(gt_df.columns)
        if not common_cols:
            print(f"   ⚠️ No common columns found between eval data and {ground_truth_file}")
            return eval_df
        
        # Use the first common column for merging (usually sample_id)
        merge_col = list(common_cols)[0]
        print(f"   Merging on column: '{merge_col}'")
        
        # Find the ground truth data column
        gt_data_col = None
        if ground_truth_column in gt_df.columns:
            gt_data_col = ground_truth_column
        else:
            # Look for any text column that could be ground truth data
            text_cols = gt_df.select_dtypes(include=["object"]).columns.tolist()
            text_cols = [col for col in text_cols if col != merge_col]
            if text_cols:
                gt_data_col = text_cols[0]
        
        if not gt_data_col:
            print(f"   ⚠️ No suitable ground truth data column found in {ground_truth_file}")
            return eval_df
        
        # Merge the ground truth data
        gt_data = gt_df[[merge_col, gt_data_col]].set_index(merge_col)[gt_data_col].to_dict()
        
        # Add to evaluation dataframe
        eval_df[ground_truth_column] = eval_df[merge_col].map(gt_data).fillna("")
        
        coverage = (eval_df[ground_truth_column] != "").sum() / len(eval_df) * 100
        print(f"   ✅ Ground truth coverage: {coverage:.1f}% ({eval_df[ground_truth_column].ne('').sum()}/{len(eval_df)} samples)")
        
        return eval_df
        
    except Exception as e:
        print(f"   ❌ Error loading ground truth for {metric_config.get('name', 'unknown')}: {e}")
        return eval_df

# Load evaluation data
print(f"\n📊 Loading evaluation data...")
eval_df_raw = load_csv_file(EVAL_DATA_PATH, "evaluation data")

if eval_df_raw is None:
    print("\n📝 Creating sample data for demonstration...")
    eval_df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "prompt": [
            "What is the capital of France?",
            "Explain machine learning in simple terms",
            "How do I bake a chocolate cake?"
        ],
        "response": [
            "The capital of France is Paris, a beautiful city known for its culture and history.",
            "Machine learning is a type of AI where computers learn patterns from data to make predictions.",
            "To bake a chocolate cake, mix flour, cocoa, eggs, and sugar, then bake at 350°F for 30 minutes."
        ]
    })
    print("✅ Sample data created")
else:
    # Standardize column names
    eval_df = eval_df_raw.copy()
    
    # Auto-detect prompt and response columns
    text_cols = eval_df.select_dtypes(include=["object"]).columns.tolist()
    
    # Look for prompt column
    prompt_col = None
    for col in text_cols:
        if any(word in col.lower() for word in ["prompt", "question", "query", "input"]):
            prompt_col = col
            break
    
    # Look for response column
    response_col = None
    for col in text_cols:
        if any(word in col.lower() for word in ["response", "answer", "output", "generated"]):
            response_col = col
            break
    
    # Rename columns if found
    if prompt_col and response_col:
        eval_df = eval_df.rename(columns={prompt_col: "prompt", response_col: "response"})
        print(f"✅ Auto-detected columns: {prompt_col} → prompt, {response_col} → response")
    else:
        print("⚠️ Could not auto-detect prompt/response columns, using as-is")

# Load metrics configuration
print(f"\n📋 Loading metrics configuration...")
metrics_config_df = load_csv_file(METRICS_CONFIG_PATH, "metrics config")

if metrics_config_df is None:
    print("❌ No metrics configuration loaded!")
    print("   Please upload your metrics CSV file")
else:
    # Load ground truth data per metric
    print("\n📚 Loading ground truth data per metric...")
    
    for idx, metric_row in metrics_config_df.iterrows():
        metric_name = metric_row.get("name", f"metric_{idx}")
        ground_truth_file = metric_row.get("ground_truth_file_path", "")
        
        if ground_truth_file:
            print(f"\n📊 Processing ground truth for metric: {metric_name}")
            eval_df = load_ground_truth_for_metric(metric_row, eval_df)
        else:
            print(f"   No ground truth file specified for metric: {metric_name}")

# Ensure ground_truth column exists
if "ground_truth" not in eval_df.columns:
    eval_df["ground_truth"] = ""

# Display preview
print(f"\n📊 Data Preview ({len(eval_df)} samples):")
print(f"Columns: {list(eval_df.columns)}")
display(eval_df.head(3))

# Store data globally
EVALUATION_DATA = eval_df
METRICS_CONFIG_DATA = metrics_config_df