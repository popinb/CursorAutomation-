# COMMAND ----------

# =============================================================================
# SIMPLE FILE LOADING CONFIGURATION
# =============================================================================

# Simple file path widgets
dbutils.widgets.text(
    "evaluation_data_path", 
    "evaluation_data.csv", 
    "📊 Evaluation Data File"
)

dbutils.widgets.text(
    "metrics_config_path",
    "metrics_config_clean.csv",
    "📋 Metrics Configuration File"
)

dbutils.widgets.text(
    "ground_truth_accuracy_path",
    "/Workspace/Users/popinb@zillowgroup.com/ground_truth_accuracy.csv",
    "📚 Ground Truth Accuracy File"
)

dbutils.widgets.text(
    "ground_truth_safety_path",
    "/Workspace/Users/popinb@zillowgroup.com/ground_truth_safety.csv",
    "📚 Ground Truth Safety File"
)

print("✅ File path widgets created")
print("📁 Evaluation Data:", dbutils.widgets.get("evaluation_data_path"))
print("📋 Metrics Config:", dbutils.widgets.get("metrics_config_path"))
print("📚 Ground Truth Accuracy:", dbutils.widgets.get("ground_truth_accuracy_path"))
print("📚 Ground Truth Safety:", dbutils.widgets.get("ground_truth_safety_path"))

# COMMAND ----------

# =============================================================================
# FILE LOADING FUNCTIONS
# =============================================================================

def find_file_in_workspace(filename):
    """Find file in Databricks workspace with flexible search."""
    # First check if it's already an absolute path
    if filename.startswith('/Workspace/'):
        if os.path.exists(filename):
            return filename
        else:
            print(f"❌ File not found: {filename}")
            return None
    
    # Search in common locations
    search_paths = [
        "/Workspace/Users/",
        "/Workspace/Shared/",
        "/Workspace/",
        "/databricks/driver/"
    ]
    
    for base_path in search_paths:
        # Try direct path
        full_path = os.path.join(base_path, filename)
        if os.path.exists(full_path):
            return full_path
        
        # Try recursive search
        try:
            for root, dirs, files in os.walk(base_path):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    return found_path
        except:
            continue
    
    print(f"❌ File not found: {filename}")
    return None

def load_evaluation_data():
    """Load evaluation data from CSV file."""
    eval_path = dbutils.widgets.get("evaluation_data_path")
    print(f"📊 Loading evaluation data from: {eval_path}")
    
    # Find the file
    full_path = find_file_in_workspace(eval_path)
    if not full_path:
        print(f"❌ Evaluation data file not found: {eval_path}")
        return None
    
    try:
        df = pd.read_csv(full_path)
        print(f"✅ Loaded evaluation data: {len(df)} samples")
        print(f"📋 Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading evaluation data: {e}")
        return None

def load_metrics_config():
    """Load metrics configuration from CSV file."""
    config_path = dbutils.widgets.get("metrics_config_path")
    print(f"📋 Loading metrics config from: {config_path}")
    
    # Find the file
    full_path = find_file_in_workspace(config_path)
    if not full_path:
        print(f"❌ Metrics config file not found: {config_path}")
        return None
    
    try:
        df = pd.read_csv(full_path)
        print(f"✅ Loaded metrics config: {len(df)} metrics")
        print(f"📋 Metrics: {list(df['name'])}")
        return df
    except Exception as e:
        print(f"❌ Error loading metrics config: {e}")
        return None

def load_ground_truth_for_metric(metric_row, evaluation_data_df):
    """Load ground truth data for a specific metric."""
    ground_truth_file = metric_row.get("ground_truth_file_path", "")
    ground_truth_column = metric_row.get("ground_truth_column", "ground_truth")
    
    if not ground_truth_file:
        print(f"   No ground truth file specified")
        return evaluation_data_df
    
    # Map filename to UI-specified path
    GROUND_TRUTH_PATH_MAPPING = {
        "ground_truth_accuracy.csv": dbutils.widgets.get("ground_truth_accuracy_path"),
        "ground_truth_safety.csv": dbutils.widgets.get("ground_truth_safety_path")
    }
    
    # Get the full path
    if ground_truth_file in GROUND_TRUTH_PATH_MAPPING:
        full_path = GROUND_TRUTH_PATH_MAPPING[ground_truth_file]
    else:
        full_path = find_file_in_workspace(ground_truth_file)
    
    if not full_path:
        print(f"   ❌ Ground truth file not found: {ground_truth_file}")
        print(f"   📋 Please upload {ground_truth_file} to your Databricks workspace")
        return evaluation_data_df
    
    try:
        # Load ground truth data
        gt_df = pd.read_csv(full_path)
        print(f"   📚 Loaded ground truth: {len(gt_df)} samples")
        print(f"   📋 GT columns: {list(gt_df.columns)}")
        
        # Find common column for merging
        common_cols = set(evaluation_data_df.columns) & set(gt_df.columns)
        if not common_cols:
            print(f"   ❌ No common columns found between eval data and {ground_truth_file}")
            print(f"   Eval columns: {list(evaluation_data_df.columns)}")
            print(f"   GT columns: {list(gt_df.columns)}")
            print(f"   📋 Please ensure both files have a common column (e.g., sample_id)")
            return evaluation_data_df
        
        merge_col = list(common_cols)[0]  # Use first common column
        print(f"   🔗 Merging on column: '{merge_col}'")
        
        # Merge ground truth data
        merged_df = evaluation_data_df.merge(gt_df, on=merge_col, how="left")
        
        # Find the specific ground truth column
        if ground_truth_column in merged_df.columns:
            print(f"   ✅ Found exact match for ground truth column: '{ground_truth_column}'")
        else:
            # Try to find similar column names
            similar_cols = [col for col in merged_df.columns if ground_truth_column.lower() in col.lower()]
            if similar_cols:
                ground_truth_column = similar_cols[0]
                print(f"   🔍 Using similar column: '{ground_truth_column}'")
            else:
                print(f"   ⚠️ Ground truth column '{ground_truth_column}' not found")
                print(f"   Available columns: {list(merged_df.columns)}")
                return evaluation_data_df
        
        # Calculate coverage
        coverage = (merged_df[ground_truth_column].notna().sum() / len(merged_df)) * 100
        print(f"   ✅ Ground truth coverage: {coverage:.1f}%")
        
        return merged_df
        
    except Exception as e:
        print(f"   ❌ Error loading ground truth: {e}")
        return evaluation_data_df

# COMMAND ----------

# =============================================================================
# LOAD ALL DATA
# =============================================================================

print("🚀 Starting data loading process...")

# Load evaluation data
EVALUATION_DATA = load_evaluation_data()
if EVALUATION_DATA is None:
    print("❌ Failed to load evaluation data. Please check the file path.")
    dbutils.notebook.exit("Data loading failed")

# Load metrics configuration
METRICS_CONFIG_DATA = load_metrics_config()
if METRICS_CONFIG_DATA is None:
    print("❌ Failed to load metrics configuration. Please check the file path.")
    dbutils.notebook.exit("Data loading failed")

# Load ground truth data per metric
print("\n📚 Loading ground truth data per metric...")

for idx, metric_row in METRICS_CONFIG_DATA.iterrows():
    metric_name = metric_row.get("name", f"metric_{idx}")
    ground_truth_file = metric_row.get("ground_truth_file_path", "")
    
    if ground_truth_file:
        print(f"\n📊 Processing ground truth for metric: {metric_name}")
        # CRITICAL: Pass EVALUATION_DATA (evaluation data) not METRICS_CONFIG_DATA
        EVALUATION_DATA = load_ground_truth_for_metric(metric_row, EVALUATION_DATA)
    else:
        print(f"   No ground truth file specified for metric: {metric_name}")

print(f"\n✅ Data loading complete!")
print(f"📊 Final evaluation data shape: {EVALUATION_DATA.shape}")
print(f"📋 Final columns: {list(EVALUATION_DATA.columns)}")

# Store data globally for other cells
globals()['EVALUATION_DATA'] = EVALUATION_DATA
globals()['METRICS_CONFIG_DATA'] = METRICS_CONFIG_DATA

print("💾 Data stored globally for other cells")
