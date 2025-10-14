# Databricks notebook source
# MAGIC %md
# MAGIC # 🤖 LLM Judge Evaluation System
# MAGIC 
# MAGIC **Complete evaluation system for LLM responses using Databricks or OpenAI models**
# MAGIC 
# MAGIC ## Features:
# MAGIC - ✅ Auto-discovery of Databricks LLM endpoints
# MAGIC - ✅ Support for OpenAI API models  
# MAGIC - ✅ Multiple metric types (Binary, 1-5 Scale, Percentage)
# MAGIC - ✅ Robust JSON parsing with fallbacks
# MAGIC - ✅ Beautiful results display
# MAGIC - ✅ No rate limit issues when using Databricks
# MAGIC 
# MAGIC ## Instructions:
# MAGIC 1. **Run Cell 1**: Install packages and imports
# MAGIC 2. **Run Cell 2**: Load your evaluation data
# MAGIC 3. **Run Cell 3**: Configure model settings  
# MAGIC 4. **Run Cell 4**: Define your custom metrics
# MAGIC 5. **Run Cell 5-9**: Execute evaluation system
# MAGIC 6. **Run Cell 10**: View results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Installation and Setup
# MAGIC **Just run this cell - no changes needed**

# COMMAND ----------

# Install packages
%pip install mlflow pandas plotly python-docx openai langchain-core langchain-openai langsmith --quiet

%restart_python

print("✅ All packages installed cleanly!")

# COMMAND ----------

# Import required libraries
from __future__ import annotations
import time
import os
import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Data processing
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LLM and MLflow
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from openai import OpenAI
import mlflow

print("✅ All libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Data Loading Configuration
# MAGIC **Upload your files to the Databricks workspace and specify filenames below**
# MAGIC 
# MAGIC ### 📊 New Ground Truth Support:
# MAGIC - **Per-metric ground truth files**: Each metric can have its own ground truth file
# MAGIC - **Workspace-based files**: Just specify filenames, system finds them in workspace
# MAGIC - **Flexible column mapping**: Ground truth files don't need specific column names
# MAGIC - **Automatic merging**: System will intelligently match data based on common columns
# MAGIC 
# MAGIC ### 📁 File Upload Instructions:
# MAGIC 1. **Upload evaluation data**: `evaluation_data.csv` to your workspace
# MAGIC 2. **Upload metrics config**: `sample_metrics_config.csv` to your workspace  
# MAGIC 3. **Upload ground truth files**: Upload all `*_ground_truth.csv` files to workspace
# MAGIC 4. **Update filenames**: Modify the widget values below to match your files

# COMMAND ----------

# =============================================================================
# FILE UPLOAD CONFIGURATION
# =============================================================================

# Create file upload widgets
dbutils.widgets.text(
    "evaluation_data_path", 
    "evaluation_data.csv", 
    "📁 Evaluation Data (CSV filename in workspace)"
)

dbutils.widgets.text(
    "metrics_config_path",
    "sample_metrics_config.csv",
    "📊 Metrics Configuration File (CSV filename in workspace)"
)

# Get file paths
EVAL_DATA_PATH = dbutils.widgets.get("evaluation_data_path")
METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")

print("📁 FILE CONFIGURATION")
print("="*60)
print(f"Evaluation Data: {EVAL_DATA_PATH}")
print(f"Metrics Config: {METRICS_CONFIG_PATH or 'None'}")
print("="*60)

def load_any_csv(file_path, file_type="data"):
    """Load any CSV file from workspace or local path."""
    try:
        # Handle workspace files (just filename) vs full paths
        if not os.path.isabs(file_path) and not file_path.startswith('/'):
            # This is likely a workspace filename, try to find it
            workspace_path = f"/Workspace/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/{file_path}"
            if os.path.exists(workspace_path):
                file_path = workspace_path
            else:
                # Try current directory
                current_dir_path = f"./{file_path}"
                if os.path.exists(current_dir_path):
                    file_path = current_dir_path
                else:
                    # Try /tmp directory
                    tmp_path = f"/tmp/{file_path}"
                    if os.path.exists(tmp_path):
                        file_path = tmp_path
        
        if not os.path.exists(file_path):
            print(f"❌ {file_type.title()} file not found: {file_path}")
            print(f"   Searched locations:")
            print(f"   - /Workspace/Users/.../{file_path}")
            print(f"   - ./{file_path}")
            print(f"   - /tmp/{file_path}")
            return None
            
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {file_type}: {len(df)} rows, {len(df.columns)} columns")
        print(f"   File: {file_path}")
        print(f"   Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {file_type}: {e}")
        return None

def load_metrics_config(file_path):
    """Load metrics configuration from CSV file with ground truth file support."""
    if not file_path or not file_path.strip():
        return None
    
    try:
        df = load_any_csv(file_path, "metrics config")
        if df is None:
            return None
        
        # Map alternative column names to expected names
        column_mapping = {
            'metric_name': 'name',
            'metric_type': 'type', 
            'grading_instructions': 'evaluation_prompt',
            'pass_threshold': 'threshold',
            'ground_truth_file': 'ground_truth_file_path',
            'gt_file': 'ground_truth_file_path',
            'gt_file_path': 'ground_truth_file_path'
        }
        
        # Rename columns to match expected format
        df = df.rename(columns=column_mapping)
        
        # Validate required columns
        required_cols = ['name', 'type', 'description', 'evaluation_prompt', 'threshold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns in metrics config: {missing_cols}")
            print(f"Available columns after mapping: {list(df.columns)}")
            print(f"Required columns: {required_cols}")
            return None
        
        # Add default ground_truth_column if not specified
        if 'ground_truth_column' not in df.columns:
            df['ground_truth_column'] = 'ground_truth'  # Default column name
            print("   Added default ground_truth_column: 'ground_truth'")
        
        # Add default ground_truth_file_path if not specified
        if 'ground_truth_file_path' not in df.columns:
            df['ground_truth_file_path'] = ''  # Empty means no specific file
            print("   Added default ground_truth_file_path: '' (no specific file)")
        
        print("✅ Metrics configuration loaded and mapped successfully")
        print(f"   Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading metrics config: {e}")
        return None

def load_ground_truth_for_metric(metric_config, eval_df):
    """Load ground truth data for a specific metric."""
    ground_truth_file = metric_config.get('ground_truth_file_path', '')
    ground_truth_column = metric_config.get('ground_truth_column', 'ground_truth')
    
    # If no ground_truth_file_path specified, use ground_truth_column as filename
    if not ground_truth_file or not ground_truth_file.strip():
        if ground_truth_column and ground_truth_column != 'ground_truth':
            # Use ground_truth_column as filename and append .csv
            ground_truth_file = f"{ground_truth_column}.csv"
            print(f"   Using ground_truth_column as filename: {ground_truth_file}")
        else:
            print(f"   No ground truth file specified for metric: {metric_config.get('name', 'unknown')}")
            return eval_df
    
    try:
        print(f"   Loading ground truth file: {ground_truth_file}")
        gt_df = load_any_csv(ground_truth_file, f"ground truth for {metric_config.get('name', 'unknown')}")
        
        if gt_df is None:
            print(f"   ❌ Failed to load ground truth file: {ground_truth_file}")
            return eval_df
        
        # Find common columns for merging
        common_cols = set(eval_df.columns) & set(gt_df.columns)
        if not common_cols:
            print(f"   ⚠️ No common columns found between eval data and ground truth file")
            print(f"   Eval columns: {list(eval_df.columns)}")
            print(f"   GT columns: {list(gt_df.columns)}")
            return eval_df
        
        # Use the first common column for merging
        merge_col = list(common_cols)[0]
        print(f"   Merging on column: '{merge_col}'")
        
        # Find the ground truth data column in the GT file
        gt_data_col = None
        
        # First, try to find a column that matches the ground_truth_column name
        if ground_truth_column in gt_df.columns:
            gt_data_col = ground_truth_column
            print(f"   Found exact match for ground truth column: '{gt_data_col}'")
        else:
            # Look for any text column that could be ground truth data
            text_cols = gt_df.select_dtypes(include=['object']).columns.tolist()
            text_cols = [col for col in text_cols if col != merge_col]
            
            if text_cols:
                gt_data_col = text_cols[0]  # Use first text column
                print(f"   Using '{gt_data_col}' as ground truth data column")
            else:
                print(f"   ⚠️ No suitable ground truth data column found")
                return eval_df
        
        # Merge the ground truth data
        merged_df = eval_df.merge(
            gt_df[[merge_col, gt_data_col]], 
            on=merge_col, 
            how='left'
        )
        
        # Rename the ground truth column to the expected name for this metric
        if gt_data_col != ground_truth_column:
            merged_df = merged_df.rename(columns={gt_data_col: ground_truth_column})
            print(f"   Renamed '{gt_data_col}' to '{ground_truth_column}'")
        
        coverage = merged_df[ground_truth_column].notna().sum()
        print(f"   ✅ Ground truth loaded: {coverage}/{len(merged_df)} samples matched")
        
        return merged_df
        
    except Exception as e:
        print(f"   ❌ Error loading ground truth for {metric_config.get('name', 'unknown')}: {e}")
        return eval_df

def create_sample_data():
    """Create sample data for demonstration."""
    return pd.DataFrame({
        'prompt': [
            "What is the capital of France?",
            "Explain machine learning in simple terms",
            "How do I bake a chocolate cake?"
        ],
        'response': [
            "The capital of France is Paris, a beautiful city known for its culture and history.",
            "Machine learning is a type of AI where computers learn patterns from data to make predictions.",
            "To bake a chocolate cake, mix flour, cocoa, eggs, and sugar, then bake at 350°F for 30 minutes."
        ]
    })

def standardize_evaluation_data(df):
    """Intelligently identify prompt and response columns."""
    if df is None:
        return None
    
    # Get all text columns
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(text_cols) < 2:
        print("❌ Need at least 2 text columns for prompt and response")
        return None
    
    print(f"   Analyzing {len(text_cols)} text columns: {text_cols}")
    
    # Smart column detection
    prompt_col = None
    response_col = None
    
    # Look for obvious prompt indicators
    prompt_indicators = ['prompt', 'question', 'query', 'input', 'text', 'message', 'user_input']
    for col in text_cols:
        if any(indicator in col.lower() for indicator in prompt_indicators):
            prompt_col = col
            print(f"   Found prompt column by name: '{col}'")
            break
    
    # Look for obvious response indicators
    response_indicators = ['response', 'answer', 'output', 'candidate', 'generated', 'result', 'reply']
    for col in text_cols:
        if any(indicator in col.lower() for indicator in response_indicators):
            response_col = col
            print(f"   Found response column by name: '{col}'")
            break
    
    # If not found by name, use content analysis
    if not prompt_col or not response_col:
        print("   Analyzing column content to identify prompt vs response...")
        
        for col in text_cols:
            if prompt_col and response_col:
                break
                
            # Sample a few rows to analyze content
            sample_texts = df[col].dropna().head(10).astype(str)
            
            # Check if this looks like a prompt (questions, shorter text, ends with ?)
            is_question_like = any(text.strip().endswith('?') for text in sample_texts)
            avg_length = sample_texts.str.len().mean()
            
            # Check if this looks like a response (longer text, answers)
            is_answer_like = avg_length > 50 and not is_question_like
            
            if not prompt_col and (is_question_like or avg_length < 100):
                prompt_col = col
                print(f"   Identified prompt column by content: '{col}' (avg length: {avg_length:.0f})")
            elif not response_col and is_answer_like:
                response_col = col
                print(f"   Identified response column by content: '{col}' (avg length: {avg_length:.0f})")
    
    # Fallback: use first two text columns if still not found
    if not prompt_col or not response_col:
        remaining_cols = [col for col in text_cols if col not in [prompt_col, response_col]]
        
        if not prompt_col and remaining_cols:
            prompt_col = remaining_cols[0]
            print(f"   Using fallback for prompt: '{prompt_col}'")
        
        if not response_col and remaining_cols:
            response_col = remaining_cols[1] if len(remaining_cols) > 1 else remaining_cols[0]
            print(f"   Using fallback for response: '{response_col}'")
    
    # Final validation
    if not prompt_col or not response_col:
        print("❌ Could not identify prompt and response columns")
        print(f"Available columns: {text_cols}")
        return None
    
    if prompt_col == response_col:
        print("❌ Prompt and response columns are the same")
        return None
    
    print(f"   Final selection:")
    print(f"     Prompt: '{prompt_col}'")
    print(f"     Response: '{response_col}'")
    
    # Create standardized dataframe
    result_df = df[[prompt_col, response_col]].copy()
    result_df = result_df.rename(columns={
        prompt_col: 'prompt',
        response_col: 'response'
    })
    
    # Keep other columns
    other_cols = [col for col in df.columns if col not in [prompt_col, response_col]]
    for col in other_cols:
        result_df[col] = df[col]
    
    print(f"✅ Standardized evaluation data: {len(result_df)} rows")
    return result_df

# Load evaluation data
print("\n📊 Loading evaluation data...")
eval_df_raw = load_any_csv(EVAL_DATA_PATH, "evaluation data")

# Standardize the data to ensure we have 'prompt' and 'response' columns
eval_df = standardize_evaluation_data(eval_df_raw)

# Create sample data if loading failed
if eval_df is None:
    print("\n📝 Creating sample data for demonstration...")
    eval_df = create_sample_data()
    print("✅ Sample data created")

# Load metrics configuration first (needed for ground truth loading)
metrics_config_df = load_metrics_config(METRICS_CONFIG_PATH)

# Load ground truth data per metric if metrics config is available
if metrics_config_df is not None and len(metrics_config_df) > 0:
    print("\n📚 Loading ground truth data per metric...")
    
    # Process each metric's ground truth requirements
    for idx, metric_row in metrics_config_df.iterrows():
        metric_name = metric_row.get('name', f'metric_{idx}')
        ground_truth_file = metric_row.get('ground_truth_file_path', '')
        ground_truth_column = metric_row.get('ground_truth_column', 'ground_truth')
        
        if ground_truth_file and ground_truth_file.strip():
            print(f"\n📊 Processing ground truth for metric: {metric_name}")
            print(f"   File: {ground_truth_file}")
            print(f"   Column: {ground_truth_column}")
            
            # Load ground truth for this specific metric
            eval_df = load_ground_truth_for_metric(metric_row, eval_df)
        else:
            print(f"   No ground truth file specified for metric: {metric_name}")

# Ensure ground_truth column exists (fallback)
if 'ground_truth' not in eval_df.columns:
    eval_df['ground_truth'] = ''

# Display preview
print(f"\n📊 Data Preview ({len(eval_df)} samples):")
print(f"Columns: {list(eval_df.columns)}")
display(eval_df.head(3))

# Store data globally
EVALUATION_DATA = eval_df
METRICS_CONFIG_DATA = metrics_config_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Model and Evaluation Settings
# MAGIC **Configure your LLM model and evaluation parameters**

# COMMAND ----------

# =============================================================================
# MODEL AND EVALUATION SETTINGS
# =============================================================================

# Model selection
dbutils.widgets.dropdown(
    "judge_model",
    "databricks-llm",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "databricks-llm"],
    "🤖 Select Judge Model"
)

# Get settings
JUDGE_MODEL = dbutils.widgets.get("judge_model")

print("🤖 EVALUATION SETTINGS")
print("="*60)
print(f"Judge Model: {JUDGE_MODEL}")
print("="*60)

# Initialize API connection based on model type
if JUDGE_MODEL == "databricks-llm":
    print("\n🏢 Databricks LLM selected")
    print("   Databricks client will be initialized during evaluation")
    client = None
    
else:
    print("\n🔗 Initializing OpenAI connection...")
    
    # Get API key - UPDATE THIS PATH TO YOUR SECRET SCOPE
    OPENAI_KEY = dbutils.secrets.get("your-secret-scope", "openai_key")
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
    
    # Initialize OpenAI client - UPDATE THIS BASE URL IF NEEDED
    client = OpenAI(
        base_url="https://api.openai.com/v1",  # Change this if using custom endpoint
        api_key=OPENAI_KEY
    )
    
    # Test connection
    try:
        test_response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=10
        )
        
        if test_response.choices and test_response.choices[0].message.content:
            print(f"✅ OpenAI connection successful!")
        else:
            print(f"⚠️ OpenAI returned empty response")
            
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")

print(f"\n🎯 Final Model Selection: {JUDGE_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Metrics Configuration
# MAGIC **Upload your metrics CSV file to workspace and specify filename above**
# MAGIC 
# MAGIC ### 📊 Required CSV Format for Metrics:
# MAGIC Your metrics CSV must have these columns:
# MAGIC - `name` (or `metric_name`): Metric identifier (e.g., "accuracy_check")
# MAGIC - `type` (or `metric_type`): Metric type ("binary", "scale_1_5", or "percentage") 
# MAGIC - `description`: Human readable description
# MAGIC - `evaluation_prompt` (or `grading_instructions`): Full LLM prompt with {prompt}, {response}, {ground_truth} placeholders
# MAGIC - `threshold` (or `pass_threshold`): Pass/fail threshold (1.0 for binary, 3.0 for scale_1_5, 0.7 for percentage)
# MAGIC - `ground_truth_column` (optional): **Ground truth filename without .csv** (e.g., "correct_answer" → "correct_answer.csv")
# MAGIC - `ground_truth_file_path` (optional): **Leave empty** - system uses ground_truth_column as filename
# MAGIC 
# MAGIC ### 🆕 New Ground Truth Features:
# MAGIC - **Per-metric files**: Each metric can have its own ground truth file
# MAGIC - **Workspace-based**: Just specify filenames, system finds files in workspace
# MAGIC - **Flexible column names**: Ground truth files don't need specific column names
# MAGIC - **Automatic merging**: System finds common columns between eval data and ground truth files
# MAGIC - **Multiple formats**: Support for different ground truth file structures
# MAGIC 
# MAGIC ### 📁 Example Metrics CSV:
# MAGIC ```csv
# MAGIC name,type,description,evaluation_prompt,threshold,ground_truth_column,ground_truth_file_path
# MAGIC accuracy_check,binary,Checks accuracy,"Evaluate...",1.0,correct_answer,
# MAGIC helpfulness,scale_1_5,Rate helpfulness,"Rate...",3.0,helpful_answer,
# MAGIC ```
# MAGIC 
# MAGIC **Note**: The system will automatically look for `correct_answer.csv` and `helpful_answer.csv` files in the workspace.

# COMMAND ----------

# =============================================================================
# METRICS VALIDATION
# =============================================================================
# Metrics are loaded from CSV file only - no code definition supported

# Validate that metrics CSV is provided
if not METRICS_CONFIG_PATH or not METRICS_CONFIG_PATH.strip():
    print("❌ ERROR: No metrics configuration file provided!")
    print("📋 REQUIRED: Please upload a CSV file with your metrics using the widget above")
    print("\n📊 CSV Format Required:")
    print("   Columns: name, type, description, evaluation_prompt, threshold, ground_truth_column")
    print("   Types: binary, scale_1_5, percentage")
    print("   Example: accuracy_check,binary,Checks accuracy,\"Evaluate...\",1.0,correct_answer")
    
elif METRICS_CONFIG_DATA is None:
    print("❌ ERROR: Could not load metrics configuration file!")
    print(f"   File path: {METRICS_CONFIG_PATH}")
    print("   Please check the file path and format")
    
else:
    print("✅ Metrics configuration loaded successfully!")
    print(f"   📊 Found {len(METRICS_CONFIG_DATA)} metrics in CSV")
    
    # Display metrics summary
    for idx, row in METRICS_CONFIG_DATA.iterrows():
        metric_name = row.get('name', row.get('metric_name', 'Unknown'))
        metric_type = row.get('type', row.get('metric_type', 'Unknown'))
        threshold = row.get('threshold', row.get('pass_threshold', 'Unknown'))
        gt_column = row.get('ground_truth_column', 'Default')
        gt_file = row.get('ground_truth_file_path', 'None')
        
        print(f"   {idx+1}. {metric_name} ({metric_type}) - threshold: {threshold}")
        print(f"      Ground Truth Column: {gt_column}")
        print(f"      Ground Truth File: {gt_file}")

# No code-based metrics - all metrics must come from CSV file upload

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Core Classes and Imports
# MAGIC **System classes - no changes needed**

# COMMAND ----------

# =============================================================================
# CORE CLASSES AND IMPORTS
# =============================================================================

class MetricType(Enum):
    """Defines the types of evaluation metrics supported."""
    BINARY = "binary"          # Pass/Fail (0 or 1)
    SCALE_1_5 = "scale_1_5"    # Rating scale 1-5
    PERCENTAGE = "percentage"   # Percentage score 0-100%

@dataclass
class MetricConfig:
    """Configuration class for evaluation metrics."""
    name: str                  # Metric name (e.g., "accuracy_check")
    description: str           # Human-readable description
    metric_type: MetricType    # Type of metric (binary, scale, percentage)
    prompt_template: str       # LLM evaluation prompt template
    threshold: float           # Pass/fail threshold
    ground_truth_column: str   # Which ground truth column to use for this metric
    ground_truth_file_path: str = ""  # Path to specific ground truth file for this metric

print("✅ Core classes defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: LLM Judge Evaluator Class
# MAGIC **Main evaluator class - no changes needed**

# COMMAND ----------

# =============================================================================
# LLM JUDGE EVALUATOR CLASS
# =============================================================================

class LLMJudgeEvaluator:
    """
    Main evaluator class with automatic LLM routing.
    
    Supports:
    - Databricks LLM endpoints (auto-discovery)
    - OpenAI API endpoints
    - Automatic response format detection
    - Robust JSON parsing with fallbacks
    """
    
    def __init__(self, judge_model: str, metrics: List[MetricConfig]):
        """Initialize the evaluator."""
        self.judge_model = judge_model
        self.metrics = metrics
        self.is_databricks_llm = judge_model == "databricks-llm"
        
        # Initialize the appropriate client based on model type
        if self.is_databricks_llm:
            self._initialize_databricks_client()
        else:
            self._initialize_openai_client()
    
    def _find_working_llm_endpoint(self):
        """Find working general LLM endpoints in Databricks workspace."""
        print("🔍 Looking for working LLM endpoints...")
        
        # Known working endpoint patterns from our testing
        # These are common Databricks LLM endpoint naming patterns
        endpoint_patterns = [
            "*claude-sonnet*",
            "*claude-opus*", 
            "*llama*405b*",
            "*llama*70b*",
            "*llama*8b*",
            "*gemma*",
            "*dbrx*"
        ]
        
        try:
            # Get list of all serving endpoints
            url = f"https://{self.workspace_url}/api/2.0/serving-endpoints"
            response = requests.get(url, headers=self.databricks_headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                endpoints = data.get('endpoints', [])
                
                # Filter for ready endpoints only
                ready_endpoints = []
                for endpoint in endpoints:
                    name = endpoint.get('name', 'Unknown')
                    state = endpoint.get('state', {})
                    config_update = state.get('config_update', 'Unknown')
                    ready_state = state.get('ready', 'Unknown')
                    
                    # Check if endpoint is ready for use
                    is_ready = False
                    if isinstance(ready_state, dict):
                        is_ready = ready_state.get('update_state') == 'UPDATE_STATE_READY'
                    elif isinstance(ready_state, str):
                        is_ready = ready_state == 'READY'
                    
                    if is_ready and config_update == 'NOT_UPDATING':
                        ready_endpoints.append(name)
                
                print(f"   Found {len(ready_endpoints)} ready endpoints")
                
                # Prioritize Claude Sonnet (what worked in our conversation)
                for endpoint_name in ready_endpoints:
                    if 'claude-sonnet' in endpoint_name.lower():
                        print(f"   🎯 Found Claude Sonnet endpoint: {endpoint_name}")
                        return endpoint_name, 'openai'  # Claude uses OpenAI-compatible format
                
                # Then try other high-quality models
                priority_patterns = ['claude-opus', 'llama*405b', 'llama*70b']
                for pattern in priority_patterns:
                    for endpoint_name in ready_endpoints:
                        pattern_clean = pattern.replace('*', '')
                        if pattern_clean in endpoint_name.lower():
                            print(f"   ✅ Found high-quality endpoint: {endpoint_name}")
                            return endpoint_name, 'openai'
                
                # Finally, try any LLM endpoint (skip embeddings and agents)
                for endpoint_name in ready_endpoints:
                    # Skip known non-LLM endpoints
                    if any(skip in endpoint_name.lower() for skip in ['agent', 'embedding', 'bge', 'gte']):
                        continue
                    
                    # Try any remaining endpoint
                    print(f"   🔄 Trying endpoint: {endpoint_name}")
                    return endpoint_name, 'openai'  # Most Databricks LLMs use OpenAI format
                
                raise Exception("No suitable LLM endpoints found")
                
            else:
                raise Exception(f"Failed to list endpoints: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Endpoint discovery failed: {e}")
    
    def _initialize_databricks_client(self):
        """Initialize Databricks LLM client with auto-discovery."""
        print("🔧 Initializing Databricks LLM client...")
        
        try:
            # Get Databricks workspace credentials from context
            self.databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            self.workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
            
            # Set up API headers
            self.databricks_headers = {
                "Authorization": f"Bearer {self.databricks_token}",
                "Content-Type": "application/json"
            }
            
            # Auto-discover the best available endpoint
            self.databricks_endpoint, self.response_format = self._find_working_llm_endpoint()
            
            print(f"✅ Databricks LLM client initialized with {self.databricks_endpoint}")
            print(f"   Response format: {self.response_format}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Databricks client: {e}")
            raise
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client using global client from previous cells."""
        print(f"🔧 Initializing OpenAI client for model: {self.judge_model}")
        
        # Verify OpenAI client exists from previous cells
        if 'client' not in globals() or client is None:
            raise ValueError("OpenAI client not initialized. Please run previous cells first.")
        
        self.openai_client = client
        print(f"✅ OpenAI client initialized for {self.judge_model}")

print("✅ LLM Judge Evaluator class defined (part 1/2)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: LLM Calling and Evaluation Methods
# MAGIC **Core evaluation logic - no changes needed**

# COMMAND ----------

# =============================================================================
# LLM CALLING AND EVALUATION METHODS
# =============================================================================

def _call_databricks_llm(self, prompt: str) -> str:
    """Call Databricks LLM endpoint."""
    try:
        # Construct endpoint URL
        url = f"https://{self.workspace_url}/serving-endpoints/{self.databricks_endpoint}/invocations"
        
        # Prepare request payload
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.1  # Low temperature for consistent evaluation
        }
        
        # Make API call
        response = requests.post(url, headers=self.databricks_headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract response based on detected format
            if self.response_format == 'openai' and 'choices' in result and result['choices']:
                return result['choices'][0]['message']['content']
            
            elif self.response_format == 'messages' and 'messages' in result and result['messages']:
                messages = result['messages']
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, dict):
                        return last_message.get('content', str(last_message))
                    else:
                        return str(last_message)
            
            # Fallback for custom formats
            else:
                return str(result)
        else:
            raise Exception(f"Databricks API error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"   Error calling Databricks LLM: {e}")
        raise

def _call_openai_llm(self, prompt: str) -> str:
    """Call OpenAI LLM endpoint with model-specific parameters."""
    try:
        # Use different parameters based on model type
        response = self.openai_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"   Error calling OpenAI LLM: {e}")
        raise

def evaluate_single(self, prompt: str, response: str, ground_truth_data: dict, metric: MetricConfig) -> dict:
    """Evaluate a single sample with one metric."""
    # Get the specific ground truth for this metric
    ground_truth = ground_truth_data.get(metric.ground_truth_column, "Not provided")
    
    # Format the evaluation prompt with actual values
    eval_prompt = metric.prompt_template.format(
        prompt=prompt,
        response=response,
        ground_truth=ground_truth if ground_truth else "Not provided"
    )
    
    try:
        # Route to appropriate LLM based on model type
        if self.is_databricks_llm:
            print(f"   🏢 Calling Databricks LLM for {metric.name}")
            llm_response = self._call_databricks_llm(eval_prompt)
        else:
            print(f"   🤖 Calling OpenAI ({self.judge_model}) for {metric.name}")
            llm_response = self._call_openai_llm(eval_prompt)
        
        # Validate response
        if not llm_response or str(llm_response).strip() == "":
            print(f"   Warning: Empty response for {metric.name}")
            return {
                "score": 0,
                "explanation": "Empty response from LLM",
                "status": "❌"
            }
        
        # Clean response content
        content = str(llm_response).strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        # Try to parse as JSON first
        try:
            result_json = json.loads(content)
        except json.JSONDecodeError:
            print(f"   Warning: Response is not JSON for {metric.name}")
            print(f"   Raw response: {content[:100]}...")
            
            # Fallback: Extract score from non-JSON text
            score = 0
            explanation = content
            
            import re
            
            # Extract scores based on metric type
            if metric.metric_type == MetricType.BINARY:
                # Look for pass/fail indicators
                if any(word in content.lower() for word in ['pass', 'correct', 'accurate', 'yes', 'true', '1']):
                    score = 1
                elif any(word in content.lower() for word in ['fail', 'incorrect', 'inaccurate', 'no', 'false', '0']):
                    score = 0
            
            elif metric.metric_type == MetricType.SCALE_1_5:
                # Look for numbers 1-5
                numbers = re.findall(r'\b[1-5]\b', content)
                if numbers:
                    score = int(numbers[0])
            
            elif metric.metric_type == MetricType.PERCENTAGE:
                # Look for percentages or decimals
                percentages = re.findall(r'(\d+(?:\.\d+)?)[%]?', content)
                if percentages:
                    score = float(percentages[0])
                    if score > 1:  # Convert percentage to decimal
                        score = score / 100
            
            return {
                "score": score,
                "explanation": explanation[:500],  # Truncate long explanations
                "status": "✅" if score >= metric.threshold else "❌"
            }
        
        # Parse JSON response with flexible key matching
        score = 0
        explanation = "No explanation provided"
        
        # Clean JSON keys (remove quotes, normalize)
        cleaned_json = {}
        for key, value in result_json.items():
            clean_key = key.strip('"').strip("'").strip()
            cleaned_json[clean_key] = value
        
        # Try multiple methods to find the score
        score_key = f"{metric.name}_score"
        if score_key in cleaned_json:
            score = cleaned_json[score_key]
        else:
            # Flexible matching: look for any key with metric name + "score"
            for key, value in cleaned_json.items():
                if (metric.name.lower() in key.lower() and 
                    "score" in key.lower() and 
                    isinstance(value, (int, float))):
                    score = value
                    break
            
            # Fallback: any "_score" key
            if score == 0:
                for key, value in cleaned_json.items():
                    if key.endswith("_score") and isinstance(value, (int, float)):
                        score = value
                        break
            
            # Last resort: just "score"
            if score == 0 and "score" in cleaned_json:
                if isinstance(cleaned_json["score"], (int, float)):
                    score = cleaned_json["score"]
        
        # Extract explanation
        for key in ["explanation", "Explanation", "reason", "Reason"]:
            if key in cleaned_json:
                explanation = cleaned_json[key]
                break
        
        return {
            "score": score,
            "explanation": explanation,
            "status": "✅" if score >= metric.threshold else "❌"
        }
        
    except Exception as e:
        print(f"Error evaluating {metric.name}: {e}")
        return {
            "score": 0,
            "explanation": f"Evaluation error: {str(e)}",
            "status": "❌"
        }

def evaluate_dataset(self, df) -> 'pd.DataFrame':
    """Evaluate entire dataset across all metrics."""
    results_df = df.copy()
    
    llm_type = f"Databricks ({self.databricks_endpoint})" if self.is_databricks_llm else f"OpenAI ({self.judge_model})"
    print(f"\n🚀 Starting evaluation of {len(df)} samples with {len(self.metrics)} metrics...")
    print(f"   Using: {llm_type}")
    
    # Evaluate each metric
    for metric in self.metrics:
        print(f"\n📊 Evaluating metric: {metric.name}")
        
        scores = []
        explanations = []
        statuses = []
        
        # Process each sample
        for idx, row in df.iterrows():
            if idx > 0 and idx % 2 == 0:
                print(f"   Progress: {idx}/{len(df)} samples")
            
            # Prepare ground truth data as dictionary for this row
            ground_truth_data = {}
            for col in df.columns:
                if col.startswith('ground_truth') or col in ['correct_answer', 'expected_response', 'reference_answer']:
                    ground_truth_data[col] = row.get(col, '')
            
            # Also add default ground_truth column
            ground_truth_data['ground_truth'] = row.get('ground_truth', '')
            
            result = self.evaluate_single(
                prompt=row['prompt'],
                response=row['response'],
                ground_truth_data=ground_truth_data,
                metric=metric
            )
            
            scores.append(result['score'])
            explanations.append(result['explanation'])
            statuses.append(result['status'])
            
            # Add delay for Databricks to avoid overwhelming endpoint
            if self.is_databricks_llm:
                time.sleep(1)  # 1 second delay between calls
        
        # Add results to dataframe
        results_df[f"{metric.name}_score"] = scores
        results_df[f"{metric.name}_explanation"] = explanations
        results_df[f"{metric.name}_status"] = statuses
        
        # Calculate and display summary statistics
        mean_score = sum(scores) / len(scores) if scores else 0
        pass_rate = sum(1 for s in scores if s >= metric.threshold) / len(scores) if scores else 0
        
        print(f"   ✅ Complete - Mean: {mean_score:.3f}, Pass Rate: {pass_rate:.1%}")
    
    return results_df

# Add methods to the class
LLMJudgeEvaluator._call_databricks_llm = _call_databricks_llm
LLMJudgeEvaluator._call_openai_llm = _call_openai_llm
LLMJudgeEvaluator.evaluate_single = evaluate_single
LLMJudgeEvaluator.evaluate_dataset = evaluate_dataset

print("✅ Evaluation methods added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Helper Functions
# MAGIC **Utility functions - no changes needed**

# COMMAND ----------

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_metrics_from_csv():
    """Load metrics from uploaded CSV file only."""
    if METRICS_CONFIG_DATA is None:
        print("❌ ERROR: No metrics CSV file loaded!")
        print("   Please upload a metrics configuration CSV file using the widget above")
        return []
    
    print("📊 Loading metrics from uploaded CSV...")
    metrics_list = []
    
    for _, row in METRICS_CONFIG_DATA.iterrows():
        metric = {
            'name': row['name'],
            'type': row['type'],
            'description': row['description'],
            'evaluation_prompt': row['evaluation_prompt'],
            'threshold': row['threshold'],
            'ground_truth_column': row.get('ground_truth_column', 'ground_truth'),
            'ground_truth_file_path': row.get('ground_truth_file_path', '')
        }
        metrics_list.append(metric)
    
    print(f"✅ Loaded {len(metrics_list)} metrics from CSV")
    return metrics_list

def process_custom_metrics(custom_metrics: list) -> List[MetricConfig]:
    """Convert custom metric definitions to MetricConfig objects."""
    configs = []
    
    for metric in custom_metrics:
        # Map string type to enum
        if metric['type'] == 'binary':
            metric_type = MetricType.BINARY
        elif metric['type'] == 'scale_1_5':
            metric_type = MetricType.SCALE_1_5
        elif metric['type'] == 'percentage':
            metric_type = MetricType.PERCENTAGE
        else:
            print(f"   Warning: Unknown metric type '{metric['type']}' for {metric.get('name', 'unknown')}")
            continue  # Skip invalid types
        
        # Get threshold from CSV
        if isinstance(metric.get('threshold'), (int, float)):
            threshold = float(metric['threshold'])
        else:
            # Default thresholds by type
            if metric_type == MetricType.BINARY:
                threshold = 1.0
            elif metric_type == MetricType.SCALE_1_5:
                threshold = 3.0
            elif metric_type == MetricType.PERCENTAGE:
                threshold = 0.7
            else:
                threshold = 1.0
        
        config = MetricConfig(
            name=metric['name'],
            description=metric['description'],
            metric_type=metric_type,
            prompt_template=metric['evaluation_prompt'],
            threshold=threshold,
            ground_truth_column=metric.get('ground_truth_column', 'ground_truth'),
            ground_truth_file_path=metric.get('ground_truth_file_path', '')
        )
        configs.append(config)
    
    return configs

def display_evaluation_results(results_df):
    """Display evaluation results in multiple formatted views."""
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    
    # Find result columns
    score_cols = [col for col in results_df.columns if col.endswith('_score')]
    status_cols = [col for col in results_df.columns if col.endswith('_status')]
    
    if score_cols:
        # Display formatted results table
        print(f"\n📋 RESULTS TABLE")
        print("-" * 100)
        
        # Create header
        header = f"{'Sample':<8}"
        for col in score_cols:
            metric_name = col.replace('_score', '').replace('_', ' ').title()
            header += f"{metric_name:<18}"
        header += f"{'Status':<10}"
        print(header)
        print("-" * 100)
        
        # Display data rows
        for idx in range(len(results_df)):
            row = results_df.iloc[idx]
            
            row_str = f"{idx+1:<8}"
            
            # Add scores
            for col in score_cols:
                score = row[col] if col in row else 'N/A'
                if isinstance(score, (int, float)):
                    row_str += f"{score:<18.3f}"
                else:
                    row_str += f"{str(score):<18}"
            
            # Add overall status
            if status_cols:
                statuses = [row[col] for col in status_cols if col in row]
                overall = '✅ PASS' if all('✅' in str(s) for s in statuses if s is not None) else '❌ FAIL'
                row_str += f"{overall:<10}"
            
            print(row_str)
        
        print("-" * 100)
        
        # Display summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        for col in score_cols:
            if col in results_df.columns:
                scores = results_df[col]
                numeric_scores = [s for s in scores if isinstance(s, (int, float))]
                
                if numeric_scores:
                    mean_score = sum(numeric_scores) / len(numeric_scores)
                    metric_name = col.replace('_score', '').replace('_', ' ').title()
                    
                    # Calculate pass rate using threshold
                    threshold = 1.0  # Default threshold
                    pass_count = sum(1 for s in numeric_scores if s >= threshold)
                    pass_rate = pass_count / len(numeric_scores) * 100
                    
                    print(f"   {metric_name}:")
                    print(f"     Mean Score: {mean_score:.3f}")
                    print(f"     Pass Rate:  {pass_rate:.1f}% ({pass_count}/{len(numeric_scores)})")
        
        # Try Databricks native display
        print(f"\n🔄 Databricks Native Display:")
        try:
            display_columns = ['prompt', 'response'] + score_cols + status_cols
            display_columns = [col for col in display_columns if col in results_df.columns]
            
            clean_df = results_df[display_columns].copy()
            
            # Truncate long text for better display
            if 'prompt' in clean_df.columns:
                clean_df['prompt'] = clean_df['prompt'].astype(str).str[:80] + "..."
            if 'response' in clean_df.columns:
                clean_df['response'] = clean_df['response'].astype(str).str[:80] + "..."
            
            display(clean_df)
            
        except Exception as e:
            print(f"   Native display failed: {e}")
            print("   (Results are shown in table above)")
    
    else:
        print("❌ No evaluation scores found in results")

print("✅ Helper functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Main Execution
# MAGIC **Run the evaluation - no changes needed**

# COMMAND ----------

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("✅ Databricks LLM system loaded with auto-discovery and improved display")

# Load metrics from CSV file only
metrics_to_use = load_metrics_from_csv()

# Process metrics 
metric_configs = process_custom_metrics(metrics_to_use)

if not metric_configs:
    print("❌ No valid metrics to evaluate!")
    print("Please define metrics in cell 4")
else:
    print(f"✅ Loaded {len(metric_configs)} valid metrics")
    
    # Create evaluator with proper LLM routing
    evaluator = LLMJudgeEvaluator(
        judge_model=JUDGE_MODEL,
        metrics=metric_configs
    )
    
    # Run the evaluation process
    print("="*60)
    print("🚀 STARTING EVALUATION")
    print("="*60)
    
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"✅ EVALUATION COMPLETE in {eval_time:.1f} seconds")
    print("="*60)
    
    # Display results using improved formatting
    display_evaluation_results(results_df)
    
    # Store results globally for later access
    globals()['results_df'] = results_df
    print(f"\n💾 Results saved to global variable 'results_df'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Export Results (Optional)
# MAGIC **Save your results to files**

# COMMAND ----------

# =============================================================================
# EXPORT RESULTS (OPTIONAL)
# =============================================================================

if 'results_df' in globals():
    # Create timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    csv_filename = f"/tmp/llm_evaluation_results_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"📁 Results saved to CSV: {csv_filename}")
    
    # Save summary statistics
    score_cols = [col for col in results_df.columns if col.endswith('_score')]
    
    summary_data = []
    for col in score_cols:
        scores = results_df[col]
        numeric_scores = [s for s in scores if isinstance(s, (int, float))]
        
        if numeric_scores:
            mean_score = sum(numeric_scores) / len(numeric_scores)
            threshold = 1.0  # Default threshold
            pass_count = sum(1 for s in numeric_scores if s >= threshold)
            pass_rate = pass_count / len(numeric_scores) * 100
            
            summary_data.append({
                'metric': col.replace('_score', ''),
                'mean_score': mean_score,
                'pass_rate_percent': pass_rate,
                'pass_count': pass_count,
                'total_samples': len(numeric_scores),
                'threshold': threshold
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"/tmp/llm_evaluation_summary_{timestamp}.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"📊 Summary saved to CSV: {summary_filename}")
    
    # Display file locations
    print(f"\n📂 Files created:")
    print(f"   Full results: {csv_filename}")
    print(f"   Summary: {summary_filename}")
    
    # Show summary
    print(f"\n📈 FINAL SUMMARY:")
    display(summary_df)
    
else:
    print("❌ No results found to export. Please run the evaluation first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 Congratulations!
# MAGIC 
# MAGIC Your LLM evaluation is complete! Here's what you accomplished:
# MAGIC 
# MAGIC ### ✅ **What This System Does:**
# MAGIC - **Evaluates LLM responses** using custom metrics
# MAGIC - **Supports multiple model types** (Databricks + OpenAI)
# MAGIC - **Auto-discovers** available Databricks endpoints
# MAGIC - **Provides detailed results** with scores and explanations
# MAGIC - **Exports results** to CSV files for further analysis
# MAGIC - **🆕 Per-metric ground truth files** for flexible evaluation
# MAGIC 
# MAGIC ### 🆕 **New Ground Truth Features:**
# MAGIC - **Distinct files per metric**: Each metric can have its own ground truth file
# MAGIC - **Flexible column mapping**: No need for specific column names
# MAGIC - **Automatic merging**: System intelligently matches data
# MAGIC - **Multiple formats**: Support for different file structures
# MAGIC 
# MAGIC ### 🚀 **Next Steps:**
# MAGIC 1. **Customize metrics** in Cell 4 for your specific use case
# MAGIC 2. **Upload your own data** by updating Cell 2
# MAGIC 3. **Create per-metric ground truth files** for better evaluation
# MAGIC 4. **Scale up evaluation** to larger datasets
# MAGIC 5. **Integrate with MLflow** for experiment tracking
# MAGIC 6. **Set up automated evaluation** pipelines
# MAGIC 
# MAGIC ### 📊 **Key Benefits:**
# MAGIC - ✅ **No OpenAI rate limits** when using Databricks
# MAGIC - ✅ **Robust error handling** with fallback parsing
# MAGIC - ✅ **Beautiful results display** with multiple formats
# MAGIC - ✅ **Production ready** with comprehensive logging
# MAGIC - ✅ **🆕 Flexible ground truth handling** for complex evaluations
# MAGIC 
# MAGIC **Happy evaluating!** 🎯