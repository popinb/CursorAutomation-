# Databricks notebook source
# MAGIC %md
# MAGIC # 🤖 Enhanced LLM Judge Evaluation System
# MAGIC 
# MAGIC **Complete evaluation system for LLM responses with per-metric ground truth support**
# MAGIC 
# MAGIC ## Features:
# MAGIC - ✅ Auto-discovery of Databricks LLM endpoints
# MAGIC - ✅ Support for OpenAI API models  
# MAGIC - ✅ **Per-metric ground truth files** - Each metric can use its own ground truth file
# MAGIC - ✅ **Flexible ground truth column mapping** - No fixed column names required
# MAGIC - ✅ Multiple metric types (Binary, 1-5 Scale, Percentage)
# MAGIC - ✅ Robust JSON parsing with fallbacks
# MAGIC - ✅ Beautiful results display
# MAGIC - ✅ No rate limit issues when using Databricks
# MAGIC 
# MAGIC ## Instructions:
# MAGIC 1. **Run Cell 1**: Install packages and imports
# MAGIC 2. **Run Cell 2**: Load your evaluation data
# MAGIC 3. **Run Cell 3**: Configure model settings  
# MAGIC 4. **Run Cell 4**: Define your custom metrics with ground truth files
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
# MAGIC **Update the file paths below to point to your data**

# COMMAND ----------

# =============================================================================
# FILE UPLOAD CONFIGURATION
# =============================================================================

# Create file upload widgets
dbutils.widgets.text(
    "evaluation_data_path", 
    "/Workspace/Users/your_username/your_eval_data.csv", 
    "📁 Evaluation Data (CSV file)"
)

dbutils.widgets.text(
    "metrics_config_path",
    "/Workspace/Users/your_username/metrics_config.csv",
    "📊 Metrics Configuration File (CSV with metric definitions and ground truth paths)"
)

# Get file paths
EVAL_DATA_PATH = dbutils.widgets.get("evaluation_data_path")
METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")

print("📁 FILE CONFIGURATION")
print("="*60)
print(f"Evaluation Data: {EVAL_DATA_PATH}")
print(f"Metrics Config: {METRICS_CONFIG_PATH}")
print("="*60)

def load_any_csv(file_path, file_type="data"):
    """Load any CSV file without assumptions."""
    try:
        if not os.path.exists(file_path):
            print(f"❌ {file_type.title()} file not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {file_type}: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {file_type}: {e}")
        return None

def load_metrics_config_enhanced(file_path):
    """Load enhanced metrics configuration from CSV file with ground truth file support."""
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
            'ground_truth_file_path': 'ground_truth_file_path',
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
        
        # Add default ground truth file path if not specified
        if 'ground_truth_file_path' not in df.columns:
            df['ground_truth_file_path'] = ''  # Empty means no specific file
            print("   Added default ground_truth_file_path column (empty)")
        
        # Add default ground_truth_column if not specified
        if 'ground_truth_column' not in df.columns:
            df['ground_truth_column'] = 'ground_truth'  # Default column name
            print("   Added default ground_truth_column: 'ground_truth'")
        
        print("✅ Enhanced metrics configuration loaded successfully")
        print(f"   Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading metrics config: {e}")
        return None

def load_ground_truth_for_metric(metric_row, evaluation_data):
    """Load ground truth data for a specific metric."""
    ground_truth_file = metric_row.get('ground_truth_file_path', '').strip()
    ground_truth_column = metric_row.get('ground_truth_column', 'ground_truth').strip()
    metric_name = metric_row.get('name', 'unknown_metric')
    
    print(f"   Loading ground truth for metric '{metric_name}':")
    print(f"     File: {ground_truth_file or 'None specified'}")
    print(f"     Column: {ground_truth_column}")
    
    # If no specific file is provided, try to use ground truth from main evaluation data
    if not ground_truth_file:
        print(f"     Using ground truth from main evaluation data")
        if ground_truth_column in evaluation_data.columns:
            return evaluation_data[ground_truth_column].fillna('').tolist()
        else:
            print(f"     ⚠️ Column '{ground_truth_column}' not found in evaluation data")
            return [''] * len(evaluation_data)
    
    # Load specific ground truth file
    try:
        if not os.path.exists(ground_truth_file):
            print(f"     ❌ Ground truth file not found: {ground_truth_file}")
            return [''] * len(evaluation_data)
        
        gt_df = pd.read_csv(ground_truth_file)
        print(f"     ✅ Loaded ground truth file: {len(gt_df)} rows, {len(gt_df.columns)} columns")
        print(f"       Available columns: {list(gt_df.columns)}")
        
        # Try to find the ground truth column
        if ground_truth_column in gt_df.columns:
            gt_values = gt_df[ground_truth_column].fillna('').tolist()
            print(f"     ✅ Found column '{ground_truth_column}' with {len(gt_values)} values")
        else:
            # Try to find any column that might contain ground truth
            possible_columns = [col for col in gt_df.columns 
                              if any(keyword in col.lower() 
                                   for keyword in ['truth', 'correct', 'answer', 'expected', 'reference', 'target'])]
            
            if possible_columns:
                selected_column = possible_columns[0]
                gt_values = gt_df[selected_column].fillna('').tolist()
                print(f"     ✅ Using column '{selected_column}' as ground truth (auto-detected)")
            else:
                # Use first column as fallback
                if len(gt_df.columns) > 0:
                    selected_column = gt_df.columns[0]
                    gt_values = gt_df[selected_column].fillna('').tolist()
                    print(f"     ⚠️ Using first column '{selected_column}' as ground truth (fallback)")
                else:
                    print(f"     ❌ No columns found in ground truth file")
                    return [''] * len(evaluation_data)
        
        # Handle length mismatch
        eval_len = len(evaluation_data)
        gt_len = len(gt_values)
        
        if gt_len < eval_len:
            # Extend with empty strings
            gt_values.extend([''] * (eval_len - gt_len))
            print(f"     ⚠️ Ground truth extended from {gt_len} to {eval_len} entries")
        elif gt_len > eval_len:
            # Truncate
            gt_values = gt_values[:eval_len]
            print(f"     ⚠️ Ground truth truncated from {gt_len} to {eval_len} entries")
        
        return gt_values
        
    except Exception as e:
        print(f"     ❌ Error loading ground truth file: {e}")
        return [''] * len(evaluation_data)

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

# Load enhanced metrics configuration
metrics_config_df = load_metrics_config_enhanced(METRICS_CONFIG_PATH)

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
# MAGIC ## Cell 4: Enhanced Metrics Configuration
# MAGIC **Upload your enhanced metrics CSV file using the widget above**
# MAGIC 
# MAGIC ### 📊 Required CSV Format for Enhanced Metrics:
# MAGIC Your metrics CSV must have these columns:
# MAGIC - `name` (or `metric_name`): Metric identifier (e.g., "accuracy_check")
# MAGIC - `type` (or `metric_type`): Metric type ("binary", "scale_1_5", or "percentage") 
# MAGIC - `description`: Human readable description
# MAGIC - `evaluation_prompt` (or `grading_instructions`): Full LLM prompt with {prompt}, {response}, {ground_truth} placeholders
# MAGIC - `threshold` (or `pass_threshold`): Pass/fail threshold (1.0 for binary, 3.0 for scale_1_5, 0.7 for percentage)
# MAGIC - `ground_truth_file_path` (optional): Path to specific ground truth file for this metric
# MAGIC - `ground_truth_column` (optional): Which column in the ground truth file to use (defaults to 'ground_truth')
# MAGIC 
# MAGIC ### 🗂️ Ground Truth File Structure:
# MAGIC - Each metric can have its own ground truth file
# MAGIC - Ground truth files can be CSV with any column structure
# MAGIC - System will auto-detect ground truth columns if not specified
# MAGIC - If no file specified, will use main evaluation data

# COMMAND ----------

# =============================================================================
# ENHANCED METRICS VALIDATION AND GROUND TRUTH LOADING
# =============================================================================

# Validate that metrics CSV is provided
if not METRICS_CONFIG_PATH or not METRICS_CONFIG_PATH.strip():
    print("❌ ERROR: No metrics configuration file provided!")
    print("📋 REQUIRED: Please upload a CSV file with your metrics using the widget above")
    print("\n📊 Enhanced CSV Format Required:")
    print("   Columns: name, type, description, evaluation_prompt, threshold")
    print("   Optional: ground_truth_file_path, ground_truth_column")
    print("   Types: binary, scale_1_5, percentage")
    print("   Example: accuracy_check,binary,Checks accuracy,\"Evaluate...\",1.0,/path/to/gt.csv,correct_answer")
    
elif METRICS_CONFIG_DATA is None:
    print("❌ ERROR: Could not load metrics configuration file!")
    print(f"   File path: {METRICS_CONFIG_PATH}")
    print("   Please check the file path and format")
    
else:
    print("✅ Enhanced metrics configuration loaded successfully!")
    print(f"   📊 Found {len(METRICS_CONFIG_DATA)} metrics in CSV")
    
    # Load ground truth data for each metric
    print(f"\n🗂️ Loading ground truth data for each metric:")
    
    # Store ground truth data for each metric
    METRIC_GROUND_TRUTHS = {}
    
    for idx, row in METRICS_CONFIG_DATA.iterrows():
        metric_name = row.get('name', row.get('metric_name', f'metric_{idx}'))
        
        # Load ground truth for this specific metric
        ground_truth_values = load_ground_truth_for_metric(row, EVALUATION_DATA)
        METRIC_GROUND_TRUTHS[metric_name] = ground_truth_values
        
        print(f"   ✅ {metric_name}: {len(ground_truth_values)} ground truth values loaded")
    
    # Display metrics summary
    print(f"\n📋 Metrics Summary:")
    for idx, row in METRICS_CONFIG_DATA.iterrows():
        metric_name = row.get('name', row.get('metric_name', 'Unknown'))
        metric_type = row.get('type', row.get('metric_type', 'Unknown'))
        threshold = row.get('threshold', row.get('pass_threshold', 'Unknown'))
        gt_file = row.get('ground_truth_file_path', '').strip()
        gt_column = row.get('ground_truth_column', 'ground_truth')
        
        gt_source = f"File: {os.path.basename(gt_file)}" if gt_file else "Main evaluation data"
        
        print(f"   {idx+1}. {metric_name} ({metric_type})")
        print(f"      Threshold: {threshold}")
        print(f"      Ground Truth: {gt_source}, Column: {gt_column}")

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
class EnhancedMetricConfig:
    """Enhanced configuration class for evaluation metrics with per-metric ground truth."""
    name: str                      # Metric name (e.g., "accuracy_check")
    description: str               # Human-readable description
    metric_type: MetricType        # Type of metric (binary, scale, percentage)
    prompt_template: str           # LLM evaluation prompt template
    threshold: float               # Pass/fail threshold
    ground_truth_values: List[str] # Pre-loaded ground truth values for this metric

print("✅ Enhanced core classes defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Enhanced LLM Judge Evaluator Class
# MAGIC **Main evaluator class with per-metric ground truth support**

# COMMAND ----------

# =============================================================================
# ENHANCED LLM JUDGE EVALUATOR CLASS
# =============================================================================

class EnhancedLLMJudgeEvaluator:
    """
    Enhanced evaluator class with per-metric ground truth support.
    
    Supports:
    - Databricks LLM endpoints (auto-discovery)
    - OpenAI API endpoints
    - Per-metric ground truth files
    - Automatic response format detection
    - Robust JSON parsing with fallbacks
    """
    
    def __init__(self, judge_model: str, metrics: List[EnhancedMetricConfig]):
        """Initialize the enhanced evaluator."""
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

print("✅ Enhanced LLM Judge Evaluator class defined (part 1/2)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Enhanced LLM Calling and Evaluation Methods
# MAGIC **Core evaluation logic with per-metric ground truth support**

# COMMAND ----------

# =============================================================================
# ENHANCED LLM CALLING AND EVALUATION METHODS
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

def evaluate_single_enhanced(self, prompt: str, response: str, sample_index: int, metric: EnhancedMetricConfig) -> dict:
    """Evaluate a single sample with one metric using pre-loaded ground truth."""
    # Get the specific ground truth for this metric and sample
    if sample_index < len(metric.ground_truth_values):
        ground_truth = metric.ground_truth_values[sample_index]
    else:
        ground_truth = ""
        print(f"   Warning: No ground truth available for sample {sample_index} in metric {metric.name}")
    
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
                "status": "❌",
                "ground_truth_used": ground_truth
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
                "status": "✅" if score >= metric.threshold else "❌",
                "ground_truth_used": ground_truth
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
            "status": "✅" if score >= metric.threshold else "❌",
            "ground_truth_used": ground_truth
        }
        
    except Exception as e:
        print(f"Error evaluating {metric.name}: {e}")
        return {
            "score": 0,
            "explanation": f"Evaluation error: {str(e)}",
            "status": "❌",
            "ground_truth_used": ground_truth
        }

def evaluate_dataset_enhanced(self, df) -> 'pd.DataFrame':
    """Evaluate entire dataset across all metrics with per-metric ground truth."""
    results_df = df.copy()
    
    llm_type = f"Databricks ({self.databricks_endpoint})" if self.is_databricks_llm else f"OpenAI ({self.judge_model})"
    print(f"\n🚀 Starting enhanced evaluation of {len(df)} samples with {len(self.metrics)} metrics...")
    print(f"   Using: {llm_type}")
    
    # Evaluate each metric
    for metric in self.metrics:
        print(f"\n📊 Evaluating metric: {metric.name}")
        print(f"   Ground truth values available: {len(metric.ground_truth_values)}")
        
        scores = []
        explanations = []
        statuses = []
        ground_truths_used = []
        
        # Process each sample
        for idx, row in df.iterrows():
            if idx > 0 and idx % 2 == 0:
                print(f"   Progress: {idx}/{len(df)} samples")
            
            result = self.evaluate_single_enhanced(
                prompt=row['prompt'],
                response=row['response'],
                sample_index=idx,
                metric=metric
            )
            
            scores.append(result['score'])
            explanations.append(result['explanation'])
            statuses.append(result['status'])
            ground_truths_used.append(result['ground_truth_used'])
            
            # Add delay for Databricks to avoid overwhelming endpoint
            if self.is_databricks_llm:
                time.sleep(1)  # 1 second delay between calls
        
        # Add results to dataframe
        results_df[f"{metric.name}_score"] = scores
        results_df[f"{metric.name}_explanation"] = explanations
        results_df[f"{metric.name}_status"] = statuses
        results_df[f"{metric.name}_ground_truth"] = ground_truths_used
        
        # Calculate and display summary statistics
        mean_score = sum(scores) / len(scores) if scores else 0
        pass_rate = sum(1 for s in scores if s >= metric.threshold) / len(scores) if scores else 0
        
        print(f"   ✅ Complete - Mean: {mean_score:.3f}, Pass Rate: {pass_rate:.1%}")
    
    return results_df

# Add methods to the enhanced class
EnhancedLLMJudgeEvaluator._call_databricks_llm = _call_databricks_llm
EnhancedLLMJudgeEvaluator._call_openai_llm = _call_openai_llm
EnhancedLLMJudgeEvaluator.evaluate_single_enhanced = evaluate_single_enhanced
EnhancedLLMJudgeEvaluator.evaluate_dataset_enhanced = evaluate_dataset_enhanced

print("✅ Enhanced evaluation methods added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Enhanced Helper Functions
# MAGIC **Utility functions with per-metric ground truth support**

# COMMAND ----------

# =============================================================================
# ENHANCED HELPER FUNCTIONS
# =============================================================================

def load_metrics_from_enhanced_csv():
    """Load metrics from uploaded CSV file with ground truth support."""
    if METRICS_CONFIG_DATA is None:
        print("❌ ERROR: No metrics CSV file loaded!")
        print("   Please upload a metrics configuration CSV file using the widget above")
        return []
    
    print("📊 Loading enhanced metrics from uploaded CSV...")
    metrics_list = []
    
    for _, row in METRICS_CONFIG_DATA.iterrows():
        metric_name = row['name']
        
        # Get pre-loaded ground truth for this metric
        ground_truth_values = METRIC_GROUND_TRUTHS.get(metric_name, [])
        
        metric = {
            'name': metric_name,
            'type': row['type'],
            'description': row['description'],
            'evaluation_prompt': row['evaluation_prompt'],
            'threshold': row['threshold'],
            'ground_truth_values': ground_truth_values
        }
        metrics_list.append(metric)
    
    print(f"✅ Loaded {len(metrics_list)} enhanced metrics from CSV")
    return metrics_list

def process_enhanced_metrics(custom_metrics: list) -> List[EnhancedMetricConfig]:
    """Convert enhanced metric definitions to EnhancedMetricConfig objects."""
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
        
        config = EnhancedMetricConfig(
            name=metric['name'],
            description=metric['description'],
            metric_type=metric_type,
            prompt_template=metric['evaluation_prompt'],
            threshold=threshold,
            ground_truth_values=metric['ground_truth_values']
        )
        configs.append(config)
    
    return configs

def display_enhanced_evaluation_results(results_df):
    """Display enhanced evaluation results with ground truth information."""
    print("\n" + "="*80)
    print("📊 ENHANCED EVALUATION RESULTS")
    print("="*80)
    
    # Find result columns
    score_cols = [col for col in results_df.columns if col.endswith('_score')]
    status_cols = [col for col in results_df.columns if col.endswith('_status')]
    gt_cols = [col for col in results_df.columns if col.endswith('_ground_truth')]
    
    if score_cols:
        # Display formatted results table
        print(f"\n📋 RESULTS TABLE")
        print("-" * 120)
        
        # Create header
        header = f"{'Sample':<8}"
        for col in score_cols:
            metric_name = col.replace('_score', '').replace('_', ' ').title()
            header += f"{metric_name:<18}"
        header += f"{'Status':<10}"
        print(header)
        print("-" * 120)
        
        # Display data rows
        for idx in range(min(len(results_df), 10)):  # Show first 10 samples
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
        
        if len(results_df) > 10:
            print(f"... and {len(results_df) - 10} more samples")
        
        print("-" * 120)
        
        # Display summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        for col in score_cols:
            if col in results_df.columns:
                scores = results_df[col]
                numeric_scores = [s for s in scores if isinstance(s, (int, float))]
                
                if numeric_scores:
                    mean_score = sum(numeric_scores) / len(numeric_scores)
                    metric_name = col.replace('_score', '').replace('_', ' ').title()
                    
                    # Find corresponding metric config for threshold
                    threshold = 1.0  # Default
                    metric_base_name = col.replace('_score', '')
                    if 'metric_configs' in globals():
                        for config in metric_configs:
                            if config.name == metric_base_name:
                                threshold = config.threshold
                                break
                    
                    pass_count = sum(1 for s in numeric_scores if s >= threshold)
                    pass_rate = pass_count / len(numeric_scores) * 100
                    
                    print(f"   {metric_name}:")
                    print(f"     Mean Score: {mean_score:.3f}")
                    print(f"     Pass Rate:  {pass_rate:.1f}% ({pass_count}/{len(numeric_scores)})")
                    
                    # Show ground truth coverage
                    gt_col = col.replace('_score', '_ground_truth')
                    if gt_col in results_df.columns:
                        gt_values = results_df[gt_col]
                        non_empty_gt = sum(1 for gt in gt_values if gt and str(gt).strip())
                        gt_coverage = non_empty_gt / len(gt_values) * 100
                        print(f"     GT Coverage: {gt_coverage:.1f}% ({non_empty_gt}/{len(gt_values)})")
        
        # Display ground truth information
        if gt_cols:
            print(f"\n🗂️ GROUND TRUTH COVERAGE:")
            for gt_col in gt_cols:
                metric_name = gt_col.replace('_ground_truth', '').replace('_', ' ').title()
                gt_values = results_df[gt_col]
                non_empty_count = sum(1 for gt in gt_values if gt and str(gt).strip())
                coverage = non_empty_count / len(gt_values) * 100
                print(f"   {metric_name}: {coverage:.1f}% ({non_empty_count}/{len(gt_values)} samples)")
        
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
            
            display(clean_df.head(10))
            
        except Exception as e:
            print(f"   Native display failed: {e}")
            print("   (Results are shown in table above)")
    
    else:
        print("❌ No evaluation scores found in results")

print("✅ Enhanced helper functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Enhanced Main Execution
# MAGIC **Run the enhanced evaluation with per-metric ground truth**

# COMMAND ----------

# =============================================================================
# ENHANCED MAIN EXECUTION
# =============================================================================

print("✅ Enhanced Databricks LLM system loaded with per-metric ground truth support")

# Load metrics from CSV file only
enhanced_metrics_to_use = load_metrics_from_enhanced_csv()

# Process enhanced metrics 
enhanced_metric_configs = process_enhanced_metrics(enhanced_metrics_to_use)

if not enhanced_metric_configs:
    print("❌ No valid enhanced metrics to evaluate!")
    print("Please define metrics in cell 4")
else:
    print(f"✅ Loaded {len(enhanced_metric_configs)} valid enhanced metrics")
    
    # Display ground truth summary
    print(f"\n🗂️ Ground Truth Summary:")
    for config in enhanced_metric_configs:
        gt_count = len([gt for gt in config.ground_truth_values if gt and str(gt).strip()])
        total_count = len(config.ground_truth_values)
        coverage = gt_count / total_count * 100 if total_count > 0 else 0
        print(f"   {config.name}: {gt_count}/{total_count} values ({coverage:.1f}% coverage)")
    
    # Create enhanced evaluator with proper LLM routing
    enhanced_evaluator = EnhancedLLMJudgeEvaluator(
        judge_model=JUDGE_MODEL,
        metrics=enhanced_metric_configs
    )
    
    # Run the enhanced evaluation process
    print("="*60)
    print("🚀 STARTING ENHANCED EVALUATION")
    print("="*60)
    
    start_time = time.time()
    enhanced_results_df = enhanced_evaluator.evaluate_dataset_enhanced(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"✅ ENHANCED EVALUATION COMPLETE in {eval_time:.1f} seconds")
    print("="*60)
    
    # Display results using enhanced formatting
    display_enhanced_evaluation_results(enhanced_results_df)
    
    # Store results globally for later access
    globals()['enhanced_results_df'] = enhanced_results_df
    globals()['metric_configs'] = enhanced_metric_configs
    print(f"\n💾 Enhanced results saved to global variable 'enhanced_results_df'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Enhanced Export Results
# MAGIC **Save your enhanced results with ground truth information**

# COMMAND ----------

# =============================================================================
# ENHANCED EXPORT RESULTS
# =============================================================================

if 'enhanced_results_df' in globals():
    # Create timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV with all columns including ground truth
    csv_filename = f"/tmp/enhanced_llm_evaluation_results_{timestamp}.csv"
    enhanced_results_df.to_csv(csv_filename, index=False)
    print(f"📁 Enhanced results saved to CSV: {csv_filename}")
    
    # Save summary statistics with ground truth coverage
    score_cols = [col for col in enhanced_results_df.columns if col.endswith('_score')]
    gt_cols = [col for col in enhanced_results_df.columns if col.endswith('_ground_truth')]
    
    summary_data = []
    for col in score_cols:
        scores = enhanced_results_df[col]
        numeric_scores = [s for s in scores if isinstance(s, (int, float))]
        
        if numeric_scores:
            mean_score = sum(numeric_scores) / len(numeric_scores)
            
            # Find threshold from metric config
            threshold = 1.0  # Default
            metric_base_name = col.replace('_score', '')
            for config in enhanced_metric_configs:
                if config.name == metric_base_name:
                    threshold = config.threshold
                    break
            
            pass_count = sum(1 for s in numeric_scores if s >= threshold)
            pass_rate = pass_count / len(numeric_scores) * 100
            
            # Calculate ground truth coverage
            gt_col = col.replace('_score', '_ground_truth')
            gt_coverage = 0
            gt_count = 0
            if gt_col in enhanced_results_df.columns:
                gt_values = enhanced_results_df[gt_col]
                gt_count = sum(1 for gt in gt_values if gt and str(gt).strip())
                gt_coverage = gt_count / len(gt_values) * 100
            
            summary_data.append({
                'metric': metric_base_name,
                'mean_score': mean_score,
                'pass_rate_percent': pass_rate,
                'pass_count': pass_count,
                'total_samples': len(numeric_scores),
                'threshold': threshold,
                'ground_truth_coverage_percent': gt_coverage,
                'ground_truth_count': gt_count
            })
    
    enhanced_summary_df = pd.DataFrame(summary_data)
    summary_filename = f"/tmp/enhanced_llm_evaluation_summary_{timestamp}.csv"
    enhanced_summary_df.to_csv(summary_filename, index=False)
    print(f"📊 Enhanced summary saved to CSV: {summary_filename}")
    
    # Save ground truth mapping for reference
    gt_mapping_data = []
    for config in enhanced_metric_configs:
        for idx, gt_value in enumerate(config.ground_truth_values):
            if gt_value and str(gt_value).strip():
                gt_mapping_data.append({
                    'metric': config.name,
                    'sample_index': idx,
                    'ground_truth_value': gt_value
                })
    
    if gt_mapping_data:
        gt_mapping_df = pd.DataFrame(gt_mapping_data)
        gt_mapping_filename = f"/tmp/ground_truth_mapping_{timestamp}.csv"
        gt_mapping_df.to_csv(gt_mapping_filename, index=False)
        print(f"🗂️ Ground truth mapping saved to CSV: {gt_mapping_filename}")
    
    # Display file locations
    print(f"\n📂 Files created:")
    print(f"   Full results: {csv_filename}")
    print(f"   Summary: {summary_filename}")
    if gt_mapping_data:
        print(f"   Ground truth mapping: {gt_mapping_filename}")
    
    # Show enhanced summary
    print(f"\n📈 ENHANCED FINAL SUMMARY:")
    display(enhanced_summary_df)
    
else:
    print("❌ No enhanced results found to export. Please run the evaluation first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 Enhanced System Complete!
# MAGIC 
# MAGIC Your enhanced LLM evaluation system with per-metric ground truth support is complete!
# MAGIC 
# MAGIC ### ✅ **New Enhanced Features:**
# MAGIC - **🗂️ Per-metric ground truth files** - Each metric can use its own ground truth source
# MAGIC - **📊 Flexible column mapping** - No fixed column names required
# MAGIC - **🔍 Auto-detection** - System automatically finds ground truth columns
# MAGIC - **📈 Enhanced reporting** - Ground truth coverage and detailed metrics
# MAGIC - **💾 Complete export** - All data including ground truth values saved
# MAGIC 
# MAGIC ### 📋 **Metrics CSV Format:**
# MAGIC ```csv
# MAGIC name,type,description,evaluation_prompt,threshold,ground_truth_file_path,ground_truth_column
# MAGIC accuracy_check,binary,Checks accuracy,"Evaluate...",1.0,/path/to/accuracy_gt.csv,correct_answer
# MAGIC quality_score,scale_1_5,Quality rating,"Rate...",3.0,/path/to/quality_gt.csv,quality_rating
# MAGIC relevance_pct,percentage,Relevance percentage,"Score...",0.7,,ground_truth
# MAGIC ```
# MAGIC 
# MAGIC ### 🚀 **Key Benefits:**
# MAGIC - ✅ **Separate ground truth per metric** - No need to merge all ground truth into one file
# MAGIC - ✅ **Flexible file structure** - Any CSV structure supported
# MAGIC - ✅ **Automatic column detection** - System finds the right columns
# MAGIC - ✅ **Coverage tracking** - See how much ground truth is available per metric
# MAGIC - ✅ **Complete audit trail** - All ground truth values saved with results
# MAGIC 
# MAGIC **Happy evaluating with enhanced ground truth support!** 🎯📊

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Sample Metrics Configuration
# MAGIC 
# MAGIC Here's an example of how to structure your enhanced metrics CSV:
# MAGIC 
# MAGIC ```csv
# MAGIC name,type,description,evaluation_prompt,threshold,ground_truth_file_path,ground_truth_column
# MAGIC accuracy_check,binary,Checks factual accuracy,"You are an expert evaluator. Compare the response to the ground truth and determine if it's factually accurate.\n\nPrompt: {prompt}\nResponse: {response}\nGround Truth: {ground_truth}\n\nReturn JSON: {{\"accuracy_check_score\": 1 or 0, \"explanation\": \"reason\"}}",1.0,/path/to/fact_check_answers.csv,correct_answer
# MAGIC helpfulness_rating,scale_1_5,Rates response helpfulness,"Rate how helpful this response is on a scale of 1-5.\n\nPrompt: {prompt}\nResponse: {response}\nReference: {ground_truth}\n\nReturn JSON: {{\"helpfulness_rating_score\": score, \"explanation\": \"reason\"}}",3.0,/path/to/helpfulness_ratings.csv,helpfulness_score
# MAGIC relevance_score,percentage,Measures relevance percentage,"Score the relevance of this response as a percentage (0-100).\n\nPrompt: {prompt}\nResponse: {response}\nExpected: {ground_truth}\n\nReturn JSON: {{\"relevance_score_score\": percentage, \"explanation\": \"reason\"}}",0.7,/path/to/relevance_data.csv,relevance_percentage
# MAGIC completeness,binary,Checks if response is complete,"Determine if the response completely addresses the prompt.\n\nPrompt: {prompt}\nResponse: {response}\nComplete Answer: {ground_truth}\n\nReturn JSON: {{\"completeness_score\": 1 or 0, \"explanation\": \"reason\"}}",1.0,,ground_truth
# MAGIC ```
# MAGIC 
# MAGIC ### 📁 Ground Truth File Examples:
# MAGIC 
# MAGIC **fact_check_answers.csv:**
# MAGIC ```csv
# MAGIC question_id,correct_answer,source
# MAGIC 1,Paris,Encyclopedia
# MAGIC 2,Machine learning uses algorithms to find patterns in data,ML Textbook
# MAGIC 3,Mix ingredients and bake at 350°F for 30 minutes,Recipe Book
# MAGIC ```
# MAGIC 
# MAGIC **helpfulness_ratings.csv:**
# MAGIC ```csv
# MAGIC prompt_id,helpfulness_score,notes
# MAGIC 1,4,Very helpful with good detail
# MAGIC 2,5,Excellent explanation
# MAGIC 3,3,Adequate but could be more detailed
# MAGIC ```