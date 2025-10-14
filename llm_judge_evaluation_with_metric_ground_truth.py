# Databricks notebook source
# MAGIC %md
# MAGIC # 🤖 LLM Judge Evaluation System with Metric-Specific Ground Truth
# MAGIC 
# MAGIC **Complete evaluation system for LLM responses using Databricks or OpenAI models**
# MAGIC 
# MAGIC ## Features:
# MAGIC - ✅ Auto-discovery of Databricks LLM endpoints
# MAGIC - ✅ Support for OpenAI API models  
# MAGIC - ✅ Multiple metric types (Binary, 1-5 Scale, Percentage)
# MAGIC - ✅ **Per-metric ground truth files** - each metric can have its own ground truth source
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
from typing import Dict, List, Optional, Any, Union
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
    "",
    "📊 Metrics Configuration File (CSV with metric definitions and ground truth paths)"
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

def load_ground_truth_file(file_path, join_key=None):
    """Load a ground truth file and return as dictionary."""
    if not file_path or not os.path.exists(file_path):
        return {}
    
    try:
        df = pd.read_csv(file_path)
        
        # If no join key specified, try to auto-detect
        if not join_key:
            # Look for common key columns
            key_candidates = ['id', 'prompt_id', 'sample_id', 'example_id', 'question_id', 'index']
            for candidate in key_candidates:
                if candidate in df.columns:
                    join_key = candidate
                    print(f"   Auto-detected join key: '{join_key}'")
                    break
        
        # If still no join key, use index
        if not join_key:
            print("   Using row index as join key")
            df['_index'] = df.index
            join_key = '_index'
        
        # Find the ground truth column (any column that's not the join key)
        gt_columns = [col for col in df.columns if col != join_key]
        if not gt_columns:
            print(f"   Warning: No ground truth column found in {file_path}")
            return {}
        
        # Use the first non-key column as ground truth
        gt_column = gt_columns[0]
        print(f"   Using '{gt_column}' as ground truth column from {os.path.basename(file_path)}")
        
        # Create dictionary mapping join_key -> ground_truth
        gt_dict = {}
        for _, row in df.iterrows():
            key = row[join_key]
            value = row[gt_column]
            gt_dict[str(key)] = str(value)  # Convert to string for consistent comparison
        
        return gt_dict
        
    except Exception as e:
        print(f"   Error loading ground truth file {file_path}: {e}")
        return {}

def load_metrics_config(file_path):
    """Load metrics configuration from CSV file."""
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
            'ground_truth_file': 'ground_truth_path',
            'ground_truth_file_path': 'ground_truth_path',
            'gt_file': 'ground_truth_path',
            'gt_path': 'ground_truth_path'
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
        
        # Add ground_truth_path if not specified
        if 'ground_truth_path' not in df.columns:
            df['ground_truth_path'] = ''  # Empty means no ground truth file
            print("   Added empty ground_truth_path column (metrics without ground truth)")
        
        # Add join_key column if not specified
        if 'join_key' not in df.columns:
            df['join_key'] = ''  # Empty means auto-detect or use index
            print("   Added empty join_key column (will auto-detect or use index)")
        
        print("✅ Metrics configuration loaded and mapped successfully")
        print(f"   Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading metrics config: {e}")
        return None

def create_sample_data():
    """Create sample data for demonstration."""
    # Create sample evaluation data
    eval_df = pd.DataFrame({
        'id': [1, 2, 3],
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
    
    # Create sample metrics configuration
    metrics_df = pd.DataFrame({
        'name': ['accuracy_check', 'completeness_check', 'style_check'],
        'type': ['binary', 'scale_1_5', 'percentage'],
        'description': [
            'Checks if the response is factually accurate',
            'Evaluates completeness of the response',
            'Assesses writing style and clarity'
        ],
        'evaluation_prompt': [
            'Is this response accurate? Prompt: {prompt}\n\nResponse: {response}\n\nExpected: {ground_truth}\n\nReturn JSON: {{"accuracy_check_score": 0 or 1, "explanation": "..."}}',
            'Rate completeness 1-5. Prompt: {prompt}\n\nResponse: {response}\n\nExpected: {ground_truth}\n\nReturn JSON: {{"completeness_check_score": 1-5, "explanation": "..."}}',
            'Rate style 0-100%. Prompt: {prompt}\n\nResponse: {response}\n\nReturn JSON: {{"style_check_score": 0.0-1.0, "explanation": "..."}}'
        ],
        'threshold': [1.0, 3.0, 0.7],
        'ground_truth_path': [
            '/tmp/accuracy_ground_truth.csv',
            '/tmp/completeness_ground_truth.csv',
            ''  # Style check doesn't need ground truth
        ],
        'join_key': ['id', 'id', '']  # How to join with eval data
    })
    
    # Create sample ground truth files
    # Accuracy ground truth
    accuracy_gt = pd.DataFrame({
        'id': [1, 2, 3],
        'correct_answer': [
            'Paris',
            'Machine learning is a subset of artificial intelligence that enables computers to learn from data',
            'Mix dry ingredients, combine with wet ingredients, bake at 350°F for 30-35 minutes'
        ]
    })
    accuracy_gt.to_csv('/tmp/accuracy_ground_truth.csv', index=False)
    
    # Completeness ground truth
    completeness_gt = pd.DataFrame({
        'id': [1, 2, 3],
        'expected_points': [
            'Should mention Paris, capital of France, cultural significance',
            'Should explain AI, learning from data, making predictions, examples',
            'Should include ingredients list, mixing instructions, baking temperature and time'
        ]
    })
    completeness_gt.to_csv('/tmp/completeness_ground_truth.csv', index=False)
    
    print("✅ Sample data and ground truth files created")
    return eval_df, metrics_df

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
    result_df = df.copy()
    
    # Rename prompt and response columns if needed
    if prompt_col != 'prompt':
        result_df = result_df.rename(columns={prompt_col: 'prompt'})
    if response_col != 'response':
        result_df = result_df.rename(columns={response_col: 'response'})
    
    print(f"✅ Standardized evaluation data: {len(result_df)} rows")
    return result_df

# Load or create data
use_sample_data = EVAL_DATA_PATH == "/Workspace/Users/your_username/your_eval_data.csv"

if use_sample_data:
    print("\n📝 Creating sample data for demonstration...")
    eval_df, metrics_config_df = create_sample_data()
    print("✅ Sample data created with ground truth files")
else:
    # Load evaluation data
    print("\n📊 Loading evaluation data...")
    eval_df_raw = load_any_csv(EVAL_DATA_PATH, "evaluation data")
    
    # Standardize the data to ensure we have 'prompt' and 'response' columns
    eval_df = standardize_evaluation_data(eval_df_raw)
    
    # Load metrics configuration if provided
    metrics_config_df = load_metrics_config(METRICS_CONFIG_PATH)

# Display preview
if eval_df is not None:
    print(f"\n📊 Evaluation Data Preview ({len(eval_df)} samples):")
    print(f"Columns: {list(eval_df.columns)}")
    display(eval_df.head(3))

if metrics_config_df is not None:
    print(f"\n📊 Metrics Configuration Preview ({len(metrics_config_df)} metrics):")
    display(metrics_config_df)

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
# MAGIC **Upload your metrics CSV file using the widget above**
# MAGIC 
# MAGIC ### 📊 Required CSV Format for Metrics:
# MAGIC Your metrics CSV must have these columns:
# MAGIC - `name` (or `metric_name`): Metric identifier (e.g., "accuracy_check")
# MAGIC - `type` (or `metric_type`): Metric type ("binary", "scale_1_5", or "percentage") 
# MAGIC - `description`: Human readable description
# MAGIC - `evaluation_prompt` (or `grading_instructions`): Full LLM prompt with {prompt}, {response}, {ground_truth} placeholders
# MAGIC - `threshold` (or `pass_threshold`): Pass/fail threshold (1.0 for binary, 3.0 for scale_1_5, 0.7 for percentage)
# MAGIC - `ground_truth_path` (or `ground_truth_file`): Path to CSV file containing ground truth for this metric
# MAGIC - `join_key` (optional): Column name to join ground truth with evaluation data (auto-detects if not specified)
# MAGIC 
# MAGIC ### 📁 Ground Truth File Format:
# MAGIC Each ground truth CSV file should contain:
# MAGIC - A key column to match with evaluation data (e.g., 'id', 'prompt_id', or use row index)
# MAGIC - One or more columns with ground truth data (first non-key column will be used)

# COMMAND ----------

# =============================================================================
# METRICS VALIDATION AND GROUND TRUTH LOADING
# =============================================================================

# Global dictionary to store ground truth data for each metric
METRIC_GROUND_TRUTH = {}

# Validate that metrics CSV is provided
if not METRICS_CONFIG_PATH or not METRICS_CONFIG_PATH.strip():
    if not use_sample_data:
        print("❌ ERROR: No metrics configuration file provided!")
        print("📋 REQUIRED: Please upload a CSV file with your metrics using the widget above")
        print("\n📊 CSV Format Required:")
        print("   Columns: name, type, description, evaluation_prompt, threshold, ground_truth_path, join_key")
        print("   Types: binary, scale_1_5, percentage")
    
elif METRICS_CONFIG_DATA is None:
    print("❌ ERROR: Could not load metrics configuration file!")
    print(f"   File path: {METRICS_CONFIG_PATH}")
    print("   Please check the file path and format")
    
else:
    print("✅ Metrics configuration loaded successfully!")
    print(f"   📊 Found {len(METRICS_CONFIG_DATA)} metrics in CSV")
    
    # Load ground truth files for each metric
    print("\n📚 Loading ground truth files for metrics...")
    
    for idx, row in METRICS_CONFIG_DATA.iterrows():
        metric_name = row.get('name', row.get('metric_name', 'Unknown'))
        gt_path = row.get('ground_truth_path', '')
        join_key = row.get('join_key', '')
        
        if gt_path and gt_path.strip():
            print(f"\n   Loading ground truth for '{metric_name}':")
            print(f"     File: {gt_path}")
            print(f"     Join key: {join_key or 'auto-detect'}")
            
            # Load the ground truth file
            gt_data = load_ground_truth_file(gt_path, join_key)
            
            if gt_data:
                METRIC_GROUND_TRUTH[metric_name] = {
                    'data': gt_data,
                    'join_key': join_key,
                    'file_path': gt_path
                }
                print(f"     ✅ Loaded {len(gt_data)} ground truth entries")
            else:
                print(f"     ❌ Failed to load ground truth")
        else:
            print(f"   ℹ️ Metric '{metric_name}' has no ground truth file (evaluation without ground truth)")
    
    # Display metrics summary
    print("\n📊 Metrics Summary:")
    for idx, row in METRICS_CONFIG_DATA.iterrows():
        metric_name = row.get('name', row.get('metric_name', 'Unknown'))
        metric_type = row.get('type', row.get('metric_type', 'Unknown'))
        threshold = row.get('threshold', row.get('pass_threshold', 'Unknown'))
        has_gt = metric_name in METRIC_GROUND_TRUTH
        
        print(f"   {idx+1}. {metric_name} ({metric_type}) - threshold: {threshold}, Ground Truth: {'✅' if has_gt else '❌'}")

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
    ground_truth_path: str     # Path to ground truth file for this metric
    join_key: str             # Column to join ground truth with eval data

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
    - Per-metric ground truth files
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

def get_ground_truth_for_sample(self, sample_row: pd.Series, metric: MetricConfig) -> str:
    """Get ground truth for a specific sample and metric."""
    # Check if this metric has ground truth data loaded
    if metric.name not in METRIC_GROUND_TRUTH:
        return "Not provided"
    
    gt_info = METRIC_GROUND_TRUTH[metric.name]
    gt_data = gt_info['data']
    join_key = gt_info['join_key']
    
    # Determine the key to use for lookup
    if join_key and join_key in sample_row:
        # Use specified join key
        lookup_key = str(sample_row[join_key])
    elif 'id' in sample_row:
        # Try 'id' column
        lookup_key = str(sample_row['id'])
    elif 'index' in sample_row:
        # Try 'index' column
        lookup_key = str(sample_row['index'])
    else:
        # Use the sample's index in the dataframe
        lookup_key = str(sample_row.name)  # .name gives the index
    
    # Look up ground truth
    ground_truth = gt_data.get(lookup_key, "Not found")
    
    if ground_truth == "Not found":
        print(f"     ⚠️ No ground truth found for key '{lookup_key}' in metric '{metric.name}'")
    
    return ground_truth

def evaluate_single(self, prompt: str, response: str, sample_row: pd.Series, metric: MetricConfig) -> dict:
    """Evaluate a single sample with one metric."""
    # Get the ground truth for this specific metric and sample
    ground_truth = self.get_ground_truth_for_sample(sample_row, metric)
    
    # Format the evaluation prompt with actual values
    eval_prompt = metric.prompt_template.format(
        prompt=prompt,
        response=response,
        ground_truth=ground_truth
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
        
        # Check if ground truth is available for this metric
        has_ground_truth = metric.name in METRIC_GROUND_TRUTH
        if has_ground_truth:
            gt_info = METRIC_GROUND_TRUTH[metric.name]
            print(f"   Using ground truth from: {os.path.basename(gt_info['file_path'])}")
        else:
            print(f"   No ground truth file (evaluating without ground truth)")
        
        scores = []
        explanations = []
        statuses = []
        
        # Process each sample
        for idx, row in df.iterrows():
            if idx > 0 and idx % 2 == 0:
                print(f"   Progress: {idx}/{len(df)} samples")
            
            result = self.evaluate_single(
                prompt=row['prompt'],
                response=row['response'],
                sample_row=row,
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
LLMJudgeEvaluator.get_ground_truth_for_sample = get_ground_truth_for_sample
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
            'ground_truth_path': row.get('ground_truth_path', ''),
            'join_key': row.get('join_key', '')
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
            ground_truth_path=metric.get('ground_truth_path', ''),
            join_key=metric.get('join_key', '')
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
                    
                    # Get threshold from metric config
                    metric_config_name = col.replace('_score', '')
                    threshold = 1.0  # Default
                    for config in metric_configs:
                        if config.name == metric_config_name:
                            threshold = config.threshold
                            break
                    
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

print("✅ Databricks LLM system loaded with metric-specific ground truth support")

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
            
            # Get threshold from metric config
            metric_name = col.replace('_score', '')
            threshold = 1.0  # Default
            for config in metric_configs:
                if config.name == metric_name:
                    threshold = config.threshold
                    break
            
            pass_count = sum(1 for s in numeric_scores if s >= threshold)
            pass_rate = pass_count / len(numeric_scores) * 100
            
            summary_data.append({
                'metric': metric_name,
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
# MAGIC - **Evaluates LLM responses** using custom metrics with per-metric ground truth files
# MAGIC - **Supports multiple model types** (Databricks + OpenAI)
# MAGIC - **Auto-discovers** available Databricks endpoints
# MAGIC - **Loads ground truth** from separate files for each metric
# MAGIC - **Provides detailed results** with scores and explanations
# MAGIC - **Exports results** to CSV files for further analysis
# MAGIC 
# MAGIC ### 📁 **Ground Truth File Structure:**
# MAGIC Each metric can have its own ground truth CSV file containing:
# MAGIC - A key column (e.g., 'id', 'prompt_id') to match with evaluation data
# MAGIC - One or more columns with ground truth data (first non-key column is used)
# MAGIC 
# MAGIC ### 🚀 **Next Steps:**
# MAGIC 1. **Create ground truth files** for each metric you want to evaluate
# MAGIC 2. **Update metrics CSV** with paths to your ground truth files
# MAGIC 3. **Upload your evaluation data** by updating Cell 2
# MAGIC 4. **Scale up evaluation** to larger datasets
# MAGIC 5. **Integrate with MLflow** for experiment tracking
# MAGIC 
# MAGIC ### 📊 **Key Benefits:**
# MAGIC - ✅ **Flexible ground truth** - each metric can use different ground truth sources
# MAGIC - ✅ **No OpenAI rate limits** when using Databricks
# MAGIC - ✅ **Robust error handling** with fallback parsing
# MAGIC - ✅ **Beautiful results display** with multiple formats
# MAGIC - ✅ **Production ready** with comprehensive logging
# MAGIC 
# MAGIC **Happy evaluating!** 🎯

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📄 Sample Metrics CSV Format
# MAGIC 
# MAGIC Here's an example of how your metrics CSV should look:
# MAGIC 
# MAGIC ```csv
# MAGIC name,type,description,evaluation_prompt,threshold,ground_truth_path,join_key
# MAGIC accuracy_check,binary,"Checks factual accuracy","Is this response accurate? Prompt: {prompt}\n\nResponse: {response}\n\nExpected: {ground_truth}\n\nReturn JSON: {{""accuracy_check_score"": 0 or 1, ""explanation"": ""...""}}",1.0,/path/to/accuracy_gt.csv,id
# MAGIC completeness_check,scale_1_5,"Evaluates response completeness","Rate completeness 1-5. Prompt: {prompt}\n\nResponse: {response}\n\nExpected points: {ground_truth}\n\nReturn JSON: {{""completeness_check_score"": 1-5, ""explanation"": ""...""}}",3.0,/path/to/completeness_gt.csv,id
# MAGIC style_check,percentage,"Assesses writing style","Rate style 0-100%. Prompt: {prompt}\n\nResponse: {response}\n\nReturn JSON: {{""style_check_score"": 0.0-1.0, ""explanation"": ""...""}}",0.7,,
# MAGIC ```
# MAGIC 
# MAGIC ### Ground Truth CSV Format Example:
# MAGIC 
# MAGIC **accuracy_gt.csv:**
# MAGIC ```csv
# MAGIC id,correct_answer
# MAGIC 1,"Paris is the capital of France"
# MAGIC 2,"Machine learning is a subset of AI"
# MAGIC 3,"Chocolate cake requires flour, cocoa, eggs"
# MAGIC ```
# MAGIC 
# MAGIC **completeness_gt.csv:**
# MAGIC ```csv
# MAGIC id,expected_points
# MAGIC 1,"Should mention Paris, capital status, cultural significance"
# MAGIC 2,"Should explain AI relationship, data learning, predictions"
# MAGIC 3,"Should list ingredients, mixing steps, baking instructions"
# MAGIC ```

# COMMAND ----------

