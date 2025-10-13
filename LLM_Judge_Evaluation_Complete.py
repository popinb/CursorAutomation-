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
    "ground_truth_paths",
    "",
    "📚 Ground Truth Files (comma-separated paths, optional)"
)

dbutils.widgets.text(
    "metrics_config_path",
    "",
    "📊 Metrics Configuration File (CSV with metric definitions, optional)"
)

# Get file paths
EVAL_DATA_PATH = dbutils.widgets.get("evaluation_data_path")
GROUND_TRUTH_PATHS = dbutils.widgets.get("ground_truth_paths")
METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")

print("📁 FILE CONFIGURATION")
print("="*60)
print(f"Evaluation Data: {EVAL_DATA_PATH}")
print(f"Ground Truth Files: {GROUND_TRUTH_PATHS or 'None'}")
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

def load_metrics_config(file_path):
    """Load metrics configuration from CSV file."""
    if not file_path or not file_path.strip():
        return None
    
    try:
        df = load_any_csv(file_path, "metrics config")
        if df is None:
            return None
        
        # Map your column names to expected names
        column_mapping = {
            'metric_name': 'name',
            'metric_type': 'type', 
            'grading_instructions': 'evaluation_prompt',
            'pass_threshold': 'threshold'
        }
        
        # Rename columns to match expected format
        df = df.rename(columns=column_mapping)
        
        # Validate required columns
        required_cols = ['name', 'type', 'description', 'evaluation_prompt', 'threshold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns in metrics config: {missing_cols}")
            print(f"Available columns after mapping: {list(df.columns)}")
            return None
        
        print("✅ Metrics configuration loaded and mapped successfully")
        return df
        
    except Exception as e:
        print(f"❌ Error loading metrics config: {e}")
        return None

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

# Load ground truth if provided
ground_truth_df = pd.DataFrame()
if GROUND_TRUTH_PATHS and GROUND_TRUTH_PATHS.strip():
    print("\n📚 Loading ground truth files...")
    
    ground_truth_files = [f.strip() for f in GROUND_TRUTH_PATHS.split(",")]
    all_ground_truth = []
    
    for file_path in ground_truth_files:
        if file_path:
            print(f"   Loading: {file_path}")
            gt_df = load_any_csv(file_path, "ground truth")
            
            if gt_df is not None:
                all_ground_truth.append(gt_df)
                print(f"   ✅ Loaded {len(gt_df)} entries")
            else:
                print(f"   ❌ Failed to load")
    
    if all_ground_truth:
        ground_truth_df = pd.concat(all_ground_truth, ignore_index=True)
        print(f"\n✅ Total ground truth entries: {len(ground_truth_df)}")
        
        # Try to merge if both datasets have a common column
        common_cols = set(eval_df.columns) & set(ground_truth_df.columns)
        if common_cols:
            merge_col = list(common_cols)[0]  # Use first common column
            print(f"   Merging on common column: '{merge_col}'")
            
            # Find ground truth column (any column that's not the merge column)
            gt_cols = [col for col in ground_truth_df.columns if col != merge_col]
            if gt_cols:
                gt_col = gt_cols[0]  # Use first non-merge column as ground truth
                print(f"   Using '{gt_col}' as ground truth column")
                
                eval_df = eval_df.merge(
                    ground_truth_df[[merge_col, gt_col]], 
                    on=merge_col, 
                    how='left'
                )
                eval_df = eval_df.rename(columns={gt_col: 'ground_truth'})
                
                coverage = eval_df['ground_truth'].notna().sum()
                print(f"✅ Ground truth matched for {coverage}/{len(eval_df)} samples")
            else:
                print("   ⚠️ No ground truth column found")
        else:
            print("   ⚠️ No common columns found for merging")

# Load metrics configuration if provided
metrics_config_df = load_metrics_config(METRICS_CONFIG_PATH)

# Ensure ground_truth column exists
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
# MAGIC ## Cell 4: Define Your Custom Metrics
# MAGIC **Option 1: Upload CSV file with metrics (use widget above)**
# MAGIC **Option 2: Define metrics in code below**
# MAGIC 
# MAGIC ### 📊 CSV Format for Metrics (Option 1):
# MAGIC Your CSV should have these columns:
# MAGIC - `name`: Metric name (e.g., "accuracy_check")
# MAGIC - `type`: Metric type ("binary", "scale_1_5", or "percentage") 
# MAGIC - `description`: Human readable description
# MAGIC - `evaluation_prompt`: Full LLM evaluation prompt with {prompt}, {response}, {ground_truth} placeholders
# MAGIC - `threshold`: Pass/fail threshold (1.0 for binary, 3.0 for scale_1_5, 0.7 for percentage)

# COMMAND ----------

# =============================================================================
# CUSTOM METRICS DEFINITION
# =============================================================================
# You have TWO options for defining metrics:
# 
# OPTION 1: Upload a CSV file using the "Metrics Configuration File" widget above
# CSV should have columns: name, type, description, evaluation_prompt, threshold
# Alternative column names supported: metric_name, metric_type, grading_instructions, pass_threshold
#
# OPTION 2: Define metrics in code below (if no CSV file provided)
# Three types supported: Binary (Pass/Fail), 1-5 Scale, and Percentage (0-100%)

CUSTOM_METRICS = [
    # ========================================
    # EXAMPLE 1: BINARY METRIC (Pass/Fail)
    # ========================================
    {
        "name": "accuracy_check",
        "type": "binary",
        "description": "Checks if the response contains accurate information",
        "evaluation_prompt": """
Evaluate if the response contains accurate information.

User Query: {prompt}
AI Response: {response}
Ground Truth (if available): {ground_truth}

Accuracy Criteria:
- All facts must be correct
- No misleading information
- Numbers and statistics must be accurate
- Procedures described correctly

Scoring:
- 1 (PASS): All information is accurate
- 0 (FAIL): Contains any inaccurate information

Return JSON:
{{
    "accuracy_check_score": 1,
    "explanation": "All facts verified as accurate. The response correctly states..."
}}
"""
    },
    
    # ========================================
    # EXAMPLE 2: 1-5 SCALE METRIC
    # ========================================
    {
        "name": "helpfulness_rating",
        "type": "scale_1_5",
        "description": "Rates how helpful the response is on a 1-5 scale",
        "evaluation_prompt": """
Rate the helpfulness of this response from 1 to 5.

User Query: {prompt}
AI Response: {response}
Ground Truth (if available): {ground_truth}

Helpfulness Scale:
5 = Extremely helpful - Comprehensive answer with actionable steps
4 = Very helpful - Good answer with useful information
3 = Moderately helpful - Adequate but could be better
2 = Slightly helpful - Limited value, missing key information
1 = Not helpful - Fails to address the question

Consider:
- Does it answer the user's question?
- Is the information actionable?
- Are next steps clear?

Return JSON:
{{
    "helpfulness_rating_score": 4,
    "explanation": "Very helpful response that answers the main question and provides clear next steps..."
}}
"""
    },
    
    # ========================================
    # EXAMPLE 3: PERCENTAGE METRIC (0-100%)
    # ========================================
    {
        "name": "completeness_percentage",
        "type": "percentage",
        "description": "Measures what percentage of the question was addressed",
        "evaluation_prompt": """
Evaluate what percentage (0-100%) of the user's question was addressed.

User Query: {prompt}
AI Response: {response}
Ground Truth (if available): {ground_truth}

Assessment Process:
1. Identify all components of the user's question
2. Check which components were addressed
3. Calculate percentage of coverage

Examples:
- 90-100%: Fully addresses all aspects
- 70-89%: Most important parts covered
- 50-69%: About half addressed
- 30-49%: Some parts addressed
- 0-29%: Minimal coverage

Return JSON (use decimal, e.g., 0.85 for 85%):
{{
    "completeness_percentage_score": 0.85,
    "explanation": "The response addresses 85% of the question. It covers the main topic well but misses..."
}}
"""
    },
    
    # ========================================
    # ADD YOUR CUSTOM METRICS HERE
    # ========================================
    # Copy any example above and modify it for your needs
    
]

# =============================================================================
# METRIC VALIDATION AND SUMMARY
# =============================================================================

print("📊 CUSTOM METRICS SUMMARY")
print("="*60)

if CUSTOM_METRICS:
    # Count metrics by type
    metric_types = {"binary": 0, "scale_1_5": 0, "percentage": 0}
    
    for i, metric in enumerate(CUSTOM_METRICS, 1):
        print(f"\n{i}. {metric['name'].upper()}")
        print(f"   Type: {metric['type']}")
        print(f"   Description: {metric['description']}")
        
        # Validate metric
        if metric['type'] not in metric_types:
            print(f"   ❌ ERROR: Invalid type '{metric['type']}'. Must be: binary, scale_1_5, or percentage")
        else:
            metric_types[metric['type']] += 1
            print(f"   ✅ Valid metric type")
        
        # Check for required placeholders
        prompt = metric.get('evaluation_prompt', '')
        if '{prompt}' not in prompt:
            print(f"   ⚠️  WARNING: Missing {{prompt}} placeholder")
        if '{response}' not in prompt:
            print(f"   ⚠️  WARNING: Missing {{response}} placeholder")
    
    print(f"\n📈 TOTAL METRICS: {len(CUSTOM_METRICS)}")
    print(f"   Binary: {metric_types['binary']}")
    print(f"   1-5 Scale: {metric_types['scale_1_5']}")
    print(f"   Percentage: {metric_types['percentage']}")
    
else:
    print("\n❌ No custom metrics defined!")

# Set thresholds based on metric types
METRIC_THRESHOLDS = {}
for metric in CUSTOM_METRICS:
    if metric['type'] == 'binary':
        METRIC_THRESHOLDS[metric['name']] = 1.0  # Pass = 1
    elif metric['type'] == 'scale_1_5':
        METRIC_THRESHOLDS[metric['name']] = 3.0  # Pass = 3+
    elif metric['type'] == 'percentage':
        METRIC_THRESHOLDS[metric['name']] = 0.7  # Pass = 70%+

print("\n🎯 Pass Thresholds:")
for name, threshold in METRIC_THRESHOLDS.items():
    print(f"   {name}: {threshold}")

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

def evaluate_single(self, prompt: str, response: str, ground_truth: str, metric: MetricConfig) -> dict:
    """Evaluate a single sample with one metric."""
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
            
            result = self.evaluate_single(
                prompt=row['prompt'],
                response=row['response'],
                ground_truth=row.get('ground_truth', ''),
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

def load_metrics_from_config_or_code():
    """Load metrics from uploaded CSV or use code-defined metrics."""
    if METRICS_CONFIG_DATA is not None:
        print("📊 Loading metrics from uploaded CSV...")
        metrics_list = []
        
        for _, row in METRICS_CONFIG_DATA.iterrows():
            metric = {
                'name': row['name'],
                'type': row['type'],
                'description': row['description'],
                'evaluation_prompt': row['evaluation_prompt'],
                'threshold': row['threshold']
            }
            metrics_list.append(metric)
        
        print(f"✅ Loaded {len(metrics_list)} metrics from CSV")
        return metrics_list
    else:
        print("📊 Using code-defined CUSTOM_METRICS...")
        return CUSTOM_METRICS

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
        
        # Get threshold - use from CSV or from METRIC_THRESHOLDS or default
        if isinstance(metric.get('threshold'), (int, float)):
            threshold = float(metric['threshold'])
        else:
            threshold = METRIC_THRESHOLDS.get(metric['name'], 1.0)
        
        config = MetricConfig(
            name=metric['name'],
            description=metric['description'],
            metric_type=metric_type,
            prompt_template=metric['evaluation_prompt'],
            threshold=threshold
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
                    threshold = METRIC_THRESHOLDS.get(col.replace('_score', ''), 1.0)
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

# Load metrics from CSV file or code definition
metrics_to_use = load_metrics_from_config_or_code()

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
            threshold = METRIC_THRESHOLDS.get(col.replace('_score', ''), 1.0)
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
# MAGIC 
# MAGIC ### 🚀 **Next Steps:**
# MAGIC 1. **Customize metrics** in Cell 4 for your specific use case
# MAGIC 2. **Upload your own data** by updating Cell 2
# MAGIC 3. **Scale up evaluation** to larger datasets
# MAGIC 4. **Integrate with MLflow** for experiment tracking
# MAGIC 5. **Set up automated evaluation** pipelines
# MAGIC 
# MAGIC ### 📊 **Key Benefits:**
# MAGIC - ✅ **No OpenAI rate limits** when using Databricks
# MAGIC - ✅ **Robust error handling** with fallback parsing
# MAGIC - ✅ **Beautiful results display** with multiple formats
# MAGIC - ✅ **Production ready** with comprehensive logging
# MAGIC 
# MAGIC **Happy evaluating!** 🎯

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧪 Test Your Setup (Optional)
# MAGIC **Run this cell to test all functionality**

# COMMAND ----------

# =============================================================================
# COMPREHENSIVE TESTING (OPTIONAL)
# =============================================================================

def test_system_functionality():
    """Test all system components."""
    print("🧪 TESTING SYSTEM FUNCTIONALITY")
    print("="*50)
    
    # Test 1: Check data loading
    print("\n1️⃣ Testing data loading...")
    if 'EVALUATION_DATA' in globals() and len(EVALUATION_DATA) > 0:
        print(f"   ✅ Evaluation data loaded: {len(EVALUATION_DATA)} samples")
        required_cols = ['prompt', 'response']
        missing_cols = [col for col in required_cols if col not in EVALUATION_DATA.columns]
        if missing_cols:
            print(f"   ❌ Missing required columns: {missing_cols}")
        else:
            print(f"   ✅ Required columns present: {required_cols}")
    else:
        print("   ❌ No evaluation data found")
    
    # Test 2: Check metrics configuration
    print("\n2️⃣ Testing metrics configuration...")
    try:
        if 'METRICS_CONFIG_DATA' in globals() and METRICS_CONFIG_DATA is not None:
            print(f"   ✅ Metrics loaded from CSV: {len(METRICS_CONFIG_DATA)} metrics")
        elif 'CUSTOM_METRICS' in globals() and CUSTOM_METRICS:
            print(f"   ✅ Metrics defined in code: {len(CUSTOM_METRICS)} metrics")
        else:
            print("   ❌ No metrics configuration found")
            
        # Test metrics processing
        metrics_to_use = load_metrics_from_config_or_code()
        metric_configs = process_custom_metrics(metrics_to_use)
        print(f"   ✅ Processed {len(metric_configs)} valid metrics")
        
        for config in metric_configs:
            print(f"     - {config.name} ({config.metric_type.value}): threshold={config.threshold}")
            
    except Exception as e:
        print(f"   ❌ Metrics processing failed: {e}")
    
    # Test 3: Check model configuration
    print("\n3️⃣ Testing model configuration...")
    if 'JUDGE_MODEL' in globals():
        print(f"   ✅ Judge model selected: {JUDGE_MODEL}")
        
        if JUDGE_MODEL == "databricks-llm":
            print("   🏢 Databricks LLM mode - will auto-discover endpoints")
        else:
            if 'client' in globals() and client is not None:
                print("   🤖 OpenAI client configured")
            else:
                print("   ❌ OpenAI client not configured")
    else:
        print("   ❌ No judge model selected")
    
    # Test 4: Check file paths
    print("\n4️⃣ Testing file paths...")
    paths_to_check = {
        'EVAL_DATA_PATH': 'Evaluation data',
        'GROUND_TRUTH_PATHS': 'Ground truth files', 
        'METRICS_CONFIG_PATH': 'Metrics configuration'
    }
    
    for var_name, description in paths_to_check.items():
        if var_name in globals():
            path = globals()[var_name]
            if path and path.strip():
                if os.path.exists(path) or any(os.path.exists(p.strip()) for p in path.split(',') if p.strip()):
                    print(f"   ✅ {description}: Found")
                else:
                    print(f"   ⚠️ {description}: Path specified but file not found")
            else:
                print(f"   ➖ {description}: Not specified (optional)")
        else:
            print(f"   ❌ {description}: Variable not defined")
    
    # Test 5: Sample evaluation (if everything is ready)
    print("\n5️⃣ Testing sample evaluation...")
    try:
        if (len(metric_configs) > 0 and 
            'EVALUATION_DATA' in globals() and 
            len(EVALUATION_DATA) > 0 and
            'JUDGE_MODEL' in globals()):
            
            print("   🚀 Running sample evaluation on first row...")
            
            # Create evaluator
            evaluator = LLMJudgeEvaluator(
                judge_model=JUDGE_MODEL,
                metrics=metric_configs[:1]  # Just test first metric
            )
            
            # Test on first row only
            sample_row = EVALUATION_DATA.iloc[0]
            result = evaluator.evaluate_single(
                prompt=sample_row['prompt'],
                response=sample_row['response'], 
                ground_truth=sample_row.get('ground_truth', ''),
                metric=metric_configs[0]
            )
            
            print(f"   ✅ Sample evaluation successful!")
            print(f"     Score: {result['score']}")
            print(f"     Status: {result['status']}")
            print(f"     Explanation: {result['explanation'][:100]}...")
            
        else:
            print("   ⚠️ Skipping sample evaluation - missing requirements")
            
    except Exception as e:
        print(f"   ❌ Sample evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("🏁 TESTING COMPLETE")
    print("If you see mostly ✅ marks above, your system is ready!")
    print("If you see ❌ marks, check the configuration in previous cells.")

# Uncomment the line below to run the test
# test_system_functionality()