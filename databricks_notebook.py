# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Installation and Setup
# MAGIC **Just run this cell - no changes needed**

# COMMAND ----------

# Install packages with --no-deps to avoid dependency conflicts
%pip install mlflow pandas plotly python-docx openai --no-deps --quiet

# Install LangChain packages with --no-deps
%pip install langchain-core langchain-openai --no-deps --quiet

# Install langsmith with --no-deps
%pip install langsmith --no-deps --quiet

%restart_python

print("✅ All packages installed cleanly!")
print("   No dependency conflicts.")
print("   LangSmith is installed and ready.")

# COMMAND ----------

# Import required libraries
from __future__ import annotations
import time
import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Data processing
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Document processing
from docx import Document

# LLM and MLflow
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from openai import OpenAI
import mlflow

print("✅ All libraries imported successfully")

# COMMAND ----------

# =============================================================================
# FILE UPLOAD CONFIGURATION
# =============================================================================

# Create file upload widgets
dbutils.widgets.text(
    "evaluation_data_path", 
    "/Workspace/Users/popinb@zillowgroup.com/eval_candidates_5.csv", 
    "📁 Evaluation Data (CSV file)"
)

dbutils.widgets.text(
    "ground_truth_paths",
    "/Workspace/Users/popinb@zillowgroup.com/ground_truth.csv",
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
print(f"Ground Truth Files: {GROUND_TRUTH_PATHS}")
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

def create_sample_data():
    """Create sample data for demonstration."""
    return pd.DataFrame({
        'prompt': ["Sample question 1", "Sample question 2", "Sample question 3"],
        'response': ["Sample answer 1", "Sample answer 2", "Sample answer 3"]
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

# =============================================================================
# MODEL AND EVALUATION SETTINGS
# =============================================================================

# Model selection
dbutils.widgets.dropdown(
    "judge_model",
    "gpt-4o-mini",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-5", "gpt-5-turbo", "gpt-5o", "gpt-5o-mini", "databricks-llm"],
    "🤖 Select Judge Model"
)

# Experiment settings
dbutils.widgets.text(
    "experiment_name",
    f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/llm_judge_evaluation",
    "🔬 MLflow Experiment Name"
)

# Evaluation settings
dbutils.widgets.dropdown(
    "include_ground_truth",
    "Yes",
    ["Yes", "No"],
    "📋 Include Ground Truth in Evaluation?"
)

# Get settings
JUDGE_MODEL = dbutils.widgets.get("judge_model")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
INCLUDE_GROUND_TRUTH = dbutils.widgets.get("include_ground_truth") == "Yes"
MAX_CONCURRENCY = 2  # Hardcoded good default for PMs

# Map UI model names to actual working model names
MODEL_MAPPING = {
    "gpt-5": "gpt-5-chat-latest",
    "gpt-5-turbo": "gpt-5-chat-latest", 
    "gpt-5o": "gpt-5-chat-latest",
    "gpt-5o-mini": "gpt-5-chat-latest"
}

# Map the selected model to the actual working model
ACTUAL_MODEL = MODEL_MAPPING.get(JUDGE_MODEL, JUDGE_MODEL)

print("🤖 EVALUATION SETTINGS")
print("="*60)
print(f"Judge Model (UI): {JUDGE_MODEL}")
print(f"Actual Model: {ACTUAL_MODEL}")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Include Ground Truth: {INCLUDE_GROUND_TRUTH}")
print(f"Parallel Evaluations: {MAX_CONCURRENCY}")
print("="*60)

# Initialize API connection
print("\n🔗 Initializing LLM connection...")

# Get API key
OPENAI_KEY = dbutils.secrets.get("popin-secure-scope", "openai_key")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.zillowlabs.com/openai/v1",
    api_key=OPENAI_KEY
)

# Test connection with proper parameters for different models
def test_model_connection(model_name):
    """Test connection with model-specific parameters."""
    try:
        print(f"   Testing {model_name}...")
        
        # Check if it's a GPT-5 variant that needs special handling
        if "gpt-5" in model_name.lower() or model_name in ["gpt-5-chat-latest"]:
            # GPT-5 variants use max_completion_tokens instead of max_tokens
            test_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_completion_tokens=10
            )
        else:
            # Other models use max_tokens
            test_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=10
            )
        
        # Check if we got a valid response
        if test_response.choices and test_response.choices[0].message.content:
            print(f"   ✅ {model_name} responded: {test_response.choices[0].message.content}")
            return True, None
        else:
            print(f"   ⚠️ {model_name} returned empty response")
            return False, "Empty response"
            
    except Exception as e:
        print(f"   ❌ {model_name} failed: {e}")
        return False, e

# Test connection with actual working model
success, error = test_model_connection(ACTUAL_MODEL)
if success:
    print(f"✅ LLM connection successful with {ACTUAL_MODEL}!")
    # Update JUDGE_MODEL to the actual working model for use in evaluation
    JUDGE_MODEL = ACTUAL_MODEL
else:
    print(f"❌ LLM connection failed with {ACTUAL_MODEL}: {error}")
    print("Trying fallback to gpt-4o-mini...")
    success, error = test_model_connection("gpt-4o-mini")
    if success:
        print("✅ Fallback connection successful with gpt-4o-mini!")
        JUDGE_MODEL = "gpt-4o-mini"  # Update to working model
    else:
        print(f"❌ Fallback connection also failed: {error}")

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: 📊 Define Your Custom Metrics
# MAGIC **Add your evaluation metrics below - supports Binary, 1-5 Scale, and Percentage**

# COMMAND ----------

# =============================================================================
# CUSTOM METRICS DEFINITION
# =============================================================================
# Define your metrics by uncommenting and modifying the examples below
# Three types supported: Binary (Pass/Fail), 1-5 Scale, and Percentage (0-100%)

CUSTOM_METRICS = [
    # ========================================
    # EXAMPLE 1: BINARY METRIC (Pass/Fail) - ACTIVE FOR DEMO
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
    # EXAMPLE 2: 1-5 SCALE METRIC - ACTIVE FOR DEMO
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
    # EXAMPLE 3: PERCENTAGE METRIC (0-100%) - ACTIVE FOR DEMO
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
    # ADD YOUR METRICS BELOW (or comment out examples above)
    # ========================================
    # To disable demo metrics, just add # at the start of each line of the metric
    # To add your own metrics, copy any example above and modify it
    
    # Your custom metrics here...
    
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
    print("\n📝 HOW TO ADD METRICS:")
    print("1. Copy one of the examples above")
    print("2. Uncomment it (remove # symbols)")
    print("3. Modify the name, description, and evaluation prompt")
    print("4. Choose type: 'binary', 'scale_1_5', or 'percentage'")
    print("5. Make sure your prompt asks for the exact JSON format shown")

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



# COMMAND ----------

# Core system classes

class MetricType(Enum):
    BINARY = "binary"
    SCALE_1_5 = "scale_1_5"
    PERCENTAGE = "percentage"

@dataclass
class MetricConfig:
    """Configuration for a metric."""
    name: str
    description: str
    metric_type: MetricType
    prompt_template: str
    threshold: float

class LLMJudgeEvaluator:
    """Main evaluator class."""
    
    def __init__(self, judge_model: str, metrics: List[MetricConfig]):
        self.judge_model = judge_model
        self.metrics = metrics
        self.client = client  # Store client reference
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM judge."""
        if self.judge_model == "databricks-llm":
            # Databricks LLM
            self.model = ChatOpenAI(
                model_name="databricks-llm",
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
        else:
            # Zillow API models using OpenAI client
            def run_chat(messages: list) -> AIMessage:
                # Extract content from first message
                content = messages[0].content if messages else ""
                
                # Use model-specific parameters
                if "gpt-5" in self.judge_model.lower() or self.judge_model in ["gpt-5-chat-latest"]:
                    # GPT-5 variants use max_completion_tokens and don't support temperature=0
                    resp = self.client.chat.completions.create(
                        model=self.judge_model,
                        messages=[{"role": "user", "content": content}],
                        max_completion_tokens=1000,
                        response_format={"type": "json_object"}
                    )
                else:
                    # Other models use max_tokens and temperature=0
                    resp = self.client.chat.completions.create(
                        model=self.judge_model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=1000,
                        temperature=0.0,
                        response_format={"type": "json_object"}
                    )
                return AIMessage(content=resp.choices[0].message.content)
            
            self.model = RunnableLambda(run_chat)
    
    def evaluate_single(self, prompt: str, response: str, ground_truth: str, metric: MetricConfig) -> dict:
        """Evaluate a single sample with one metric."""
        # Format the evaluation prompt
        eval_prompt = metric.prompt_template.format(
            prompt=prompt,
            response=response,
            ground_truth=ground_truth if ground_truth else "Not provided"
        )
        
        try:
            # Get evaluation from LLM
            result = self.model.invoke([HumanMessage(content=eval_prompt)])
            
            # Check if we got a valid response
            if not result or not result.content or result.content.strip() == "":
                print(f"   Warning: Empty response for {metric.name}")
                return {
                    "score": 0,
                    "explanation": "Empty response from LLM",
                    "status": "❌"
                }
            
            # Parse JSON response
            try:
                result_json = json.loads(result.content)
            except json.JSONDecodeError as json_err:
                print(f"   Warning: Invalid JSON for {metric.name}: {result.content[:100]}...")
                return {
                    "score": 0,
                    "explanation": f"Invalid JSON response: {result.content[:200]}",
                    "status": "❌"
                }
            
            # Extract score - NEVER use direct key access that could cause KeyError
            score = 0
            explanation = "No explanation provided"
            
            # Method 1: Try exact key match
            score_key = f"{metric.name}_score"
            if score_key in result_json:
                score = result_json[score_key]
            
            # Method 2: Search for any key containing the metric name and "score"
            elif any(metric.name in key and "score" in key.lower() for key in result_json.keys()):
                for key, value in result_json.items():
                    if metric.name in key and "score" in key.lower() and isinstance(value, (int, float)):
                        score = value
                        break
            
            # Method 3: Search for any key ending with "_score"
            elif any(key.endswith("_score") for key in result_json.keys()):
                for key, value in result_json.items():
                    if key.endswith("_score") and isinstance(value, (int, float)):
                        score = value
                        break
            
            # Method 4: Search for any key containing "score" (last resort)
            else:
                for key, value in result_json.items():
                    if "score" in key.lower() and isinstance(value, (int, float)):
                        score = value
                        break
            
            # Extract explanation safely
            for key in ["explanation", "Explanation", "reason", "Reason"]:
                if key in result_json:
                    explanation = result_json[key]
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
    
    def evaluate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate entire dataset."""
        results_df = df.copy()
        
        print(f"\n🚀 Starting evaluation of {len(df)} samples with {len(self.metrics)} metrics...")
        print(f"   Using model: {self.judge_model}")
        
        for metric in self.metrics:
            print(f"\n📊 Evaluating metric: {metric.name}")
            
            scores = []
            explanations = []
            statuses = []
            
            for idx, row in df.iterrows():
                if idx > 0 and idx % 10 == 0:
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
            
            # Add results to dataframe
            results_df[f"{metric.name}_score"] = scores
            results_df[f"{metric.name}_explanation"] = explanations
            results_df[f"{metric.name}_status"] = statuses
            
            # Calculate summary statistics
            mean_score = sum(scores) / len(scores)
            pass_rate = sum(1 for s in scores if s >= metric.threshold) / len(scores)
            
            print(f"   ✅ Complete - Mean: {mean_score:.3f}, Pass Rate: {pass_rate:.1%}")
        
        return results_df

# Convert custom metrics to MetricConfig objects
def process_custom_metrics(custom_metrics: list) -> List[MetricConfig]:
    """Convert custom metric definitions to MetricConfig objects."""
    configs = []
    
    for metric in custom_metrics:
        # Map metric type
        if metric['type'] == 'binary':
            metric_type = MetricType.BINARY
        elif metric['type'] == 'scale_1_5':
            metric_type = MetricType.SCALE_1_5
        elif metric['type'] == 'percentage':
            metric_type = MetricType.PERCENTAGE
        else:
            print(f"   Warning: Unknown metric type '{metric['type']}' for {metric.get('name', 'unknown')}")
            continue  # Skip invalid types
        
        # Safely get threshold with fallback
        threshold = metric.get('threshold', 1.0)
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            print(f"   Warning: Invalid threshold '{threshold}' for {metric.get('name', 'unknown')}, using default 1.0")
            threshold = 1.0
        
        config = MetricConfig(
            name=metric['name'],
            description=metric['description'],
            metric_type=metric_type,
            prompt_template=metric['evaluation_prompt'],
            threshold=threshold
        )
        configs.append(config)
    
    return configs

# Load metrics from CSV or use default
def load_metrics_from_config():
    """Load metrics from uploaded CSV or use default metrics."""
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
        print("📊 Using default metrics...")
        # Default metrics if no CSV uploaded
        return [
            {
                "name": "accuracy",
                "type": "binary",
                "description": "Checks if the response contains accurate information",
                "evaluation_prompt": """
Evaluate if the response contains accurate information.

User Query: {prompt}
AI Response: {response}
Ground Truth (if available): {ground_truth}

Criteria:
- All facts must be correct
- No misleading information
- Numbers and statistics must be accurate

Return JSON:
{{
    "accuracy_score": 1,
    "explanation": "All information is accurate and verified."
}}
""",
                "threshold": 1.0
            },
            {
                "name": "helpfulness",
                "type": "scale_1_5",
                "description": "Rates how helpful the response is (1-5 scale)",
                "evaluation_prompt": """
Rate the helpfulness of this response from 1 to 5.

User Query: {prompt}
AI Response: {response}
Ground Truth (if available): {ground_truth}

Scale:
5 = Extremely helpful - Comprehensive answer with actionable steps
4 = Very helpful - Good answer with useful information
3 = Moderately helpful - Adequate but could be better
2 = Slightly helpful - Limited value, missing key information
1 = Not helpful - Fails to address the question

Return JSON:
{{
    "helpfulness_score": 4,
    "explanation": "Very helpful response that answers the main question..."
}}
""",
                "threshold": 3.0
            }
        ]

print("✅ System classes loaded")

# COMMAND ----------


CUSTOM_METRICS = load_metrics_from_config()

# Process metrics and create evaluator
metric_configs = process_custom_metrics(CUSTOM_METRICS)

if not metric_configs:
    print("❌ No valid metrics to evaluate!")
    print("Please upload a metrics configuration CSV file")
else:
    # Create evaluator
    evaluator = LLMJudgeEvaluator(
        judge_model=JUDGE_MODEL,
        metrics=metric_configs
    )
    
    # Run evaluation
    print("="*60)
    print("🚀 STARTING EVALUATION")
    print("="*60)
    
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"✅ EVALUATION COMPLETE in {eval_time:.1f} seconds")
    print("="*60)
    
    # Display sample results
    print("\n📊 Sample Results:")
    display_cols = ['prompt', 'response'] + [f"{m.name}_score" for m in metric_configs] + [f"{m.name}_status" for m in metric_configs]
    display_cols = [col for col in display_cols if col in results_df.columns]
    display(results_df[display_cols].head())