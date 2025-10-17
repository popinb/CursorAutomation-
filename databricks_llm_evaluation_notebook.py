# Databricks notebook source
# MAGIC %md
# MAGIC ## Cell 1: Installation and Setup
# MAGIC 
# MAGIC **Purpose**: This cell installs and imports all necessary Python libraries required for the LLM evaluation system.
#MAGIC 
#MAGIC **What it does**:
#MAGIC - Installs required packages including MLflow (for experiment tracking), Pandas (for data manipulation), Plotly (for visualizations), OpenAI libraries (for LLM integration), and other dependencies
#MAGIC - Restarts the Python kernel to ensure all packages are properly loaded
#MAGIC - Imports all necessary libraries for data processing, machine learning evaluation, and visualization
#MAGIC 
#MAGIC **When to run**: Always run this cell first before executing any other cells in the notebook
#MAGIC 
#MAGIC **Expected output**: Confirmation messages showing successful package installation and library imports

# COMMAND ----------

# Install packages
%pip install mlflow pandas plotly python-docx openai langchain-core langchain-openai langsmith --quiet

%restart_python

print("✅ All packages installed!")

# COMMAND ----------

# Import required libraries
import time
import os
import json
import requests
import glob
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from openai import OpenAI
import mlflow

print("✅ All libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: File Configuration
#MAGIC 
#MAGIC **Purpose**: This cell sets up file paths and loads the evaluation data, metrics configuration, and ground truth files.
#MAGIC 
#MAGIC **What it does**:
#MAGIC - Creates interactive widgets for users to specify file paths for evaluation data, metrics configuration, and ground truth files
#MAGIC - Automatically searches for files in common Databricks workspace locations if files are not found in the specified path
#MAGIC - Loads CSV files containing evaluation samples, metric definitions, and reference answers
#MAGIC - Creates sample data if the specified files are not found, allowing users to test the system
#MAGIC - Displays a summary of loaded data including number of samples, metrics, and ground truth files
#MAGIC 
#MAGIC **When to run**: Run this cell after Cell 1 to configure your data sources
#MAGIC 
#MAGIC **User input required**: Update the widget values with your actual file paths before running
#MAGIC 
#MAGIC **Expected output**: File loading status messages and data summary statistics

# COMMAND ----------

# Widgets for file paths
dbutils.widgets.text(
    "evaluation_data_path", 
    "evaluation_data.csv", 
    "📊 Evaluation Data (CSV)"
)

dbutils.widgets.text(
    "metrics_config_path",
    "sample_metrics_config_simplified.csv",
    "📋 Metrics Configuration (CSV)"
)

dbutils.widgets.text(
    "ground_truth_files",
    "/Workspace/Users/popinb@zillowgroup.com/ground_truth_accuracy.csv;/Workspace/Users/popinb@zillowgroup.com/ground_truth_safety.csv",
    "📚 Ground Truth Files (semicolon separated)"
)

# Get settings
EVAL_DATA_PATH = dbutils.widgets.get("evaluation_data_path")
METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")
GROUND_TRUTH_FILES_STRING = dbutils.widgets.get("ground_truth_files")

print("📁 FILE CONFIGURATION")
print("="*60)
print(f"Evaluation Data: {EVAL_DATA_PATH}")
print(f"Metrics Config: {METRICS_CONFIG_PATH}")
print(f"Ground Truth Files: {GROUND_TRUTH_FILES_STRING}")
print("="*60)

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
    """Auto-detect file in common Databricks locations."""
    if os.path.exists(filename):
        return filename
    
    user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    search_locations = [
        f"/Workspace/Users/{user_name}/{filename}",
        f"./{filename}",
        f"/tmp/{filename}"
    ]
    
    for location in search_locations:
        if os.path.exists(location):
            return location
    
    return None

def load_csv_file(filename, file_type="data"):
    """Load CSV file with auto-detection."""
    try:
        file_path = find_file_in_workspace(filename)
        
        if not file_path:
            print(f"❌ {file_type.title()} file not found: {filename}")
            return None
        
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {file_type}: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {file_type}: {e}")
        return None

# Load evaluation data
print(f"\n📊 Loading evaluation data...")
evaluation_data_df = load_csv_file(EVAL_DATA_PATH, "evaluation data")

if evaluation_data_df is None:
    print("\n📝 Creating sample data...")
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
            "To bake a chocolate cake, mix flour, cocoa, eggs, and sugar, then bake at 350°F for 30 minutes."
        ]
    })
    print("✅ Sample data created")

# Load metrics configuration
print(f"\n📋 Loading metrics configuration...")
metrics_config_df = load_csv_file(METRICS_CONFIG_PATH, "metrics config")

if metrics_config_df is None:
    print("❌ No metrics configuration loaded!")
else:
    print(f"✅ Loaded {len(metrics_config_df)} metrics")

# Load ground truth files
GROUND_TRUTH_FILES_LIST = parse_ground_truth_files(GROUND_TRUTH_FILES_STRING)
ground_truth_data = {}

if GROUND_TRUTH_FILES_LIST:
    print(f"\n📚 Loading {len(GROUND_TRUTH_FILES_LIST)} ground truth files...")
    print("💡 TIP: In your metrics CSV, just specify the filename (e.g., 'ground_truth_accuracy.csv')")
    print("        Code will automatically match it to the uploaded files below:\n")
    
    for file_path in GROUND_TRUTH_FILES_LIST:
        if file_path and os.path.exists(file_path):
            filename = os.path.basename(file_path)
            try:
                df = pd.read_csv(file_path)
                ground_truth_data[filename] = df
                print(f"✅ {filename} → {len(df)} rows, columns: {', '.join(df.columns.tolist())}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    print(f"\n✅ Loaded {len(ground_truth_data)} ground truth files")

# Store data globally
EVALUATION_DATA = evaluation_data_df
METRICS_CONFIG_DATA = metrics_config_df

print(f"\n📊 Data loaded successfully!")
print(f"   Evaluation samples: {len(EVALUATION_DATA)}")
print(f"   Metrics configured: {len(METRICS_CONFIG_DATA) if METRICS_CONFIG_DATA is not None else 0}")
print(f"   Ground truth files: {len(ground_truth_data)}")

# COMMAND ----------

# MAGIC %md
#MAGIC ## Cell 3: Model Configuration
#MAGIC 
#MAGIC **Purpose**: This cell configures the language model that will be used as a judge to evaluate the AI responses.
#MAGIC 
#MAGIC **What it does**:
#MAGIC - Provides a dropdown widget to select the judge model (GPT-4, GPT-3.5, or Databricks LLM)
#MAGIC - Initializes the appropriate API connection based on the selected model
#MAGIC - Tests the connection to ensure the model is accessible and working properly
#MAGIC - Sets up authentication and configuration for both OpenAI and Databricks LLM endpoints
#MAGIC 
#MAGIC **When to run**: Run this cell after Cell 2 to configure your evaluation model
#MAGIC 
#MAGIC **User input required**: Select your preferred judge model from the dropdown
#MAGIC 
#MAGIC **Expected output**: Model selection confirmation and connection test results

# COMMAND ----------

# Model selection
dbutils.widgets.dropdown(
    "judge_model",
    "databricks-llm",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "databricks-llm"],
    "🤖 Judge Model"
)

# Get settings
JUDGE_MODEL = dbutils.widgets.get("judge_model")

print("🤖 MODEL SETTINGS")
print("="*60)
print(f"Judge Model: {JUDGE_MODEL}")
print("="*60)

# Initialize API connection
if JUDGE_MODEL == "databricks-llm":
    print("\n🏢 Databricks LLM selected")
    client = None
else:
    print("\n🔗 Initializing OpenAI connection...")
    try:
        OPENAI_KEY = dbutils.secrets.get("popin-secure-scope", "openai_key")
        os.environ["OPENAI_API_KEY"] = OPENAI_KEY
        
        client = OpenAI(
            base_url="https://api.zillowlabs.com/openai/v1",
            api_key=OPENAI_KEY
        )
        
        # Test connection
        test_response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=10
        )
        print(f"✅ OpenAI connection successful!")
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")

# COMMAND ----------

# MAGIC %md
#MAGIC ## Cell 4: Verify Metrics
#MAGIC 
#MAGIC **Purpose**: This cell displays the loaded metrics configuration to verify that all evaluation criteria are properly set up.
#MAGIC 
#MAGIC **What it does**:
#MAGIC - Lists all metrics that will be used to evaluate the AI responses
#MAGIC - Shows the metric name, type (binary, scale, percentage), and threshold values
#MAGIC - Helps users verify that their metrics configuration file was loaded correctly
#MAGIC - Provides a quick overview of what will be evaluated before running the actual evaluation
#MAGIC 
#MAGIC **When to run**: Run this cell after Cell 2 to review your metrics configuration
#MAGIC 
#MAGIC **Expected output**: A numbered list of all configured metrics with their types and thresholds

# COMMAND ----------

if METRICS_CONFIG_DATA is not None:
    print("✅ Metrics Configuration")
    print("="*60)
    for idx, row in METRICS_CONFIG_DATA.iterrows():
        print(f"{idx+1}. {row['name']} ({row['type']}) - threshold: {row['threshold']}")
else:
    print("❌ No metrics configuration loaded!")

# COMMAND ----------

# MAGIC %md
#MAGIC ## Cell 5: Core Classes
#MAGIC 
#MAGIC **Purpose**: This cell defines the fundamental data structures and classes used throughout the evaluation system.
#MAGIC 
#MAGIC **What it does**:
#MAGIC - Defines the MetricType enumeration for different types of evaluation metrics (binary, scale, percentage)
#MAGIC - Creates the MetricConfig dataclass to store configuration for each evaluation metric
#MAGIC - Establishes the data structures that will be used by the evaluation engine
#MAGIC - Sets up the foundation for the evaluation logic in subsequent cells
#MAGIC 
#MAGIC **When to run**: Run this cell after Cell 4 to initialize the core data structures
#MAGIC 
#MAGIC **Expected output**: Confirmation message that core classes have been defined

# COMMAND ----------

class MetricType(Enum):
    BINARY = "binary"
    SCALE_1_5 = "1-5_scale"
    PERCENTAGE = "percentage"

@dataclass
class MetricConfig:
    name: str
    description: str
    metric_type: MetricType
    prompt_template: str
    threshold: float
    ground_truth_column: str
    ground_truth_file_path: str = ""

print("✅ Core classes defined")

# COMMAND ----------

# MAGIC %md
#MAGIC ## Cell 6: LLM Judge Evaluator (Main Logic)
#MAGIC 
#MAGIC **Purpose**: This cell contains the core evaluation engine that uses a language model to judge AI responses against defined metrics.
#MAGIC 
#MAGIC **What it does**:
#MAGIC - Implements the LLMJudgeEvaluator class that handles all evaluation logic
#MAGIC - Provides robust JSON parsing that works with any custom metric naming convention
#MAGIC - Handles both OpenAI and Databricks LLM endpoints for evaluation
#MAGIC - Implements smart score extraction that can find scores regardless of how the LLM formats its response
#MAGIC - Includes fallback parsing methods for when JSON parsing fails
#MAGIC - Manages ground truth data integration for reference-based evaluations
#MAGIC - Provides comprehensive error handling and status reporting
#MAGIC 
#MAGIC **When to run**: Run this cell after Cell 5 to initialize the evaluation engine
#MAGIC 
#MAGIC **Expected output**: Confirmation message that the LLM Judge Evaluator has been defined with bulletproof JSON parsing

# COMMAND ----------

import json
import time
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import os

class LLMJudgeEvaluator:
    """
    Main evaluator class with bulletproof JSON parsing for ANY custom metric name.
    
    Key Innovation: Smart key detection that works with any metric naming convention.
    """
    
    def __init__(self, judge_model: str, metrics: List[MetricConfig], ground_truth_data: Dict[str, pd.DataFrame] = None):
        self.judge_model = judge_model
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data or {}
        self.is_databricks_llm = judge_model == "databricks-llm"
        
        if self.is_databricks_llm:
            self._init_databricks_llm()
        else:
            self._init_openai_llm()
    
    def _init_databricks_llm(self):
        """Initialize Databricks LLM client."""
        try:
            self.databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            self.workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
            
            self.databricks_headers = {
                "Authorization": f"Bearer {self.databricks_token}",
                "Content-Type": "application/json"
            }
            
            import requests
            url = f"https://{self.workspace_url}/api/2.0/serving-endpoints"
            response = requests.get(url, headers=self.databricks_headers, timeout=10)
            
            if response.status_code == 200:
                endpoints = response.json().get('endpoints', [])
                available_models = [ep['name'] for ep in endpoints]
                
                # Find Claude Sonnet (preferred)
                for endpoint_name in available_models:
                    if 'claude-sonnet' in endpoint_name.lower():
                        self.databricks_endpoint = endpoint_name
                        self.response_format = 'openai'
                        print(f"✅ Using Databricks endpoint: {endpoint_name}")
                        return
                
                # Fallback to first available
                if available_models:
                    self.databricks_endpoint = available_models[0]
                    self.response_format = 'openai'
                    print(f"✅ Using Databricks endpoint: {available_models[0]}")
                else:
                    raise Exception("No endpoints found")
            else:
                raise Exception(f"Failed to list endpoints: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Databricks init failed: {e}")
            raise
    
    def _init_openai_llm(self):
        """Initialize OpenAI LLM client."""
        if 'client' in globals() and client is not None:
            self.llm_client = client
            print(f"✅ OpenAI client initialized")
        else:
            raise ValueError("OpenAI client not found")
    
    def _call_databricks_llm(self, prompt: str) -> str:
        """Call Databricks LLM endpoint."""
        try:
            import requests
            url = f"https://{self.workspace_url}/serving-endpoints/{self.databricks_endpoint}/invocations"
            
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            response = requests.post(url, headers=self.databricks_headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
            
            return ""
                
        except Exception as e:
            print(f"Error calling Databricks LLM: {e}")
            return ""
    
    def _call_openai_llm(self, prompt: str) -> str:
        """Call OpenAI LLM endpoint."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI LLM: {e}")
            return ""
    
    def _get_ground_truth_for_metric(self, metric: MetricConfig, sample_idx: int) -> str:
        """
        Get ground truth for a metric from ground truth files.
        
        How it works:
        1. PM specifies just the FILENAME in CSV (e.g., 'ground_truth_accuracy.csv')
        2. Full paths come from UI widget (e.g., '/Workspace/Users/email/ground_truth_accuracy.csv')
        3. Code matches filename from CSV to actual uploaded files
        
        This means PMs don't need to know full paths - just the filename!
        """
        try:
            # PM can specify filename or full path - we handle both
            files = []
            if ';' in metric.ground_truth_file_path:
                files = [f.strip() for f in metric.ground_truth_file_path.split(';') if f.strip()]
            elif ',' in metric.ground_truth_file_path:
                files = [f.strip() for f in metric.ground_truth_file_path.split(',') if f.strip()]
            else:
                files = [metric.ground_truth_file_path.strip()]
            
            if not files or not files[0]:
                return "Not provided"
            
            # Match by filename (PM provides filename, widget provides full paths)
            for file_path in files:
                filename = os.path.basename(file_path)  # Extract just the filename
                if filename in self.ground_truth_data:
                    df = self.ground_truth_data[filename]
                    if metric.ground_truth_column in df.columns:
                        if sample_idx < len(df):
                            ground_truth = df[metric.ground_truth_column].iloc[sample_idx]
                            return str(ground_truth) if pd.notna(ground_truth) else "Not provided"
            
            return "Not provided"
            
        except Exception as e:
            return "Not provided"
    
    def _escape_prompt_template(self, template: str) -> str:
        """
        Properly escape prompt template to handle JSON examples.
        
        Protects {prompt}, {response}, {ground_truth} while escaping other braces.
        """
        actual_placeholders = {
            '{prompt}': '<<<PROMPT_PLACEHOLDER>>>',
            '{response}': '<<<RESPONSE_PLACEHOLDER>>>',
            '{ground_truth}': '<<<GROUND_TRUTH_PLACEHOLDER>>>'
        }
        
        escaped_template = template
        for placeholder, marker in actual_placeholders.items():
            escaped_template = escaped_template.replace(placeholder, marker)
        
        # Escape all remaining curly braces
        escaped_template = escaped_template.replace('{', '{{').replace('}', '}}')
        
        # Restore actual placeholders
        for placeholder, marker in actual_placeholders.items():
            escaped_template = escaped_template.replace(marker, placeholder)
        
        return escaped_template
    
    def evaluate_single(self, prompt: str, response: str, ground_truth_data: dict, metric: MetricConfig, sample_idx: int = 0) -> dict:
        """Evaluate a single sample with one metric."""
        try:
            ground_truth = self._get_ground_truth_for_metric(metric, sample_idx)
            
            # Escape prompt template
            safe_template = self._escape_prompt_template(metric.prompt_template)
            
            # Format prompt
            eval_prompt = safe_template.format(
                prompt=prompt,
                response=response,
                ground_truth=ground_truth if ground_truth else "Not provided"
            )
            
            # Call LLM
            if self.is_databricks_llm:
                llm_response = self._call_databricks_llm(eval_prompt)
            else:
                llm_response = self._call_openai_llm(eval_prompt)
            
            if not llm_response or str(llm_response).strip() == "":
                return {
                    "score": 0,
                    "explanation": "Empty response from LLM",
                    "status": "❌"
                }
            
            # Parse response
            score, explanation = self._parse_llm_response(llm_response, metric)
            
            status = "✅" if score >= metric.threshold else "❌"
            
            return {
                "score": score,
                "explanation": explanation,
                "status": status
            }
            
        except Exception as e:
            return {
                "score": 0,
                "explanation": f"Evaluation error: {str(e)}",
                "status": "❌"
            }
    
    def _parse_llm_response(self, llm_response: str, metric: MetricConfig) -> tuple:
        """
        BULLETPROOF JSON parsing that works with ANY custom metric name.
        
        Strategy:
        1. Try to parse as JSON
        2. Look for score using intelligent key detection
        3. Fall back to text parsing if JSON fails
        """
        content = str(llm_response).strip()
        
        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        # Try JSON parsing
        try:
            result_json = json.loads(content)
            
            # BULLETPROOF: Smart score extraction
            score = self._smart_extract_score(result_json, metric)
            
            # BULLETPROOF: Smart explanation extraction
            explanation = self._smart_extract_explanation(result_json)
            
        except json.JSONDecodeError:
            # Fallback to text parsing
            score, explanation = self._fallback_parse(content, metric)
        
        # Normalize score
        score = self._normalize_score(score, metric.metric_type)
        
        return score, str(explanation)[:500]
    
    def _smart_extract_score(self, json_obj: dict, metric: MetricConfig) -> Any:
        """
        BULLETPROOF score extraction that handles ANY custom metric name.
        
        Strategy:
        1. Try standard keys first (score, value, rating)
        2. Try the exact metric name
        3. Try metric name variations (with/without common suffixes)
        4. Try case-insensitive search through all keys
        5. Try finding any numeric value in the JSON
        """
        # Step 1: Try standard keys first (most common)
        standard_keys = ["score", "Score", "value", "Value", "rating", "Rating", "result", "Result"]
        for key in standard_keys:
            if key in json_obj:
                return json_obj[key]
        
        # Step 2: Try exact metric name
        if metric.name in json_obj:
            return json_obj[metric.name]
        
        # Step 3: Try metric name with common suffix variations
        # Remove common suffixes if they exist
        base_name = metric.name
        common_suffixes = ['_score', '_rating', '_check', '_value', '_result', 'Score', 'Rating', 'Check', 'Value', 'Result']
        
        for suffix in common_suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        # Try base name without suffix
        if base_name in json_obj:
            return json_obj[base_name]
        
        # Try base name with different suffixes
        for suffix in ['_score', '_rating', '_value', 'Score', 'Rating', 'Value']:
            key = f"{base_name}{suffix}"
            if key in json_obj:
                return json_obj[key]
        
        # Step 4: Case-insensitive search
        metric_name_lower = metric.name.lower()
        for key, value in json_obj.items():
            if key.lower() == metric_name_lower:
                return value
        
        # Step 5: Look for keys containing the metric name or common score words
        score_keywords = ['score', 'rating', 'value', 'result', metric.name.lower()]
        for key, value in json_obj.items():
            key_lower = key.lower()
            for keyword in score_keywords:
                if keyword in key_lower and isinstance(value, (int, float, str)):
                    try:
                        # Try to convert to number
                        return float(value) if '.' in str(value) else int(value)
                    except:
                        pass
        
        # Step 6: Last resort - find ANY numeric value in the JSON
        for key, value in json_obj.items():
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                try:
                    return float(value) if '.' in value else int(value)
                except:
                    pass
        
        # If nothing found, return 0
        return 0
    
    def _smart_extract_explanation(self, json_obj: dict) -> str:
        """
        BULLETPROOF explanation extraction.
        
        Tries multiple common keys for explanations.
        """
        explanation_keys = [
            "explanation", "Explanation", 
            "reason", "Reason", 
            "comment", "Comment", 
            "rationale", "Rationale", 
            "justification", "Justification",
            "reasoning", "Reasoning",
            "details", "Details",
            "description", "Description"
        ]
        
        # Try standard keys
        for key in explanation_keys:
            if key in json_obj:
                return json_obj[key]
        
        # Case-insensitive search
        for key, value in json_obj.items():
            key_lower = key.lower()
            if any(exp_key.lower() in key_lower for exp_key in explanation_keys):
                if isinstance(value, str):
                    return value
        
        # Look for any string value that's not too short
        for key, value in json_obj.items():
            if isinstance(value, str) and len(value) > 10:
                return value
        
        return "No explanation provided"
    
    def _fallback_parse(self, content: str, metric: MetricConfig) -> tuple:
        """Fallback parsing when JSON parsing fails."""
        score = 0
        explanation = content
        
        if metric.metric_type == MetricType.BINARY:
            if any(word in content.lower() for word in ['pass', 'correct', 'accurate', 'yes', 'true', '1']):
                score = 1
        
        elif metric.metric_type == MetricType.SCALE_1_5:
            numbers = re.findall(r'\b[1-5]\b', content)
            if numbers:
                score = int(numbers[0])
        
        elif metric.metric_type == MetricType.PERCENTAGE:
            percentages = re.findall(r'(\d+(?:\.\d+)?)[%]?', content)
            if percentages:
                score = float(percentages[0])
                # Keep raw score - _normalize_score will handle range conversion
        
        return score, explanation
    
    def _normalize_score(self, score: Any, metric_type: MetricType) -> float:
        """Normalize score to valid range."""
        try:
            score = float(score)
        except (ValueError, TypeError):
            return 0.0
        
        if metric_type == MetricType.BINARY:
            return 1.0 if score > 0.5 else 0.0
        elif metric_type == MetricType.SCALE_1_5:
            return max(1.0, min(5.0, score))
        elif metric_type == MetricType.PERCENTAGE:
            # Keep percentage in 0-100 range to match thresholds in CSV
            # If LLM returns 0-1 range, convert to 0-100
            if score <= 1.0:
                score = score * 100
            return max(0.0, min(100.0, score))
        
        return score
    
    def evaluate_dataset(self, evaluation_data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate entire dataset with all metrics."""
        print(f"\n🔍 Evaluating {len(evaluation_data)} samples with {len(self.metrics)} metrics...")
        
        results = []
        
        for idx, row in evaluation_data.iterrows():
            sample_id = row.get('sample_id', f'sample_{idx}')
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"\n📝 Sample {idx + 1}/{len(evaluation_data)}: {sample_id}")
            
            for metric in self.metrics:
                print(f"   📊 {metric.name}...", end=' ')
                
                ground_truth_data = {}
                for col in evaluation_data.columns:
                    if col not in ['sample_id', 'prompt', 'response']:
                        ground_truth_data[col] = row.get(col, '')
                
                result = self.evaluate_single(prompt, response, ground_truth_data, metric, idx)
                
                print(f"{result['status']}")
                
                results.append({
                    'sample_id': sample_id,
                    'metric_name': metric.name,
                    'metric_type': metric.metric_type.value,
                    'score': result['score'],
                    'threshold': metric.threshold,
                    'status': result['status'],
                    'explanation': result['explanation'],
                    'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    'response': response[:100] + "..." if len(response) > 100 else response
                })
        
        print(f"\n Evaluation complete!")
        return pd.DataFrame(results)

print(" LLM Judge Evaluator defined with  JSON parsing")

# COMMAND ----------

# MAGIC %md
#MAGIC ## Cell 7: Run Evaluation
#MAGIC 
#MAGIC **Purpose**: This cell executes the actual evaluation process, running all configured metrics against all evaluation samples.
#MAGIC 
#MAGIC **What it does**:
#MAGIC - Loads metrics configuration from the CSV file and auto-generates evaluation prompts from grading rubrics
#MAGIC - Creates the LLM Judge Evaluator instance with the selected model and metrics
#MAGIC - Runs the evaluation process for all samples and all metrics
#MAGIC - Displays real-time progress as each sample and metric is evaluated
#MAGIC - Generates comprehensive results including scores, explanations, and pass/fail status
#MAGIC - Logs all results to MLflow for experiment tracking and comparison
#MAGIC - Provides detailed summary statistics including overall pass rates and per-metric performance
#MAGIC 
#MAGIC **When to run**: Run this cell after Cells 1-6 to execute the evaluation
#MAGIC 
#MAGIC **Expected output**: Real-time evaluation progress, comprehensive results summary, and MLflow logging confirmation

# COMMAND ----------

def auto_generate_evaluation_prompt(metric_name: str, metric_type: str, description: str, grading_rubric: str) -> str:
    """
    Auto-generate evaluation prompt from grading rubric.
    PM just provides grading rubric, code handles the rest!
    """
    
    # Add grading rubric section
    rubric_section = f"\n**Grading Rubric:**\n{grading_rubric}\n" if grading_rubric else ""
    
    # Build complete evaluation prompt
    prompt = f"""You are an expert evaluator. Your task: {description}

{rubric_section}
**Evaluation Details:**
- User Query: {{prompt}}
- AI Response: {{response}}
- Ground Truth Reference: {{ground_truth}}

**Instructions:**
Carefully evaluate the AI response using the grading rubric above.

**Required Output Format:**
Return ONLY a valid JSON object with these two fields:
{{
  "score": <your_score>,
  "explanation": "Brief explanation of your score"
}}

Do not include any other text outside the JSON object."""
    
    return prompt

def load_metrics_from_csv():
    """Load metrics from CSV and auto-generate evaluation prompts."""
    if METRICS_CONFIG_DATA is None:
        return []
    
    metric_configs = []
    for _, row in METRICS_CONFIG_DATA.iterrows():
        metric_type_str = row['type'].strip().lower()
        if metric_type_str == 'binary':
            metric_type = MetricType.BINARY
        elif metric_type_str in ['1-5_scale', 'scale_1_5']:
            metric_type = MetricType.SCALE_1_5
        elif metric_type_str == 'percentage':
            metric_type = MetricType.PERCENTAGE
        else:
            metric_type = MetricType.BINARY
        
        # Check if using new format (grading_rubric) or old format (evaluation_prompt)
        if 'grading_rubric' in row and pd.notna(row.get('grading_rubric')):
            # NEW FORMAT: Auto-generate evaluation prompt from grading rubric
            grading_rubric = str(row['grading_rubric']).strip()
            prompt_template = auto_generate_evaluation_prompt(
                metric_name=row['name'].strip(),
                metric_type=metric_type_str,
                description=row.get('description', '').strip(),
                grading_rubric=grading_rubric
            )
            print(f"✅ {row['name'].strip()} - Auto-generated prompt from grading rubric")
        elif 'evaluation_prompt' in row and pd.notna(row.get('evaluation_prompt')):
            # OLD FORMAT: Use evaluation_prompt directly (backward compatibility)
            prompt_template = row['evaluation_prompt'].strip()
            print(f"✅ {row['name'].strip()} - Using provided evaluation_prompt")
        else:
            # Fallback: Generate basic prompt from description
            prompt_template = auto_generate_evaluation_prompt(
                metric_name=row['name'].strip(),
                metric_type=metric_type_str,
                description=row.get('description', '').strip(),
                grading_rubric=""
            )
            print(f"⚠️ {row['name'].strip()} - No rubric or prompt provided, using basic prompt")
        
        # Ground truth file matching (PM provides filename, code matches to uploaded files)
        gt_file_value = row.get('ground_truth_file_path', '').strip()
        
        metric_config = MetricConfig(
            name=row['name'].strip(),
            metric_type=metric_type,
            description=row.get('description', '').strip(),
            prompt_template=prompt_template,
            threshold=float(row['threshold']),
            ground_truth_column=row['ground_truth_column'].strip(),
            ground_truth_file_path=gt_file_value
        )
        
        # Show which GT file this metric will use
        if gt_file_value:
            gt_filename = os.path.basename(gt_file_value)
            print(f"   📚 Ground truth: {gt_filename} → column '{row['ground_truth_column'].strip()}'")
        
        metric_configs.append(metric_config)
    
    return metric_configs

# Load metrics
metric_configs = load_metrics_from_csv()

if not metric_configs:
    print("❌ No valid metrics!")
else:
    # Create evaluator
    evaluator = LLMJudgeEvaluator(
        judge_model=JUDGE_MODEL,
        metrics=metric_configs,
        ground_truth_data=ground_truth_data
    )
    
    # Run evaluation
    print("="*60)
    print("🚀 STARTING EVALUATION")
    print("="*60)
    
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE in {eval_time:.1f}s")
    print(f"{'='*60}")
    
    # Display summary
    total = len(results_df)
    passed = len(results_df[results_df['status'] == '✅'])
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Evaluations: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"{'='*60}")
    
    # Per-metric results
    for metric_name in results_df['metric_name'].unique():
        metric_results = results_df[results_df['metric_name'] == metric_name]
        metric_passed = len(metric_results[metric_results['status'] == '✅'])
        metric_total = len(metric_results)
        metric_pass_rate = (metric_passed / metric_total) * 100 if metric_total > 0 else 0
        avg_score = metric_results['score'].mean()
        
        print(f"\n🔹 {metric_name}:")
        print(f"   Pass Rate: {metric_pass_rate:.1f}% ({metric_passed}/{metric_total})")
        print(f"   Avg Score: {avg_score:.2f}")
        print(f"   Threshold: {metric_results['threshold'].iloc[0]}")
    
    # Store globally
    globals()['results_df'] = results_df
    
    # Log to MLflow
    mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/llm_evaluation_experiment")
    
    with mlflow.start_run(run_name=f"eval_{JUDGE_MODEL}_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("judge_model", JUDGE_MODEL)
        mlflow.log_param("num_samples", len(EVALUATION_DATA))
        mlflow.log_param("metrics", ",".join([cfg.name for cfg in metric_configs]))
        
        mlflow.log_metric("overall_pass_rate", pass_rate)
        mlflow.log_metric("total_evaluations", total)
        mlflow.log_metric("passed_evaluations", passed)
        
        for metric_name in results_df['metric_name'].unique():
            metric_results = results_df[results_df['metric_name'] == metric_name]
            scores = metric_results['score'].tolist()
            numeric_scores = [s for s in scores if isinstance(s, (int, float))]
            
            if numeric_scores:
                mean_score = sum(numeric_scores) / len(numeric_scores)
                threshold = metric_results['threshold'].iloc[0]
                pass_count = sum(1 for s in numeric_scores if s >= threshold)
                pass_rate_metric = (pass_count / len(numeric_scores)) * 100
                
                mlflow.log_metric(f"{metric_name}_mean", float(mean_score))
                mlflow.log_metric(f"{metric_name}_pass_rate", float(pass_rate_metric))
        
        # Save results
        out_dir = "/tmp/llm_eval_artifacts"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"results_{int(time.time())}.csv")
        results_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="tables")
    
    print(f"\n✅ Results logged to MLflow")

# COMMAND ----------

# MAGIC %md
#MAGIC ## Cell 8: Export Results
#MAGIC 
#MAGIC **Purpose**: This cell exports the evaluation results to a CSV file for further analysis and sharing.
#MAGIC 
#MAGIC **What it does**:
#MAGIC - Saves the complete evaluation results to a timestamped CSV file in the user's workspace
#MAGIC - Displays a sample of the results for quick review
#MAGIC - Provides the file path where results are saved for easy access
#MAGIC - Shows the total number of evaluations that were exported
#MAGIC 
#MAGIC **When to run**: Run this cell after Cell 7 to save your evaluation results
#MAGIC 
#MAGIC **Expected output**: File path confirmation and sample results display

# COMMAND ----------

if 'results_df' in globals() and not results_df.empty:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    workspace_path = f"/Workspace/Users/{user_name}"
    
    filename = f"{workspace_path}/llm_evaluation_results_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    
    print(f"📊 Results exported to: {filename}")
    print(f"✅ {len(results_df)} evaluations saved")
    
    # Display first few rows
    print(f"\n📋 Sample Results:")
    display(results_df.head(10))
else:
    print("❌ No results to export")

# COMMAND ----------

# MAGIC %md
#MAGIC ## Cell 9: MLflow Dashboard
#MAGIC 
#MAGIC **Purpose**: This cell creates comprehensive visualizations and dashboards using MLflow experiment data to track evaluation performance over time.
#MAGIC 
#MAGIC **What it does**:
#MAGIC - Loads historical evaluation data from MLflow experiments
#MAGIC - Creates multiple interactive visualizations including heatmaps, line charts, and comparison charts
#MAGIC - Displays run-to-run performance comparisons and metric evolution over time
#MAGIC - Shows model comparison charts if multiple models have been used
#MAGIC - Provides improvement tracking to see how metrics have changed over time
#MAGIC - Generates summary statistics and performance breakdowns
#MAGIC - Creates professional dashboards suitable for stakeholder presentations
#MAGIC 
#MAGIC **When to run**: Run this cell after Cell 7 to view historical performance and trends
#MAGIC 
#MAGIC **Expected output**: Multiple interactive charts and comprehensive performance analysis

# COMMAND ----------

# Databricks notebook source
# MAGIC %md
#MAGIC ## Cell 9: MLflow Dashboard 📊🔥
#MAGIC **Beautiful visualizations tracking evaluations over time**

# COMMAND ----------

# MLflow is pre-installed in Databricks - just import it
import mlflow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime

# Get experiment
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
MLFLOW_EXPERIMENT_PATH = f"/Users/{user_name}/llm_evaluation_experiment"

print("🔍 Loading MLflow experiment data...\n")

exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_PATH)

if exp is None:
    print(f"❌ Experiment not found: {MLFLOW_EXPERIMENT_PATH}")
    print(f"💡 Run an evaluation first (Cell 7) to create the experiment.")
else:
    # Get all runs
    runs_df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=100
    )
    
    num_runs = len(runs_df)
    print(f"✅ Found {num_runs} experiment run(s)\n")
    
    if num_runs == 0:
        print("💡 No completed runs yet. Run an evaluation first (Cell 7).")
    
    elif num_runs > 0:
        
        # Get metric columns dynamically (works with ANY custom metrics!)
        metric_mean_cols = [c for c in runs_df.columns if c.startswith("metrics.") and c.endswith("_mean")]
        metric_pass_cols = [c for c in runs_df.columns if c.startswith("metrics.") and c.endswith("_pass_rate")]
        
        # Extract metric names
        metric_names = [c.replace('metrics.', '').replace('_pass_rate', '') for c in metric_pass_cols]
        
        print(f"📊 Tracking {len(metric_names)} custom metrics:")
        for m in metric_names:
            print(f"   • {m}")
        print()
        
        # ============================================================
        # 1. LATEST RUN SUMMARY
        # ============================================================
        print("="*70)
        print("📈 LATEST RUN SUMMARY")
        print("="*70)
        
        latest = runs_df.iloc[0]
        
        print(f"\n🆔 Run ID: {latest['run_id'][:12]}...")
        print(f"🤖 Model: {latest.get('params.model', latest.get('params.judge_model', 'N/A'))}")
        print(f"📊 Samples: {latest.get('params.samples', latest.get('params.num_samples', 'N/A'))}")
        print(f"📅 Time: {latest['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duration: {(latest['end_time'] - latest['start_time']).total_seconds():.1f}s")
        
        if 'metrics.pass_rate' in latest:
            print(f"\n🎯 Overall Pass Rate: {latest['metrics.pass_rate']:.1f}%")
        
        print(f"\n📊 Metric Performance:")
        for col in metric_pass_cols:
            metric_name = col.replace('metrics.', '').replace('_pass_rate', '')
            if pd.notna(latest[col]):
                pass_rate = latest[col]
                mean_col = f"metrics.{metric_name}_mean"
                mean_score = latest.get(mean_col, 0)
                
                # Status icon
                icon = "✅" if pass_rate >= 70 else "⚠️" if pass_rate >= 50 else "❌"
                print(f"   {icon} {metric_name:30s} - Pass: {pass_rate:5.1f}% | Avg: {mean_score:.2f}")
        
        print("\n" + "="*70 + "\n")
        
        # ============================================================
        # 2. RUN-TO-RUN COMPARISON HEATMAP
        # ============================================================
        if num_runs >= 2:
            print("🔥 Creating Run-to-Run Comparison Heatmap...\n")
            
            # Create matrix: rows = runs, columns = metrics
            heatmap_data = []
            run_labels = []
            
            for idx, row in runs_df.iterrows():
                run_id_short = row['run_id'][:8]
                run_time = row['start_time'].strftime('%m/%d %H:%M')
                run_label = f"{run_time}\n{run_id_short}"
                run_labels.append(run_label)
                
                # Get pass rates for this run
                pass_rates = []
                for metric in metric_names:
                    col = f"metrics.{metric}_pass_rate"
                    if col in row and pd.notna(row[col]):
                        pass_rates.append(row[col])
                    else:
                        pass_rates.append(0)
                
                heatmap_data.append(pass_rates)
            
            # Convert to numpy array
            heatmap_matrix = np.array(heatmap_data)
            
            # Create heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_matrix,
                x=metric_names,
                y=run_labels,
                colorscale=[
                    [0.0, '#d73027'],   # Red
                    [0.25, '#fc8d59'],  # Orange
                    [0.5, '#fee090'],   # Yellow
                    [0.75, '#91cf60'],  # Light green
                    [1.0, '#1a9850']    # Dark green
                ],
                text=heatmap_matrix,
                texttemplate='%{text:.0f}%',
                textfont={"size": 10},
                colorbar=dict(title="Pass Rate %"),
                hoverongaps=False,
                hovertemplate='Run: %{y}<br>Metric: %{x}<br>Pass Rate: %{z:.1f}%<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title={
                    'text': "🔥 Run-to-Run Performance Heatmap (All Custom Metrics)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2c3e50'}
                },
                xaxis_title="Metrics",
                yaxis_title="Experiment Runs (Time & Run ID)",
                height=max(400, num_runs * 50),
                width=max(900, len(metric_names) * 120),
                xaxis={'tickangle': -45, 'side': 'bottom'},
                yaxis={'autorange': 'reversed'},  # Latest run at top
                font=dict(size=11)
            )
            
            displayHTML(pio.to_html(fig_heatmap, include_plotlyjs='cdn'))
        
        # ============================================================
        # 3. METRIC EVOLUTION OVER TIME
        # ============================================================
        if num_runs >= 2:
            print("📈 Creating Metric Evolution Timeline...\n")
            
            fig_evolution = go.Figure()
            
            for metric in metric_names:
                col = f"metrics.{metric}_pass_rate"
                if col in runs_df.columns:
                    # Sort by time for proper line chart
                    sorted_df = runs_df.sort_values('start_time')
                    
                    fig_evolution.add_trace(go.Scatter(
                        x=sorted_df['start_time'],
                        y=sorted_df[col],
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=3),
                        marker=dict(size=10),
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Pass Rate: %{y:.1f}%<extra></extra>'
                    ))
            
            fig_evolution.update_layout(
                title={
                    'text': "📈 Metric Performance Evolution Over Time",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2c3e50'}
                },
                xaxis_title="Experiment Run Time",
                yaxis_title="Pass Rate (%)",
                height=600,
                hovermode='x unified',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                font=dict(size=12)
            )
            
            # Add threshold lines
            fig_evolution.add_hline(y=70, line_dash="dash", line_color="green", 
                                   annotation_text="Good (70%)", annotation_position="right")
            fig_evolution.add_hline(y=50, line_dash="dash", line_color="orange", 
                                   annotation_text="Okay (50%)", annotation_position="right")
            
            displayHTML(pio.to_html(fig_evolution, include_plotlyjs='cdn'))
        
        # ============================================================
        # 4. METRIC PERFORMANCE COMPARISON (Latest vs Best)
        # ============================================================
        if num_runs >= 2:
            print("🏆 Creating Latest vs Best Performance Comparison...\n")
            
            # Get best run for each metric
            comparison_data = []
            
            for metric in metric_names:
                pass_col = f"metrics.{metric}_pass_rate"
                mean_col = f"metrics.{metric}_mean"
                
                # FIXED: Check if columns exist before accessing
                if pass_col in runs_df.columns:
                    latest_pass = latest.get(pass_col, 0) if pd.notna(latest.get(pass_col)) else 0
                    best_pass = runs_df[pass_col].max() if pd.notna(runs_df[pass_col].max()) else 0
                    
                    # Only include mean if column exists
                    if mean_col in runs_df.columns:
                        latest_mean = latest.get(mean_col, 0) if pd.notna(latest.get(mean_col)) else 0
                        best_mean = runs_df[mean_col].max() if pd.notna(runs_df[mean_col].max()) else 0
                    else:
                        latest_mean = 0
                        best_mean = 0
                    
                    comparison_data.append({
                        'metric': metric,
                        'latest_pass': latest_pass,
                        'best_pass': best_pass,
                        'latest_mean': latest_mean,
                        'best_mean': best_mean
                    })
            
            if comparison_data:  # Only create chart if we have data
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create grouped bar chart
                fig_comparison = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Pass Rate: Latest vs Best', 'Average Score: Latest vs Best'),
                    specs=[[{"type": "bar"}, {"type": "bar"}]]
                )
                
                # Pass rates
                fig_comparison.add_trace(
                    go.Bar(
                        x=comparison_df['metric'],
                        y=comparison_df['latest_pass'],
                        name='Latest Run',
                        marker_color='#3498db',
                        text=comparison_df['latest_pass'].round(1),
                        texttemplate='%{text}%',
                        textposition='outside'
                    ),
                    row=1, col=1
                )
                
                fig_comparison.add_trace(
                    go.Bar(
                        x=comparison_df['metric'],
                        y=comparison_df['best_pass'],
                        name='Best Ever',
                        marker_color='#2ecc71',
                        text=comparison_df['best_pass'].round(1),
                        texttemplate='%{text}%',
                        textposition='outside'
                    ),
                    row=1, col=1
                )
                
                # Average scores
                fig_comparison.add_trace(
                    go.Bar(
                        x=comparison_df['metric'],
                        y=comparison_df['latest_mean'],
                        name='Latest Run',
                        marker_color='#3498db',
                        text=comparison_df['latest_mean'].round(2),
                        texttemplate='%{text}',
                        textposition='outside',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                fig_comparison.add_trace(
                    go.Bar(
                        x=comparison_df['metric'],
                        y=comparison_df['best_mean'],
                        name='Best Ever',
                        marker_color='#2ecc71',
                        text=comparison_df['best_mean'].round(2),
                        texttemplate='%{text}',
                        textposition='outside',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                fig_comparison.update_layout(
                    title={
                        'text': "🏆 Latest Run vs Historical Best Performance",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18, 'color': '#2c3e50'}
                    },
                    height=600,
                    barmode='group',
                    font=dict(size=11)
                )
                
                fig_comparison.update_xaxes(tickangle=-45, row=1, col=1)
                fig_comparison.update_xaxes(tickangle=-45, row=1, col=2)
                fig_comparison.update_yaxes(title_text="Pass Rate (%)", row=1, col=1)
                fig_comparison.update_yaxes(title_text="Average Score", row=1, col=2)
                
                displayHTML(pio.to_html(fig_comparison, include_plotlyjs='cdn'))
        
        # ============================================================
        # 5. MODEL COMPARISON (if multiple models used)
        # ============================================================
        if 'params.model' in runs_df.columns or 'params.judge_model' in runs_df.columns:
            model_col = 'params.model' if 'params.model' in runs_df.columns else 'params.judge_model'
            unique_models = runs_df[model_col].dropna().unique()
            
            if len(unique_models) > 1:
                print("🤖 Creating Model Comparison...\n")
                
                # Group by model and calculate average pass rates
                model_comparison = []
                
                for model in unique_models:
                    model_runs = runs_df[runs_df[model_col] == model]
                    
                    for metric in metric_names:
                        pass_col = f"metrics.{metric}_pass_rate"
                        if pass_col in model_runs.columns:
                            avg_pass = model_runs[pass_col].mean()
                            if pd.notna(avg_pass):
                                model_comparison.append({
                                    'model': model,
                                    'metric': metric,
                                    'avg_pass_rate': avg_pass
                                })
                
                if model_comparison:  # Only create chart if we have data
                    model_comp_df = pd.DataFrame(model_comparison)
                    
                    # Create grouped bar chart
                    fig_models = go.Figure()
                    
                    for model in unique_models:
                        model_data = model_comp_df[model_comp_df['model'] == model]
                        
                        if len(model_data) > 0:
                            fig_models.add_trace(go.Bar(
                                x=model_data['metric'],
                                y=model_data['avg_pass_rate'],
                                name=model,
                                text=model_data['avg_pass_rate'].round(1),
                                texttemplate='%{text}%',
                                textposition='outside'
                            ))
                    
                    fig_models.update_layout(
                        title={
                            'text': "🤖 Model Comparison: Average Pass Rates",
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'size': 18, 'color': '#2c3e50'}
                        },
                        xaxis_title="Metrics",
                        yaxis_title="Average Pass Rate (%)",
                        height=600,
                        barmode='group',
                        xaxis={'tickangle': -45},
                        font=dict(size=12)
                    )
                    
                    displayHTML(pio.to_html(fig_models, include_plotlyjs='cdn'))
        
        # ============================================================
        # 6. IMPROVEMENT TRACKING
        # ============================================================
        if num_runs >= 3:
            print("📊 Creating Improvement Tracking Dashboard...\n")
            
            # Calculate improvement from first run to latest
            first_run = runs_df.iloc[-1]  # Oldest run
            latest_run = runs_df.iloc[0]  # Newest run
            
            improvements = []
            
            for metric in metric_names:
                pass_col = f"metrics.{metric}_pass_rate"
                
                if pass_col in runs_df.columns:
                    first_val = first_run.get(pass_col, 0)
                    latest_val = latest_run.get(pass_col, 0)
                    
                    if pd.notna(first_val) and pd.notna(latest_val):
                        improvement = latest_val - first_val
                        
                        improvements.append({
                            'metric': metric,
                            'first': first_val,
                            'latest': latest_val,
                            'improvement': improvement
                        })
            
            if improvements:  # Only create chart if we have data
                improve_df = pd.DataFrame(improvements).sort_values('improvement')
                
                # Create waterfall-style chart
                fig_improve = go.Figure()
                
                colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in improve_df['improvement']]
                
                fig_improve.add_trace(go.Bar(
                    x=improve_df['metric'],
                    y=improve_df['improvement'],
                    marker_color=colors,
                    text=improve_df['improvement'].round(1),
                    texttemplate='%{text:+.1f}%',
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>First: %{customdata[0]:.1f}%<br>Latest: %{customdata[1]:.1f}%<br>Change: %{y:+.1f}%<extra></extra>',
                    customdata=improve_df[['first', 'latest']].values
                ))
                
                fig_improve.update_layout(
                    title={
                        'text': f"📊 Improvement Tracking: First Run vs Latest ({num_runs} runs)",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18, 'color': '#2c3e50'}
                    },
                    xaxis_title="Metrics",
                    yaxis_title="Change in Pass Rate (%)",
                    height=600,
                    xaxis={'tickangle': -45},
                    font=dict(size=12)
                )
                
                # Add zero line
                fig_improve.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
                
                displayHTML(pio.to_html(fig_improve, include_plotlyjs='cdn'))
                
                # Print summary
                print("\n" + "="*70)
                print("📊 IMPROVEMENT SUMMARY")
                print("="*70)
                
                improved = improve_df[improve_df['improvement'] > 0]
                declined = improve_df[improve_df['improvement'] < 0]
                unchanged = improve_df[improve_df['improvement'] == 0]
                
                print(f"\n✅ Improved: {len(improved)} metrics")
                for _, row in improved.iterrows():
                    print(f"   🔼 {row['metric']:30s} +{row['improvement']:5.1f}% ({row['first']:.1f}% → {row['latest']:.1f}%)")
                
                if len(declined) > 0:
                    print(f"\n⚠️  Declined: {len(declined)} metrics")
                    for _, row in declined.iterrows():
                        print(f"   🔽 {row['metric']:30s} {row['improvement']:5.1f}% ({row['first']:.1f}% → {row['latest']:.1f}%)")
                
                if len(unchanged) > 0:
                    print(f"\n➡️  Unchanged: {len(unchanged)} metrics")
                
                print("\n" + "="*70 + "\n")
        
        # ============================================================
        # 7. SUMMARY STATISTICS TABLE
        # ============================================================
        print("="*70)
        print("📊 MLFLOW EXPERIMENT SUMMARY")
        print("="*70)
        
        print(f"\n🆔 Experiment: {exp.name}")
        print(f"📁 Experiment ID: {exp.experiment_id}")
        print(f"📊 Total Runs: {num_runs}")
        print(f"📅 First Run: {runs_df.iloc[-1]['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 Latest Run: {runs_df.iloc[0]['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Metrics Tracked: {len(metric_names)}")
        
        if 'metrics.pass_rate' in runs_df.columns:
            overall_best = runs_df['metrics.pass_rate'].max()
            overall_avg = runs_df['metrics.pass_rate'].mean()
            if pd.notna(overall_best) and pd.notna(overall_avg):
                print(f"\n🏆 Best Overall Pass Rate: {overall_best:.1f}%")
                print(f"📊 Average Overall Pass Rate: {overall_avg:.1f}%")
        
        print("\n" + "="*70)
        print("✅ MLflow Dashboard Complete!")
        print("="*70)

# COMMAND ----------

# MAGIC %md
#MAGIC ## Cell 10: Advanced Visualizations
#MAGIC 
#MAGIC **Purpose**: This cell creates detailed, interactive visualizations to analyze the current evaluation results in depth.
#MAGIC 
#MAGIC **What it does**:
#MAGIC - Creates a comprehensive heatmap showing sample vs metric performance
#MAGIC - Generates metric performance overview charts with pass rates and average scores
#MAGIC - Produces violin plots to show score distributions for each metric
#MAGIC - Creates sample performance scorecards ranking samples by overall performance
#MAGIC - Generates correlation heatmaps to identify relationships between metrics
#MAGIC - Displays pass/fail breakdown charts by metric
#MAGIC - Creates metric type analysis showing performance by evaluation type
#MAGIC - Provides detailed summary statistics and performance breakdowns
#MAGIC 
#MAGIC **When to run**: Run this cell after Cell 7 to analyze your current evaluation results
#MAGIC 
#MAGIC **Expected output**: Multiple interactive charts and comprehensive statistical analysis

# COMMAND ----------

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

if 'results_df' not in globals() or results_df.empty:
    print("❌ No results available. Run evaluation first (Cell 7).")
else:
    print("🎨 Creating advanced visualizations...\n")
    
    # ============================================================
    # 1. HEATMAP: Sample vs Metric Performance
    # ============================================================
    print("📊 1. Creating Sample vs Metric Heatmap...")
    
    # Create pivot table for heatmap
    heatmap_data = results_df.pivot_table(
        values='score',
        index='sample_id',
        columns='metric_name',
        aggfunc='first'
    )
    
    # Create custom colorscale
    # Green for high scores, Yellow for medium, Red for low
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[
            [0.0, '#d73027'],   # Red for 0
            [0.25, '#fc8d59'],  # Orange
            [0.5, '#fee090'],   # Yellow
            [0.75, '#91cf60'],  # Light green
            [1.0, '#1a9850']    # Dark green for 1
        ],
        text=heatmap_data.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Score"),
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title={
            'text': "🔥 Sample vs Metric Performance Heatmap",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="Metrics",
        yaxis_title="Samples",
        height=max(400, len(heatmap_data) * 40),
        width=max(800, len(heatmap_data.columns) * 100),
        xaxis={'tickangle': -45},
        font=dict(size=12)
    )
    
    displayHTML(pio.to_html(fig_heatmap, include_plotlyjs='cdn'))
    
    # ============================================================
    # 2. METRIC PERFORMANCE OVERVIEW
    # ============================================================
    print("📊 2. Creating Metric Performance Overview...")
    
    # Calculate metrics
    metric_summary = results_df.groupby('metric_name').agg({
        'score': ['mean', 'std', 'min', 'max'],
        'status': lambda x: (x == '✅').sum() / len(x) * 100  # Pass rate
    }).round(2)
    
    metric_summary.columns = ['Mean Score', 'Std Dev', 'Min', 'Max', 'Pass Rate %']
    metric_summary = metric_summary.reset_index()
    
    # Create subplot with 2 charts
    fig_metrics = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Pass Rate by Metric', 'Average Score by Metric'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Pass rate chart
    fig_metrics.add_trace(
        go.Bar(
            x=metric_summary['metric_name'],
            y=metric_summary['Pass Rate %'],
            marker=dict(
                color=metric_summary['Pass Rate %'],
                colorscale='RdYlGn',
                showscale=False,
                line=dict(color='black', width=1)
            ),
            text=metric_summary['Pass Rate %'].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            name='Pass Rate'
        ),
        row=1, col=1
    )
    
    # Average score chart
    fig_metrics.add_trace(
        go.Bar(
            x=metric_summary['metric_name'],
            y=metric_summary['Mean Score'],
            marker=dict(
                color=metric_summary['Mean Score'],
                colorscale='Viridis',
                showscale=False,
                line=dict(color='black', width=1)
            ),
            text=metric_summary['Mean Score'].round(2),
            texttemplate='%{text}',
            textposition='outside',
            name='Avg Score',
            error_y=dict(
                type='data',
                array=metric_summary['Std Dev'],
                visible=True,
                color='rgba(0,0,0,0.3)'
            )
        ),
        row=1, col=2
    )
    
    fig_metrics.update_layout(
        title={
            'text': "📈 Metric Performance Dashboard",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        height=500,
        showlegend=False,
        font=dict(size=11)
    )
    
    fig_metrics.update_xaxes(tickangle=-45, row=1, col=1)
    fig_metrics.update_xaxes(tickangle=-45, row=1, col=2)
    fig_metrics.update_yaxes(title_text="Pass Rate (%)", row=1, col=1)
    fig_metrics.update_yaxes(title_text="Average Score", row=1, col=2)
    
    displayHTML(pio.to_html(fig_metrics, include_plotlyjs='cdn'))
    
    # ============================================================
    # 3. SCORE DISTRIBUTION: Violin Plots
    # ============================================================
    print("📊 3. Creating Score Distribution Analysis...")
    
    fig_violin = go.Figure()
    
    for metric in results_df['metric_name'].unique():
        metric_data = results_df[results_df['metric_name'] == metric]
        
        fig_violin.add_trace(go.Violin(
            y=metric_data['score'],
            name=metric,
            box_visible=True,
            meanline_visible=True,
            fillcolor='rgba(0,100,200,0.3)',
            line_color='rgb(0,100,200)',
            opacity=0.7,
            points='all',
            jitter=0.3,
            pointpos=-0.5,
            hovertemplate='<b>%{fullData.name}</b><br>Score: %{y:.2f}<extra></extra>'
        ))
    
    fig_violin.update_layout(
        title={
            'text': "🎻 Score Distribution by Metric (Violin Plot)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        yaxis_title="Score",
        xaxis_title="Metric",
        height=600,
        showlegend=False,
        xaxis={'tickangle': -45},
        font=dict(size=12)
    )
    
    displayHTML(pio.to_html(fig_violin, include_plotlyjs='cdn'))
    
    # ============================================================
    # 4. SAMPLE PERFORMANCE SCORECARD
    # ============================================================
    print("📊 4. Creating Sample Performance Scorecard...")
    
    sample_summary = results_df.groupby('sample_id').agg({
        'score': 'mean',
        'status': lambda x: (x == '✅').sum() / len(x) * 100
    }).round(2)
    
    sample_summary.columns = ['Avg Score', 'Pass Rate %']
    sample_summary = sample_summary.reset_index()
    sample_summary = sample_summary.sort_values('Pass Rate %', ascending=True)
    
    fig_samples = go.Figure()
    
    # Add bars
    fig_samples.add_trace(go.Bar(
        y=sample_summary['sample_id'].astype(str),
        x=sample_summary['Pass Rate %'],
        orientation='h',
        marker=dict(
            color=sample_summary['Pass Rate %'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Pass Rate %"),
            line=dict(color='black', width=1)
        ),
        text=sample_summary['Pass Rate %'].round(1),
        texttemplate='%{text}%',
        textposition='outside',
        hovertemplate='<b>Sample %{y}</b><br>Pass Rate: %{x:.1f}%<br>Avg Score: %{customdata:.2f}<extra></extra>',
        customdata=sample_summary['Avg Score']
    ))
    
    fig_samples.update_layout(
        title={
            'text': "📋 Sample Performance Scorecard",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="Pass Rate (%)",
        yaxis_title="Sample ID",
        height=max(400, len(sample_summary) * 40),
        showlegend=False,
        font=dict(size=12)
    )
    
    displayHTML(pio.to_html(fig_samples, include_plotlyjs='cdn'))
    
    # ============================================================
    # 5. METRIC CORRELATION HEATMAP
    # ============================================================
    print("📊 5. Creating Metric Correlation Heatmap...")
    
    # Calculate correlation between metrics
    correlation_data = heatmap_data.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_data.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation"),
        hoverongaps=False
    ))
    
    fig_corr.update_layout(
        title={
            'text': "🔗 Metric Correlation Matrix",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="Metrics",
        yaxis_title="Metrics",
        height=max(500, len(correlation_data) * 50),
        width=max(700, len(correlation_data) * 50),
        xaxis={'tickangle': -45},
        font=dict(size=11)
    )
    
    displayHTML(pio.to_html(fig_corr, include_plotlyjs='cdn'))
    
    # ============================================================
    # 6. PASS/FAIL BREAKDOWN BY METRIC
    # ============================================================
    print("📊 6. Creating Pass/Fail Breakdown...")
    
    # Calculate pass/fail counts
    pass_fail = results_df.groupby(['metric_name', 'status']).size().unstack(fill_value=0)
    
    if '✅' not in pass_fail.columns:
        pass_fail['✅'] = 0
    if '❌' not in pass_fail.columns:
        pass_fail['❌'] = 0
    
    pass_fail = pass_fail.reset_index()
    
    fig_passfail = go.Figure()
    
    fig_passfail.add_trace(go.Bar(
        x=pass_fail['metric_name'],
        y=pass_fail['✅'],
        name='Pass ✅',
        marker_color='rgb(26,152,80)',
        text=pass_fail['✅'],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))
    
    fig_passfail.add_trace(go.Bar(
        x=pass_fail['metric_name'],
        y=pass_fail['❌'],
        name='Fail ❌',
        marker_color='rgb(215,48,39)',
        text=pass_fail['❌'],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))
    
    fig_passfail.update_layout(
        title={
            'text': "✅❌ Pass/Fail Breakdown by Metric",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        barmode='stack',
        xaxis_title="Metric",
        yaxis_title="Count",
        height=500,
        xaxis={'tickangle': -45},
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    displayHTML(pio.to_html(fig_passfail, include_plotlyjs='cdn'))
    
    # ============================================================
    # 7. METRIC TYPE COMPARISON
    # ============================================================
    print("📊 7. Creating Metric Type Comparison...")
    
    type_summary = results_df.groupby('metric_type').agg({
        'score': 'mean',
        'status': lambda x: (x == '✅').sum() / len(x) * 100
    }).round(2)
    
    type_summary.columns = ['Avg Score', 'Pass Rate %']
    type_summary = type_summary.reset_index()
    
    fig_types = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "bar"}]],
        subplot_titles=('Evaluations by Metric Type', 'Pass Rate by Type')
    )
    
    # Pie chart
    type_counts = results_df['metric_type'].value_counts()
    fig_types.add_trace(
        go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.4,
            marker=dict(colors=['#3498db', '#e74c3c', '#2ecc71']),
            textinfo='label+percent',
            textfont=dict(size=12)
        ),
        row=1, col=1
    )
    
    # Bar chart
    fig_types.add_trace(
        go.Bar(
            x=type_summary['metric_type'],
            y=type_summary['Pass Rate %'],
            marker=dict(
                color=type_summary['Pass Rate %'],
                colorscale='RdYlGn',
                showscale=False,
                line=dict(color='black', width=1)
            ),
            text=type_summary['Pass Rate %'].round(1),
            texttemplate='%{text}%',
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig_types.update_layout(
        title={
            'text': "📊 Metric Type Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        height=500,
        showlegend=False,
        font=dict(size=12)
    )
    
    fig_types.update_yaxes(title_text="Pass Rate (%)", row=1, col=2)
    
    displayHTML(pio.to_html(fig_types, include_plotlyjs='cdn'))
    
    # ============================================================
    # 8. SUMMARY STATISTICS TABLE
    # ============================================================
    print("📊 8. Creating Summary Statistics...")
    
    print("\n" + "="*80)
    print("📈 EVALUATION SUMMARY STATISTICS")
    print("="*80)
    
    total_evals = len(results_df)
    total_pass = (results_df['status'] == '✅').sum()
    total_fail = (results_df['status'] == '❌').sum()
    overall_pass_rate = (total_pass / total_evals * 100) if total_evals > 0 else 0
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Total Evaluations: {total_evals}")
    print(f"   Passed: {total_pass} ({overall_pass_rate:.1f}%)")
    print(f"   Failed: {total_fail} ({100-overall_pass_rate:.1f}%)")
    print(f"   Average Score: {results_df['score'].mean():.2f}")
    print(f"   Score Std Dev: {results_df['score'].std():.2f}")
    
    print(f"\n📊 By Metric:")
    for metric in results_df['metric_name'].unique():
        metric_data = results_df[results_df['metric_name'] == metric]
        metric_pass = (metric_data['status'] == '✅').sum()
        metric_total = len(metric_data)
        metric_rate = (metric_pass / metric_total * 100) if metric_total > 0 else 0
        avg_score = metric_data['score'].mean()
        
        status_icon = "✅" if metric_rate >= 70 else "⚠️" if metric_rate >= 50 else "❌"
        # FIXED: Convert metric to string and use proper formatting
        metric_str = str(metric)
        print(f"   {status_icon} {metric_str:<30} - Pass: {metric_pass}/{metric_total} ({metric_rate:5.1f}%) | Avg: {avg_score:.2f}")
    
    print(f"\n📋 By Sample:")
    for sample in sorted(results_df['sample_id'].unique()):
        sample_data = results_df[results_df['sample_id'] == sample]
        sample_pass = (sample_data['status'] == '✅').sum()
        sample_total = len(sample_data)
        sample_rate = (sample_pass / sample_total * 100) if sample_total > 0 else 0
        
        status_icon = "✅" if sample_rate >= 70 else "⚠️" if sample_rate >= 50 else "❌"
        # FIXED: Convert sample to string
        sample_str = str(sample)
        print(f"   {status_icon} Sample {sample_str:<10} - Pass: {sample_pass}/{sample_total} ({sample_rate:5.1f}%)")
    
    print("\n" + "="*80)
    print("✅ All visualizations generated successfully!")
    print("="*80)