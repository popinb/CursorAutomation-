# Databricks notebook source
# MAGIC %md
# MAGIC # ?? LLM-as-a-Judge Evaluation System with Interactive Metrics Editor
# MAGIC 
# MAGIC ## Enhanced Version with User-Friendly Metrics Management
# MAGIC 
# MAGIC **New Features:**
# MAGIC - ? Interactive metrics editor for non-technical users
# MAGIC - ? Add/Edit/Delete metrics via simple forms
# MAGIC - ? Quick-add standard metrics
# MAGIC - ? Automatic CSV saving
# MAGIC - ? All original features preserved

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Installation and Setup

# COMMAND ----------

# Install packages
%pip install mlflow pandas plotly python-docx openai langchain-core langchain-openai langsmith --quiet

%restart_python

print("? All packages installed!")

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

print("? All libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: ?? Interactive Metrics Configuration Editor
# MAGIC 
# MAGIC **Purpose**: User-friendly interface for managing evaluation metrics
# MAGIC 
# MAGIC **Features**:
# MAGIC - View current metrics in a table
# MAGIC - Add new metrics with simple forms
# MAGIC - Edit existing metrics
# MAGIC - Delete unwanted metrics
# MAGIC - Quick-add standard metrics
# MAGIC - Save changes directly to CSV

# COMMAND ----------

# ============================================================
# STEP 1: File Configuration
# ============================================================

print("=" * 80)
print("?? INTERACTIVE METRICS EDITOR")
print("=" * 80)

# Widgets for file paths
dbutils.widgets.text(
    "metrics_config_path",
    "test_metrics_config.csv",
    "?? Metrics Configuration File"
)

dbutils.widgets.text(
    "evaluation_data_path", 
    "test_evaluation_data.csv", 
    "?? Evaluation Data (CSV)"
)

dbutils.widgets.text(
    "ground_truth_files",
    "ground_truth_accuracy.csv",
    "?? Ground Truth Files (semicolon separated)"
)

# Get settings
METRICS_CONFIG_PATH = dbutils.widgets.get("metrics_config_path")
EVAL_DATA_PATH = dbutils.widgets.get("evaluation_data_path")
GROUND_TRUTH_FILES_STRING = dbutils.widgets.get("ground_truth_files")

print(f"?? Metrics File: {METRICS_CONFIG_PATH}")
print(f"?? Evaluation Data: {EVAL_DATA_PATH}")
print(f"?? Ground Truth: {GROUND_TRUTH_FILES_STRING}")
print("=" * 80)

# ============================================================
# STEP 2: Helper Functions
# ============================================================

def load_metrics_file(file_path):
    """Load metrics CSV file."""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"? Loaded {len(df)} metrics from file")
            return df
        else:
            print(f"??  File not found. Creating new metrics file.")
            df = pd.DataFrame(columns=[
                'name', 'type', 'description', 'grading_rubric', 
                'threshold', 'ground_truth_file_path', 'ground_truth_column'
            ])
            return df
    except Exception as e:
        print(f"? Error loading file: {e}")
        return pd.DataFrame(columns=[
            'name', 'type', 'description', 'grading_rubric', 
            'threshold', 'ground_truth_file_path', 'ground_truth_column'
        ])

def save_metrics_file(df, file_path):
    """Save metrics DataFrame to CSV."""
    try:
        df.to_csv(file_path, index=False)
        print(f"? Successfully saved {len(df)} metrics to: {file_path}")
        return True
    except Exception as e:
        print(f"? Error saving file: {e}")
        return False

# ============================================================
# STEP 3: Load and Display Current Metrics
# ============================================================

current_metrics_df = load_metrics_file(METRICS_CONFIG_PATH)

print("\n" + "=" * 80)
print("?? CURRENT METRICS")
print("=" * 80)

if len(current_metrics_df) > 0:
    display(current_metrics_df)
    
    print(f"\n? Total Metrics: {len(current_metrics_df)}")
    
    if 'type' in current_metrics_df.columns:
        print(f"\n?? Metrics by Type:")
        type_counts = current_metrics_df['type'].value_counts()
        for metric_type, count in type_counts.items():
            print(f"   ? {metric_type}: {count}")
else:
    print("??  No metrics found. Use the form below to add your first metric!")

print("\n" + "=" * 80)

# Store globally for other cells
METRICS_CONFIG_DATA = current_metrics_df

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ?? Add, Edit, or Delete Metrics
# MAGIC 
# MAGIC Use the widgets below to manage your metrics. After making changes, run the next cell to save.

# COMMAND ----------

# ============================================================
# STEP 4: Interactive Form Widgets
# ============================================================

# Create form widgets
dbutils.widgets.dropdown(
    "action",
    "add_new",
    ["add_new", "edit_existing", "delete_existing", "view_only"],
    "?? Action"
)

# For editing/deleting: show existing metric names
current_metrics_df = load_metrics_file(METRICS_CONFIG_PATH)
if len(current_metrics_df) > 0:
    metric_names = ["(select metric)"] + current_metrics_df['name'].tolist()
    default_metric = metric_names[0]
else:
    metric_names = ["(no metrics yet)"]
    default_metric = metric_names[0]

dbutils.widgets.dropdown(
    "select_metric",
    default_metric,
    metric_names,
    "?? Select Metric (for edit/delete)"
)

# Metric definition widgets
dbutils.widgets.text("metric_name", "", "1?? Metric Name *")
dbutils.widgets.dropdown(
    "metric_type",
    "binary",
    ["binary", "1-5_scale", "percentage"],
    "2?? Metric Type *"
)
dbutils.widgets.text("metric_description", "", "3?? Description")
dbutils.widgets.text("metric_grading_rubric", "", "4?? Grading Rubric (detailed criteria) *")
dbutils.widgets.text("metric_threshold", "1", "5?? Threshold (e.g., 1 for binary, 4 for scale, 70 for %) *")
dbutils.widgets.text("metric_ground_truth_file", "", "6?? Ground Truth File (optional)")
dbutils.widgets.text("metric_ground_truth_column", "", "7?? Ground Truth Column (optional)")

print("\n" + "=" * 80)
print("?? INTERACTIVE METRIC FORM")
print("=" * 80)
print("\n?? Instructions:")
print("1. Select action: add_new, edit_existing, delete_existing, or view_only")
print("2. If editing/deleting: select the metric from dropdown")
print("3. Fill in the form fields above")
print("4. Run the next cell to save your changes")
print("\n?? Tips:")
print("   ? Metric Name: Short, descriptive (e.g., 'Accuracy', 'Tone')")
print("   ? Type: binary (0/1), 1-5_scale (1-5), percentage (0-100)")
print("   ? Threshold: Minimum passing score")
print("   ? Grading Rubric: Detailed evaluation criteria for LLM judge")
print("\n?? Ground Truth (optional):")
print("   ? File: Just the filename (e.g., 'ground_truth_accuracy.csv')")
print("   ? Column: Column name with reference answers")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ?? Save Changes
# MAGIC 
# MAGIC Run this cell to save your metric changes to the CSV file.

# COMMAND ----------

# ============================================================
# STEP 5: Process Form and Save Changes
# ============================================================

action = dbutils.widgets.get("action")
selected_metric = dbutils.widgets.get("select_metric")
metric_name = dbutils.widgets.get("metric_name").strip()
metric_type = dbutils.widgets.get("metric_type")
metric_description = dbutils.widgets.get("metric_description").strip()
metric_grading_rubric = dbutils.widgets.get("metric_grading_rubric").strip()
metric_threshold = dbutils.widgets.get("metric_threshold").strip()
metric_gt_file = dbutils.widgets.get("metric_ground_truth_file").strip()
metric_gt_column = dbutils.widgets.get("metric_ground_truth_column").strip()

# Reload current metrics
current_metrics_df = load_metrics_file(METRICS_CONFIG_PATH)

print("\n" + "=" * 80)
print("?? PROCESSING CHANGES")
print("=" * 80)
print(f"Action: {action}")
print()

if action == "view_only":
    print("?? View mode - No changes made")
    print(f"\n?? Current metrics count: {len(current_metrics_df)}")

elif action == "add_new":
    if not metric_name:
        print("? Error: Metric name is required!")
    elif not metric_grading_rubric:
        print("? Error: Grading rubric is required!")
    else:
        # Check if metric already exists
        if metric_name in current_metrics_df['name'].values:
            print(f"??  Metric '{metric_name}' already exists!")
            print("   ?? Use 'edit_existing' to modify it, or choose a different name.")
        else:
            # Add new row
            new_row = {
                'name': metric_name,
                'type': metric_type,
                'description': metric_description if metric_description else f"Evaluation metric: {metric_name}",
                'grading_rubric': metric_grading_rubric,
                'threshold': metric_threshold,
                'ground_truth_file_path': metric_gt_file,
                'ground_truth_column': metric_gt_column
            }
            
            current_metrics_df = pd.concat([current_metrics_df, pd.DataFrame([new_row])], ignore_index=True)
            
            if save_metrics_file(current_metrics_df, METRICS_CONFIG_PATH):
                print(f"? Successfully added new metric: '{metric_name}'")
                print(f"   Type: {metric_type}")
                print(f"   Threshold: {metric_threshold}")
                print(f"   Ground Truth: {metric_gt_file if metric_gt_file else 'None'}")
                print(f"\n?? Total metrics: {len(current_metrics_df)}")

elif action == "edit_existing":
    if selected_metric == "(select metric)" or selected_metric == "(no metrics yet)":
        print("? Error: Please select a metric to edit!")
    else:
        # Find the metric to edit
        metric_idx = current_metrics_df[current_metrics_df['name'] == selected_metric].index
        
        if len(metric_idx) == 0:
            print(f"? Error: Metric '{selected_metric}' not found!")
        else:
            idx = metric_idx[0]
            print(f"?? Editing metric: '{selected_metric}'")
            
            # Update fields (only update non-empty fields)
            updated_fields = []
            if metric_name and metric_name != selected_metric:
                current_metrics_df.at[idx, 'name'] = metric_name
                updated_fields.append(f"name ? '{metric_name}'")
            if metric_type:
                current_metrics_df.at[idx, 'type'] = metric_type
                updated_fields.append(f"type ? '{metric_type}'")
            if metric_description:
                current_metrics_df.at[idx, 'description'] = metric_description
                updated_fields.append("description")
            if metric_grading_rubric:
                current_metrics_df.at[idx, 'grading_rubric'] = metric_grading_rubric
                updated_fields.append("grading_rubric")
            if metric_threshold:
                current_metrics_df.at[idx, 'threshold'] = metric_threshold
                updated_fields.append(f"threshold ? {metric_threshold}")
            if metric_gt_file or metric_gt_file == "":  # Allow clearing
                current_metrics_df.at[idx, 'ground_truth_file_path'] = metric_gt_file
                updated_fields.append("ground_truth_file")
            if metric_gt_column or metric_gt_column == "":  # Allow clearing
                current_metrics_df.at[idx, 'ground_truth_column'] = metric_gt_column
                updated_fields.append("ground_truth_column")
            
            if updated_fields:
                if save_metrics_file(current_metrics_df, METRICS_CONFIG_PATH):
                    print(f"? Successfully updated metric: '{selected_metric}'")
                    print(f"   Updated fields: {', '.join(updated_fields)}")
                    print(f"\n?? Total metrics: {len(current_metrics_df)}")
            else:
                print("??  No fields to update. Fill in the form fields with new values.")

elif action == "delete_existing":
    if selected_metric == "(select metric)" or selected_metric == "(no metrics yet)":
        print("? Error: Please select a metric to delete!")
    else:
        print(f"???  Deleting metric: '{selected_metric}'")
        
        # Remove the metric
        current_metrics_df = current_metrics_df[current_metrics_df['name'] != selected_metric]
        
        if save_metrics_file(current_metrics_df, METRICS_CONFIG_PATH):
            print(f"? Successfully deleted metric: '{selected_metric}'")
            print(f"\n?? Remaining metrics: {len(current_metrics_df)}")

else:
    print(f"? Unknown action: {action}")

print("=" * 80)

# Display updated metrics
print("\n?? UPDATED METRICS TABLE:")
print("=" * 80)
current_metrics_df = load_metrics_file(METRICS_CONFIG_PATH)
if len(current_metrics_df) > 0:
    display(current_metrics_df)
else:
    print("??  No metrics in file.")

# Update global variable
METRICS_CONFIG_DATA = current_metrics_df

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ?? Quick Add Standard Metrics
# MAGIC 
# MAGIC Use this to quickly add common evaluation metrics.

# COMMAND ----------

# ============================================================
# OPTIONAL: Quick Add Standard Metrics
# ============================================================

dbutils.widgets.multiselect(
    "quick_add_metrics",
    "Completeness,Conciseness",
    ["Completeness", "Conciseness", "Tone", "Helpfulness", "Clarity"],
    "?? Quick Add Standard Metrics"
)

quick_add = dbutils.widgets.get("quick_add_metrics")

if quick_add and quick_add.strip():
    selected_metrics = [m.strip() for m in quick_add.split(",")]
    
    current_metrics_df = load_metrics_file(METRICS_CONFIG_PATH)
    
    print("=" * 80)
    print("?? QUICK ADD STANDARD METRICS")
    print("=" * 80)
    
    # Standard metric templates
    standard_metrics = {
        "Completeness": {
            'type': '1-5_scale',
            'description': 'Evaluate if the response fully addresses the query',
            'grading_rubric': '5=Fully complete answer addressing all aspects; 4=Mostly complete with minor gaps; 3=Partially complete, missing some key points; 2=Barely complete, significant gaps; 1=Incomplete, fails to address query',
            'threshold': '4',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        },
        "Conciseness": {
            'type': '1-5_scale',
            'description': 'Evaluate if the response is concise without unnecessary information',
            'grading_rubric': '5=Perfectly concise, no fluff; 4=Mostly concise with minor verbosity; 3=Somewhat verbose with extra info; 2=Very verbose with much unnecessary content; 1=Extremely verbose and unfocused',
            'threshold': '3',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        },
        "Tone": {
            'type': '1-5_scale',
            'description': 'Evaluate if the tone is professional and appropriate',
            'grading_rubric': '5=Perfect professional tone, appropriate for context; 4=Good tone with minor issues; 3=Acceptable but could be improved; 2=Somewhat inappropriate tone; 1=Very inappropriate or unprofessional tone',
            'threshold': '4',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        },
        "Helpfulness": {
            'type': '1-5_scale',
            'description': 'Evaluate how helpful the response is to the user',
            'grading_rubric': '5=Extremely helpful, provides exactly what user needs; 4=Very helpful with good information; 3=Somewhat helpful but could be better; 2=Minimally helpful; 1=Not helpful at all',
            'threshold': '4',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        },
        "Clarity": {
            'type': '1-5_scale',
            'description': 'Evaluate how clear and understandable the response is',
            'grading_rubric': '5=Perfectly clear and easy to understand; 4=Mostly clear with minor ambiguity; 3=Somewhat clear but confusing in parts; 2=Mostly unclear or confusing; 1=Completely unclear',
            'threshold': '4',
            'ground_truth_file_path': '',
            'ground_truth_column': ''
        }
    }
    
    added_count = 0
    skipped_count = 0
    
    for metric_name in selected_metrics:
        if metric_name in standard_metrics:
            if metric_name in current_metrics_df['name'].values:
                print(f"??  Skipped '{metric_name}' (already exists)")
                skipped_count += 1
            else:
                new_row = standard_metrics[metric_name].copy()
                new_row['name'] = metric_name
                
                current_metrics_df = pd.concat([current_metrics_df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"? Added '{metric_name}'")
                added_count += 1
    
    if added_count > 0:
        if save_metrics_file(current_metrics_df, METRICS_CONFIG_PATH):
            print(f"\n?? Successfully added {added_count} new metrics!")
            if skipped_count > 0:
                print(f"??  Skipped {skipped_count} existing metrics")
            print(f"?? Total metrics: {len(current_metrics_df)}")
    else:
        print(f"\n??  No new metrics added")
        if skipped_count > 0:
            print(f"   All {skipped_count} selected metrics already exist")
    
    print("=" * 80)
    
    display(current_metrics_df)
    METRICS_CONFIG_DATA = current_metrics_df
else:
    print("?? Select metrics from the 'Quick Add' widget above to add them")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ?? Load Evaluation Data and Ground Truth Files
# MAGIC 
# MAGIC Now that metrics are configured, let's load the data files.

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

def load_csv_file(filename, file_type="data"):
    """Load CSV file."""
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"? Loaded {file_type}: {len(df)} rows, {len(df.columns)} columns")
            return df
        else:
            print(f"? {file_type.title()} file not found: {filename}")
            return None
    except Exception as e:
        print(f"? Error loading {file_type}: {e}")
        return None

# Load evaluation data
print("=" * 80)
print("?? LOADING DATA FILES")
print("=" * 80)

print(f"\n?? Loading evaluation data...")
evaluation_data_df = load_csv_file(EVAL_DATA_PATH, "evaluation data")

if evaluation_data_df is not None:
    print(f"\n?? Evaluation Data Preview:")
    display(evaluation_data_df.head())
else:
    print("\n? No evaluation data loaded!")

# Load ground truth files
GROUND_TRUTH_FILES_LIST = parse_ground_truth_files(GROUND_TRUTH_FILES_STRING)
ground_truth_data = {}

if GROUND_TRUTH_FILES_LIST:
    print(f"\n?? Loading {len(GROUND_TRUTH_FILES_LIST)} ground truth file(s)...")
    
    for file_path in GROUND_TRUTH_FILES_LIST:
        if file_path and os.path.exists(file_path):
            filename = os.path.basename(file_path)
            try:
                df = pd.read_csv(file_path)
                ground_truth_data[filename] = df
                print(f"? {filename} ? {len(df)} rows, columns: {', '.join(df.columns.tolist())}")
            except Exception as e:
                print(f"? Error loading {filename}: {e}")
        else:
            print(f"??  File not found: {file_path}")
    
    print(f"\n? Loaded {len(ground_truth_data)} ground truth file(s)")
    
    # Preview ground truth
    for filename, df in ground_truth_data.items():
        print(f"\n?? Ground Truth Preview ({filename}):")
        display(df.head(3))

# Store globally
EVALUATION_DATA = evaluation_data_df
GROUND_TRUTH_DATA = ground_truth_data

print("\n" + "=" * 80)
print("?? DATA LOADING SUMMARY")
print("=" * 80)
print(f"? Metrics configured: {len(METRICS_CONFIG_DATA) if METRICS_CONFIG_DATA is not None else 0}")
print(f"? Evaluation samples: {len(EVALUATION_DATA) if EVALUATION_DATA is not None else 0}")
print(f"? Ground truth files: {len(GROUND_TRUTH_DATA)}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Model Configuration
# MAGIC 
# MAGIC Select the judge model to use for evaluation.

# COMMAND ----------

# Model selection
dbutils.widgets.dropdown(
    "judge_model",
    "gpt-4o-mini",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "databricks-llm"],
    "?? Judge Model"
)

JUDGE_MODEL = dbutils.widgets.get("judge_model")

print("=" * 80)
print("?? MODEL CONFIGURATION")
print("=" * 80)
print(f"Judge Model: {JUDGE_MODEL}")
print("=" * 80)

# Initialize API connection
if JUDGE_MODEL == "databricks-llm":
    print("\n?? Databricks LLM selected")
    client = None
else:
    print("\n?? Initializing OpenAI connection...")
    
    # Try multiple sources for API key
    OPENAI_KEY = None
    key_source = None
    
    try:
        OPENAI_KEY = dbutils.secrets.get("popin-secure-scope", "openai_key")
        key_source = "popin-secure-scope"
        print("? Using OpenAI key from popin-secure-scope")
    except:
        try:
            username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
            clean_username = username.replace("@", "_at_").replace(".", "_")
            user_scope = f"user_{clean_username}_secrets"
            OPENAI_KEY = dbutils.secrets.get(user_scope, "openai_key")
            key_source = user_scope
            print(f"? Using OpenAI key from {user_scope}")
        except:
            print("??  No API key found in secrets")
    
    if OPENAI_KEY:
        try:
            os.environ["OPENAI_API_KEY"] = OPENAI_KEY
            
            client = OpenAI(api_key=OPENAI_KEY)
            
            # Test connection
            test_response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=10
            )
            print(f"? OpenAI connection successful!")
            print(f"   Using key from: {key_source}")
            
        except Exception as e:
            print(f"? OpenAI connection failed: {e}")
            client = None
    else:
        print("? No OpenAI API key found. Please configure secrets or use 'databricks-llm'")
        client = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Core Classes
# MAGIC 
# MAGIC Define the evaluation system classes.

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
    use_all_columns: bool = True

print("? Core classes defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: LLM Judge Evaluator
# MAGIC 
# MAGIC Main evaluation engine with enhanced ground truth access.

# COMMAND ----------

import json
import re
import numpy as np

class LLMJudgeEvaluator:
    """
    Enhanced LLM Judge Evaluator with comprehensive ground truth access.
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
            
            print(f"? Databricks LLM initialized")
                
        except Exception as e:
            print(f"? Databricks init failed: {e}")
            raise
    
    def _init_openai_llm(self):
        """Initialize OpenAI LLM client."""
        if 'client' in globals() and client is not None:
            self.llm_client = client
            print(f"? OpenAI client initialized")
        else:
            raise ValueError("OpenAI client not found")
    
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
        """Get ALL columns from ground truth files for comprehensive context."""
        try:
            if not metric.ground_truth_file_path:
                return "Not provided"
            
            files = []
            if ';' in metric.ground_truth_file_path:
                files = [f.strip() for f in metric.ground_truth_file_path.split(';') if f.strip()]
            elif ',' in metric.ground_truth_file_path:
                files = [f.strip() for f in metric.ground_truth_file_path.split(',') if f.strip()]
            else:
                files = [metric.ground_truth_file_path.strip()]
            
            if not files or not files[0]:
                return "Not provided"
            
            for file_path in files:
                filename = os.path.basename(file_path)
                if filename in self.ground_truth_data:
                    df = self.ground_truth_data[filename]
                    
                    if sample_idx >= len(df):
                        return f"Sample index {sample_idx} out of range"
                    
                    row_data = df.iloc[sample_idx]
                    
                    # Format all columns
                    all_data = []
                    for col, value in row_data.items():
                        if pd.notna(value) and str(value).strip():
                            clean_col = str(col).strip()
                            clean_value = str(value).strip()
                            all_data.append(f"? {clean_col}: {clean_value}")
                    
                    if all_data:
                        result = "?? Ground Truth Reference:\n" + "\n".join(all_data)
                        return result
                    else:
                        return "No ground truth data available"
            
            return f"Ground truth file not found: {filename}"
            
        except Exception as e:
            return f"Error accessing ground truth: {str(e)}"
    
    def evaluate_single(self, prompt: str, response: str, ground_truth_data: dict, metric: MetricConfig, sample_idx: int = 0) -> dict:
        """Evaluate a single sample with one metric."""
        try:
            ground_truth = self._get_ground_truth_for_metric(metric, sample_idx)
            
            eval_prompt = metric.prompt_template.format(
                prompt=prompt,
                response=response,
                ground_truth=ground_truth if ground_truth else "Not provided"
            )
            
            if self.is_databricks_llm:
                llm_response = ""  # Implement Databricks call if needed
            else:
                llm_response = self._call_openai_llm(eval_prompt)
            
            if not llm_response or str(llm_response).strip() == "":
                return {
                    "score": 0,
                    "explanation": "Empty response from LLM",
                    "status": "?"
                }
            
            score, explanation = self._parse_llm_response(llm_response, metric)
            status = "?" if score >= metric.threshold else "?"
            
            return {
                "score": score,
                "explanation": explanation,
                "status": status
            }
            
        except Exception as e:
            return {
                "score": 0,
                "explanation": f"Evaluation error: {str(e)}",
                "status": "?"
            }
    
    def _parse_llm_response(self, llm_response: str, metric: MetricConfig) -> tuple:
        """Parse LLM response with bulletproof JSON handling."""
        content = str(llm_response).strip()
        
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        try:
            result_json = json.loads(content)
            score = self._smart_extract_score(result_json, metric)
            explanation = self._smart_extract_explanation(result_json)
        except json.JSONDecodeError:
            score, explanation = self._fallback_parse(content, metric)
        
        score = self._normalize_score(score, metric.metric_type)
        return score, str(explanation)[:500]
    
    def _smart_extract_score(self, json_obj: dict, metric: MetricConfig) -> Any:
        """Smart score extraction."""
        standard_keys = ["score", "Score", "value", "Value", "rating", "Rating"]
        for key in standard_keys:
            if key in json_obj:
                return json_obj[key]
        
        for key, value in json_obj.items():
            if isinstance(value, (int, float)):
                return value
        
        return 0
    
    def _smart_extract_explanation(self, json_obj: dict) -> str:
        """Smart explanation extraction."""
        explanation_keys = ["explanation", "Explanation", "reason", "Reason"]
        for key in explanation_keys:
            if key in json_obj:
                return json_obj[key]
        
        for key, value in json_obj.items():
            if isinstance(value, str) and len(value) > 10:
                return value
        
        return "No explanation provided"
    
    def _fallback_parse(self, content: str, metric: MetricConfig) -> tuple:
        """Fallback parsing when JSON fails."""
        score = 0
        explanation = content
        
        if metric.metric_type == MetricType.BINARY:
            if any(word in content.lower() for word in ['pass', 'correct', 'yes', 'true', '1']):
                score = 1
        elif metric.metric_type == MetricType.SCALE_1_5:
            numbers = re.findall(r'\b[1-5]\b', content)
            if numbers:
                score = int(numbers[0])
        elif metric.metric_type == MetricType.PERCENTAGE:
            percentages = re.findall(r'(\d+(?:\.\d+)?)[%]?', content)
            if percentages:
                score = float(percentages[0])
        
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
            if score <= 1.0:
                score = score * 100
            return max(0.0, min(100.0, score))
        
        return score
    
    def evaluate_dataset(self, evaluation_data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate entire dataset with all metrics."""
        print(f"\n?? Evaluating {len(evaluation_data)} samples with {len(self.metrics)} metrics...")
        
        results = []
        
        for idx, row in evaluation_data.iterrows():
            sample_id = row.get('sample_id', f'sample_{idx}')
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"\n?? Sample {idx + 1}/{len(evaluation_data)}: {sample_id}")
            
            for metric in self.metrics:
                print(f"   ?? {metric.name}...", end=' ')
                
                ground_truth_data = {}
                result = self.evaluate_single(prompt, response, ground_truth_data, metric, idx)
                
                print(f"{result['status']} (score: {result['score']:.2f})")
                
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
        
        print(f"\n? Evaluation complete!")
        return pd.DataFrame(results)

print("? LLM Judge Evaluator defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Run Evaluation
# MAGIC 
# MAGIC Execute the evaluation with current metrics.

# COMMAND ----------

def auto_generate_evaluation_prompt(metric_name: str, metric_type: str, description: str, grading_rubric: str) -> str:
    """Auto-generate evaluation prompt from grading rubric."""
    rubric_section = f"\n**Grading Rubric:**\n{grading_rubric}\n" if grading_rubric else ""
    
    prompt = f"""You are an expert evaluator. Your task: {description}

{rubric_section}
**Evaluation Details:**
- User Query: {{prompt}}
- AI Response: {{response}}
- Ground Truth Reference: {{ground_truth}}

**Instructions:**
Carefully evaluate the AI response using the grading rubric above and the ground truth reference data.

**Required Output Format:**
Return ONLY a valid JSON object:
{{
  "score": <your_score>,
  "explanation": "Brief explanation of your score"
}}

Do not include any other text."""
    
    return prompt

def safe_float_conversion(value, default=0.0):
    """Safely convert value to float."""
    if pd.isna(value) or value is None:
        return default
    
    str_value = str(value).strip().lower()
    
    if str_value in ['true', '==true', 'yes', '1']:
        return 1.0
    elif str_value in ['false', '==false', 'no', '0']:
        return 0.0
    elif str_value.endswith('%'):
        try:
            return float(str_value[:-1])
        except:
            return default
    
    try:
        return float(str_value)
    except:
        return default

def load_metrics_from_csv():
    """Load metrics from CSV with error handling."""
    if METRICS_CONFIG_DATA is None or len(METRICS_CONFIG_DATA) == 0:
        print("? No metrics configuration data available")
        return []
    
    print(f"?? Processing {len(METRICS_CONFIG_DATA)} metrics from CSV...")
    
    metric_configs = []
    for idx, row in METRICS_CONFIG_DATA.iterrows():
        try:
            name = str(row.get('name', '')).strip()
            if not name:
                continue
            
            metric_type_str = str(row.get('type', 'binary')).strip().lower()
            if metric_type_str in ['binary', 'bool', 'boolean']:
                metric_type = MetricType.BINARY
            elif metric_type_str in ['1-5_scale', 'scale']:
                metric_type = MetricType.SCALE_1_5
            elif metric_type_str in ['percentage', 'percent']:
                metric_type = MetricType.PERCENTAGE
            else:
                metric_type = MetricType.BINARY
            
            description = str(row.get('description', '')).strip()
            grading_rubric = str(row.get('grading_rubric', '')).strip()
            
            if metric_type == MetricType.BINARY:
                default_threshold = 0.5
            elif metric_type == MetricType.SCALE_1_5:
                default_threshold = 3.0
            else:
                default_threshold = 70.0
            
            threshold = safe_float_conversion(row.get('threshold', default_threshold), default_threshold)
            
            gt_file = str(row.get('ground_truth_file_path', '')).strip()
            gt_column = str(row.get('ground_truth_column', '')).strip()
            
            prompt_template = auto_generate_evaluation_prompt(
                metric_name=name,
                metric_type=metric_type.value,
                description=description,
                grading_rubric=grading_rubric
            )
            
            metric_config = MetricConfig(
                name=name,
                metric_type=metric_type,
                description=description,
                prompt_template=prompt_template,
                threshold=threshold,
                ground_truth_column=gt_column,
                ground_truth_file_path=gt_file,
                use_all_columns=True
            )
            
            metric_configs.append(metric_config)
            print(f"? Loaded metric: {name} ({metric_type.value}, threshold: {threshold})")
            
        except Exception as e:
            print(f"? Error processing metric {idx + 1}: {e}")
            continue
    
    return metric_configs

# Load metrics
print("=" * 80)
print("?? LOADING METRICS FOR EVALUATION")
print("=" * 80)

metric_configs = load_metrics_from_csv()

if not metric_configs:
    print("? No valid metrics loaded! Please configure metrics first.")
else:
    print(f"\n? Loaded {len(metric_configs)} metrics")
    
    # Create evaluator
    evaluator = LLMJudgeEvaluator(
        judge_model=JUDGE_MODEL,
        metrics=metric_configs,
        ground_truth_data=GROUND_TRUTH_DATA
    )
    
    # Run evaluation
    print("\n" + "=" * 80)
    print("?? STARTING EVALUATION")
    print("=" * 80)
    
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"? EVALUATION COMPLETE in {eval_time:.1f}s")
    print(f"{'='*80}")
    
    # Display results
    total = len(results_df)
    passed = len(results_df[results_df['status'] == '?'])
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n?? RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"?? Total Evaluations: {total}")
    print(f"? Passed: {passed}")
    print(f"? Failed: {total - passed}")
    print(f"?? Overall Pass Rate: {pass_rate:.1f}%")
    print(f"??  Evaluation Time: {eval_time:.1f} seconds")
    print(f"{'='*80}")
    
    # Display results table
    print("\n?? DETAILED RESULTS:")
    display(results_df)
    
    # Store globally
    globals()['results_df'] = results_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## ? Evaluation Complete!
# MAGIC 
# MAGIC Your LLM evaluation is complete. Results are stored in `results_df`.
# MAGIC 
# MAGIC **Next steps:**
# MAGIC - Review the results table above
# MAGIC - Export results to CSV if needed
# MAGIC - Adjust metrics and re-run if needed
