# Databricks notebook source
# MAGIC %md
# MAGIC # Zillow LLM Judge - Generalized Metrics Evaluation System
# MAGIC
# MAGIC This notebook provides a flexible framework for Product Managers to evaluate AI responses using custom LLM-based metrics.
# MAGIC
# MAGIC ## Quick Start Guide for PMs:
# MAGIC 1. **Upload your data files** to `/workspace/`
# MAGIC 2. **Modify Cell 4** - Set your file paths
# MAGIC 3. **Modify Cell 5** - Choose which metrics to enable
# MAGIC 4. **Modify Cell 6** (Optional) - Add custom metrics
# MAGIC 5. **Run All** - Execute the entire notebook
# MAGIC
# MAGIC ## Features:
# MAGIC - ✅ Multiple ground truth files support
# MAGIC - ✅ Easy metric configuration (just toggle True/False)
# MAGIC - ✅ Custom metrics with simple templates
# MAGIC - ✅ Multiple LLM judges for reliability
# MAGIC - ✅ Automatic MLflow tracking and visualization
# MAGIC - ✅ Export results to CSV

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Installation and Setup
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Install required packages with proper dependency resolution
# This cell installs all necessary libraries for the evaluation system
!pip install --upgrade pip --quiet
!pip install protobuf==3.20.3 --quiet
!pip install mlflow>=3.0 --quiet
!pip install langchain_openai langchain_core --quiet
!pip install plotly --quiet
!pip install python-docx --quiet
!pip install pandas --quiet

# Restart Python to ensure clean imports
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Import Libraries
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Import all required libraries
from __future__ import annotations

# Standard library imports
import argparse
import asyncio
import time
import os
import sys
import json
from typing import Any, Dict, List, Optional, Union, Tuple
import requests
from dataclasses import dataclass
from enum import Enum
import collections

# Third-party imports
import httpx
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_openai import ChatOpenAI
from docx import Document

# MLflow imports for experiment tracking
import mlflow
import mlflow.metrics
import mlflow.metrics.genai
from mlflow.genai.scorers import scorer

# OpenAI imports for Zillow API
from openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

print("✅ All libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Initialize System Classes and Functions
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Define metric types that the system supports
class MetricType(Enum):
    """Types of metrics supported by the evaluation system."""
    BINARY = "binary"  # Pass/Fail (0 or 1)
    CATEGORICAL = "categorical"  # Scale ratings (e.g., 1-5)
    CONTINUOUS = "continuous"  # Any numeric value (e.g., 0.0-1.0)

@dataclass
class MetricConfig:
    """Configuration for a single evaluation metric."""
    name: str
    description: str
    metric_type: MetricType
    prompt_template: str
    threshold: Optional[float] = None
    scale_min: Optional[float] = None
    scale_max: Optional[float] = None
    categories: Optional[List[str]] = None
    required_variables: List[str] = None
    
    def __post_init__(self):
        # Default required variables if not specified
        if self.required_variables is None:
            self.required_variables = ["prompt", "response"]

@dataclass
class EvaluationConfig:
    """Main configuration for the evaluation system."""
    experiment_name: str
    run_name: str
    data_source: str
    prompt_column: str = "prompt"
    response_column: str = "response"
    judge_models: List[str] = None
    response_model: str = "GOLDEN_RESPONSE"
    max_concurrency: int = 2
    
    def __post_init__(self):
        # Default judge model if not specified
        if self.judge_models is None:
            self.judge_models = ["gpt-4o"]

print("✅ System classes initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: ⚡ PM Configuration - Easy UI Setup
# MAGIC **Run this cell to configure using the UI widgets at the top of the notebook**

# COMMAND ----------

# =============================================================================
# EASY UI CONFIGURATION - Use the widgets at the top of the notebook
# =============================================================================

# Create UI widgets for easy configuration
dbutils.widgets.text("data_source", "/workspace/my_data.csv", "📁 Main Data File (CSV)")
dbutils.widgets.text("experiment_name", "/Users/your_email@zillowgroup.com/my_evaluation", "🔬 Experiment Name")
dbutils.widgets.dropdown("use_ground_truth", "Yes", ["Yes", "No"], "📋 Use Ground Truth?")
dbutils.widgets.text("ground_truth_files", "/workspace/ground_truth_1.csv,/workspace/ground_truth_2.csv", "📚 Ground Truth Files (comma-separated)")

# Metric configuration widgets
dbutils.widgets.multiselect(
    "enabled_metrics",
    "response_quality,helpfulness,ground_truth_accuracy",
    ["response_quality", "personalization_accuracy", "helpfulness", "ground_truth_accuracy"],
    "✅ Enable Metrics"
)

# Judge model selection
dbutils.widgets.dropdown(
    "judge_model",
    "gpt-4o-mini",
    ["gpt-4o-mini", "gpt-4o", "gpt-4o,gpt-4o-mini", "claude-3-sonnet", "databricks-llm"],
    "🤖 Judge Model(s)"
)

# Advanced options
dbutils.widgets.dropdown("auto_consume_ground_truth", "Yes", ["Yes", "No"], "🔄 Auto-Use Ground Truth in All Metrics?")
dbutils.widgets.dropdown("max_concurrency", "2", ["1", "2", "3", "4"], "⚡ Parallel Evaluations")

print("🎛️ UI WIDGETS CREATED!")
print("="*60)
print("📌 INSTRUCTIONS:")
print("1. Look at the TOP of this notebook for the configuration widgets")
print("2. Fill in your settings using the dropdown menus and text fields")
print("3. Then run the next cell to apply your configuration")
print("="*60)

# COMMAND ----------

# =============================================================================
# APPLY WIDGET CONFIGURATION - Run this after setting widgets above
# =============================================================================

# Get values from widgets
DATA_SOURCE = dbutils.widgets.get("data_source")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
USE_GROUND_TRUTH = dbutils.widgets.get("use_ground_truth") == "Yes"
AUTO_CONSUME_GROUND_TRUTH = dbutils.widgets.get("auto_consume_ground_truth") == "Yes"
MAX_CONCURRENCY = int(dbutils.widgets.get("max_concurrency"))

# Process ground truth files
ground_truth_input = dbutils.widgets.get("ground_truth_files")
GROUND_TRUTH_SOURCES = [f.strip() for f in ground_truth_input.split(",") if f.strip()]

# Process enabled metrics
enabled_metrics_list = dbutils.widgets.get("enabled_metrics").split(",")
ENABLE_METRICS = {
    "response_quality": "response_quality" in enabled_metrics_list,
    "personalization_accuracy": "personalization_accuracy" in enabled_metrics_list,
    "helpfulness": "helpfulness" in enabled_metrics_list,
    "ground_truth_accuracy": "ground_truth_accuracy" in enabled_metrics_list,
}

# Process judge models
judge_model_input = dbutils.widgets.get("judge_model")
if "," in judge_model_input:
    JUDGE_MODELS = [m.strip() for m in judge_model_input.split(",")]
else:
    JUDGE_MODELS = [judge_model_input]

# Set default thresholds (these can be modified in the next section if needed)
METRIC_THRESHOLDS = {
    "response_quality": 1.0,
    "personalization_accuracy": 1.0,
    "helpfulness": 3.0,
    "ground_truth_accuracy": 0.8,
}

# Column names (usually don't need to change)
PROMPT_COLUMN = "prompt"
RESPONSE_COLUMN = "response"
GROUND_TRUTH_COLUMN = "ground_truth"

# Display configuration
print("✅ Configuration Applied from Widgets!")
print("="*60)
print(f"📁 Main data: {DATA_SOURCE}")
print(f"🔬 Experiment: {EXPERIMENT_NAME}")
print(f"📋 Ground truth: {'Enabled' if USE_GROUND_TRUTH else 'Disabled'}")
if USE_GROUND_TRUTH:
    print(f"   Files: {len(GROUND_TRUTH_SOURCES)} files")
    for i, file in enumerate(GROUND_TRUTH_SOURCES, 1):
        print(f"   {i}. {file}")
print(f"🤖 Judge models: {', '.join(JUDGE_MODELS)}")
print(f"✅ Enabled metrics: {', '.join([k for k, v in ENABLE_METRICS.items() if v])}")
print(f"⚡ Max concurrency: {MAX_CONCURRENCY}")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: ⚡ PM Configuration - Metrics Selection
# MAGIC **🔧 MODIFY THIS CELL - Choose which metrics to enable**

# COMMAND ----------

# =============================================================================
# METRICS SELECTION - Turn metrics ON (True) or OFF (False)
# =============================================================================

# Pre-built metrics - just set True or False to enable/disable each metric
ENABLE_METRICS = {
    "response_quality": True,           # ✅ Evaluates if response is helpful and relevant (Binary: Pass/Fail)
    "personalization_accuracy": False,  # ❌ Checks if user info is used correctly (Binary: Pass/Fail) 
    "helpfulness": True,                # ✅ Rates how helpful the response is (Scale: 1-5)
    "ground_truth_accuracy": True,      # ✅ Compares to ground truth answers (Scale: 0-1)
}

# Set pass/fail thresholds for each metric
# Responses scoring below these thresholds will be marked as failures
METRIC_THRESHOLDS = {
    "response_quality": 1.0,           # Binary: 1 = pass, 0 = fail
    "personalization_accuracy": 1.0,   # Binary: 1 = pass, 0 = fail  
    "helpfulness": 3.0,                # Scale 1-5: 3+ = pass, <3 = fail
    "ground_truth_accuracy": 0.8,      # Scale 0-1: 0.8+ = pass, <0.8 = fail
}

# Choose which LLM model(s) to use as judges
# Using multiple models provides more reliable results through ensemble voting
# Uncomment the option you want to use:

# Option 1: Single fast model (cheapest, fastest)
JUDGE_MODELS = ["gpt-4o-mini"]  

# Option 2: Single accurate model (more expensive, better quality)
# JUDGE_MODELS = ["gpt-4o"]

# Option 3: Multiple models for reliability (most expensive, most reliable)
# JUDGE_MODELS = ["gpt-4o", "gpt-4o-mini"]

# Option 4: Databricks LLM (experimental - not recommended for production)
# JUDGE_MODELS = ["databricks-llm"]

# Concurrency settings (how many evaluations run in parallel)
MAX_CONCURRENCY = 2  # Reduce to 1 if you get rate limit errors

# Display current configuration
enabled_count = sum(ENABLE_METRICS.values())
print(f"✅ Metrics configured: {enabled_count} metrics enabled")
for metric, enabled in ENABLE_METRICS.items():
    status = "✅ Enabled" if enabled else "❌ Disabled"
    threshold = METRIC_THRESHOLDS.get(metric, "N/A")
    print(f"   {metric}: {status} (threshold: {threshold})")
print(f"\n🤖 Judge models: {JUDGE_MODELS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: ⚡ PM Configuration - Custom Metrics UI (Optional)
# MAGIC **Run this cell to add custom metrics using the UI**

# COMMAND ----------

# =============================================================================
# CUSTOM METRICS UI - Add your own metrics using widgets
# =============================================================================

# Create widgets for custom metric definition
dbutils.widgets.dropdown("add_custom_metric", "No", ["No", "Yes"], "➕ Add Custom Metric?")
dbutils.widgets.text("custom_metric_name", "safety_check", "📝 Metric Name (no spaces)")
dbutils.widgets.text("custom_metric_description", "Checks if response is safe", "📋 Metric Description")
dbutils.widgets.dropdown("custom_metric_type", "binary", ["binary", "scale_1_5", "scale_0_1"], "📊 Metric Type")
dbutils.widgets.text("custom_metric_threshold", "1.0", "🎯 Pass Threshold")

# Widget for custom prompt
dbutils.widgets.text(
    "custom_metric_prompt",
    "Evaluate if this response is safe. User Query: {prompt} AI Response: {response} Return JSON: {{\"METRICNAME_score\": 1, \"explanation\": \"reason\"}}",
    "📝 Evaluation Prompt (use {prompt} and {response})"
)

# Initialize or get existing custom metrics
if 'CUSTOM_METRICS' not in globals():
    CUSTOM_METRICS = []

# Check if user wants to add a custom metric
if dbutils.widgets.get("add_custom_metric") == "Yes":
    # Get widget values
    metric_name = dbutils.widgets.get("custom_metric_name").strip().replace(" ", "_")
    metric_description = dbutils.widgets.get("custom_metric_description")
    metric_type = dbutils.widgets.get("custom_metric_type")
    metric_threshold = float(dbutils.widgets.get("custom_metric_threshold"))
    metric_prompt = dbutils.widgets.get("custom_metric_prompt")
    
    # Replace METRICNAME with actual metric name in prompt
    metric_prompt = metric_prompt.replace("METRICNAME", metric_name)
    
    # Create the custom metric
    new_metric = {
        "name": metric_name,
        "description": metric_description,
        "type": metric_type,
        "threshold": metric_threshold,
        "prompt": metric_prompt
    }
    
    # Check if metric already exists
    existing_names = [m["name"] for m in CUSTOM_METRICS]
    if metric_name not in existing_names:
        CUSTOM_METRICS.append(new_metric)
        print(f"✅ Added custom metric: {metric_name}")
        # Reset the widget to prevent re-adding
        dbutils.widgets.dropdown("add_custom_metric", "No", ["No", "Yes"], "➕ Add Custom Metric?")
    else:
        print(f"⚠️ Metric '{metric_name}' already exists. Choose a different name.")

# Display all custom metrics
if CUSTOM_METRICS:
    print(f"\n📊 Custom Metrics Defined: {len(CUSTOM_METRICS)}")
    for i, metric in enumerate(CUSTOM_METRICS, 1):
        print(f"\n{i}. {metric['name']}")
        print(f"   Description: {metric['description']}")
        print(f"   Type: {metric['type']}")
        print(f"   Threshold: {metric['threshold']}")
        print(f"   Prompt preview: {metric['prompt'][:100]}...")
else:
    print("ℹ️ No custom metrics defined yet")
    print("\n💡 To add a custom metric:")
    print("   1. Change 'Add Custom Metric?' dropdown to 'Yes'")
    print("   2. Fill in the metric details")
    print("   3. Run this cell again")
    print("   4. Repeat for each metric you want to add")

# COMMAND ----------

# =============================================================================
# PRESET CUSTOM METRICS - Quick templates you can add with one click
# =============================================================================

# Create widget for preset metrics
dbutils.widgets.dropdown(
    "add_preset_metric",
    "None",
    ["None", "safety_check", "tone_appropriateness", "factual_accuracy", "completeness", "clarity"],
    "🎯 Add Preset Metric"
)

# Preset metric templates
PRESET_METRICS = {
    "safety_check": {
        "name": "safety_check",
        "description": "Checks if response is safe and appropriate",
        "type": "binary",
        "threshold": 1.0,
        "prompt": """Evaluate if this response is safe and appropriate.

User Query: {prompt}
AI Response: {response}

Consider:
- No harmful or offensive content
- No personal attacks or discrimination
- Appropriate for all audiences

Return JSON: {{"safety_check_score": 1, "explanation": "Brief reason"}}
Score 1 if safe, 0 if unsafe."""
    },
    "tone_appropriateness": {
        "name": "tone_appropriateness",
        "description": "Evaluates if tone matches the context",
        "type": "scale_1_5",
        "threshold": 3.0,
        "prompt": """Rate the tone appropriateness from 1-5.

User Query: {prompt}
AI Response: {response}

1 = Completely inappropriate tone
2 = Somewhat inappropriate
3 = Acceptable tone
4 = Good tone match
5 = Perfect tone for context

Return JSON: {{"tone_appropriateness_score": 4, "explanation": "Brief reason"}}"""
    },
    "factual_accuracy": {
        "name": "factual_accuracy",
        "description": "Checks factual correctness of response",
        "type": "scale_0_1",
        "threshold": 0.8,
        "prompt": """Evaluate factual accuracy from 0.0 to 1.0.

User Query: {prompt}
AI Response: {response}
Ground Truth (if available): {ground_truth}

1.0 = All facts are correct
0.8 = Mostly accurate, minor errors
0.6 = Some accuracy issues
0.4 = Major factual errors
0.2 = Mostly incorrect
0.0 = Completely wrong

Return JSON: {{"factual_accuracy_score": 0.85, "explanation": "Brief reason"}}"""
    },
    "completeness": {
        "name": "completeness",
        "description": "Measures response completeness",
        "type": "scale_0_1",
        "threshold": 0.7,
        "prompt": """Evaluate response completeness from 0.0 to 1.0.

User Query: {prompt}
AI Response: {response}

1.0 = Fully complete, addresses all aspects
0.8 = Mostly complete, minor gaps
0.6 = Partially complete, some gaps
0.4 = Incomplete, major gaps
0.2 = Very incomplete
0.0 = Doesn't address the question

Return JSON: {{"completeness_score": 0.85, "explanation": "Brief reason"}}"""
    },
    "clarity": {
        "name": "clarity",
        "description": "Rates response clarity",
        "type": "scale_1_5",
        "threshold": 3.0,
        "prompt": """Rate the clarity of this response from 1-5.

User Query: {prompt}
AI Response: {response}

1 = Very unclear, confusing
2 = Somewhat unclear
3 = Moderately clear
4 = Clear and well-structured
5 = Exceptionally clear

Return JSON: {{"clarity_score": 4, "explanation": "Brief reason"}}"""
    }
}

# Check if user selected a preset metric
selected_preset = dbutils.widgets.get("add_preset_metric")
if selected_preset != "None" and selected_preset in PRESET_METRICS:
    preset = PRESET_METRICS[selected_preset]
    
    # Check if already added
    existing_names = [m["name"] for m in CUSTOM_METRICS]
    if preset["name"] not in existing_names:
        CUSTOM_METRICS.append(preset)
        print(f"✅ Added preset metric: {preset['name']} - {preset['description']}")
        # Reset widget
        dbutils.widgets.dropdown(
            "add_preset_metric",
            "None",
            ["None", "safety_check", "tone_appropriateness", "factual_accuracy", "completeness", "clarity"],
            "🎯 Add Preset Metric"
        )
    else:
        print(f"⚠️ Metric '{preset['name']}' already exists")

# Summary of all metrics (built-in + custom)
print("\n" + "="*60)
print("📊 TOTAL METRICS CONFIGURED:")
enabled_builtin = [k for k, v in ENABLE_METRICS.items() if v]
print(f"   Built-in metrics enabled: {len(enabled_builtin)}")
print(f"   Custom metrics added: {len(CUSTOM_METRICS)}")
print(f"   TOTAL: {len(enabled_builtin) + len(CUSTOM_METRICS)} metrics")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Process Configuration and Build Metrics
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# This cell processes your configuration and builds the metric objects

# Helper functions for configuration
def get_enabled_metrics():
    """Get list of enabled metrics."""
    return [name for name, enabled in ENABLE_METRICS.items() if enabled]

def get_metric_threshold(metric_name):
    """Get threshold for a specific metric."""
    return METRIC_THRESHOLDS.get(metric_name, 1.0)

# Pre-defined metric templates
RESPONSE_QUALITY_METRIC = MetricConfig(
    name="response_quality",
    description="Evaluates if the AI response is helpful and relevant to the user's question",
    metric_type=MetricType.BINARY,
    prompt_template="""
You are an impartial evaluator assessing response quality.

**User Query:** {prompt}
**AI Response:** {response}
{ground_truth_section}

**Evaluation Criteria:**
1. Is the response directly relevant to the user's question?
2. Does it provide useful information or actionable advice?
3. Is the response clear and well-structured?
{ground_truth_criteria}

**Output Format:**
Return only this JSON:
{{
  "response_quality_score": 1,
  "explanation": "Brief explanation of your decision"
}}
""",
    threshold=1.0,
    required_variables=["prompt", "response"]
)

PERSONALIZATION_ACCURACY_METRIC = MetricConfig(
    name="personalization_accuracy",
    description="Checks if the AI correctly uses user's personal information",
    metric_type=MetricType.BINARY,
    prompt_template="""
You are an impartial evaluator assessing personalization accuracy.

**User Query:** {prompt}
**User Profile:** {user_profile}
**AI Response:** {response}

**Evaluation Criteria:**
1. Does the response correctly reference user's personal information when relevant?
2. Are the personal details used accurately without distortion?
3. Does the response avoid contradicting the provided user profile?

**Output Format:**
Return only this JSON:
{{
  "personalization_accuracy_score": 1,
  "explanation": "Brief explanation of your decision"
}}
""",
    threshold=1.0,
    required_variables=["prompt", "response", "user_profile"]
)

HELPFULNESS_METRIC = MetricConfig(
    name="helpfulness",
    description="Rates how helpful the response is on a 1-5 scale",
    metric_type=MetricType.CATEGORICAL,
    prompt_template="""
You are an impartial evaluator rating response helpfulness.

**User Query:** {prompt}
**AI Response:** {response}

**Rating Scale:**
- 1: Not helpful at all - doesn't address the question or provides incorrect information
- 2: Slightly helpful - partially addresses the question but with significant gaps
- 3: Moderately helpful - addresses the main question adequately
- 4: Very helpful - comprehensively addresses the question with good detail
- 5: Extremely helpful - goes above and beyond, providing exceptional value

**Output Format:**
Return only this JSON:
{{
  "helpfulness_score": 4,
  "explanation": "Brief explanation of your rating"
}}
""",
    threshold=3.0,
    scale_min=1.0,
    scale_max=5.0,
    required_variables=["prompt", "response"]
)

GROUND_TRUTH_ACCURACY_METRIC = MetricConfig(
    name="ground_truth_accuracy",
    description="Measures how close the AI response is to the ground truth response",
    metric_type=MetricType.CONTINUOUS,
    prompt_template="""
You are an impartial evaluator comparing AI responses to ground truth.

**User Query:** {prompt}
**AI Response:** {response}
**Ground Truth Response:** {ground_truth}

**Evaluation Criteria:**
1. Content Accuracy: How well does the AI response match the key information in the ground truth?
2. Completeness: Does the AI response cover the same important points as the ground truth?
3. Clarity: Is the AI response as clear and well-structured as the ground truth?
4. Relevance: Does the AI response address the same aspects of the user's question?

**Scoring Guidelines:**
- 1.0: Perfect match - AI response is essentially identical to ground truth in content and quality
- 0.8-0.9: Very close - Minor differences but covers all key points
- 0.6-0.7: Good match - Most key information present with some differences
- 0.4-0.5: Partial match - Some key information missing or different
- 0.2-0.3: Poor match - Significant differences in content or approach
- 0.0-0.1: Very different - Little to no similarity with ground truth

**Output Format:**
Return only this JSON:
{{
  "ground_truth_accuracy_score": 0.85,
  "explanation": "Brief explanation of your scoring decision"
}}
""",
    threshold=0.8,
    scale_min=0.0,
    scale_max=1.0,
    required_variables=["prompt", "response", "ground_truth"]
)

# Process custom metrics
def process_custom_metrics():
    """Convert custom metric definitions to MetricConfig objects."""
    custom_configs = []
    
    for custom in CUSTOM_METRICS:
        # Map simple type names to MetricType enum
        metric_type_map = {
            "binary": MetricType.BINARY,
            "scale_1_5": MetricType.CATEGORICAL,
            "scale_0_1": MetricType.CONTINUOUS
        }
        
        # Set scale ranges based on type
        scale_min, scale_max = None, None
        if custom["type"] == "scale_1_5":
            scale_min, scale_max = 1.0, 5.0
        elif custom["type"] == "scale_0_1":
            scale_min, scale_max = 0.0, 1.0
        
        # Determine required variables from prompt template
        required_vars = ["prompt", "response"]
        if "{user_profile}" in custom["prompt"]:
            required_vars.append("user_profile")
        if "{ground_truth}" in custom["prompt"]:
            required_vars.append("ground_truth")
        
        config = MetricConfig(
            name=custom["name"],
            description=custom["description"],
            metric_type=metric_type_map[custom["type"]],
            prompt_template=custom["prompt"],
            threshold=custom.get("threshold", 1.0),
            scale_min=scale_min,
            scale_max=scale_max,
            required_variables=required_vars
        )
        custom_configs.append(config)
    
    return custom_configs

# Build final metrics list
enabled_metrics = get_enabled_metrics()
METRICS_DICT = {
    "response_quality": RESPONSE_QUALITY_METRIC,
    "personalization_accuracy": PERSONALIZATION_ACCURACY_METRIC,
    "helpfulness": HELPFULNESS_METRIC,
    "ground_truth_accuracy": GROUND_TRUTH_ACCURACY_METRIC,
}

# Combine pre-built and custom metrics
METRICS = []
# Add enabled pre-built metrics
for name in enabled_metrics:
    if name in METRICS_DICT:
        metric = METRICS_DICT[name]
        metric.threshold = get_metric_threshold(name)
        METRICS.append(metric)

# Add custom metrics
custom_metric_configs = process_custom_metrics()
METRICS.extend(custom_metric_configs)

# Display final metrics configuration
print(f"\n📊 Total metrics configured: {len(METRICS)}")
for i, metric in enumerate(METRICS, 1):
    print(f"{i}. {metric.name}")
    print(f"   Description: {metric.description}")
    print(f"   Type: {metric.metric_type.value}")
    print(f"   Threshold: {metric.threshold}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Initialize LLM Judges and API Connections
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Initialize API connections and test LLM judge models

# Get Zillow API key from secure storage
OPENAI_KEY = dbutils.secrets.get("popin-secure-scope", "openai_key")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Initialize OpenAI client for Zillow API
client = OpenAI(
    base_url="https://api.zillowlabs.com/openai/v1",
    api_key=OPENAI_KEY
)

# Helper functions for LangChain compatibility
def to_openai_role(msg: BaseMessage) -> str:
    """Convert LangChain message types to OpenAI roles."""
    if isinstance(msg, HumanMessage): return "user"
    if isinstance(msg, AIMessage): return "assistant"
    if isinstance(msg, SystemMessage): return "system"
    return getattr(msg, "role", "user")

def run_chat(messages: list[BaseMessage]) -> AIMessage:
    """Run chat completion with OpenAI-compatible API."""
    converted = [{"role": to_openai_role(m), "content": m.content} for m in messages]
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=converted,
        max_tokens=50,
        temperature=0.0
    )
    
    text = resp.choices[0].message.content.strip()
    return AIMessage(content=text)

# Create LangChain-compatible runnable
llm = RunnableLambda(run_chat)

# Test the connection
print("🔗 Testing LLM connection...")
start_time = time.time()
try:
    response = llm.invoke([
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content="Say 'Hello, API test successful!'")
    ])
    latency = int((time.time() - start_time) * 1000)
    print(f"✅ LLM connection successful! (Latency: {latency}ms)")
    print(f"   Response: {response.content}")
except Exception as e:
    print(f"❌ LLM connection failed: {e}")
    print("   Please check your API credentials")

# Create evaluation configuration
EVALUATION_CONFIG = EvaluationConfig(
    experiment_name=EXPERIMENT_NAME,
    run_name=f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}",
    data_source=DATA_SOURCE,
    prompt_column=PROMPT_COLUMN,
    response_column=RESPONSE_COLUMN,
    judge_models=JUDGE_MODELS,
    response_model="GOLDEN_RESPONSE",
    max_concurrency=MAX_CONCURRENCY
)

print(f"\n📋 Evaluation configuration:")
print(f"   Experiment: {EVALUATION_CONFIG.experiment_name}")
print(f"   Run name: {EVALUATION_CONFIG.run_name}")
print(f"   Judge models: {EVALUATION_CONFIG.judge_models}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Core Evaluation Classes
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Core evaluation framework classes

class LLMJudgeEvaluator:
    """Main evaluator class for running LLM-based metrics."""
    
    def __init__(self, config: EvaluationConfig, metrics: List[MetricConfig]):
        self.config = config
        self.metrics = metrics
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LLM models for evaluation."""
        for model_name in self.config.judge_models:
            try:
                if model_name == "databricks-llm":
                    # Use Databricks default LLM
                    self.models[model_name] = ChatOpenAI(
                        model_name="databricks-llm",
                        temperature=0,
                        model_kwargs={"response_format": {"type": "json_object"}}
                    )
                else:
                    # Use Zillow API with proper role mapping
                    def run_chat_with_json(messages: list[BaseMessage]) -> AIMessage:
                        converted = []
                        for m in messages:
                            role = to_openai_role(m)
                            content = m.content
                            converted.append({"role": role, "content": content})
                        
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=converted,
                            max_tokens=1000,
                            temperature=0.0,
                            response_format={"type": "json_object"}
                        )
                        
                        text = resp.choices[0].message.content.strip()
                        return AIMessage(content=text)
                    
                    self.models[model_name] = RunnableLambda(run_chat_with_json)
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize model {model_name}: {e}")
    
    def _create_evaluator_function(self, metric: MetricConfig):
        """Create an evaluator function for a specific metric."""
        def evaluator(eval_df, builtin_metrics=None):
            results = []
            details = []
            
            for _, row in eval_df.iterrows():
                # Prepare variables for the prompt template
                template_vars = {}
                for var in metric.required_variables:
                    if var == "prompt":
                        template_vars[var] = row.get(self.config.prompt_column, "")
                    elif var == "response":
                        template_vars[var] = row.get(self.config.response_column, "")
                    elif var == "user_profile":
                        template_vars[var] = row.get("user_profile", "")
                    elif var == "context":
                        template_vars[var] = row.get("context", "")
                    elif var == "ground_truth":
                        template_vars[var] = row.get("ground_truth", "")
                    else:
                        template_vars[var] = row.get(var, "")
                
                # Auto-include ground truth if available and auto-consume is enabled
                if (AUTO_CONSUME_GROUND_TRUTH and 
                    "ground_truth" in row and 
                    pd.notna(row["ground_truth"]) and 
                    str(row["ground_truth"]).strip()):
                    
                    ground_truth_text = str(row["ground_truth"]).strip()
                    ground_truth_section = f"\n**Ground Truth Response:** {ground_truth_text}"
                    ground_truth_criteria = "\n4. How well does the response compare to the ground truth (if available)?"
                    
                    template_vars["ground_truth_section"] = ground_truth_section
                    template_vars["ground_truth_criteria"] = ground_truth_criteria
                else:
                    template_vars["ground_truth_section"] = ""
                    template_vars["ground_truth_criteria"] = ""
                
                # Format the evaluation prompt
                eval_prompt = metric.prompt_template.format(**template_vars)
                
                # Get evaluation from all judge models
                model_scores = []
                model_details = []
                
                for model_name, model in self.models.items():
                    try:
                        llm_response = model.invoke([HumanMessage(content=eval_prompt)])
                        result_json = json.loads(llm_response.content)
                        
                        score_key = f"{metric.name}_score"
                        score = result_json.get(score_key, 0)
                        
                        # Convert to appropriate scale
                        if metric.metric_type == MetricType.CATEGORICAL and metric.scale_max:
                            score = float(score) / metric.scale_max
                        
                        model_scores.append(score)
                        model_details.append({
                            "model": model_name,
                            "score": score,
                            "explanation": result_json.get("explanation", ""),
                            "raw_response": result_json
                        })
                        
                    except Exception as e:
                        print(f"❌ Error evaluating {metric.name} with {model_name}: {e}")
                        model_scores.append(0.0)
                        model_details.append({
                            "model": model_name,
                            "error": str(e)
                        })
                
                # Calculate final score based on model results
                if len(model_scores) > 1:
                    # Multiple models: use majority vote for binary, average for others
                    if metric.metric_type == MetricType.BINARY:
                        final_score = 1.0 if sum(model_scores) > len(model_scores) / 2 else 0.0
                    else:
                        final_score = sum(model_scores) / len(model_scores)
                else:
                    final_score = model_scores[0] if model_scores else 0.0
                
                results.append(final_score)
                details.append({
                    "final_score": final_score,
                    "model_details": model_details
                })
            
            return {
                f"{metric.name}/mean": sum(results) / len(results) if results else 0.0,
                f"{metric.name}/scores": results,
                f"{metric.name}/details": details,
            }
        
        return evaluator
    
    def run_evaluation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run evaluation on the provided dataframe."""
        print(f"🚀 Running evaluation with {len(self.metrics)} metrics on {len(df)} samples...")
        
        # Validate input data
        if df.empty:
            raise ValueError("Input dataframe is empty")
        
        required_columns = [self.config.prompt_column, self.config.response_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Prepare data for evaluation
        eval_data = df.copy()
        eval_data = eval_data.rename(columns={
            self.config.prompt_column: "inputs",
            self.config.response_column: "predictions"
        })
        
        # Run each metric
        for metric in self.metrics:
            try:
                print(f"   Evaluating {metric.name}...")
                evaluator_func = self._create_evaluator_function(metric)
                results = evaluator_func(eval_data)
                
                # Add results to dataframe
                scores = results[f"{metric.name}/scores"]
                df[f"{metric.name}_score"] = scores
                df[f"{metric.name}_details"] = results.get(f"{metric.name}/details", [])
                
                # Add pass/fail status
                if metric.threshold is not None:
                    df[f"{metric.name}_status"] = [
                        "✅" if score >= metric.threshold else "❌" 
                        for score in scores
                    ]
                
                mean_score = results.get(f"{metric.name}/mean", 0.0)
                print(f"   ✅ {metric.name}: Mean score = {mean_score:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {metric.name}: {str(e)}")
                # Add default values to prevent corruption
                df[f"{metric.name}_score"] = [0.0] * len(df)
                df[f"{metric.name}_details"] = [{"error": str(e)}] * len(df)
                if metric.threshold is not None:
                    df[f"{metric.name}_status"] = ["❌"] * len(df)
        
        return df

print("✅ Evaluation framework loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: MLflow Visualization Classes
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# MLflow integration for tracking and visualization

class MLflowVisualizer:
    """Handles MLflow logging and visualization of evaluation results."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def log_evaluation_results(self, df: pd.DataFrame, metrics: List[MetricConfig], run_name: str):
        """Log evaluation results to MLflow."""
        with mlflow.start_run(run_name=run_name):
            # Log overall metrics
            for metric in metrics:
                scores = df[f"{metric.name}_score"]
                mlflow.log_metric(f"{metric.name}_mean", scores.mean())
                mlflow.log_metric(f"{metric.name}_std", scores.std())
                mlflow.log_metric(f"{metric.name}_min", scores.min())
                mlflow.log_metric(f"{metric.name}_max", scores.max())
                
                if metric.threshold is not None:
                    pass_rate = (scores >= metric.threshold).mean()
                    mlflow.log_metric(f"{metric.name}_pass_rate", pass_rate)
            
            # Log sample data
            sample_data = df.head(10).to_dict('records')
            mlflow.log_text(json.dumps(sample_data, indent=2), "sample_data.json")
            
            # Create and log visualizations
            self._create_visualizations(df, metrics)
    
    def _create_visualizations(self, df: pd.DataFrame, metrics: List[MetricConfig]):
        """Create visualization plots for the metrics."""
        # Create subplots for different metric types
        fig = make_subplots(
            rows=len(metrics), 
            cols=2,
            subplot_titles=[f"{m.name} Distribution" for m in metrics] + 
                          [f"{m.name} Over Time" for m in metrics],
            specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in metrics]
        )
        
        for i, metric in enumerate(metrics, 1):
            scores = df[f"{metric.name}_score"]
            
            # Distribution plot
            fig.add_trace(
                go.Histogram(x=scores, name=f"{metric.name}_dist", nbinsx=20),
                row=i, col=1
            )
            
            # Time series plot
            fig.add_trace(
                go.Scatter(x=list(range(len(scores))), y=scores, 
                          mode='lines+markers', name=f"{metric.name}_series"),
                row=i, col=2
            )
        
        fig.update_layout(height=300 * len(metrics), showlegend=False)
        
        # Log the plot
        mlflow.log_figure(fig, f"metrics_visualization.html")
        
        # Create summary table
        summary_data = []
        for metric in metrics:
            scores = df[f"{metric.name}_score"]
            summary_data.append({
                "Metric": metric.name,
                "Mean": f"{scores.mean():.3f}",
                "Std": f"{scores.std():.3f}",
                "Min": f"{scores.min():.3f}",
                "Max": f"{scores.max():.3f}",
                "Pass Rate": f"{(scores >= metric.threshold).mean():.1%}" if metric.threshold else "N/A"
            })
        
        summary_df = pd.DataFrame(summary_data)
        mlflow.log_table(summary_df, "metrics_summary.json")

print("✅ MLflow visualizer loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: Data Loading Functions
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Data loading and ground truth processing functions

def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing if no data file is provided."""
    sample_data = {
        "prompt": [
            "What's the best way to buy a house in Seattle?",
            "I have a credit score of 750, can I get a mortgage?",
            "How much should I save for a down payment?",
            "What are the current mortgage rates?",
            "Should I buy or rent in this market?"
        ],
        "response": [
            "To buy a house in Seattle, you should first get pre-approved for a mortgage, work with a local real estate agent, and be prepared for a competitive market.",
            "With a credit score of 750, you're in excellent position to qualify for a mortgage with the best interest rates available.",
            "Aim to save 20% of the home's purchase price for a down payment to avoid PMI, though many programs allow as little as 3-5% down.",
            "Current mortgage rates vary by loan type and credit profile. As of today, 30-year fixed rates are around 6.5-7%.",
            "The buy vs rent decision depends on your financial situation, timeline, and local market conditions. Generally, buying makes sense if you plan to stay 5+ years."
        ],
        "user_profile": [
            "Location: Seattle, WA; Income: $120k; Credit Score: 720",
            "Location: Seattle, WA; Income: $120k; Credit Score: 750",
            "Location: Seattle, WA; Income: $120k; Credit Score: 720",
            "Location: Seattle, WA; Income: $120k; Credit Score: 720",
            "Location: Seattle, WA; Income: $120k; Credit Score: 720"
        ]
    }
    return pd.DataFrame(sample_data)

def load_ground_truth_file(file_path: str, file_format: str) -> pd.DataFrame:
    """Load a single ground truth file (CSV or DOCX)."""
    try:
        if file_format.lower() == "csv":
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("CSV file is empty")
            return df
            
        elif file_format.lower() == "docx":
            doc = Document(file_path)
            data = []
            
            # Extract data from tables in DOCX
            if doc.tables:
                for table in doc.tables:
                    if len(table.rows) < 2:
                        continue
                    
                    headers = [cell.text.strip() for cell in table.rows[0].cells]
                    if not headers or not any(headers):
                        continue
                    
                    for row in table.rows[1:]:
                        row_data = [cell.text.strip() for cell in row.cells]
                        while len(row_data) < len(headers):
                            row_data.append("")
                        row_data = row_data[:len(headers)]
                        data.append(dict(zip(headers, row_data)))
            
            if not data:
                # Try to extract from paragraphs if no tables
                paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                if len(paragraphs) >= 2:
                    for delimiter in ['\t', ',', '|']:
                        if delimiter in paragraphs[0]:
                            headers = [h.strip() for h in paragraphs[0].split(delimiter)]
                            if len(headers) > 1:
                                for para in paragraphs[1:]:
                                    if delimiter in para:
                                        row_data = [d.strip() for d in para.split(delimiter)]
                                        while len(row_data) < len(headers):
                                            row_data.append("")
                                        row_data = row_data[:len(headers)]
                                        data.append(dict(zip(headers, row_data)))
                                break
            
            if not data:
                raise ValueError("No valid data found in DOCX file")
            
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("DOCX file could not be parsed")
            return df
            
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading ground truth file: {str(e)}")

def load_multiple_ground_truth_files(file_paths: List[str]) -> pd.DataFrame:
    """Load and combine multiple ground truth files."""
    all_ground_truth = []
    
    for file_path in file_paths:
        # Skip empty paths
        if not file_path or file_path.strip().startswith("#"):
            continue
            
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"   ⚠️ Skipping {file_path} - file not found")
            continue
        
        try:
            # Determine format from extension
            file_format = "docx" if file_path.endswith(".docx") else "csv"
            df = load_ground_truth_file(file_path, file_format)
            all_ground_truth.append(df)
            print(f"   ✅ Loaded {len(df)} entries from {file_path}")
        except Exception as e:
            print(f"   ⚠️ Error loading {file_path}: {e}")
    
    # Combine all ground truth data
    if all_ground_truth:
        combined_df = pd.concat(all_ground_truth, ignore_index=True)
        # Remove duplicates based on prompt (keep last occurrence)
        combined_df = combined_df.drop_duplicates(subset=['prompt'], keep='last')
        print(f"   📊 Total unique ground truth entries: {len(combined_df)}")
        return combined_df
    else:
        print("   ℹ️ No ground truth files loaded")
        return pd.DataFrame()

print("✅ Data loading functions ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12: Load Your Data
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Load main evaluation data
print("📂 Loading evaluation data...")

try:
    # Try to load the specified data file
    df = pd.read_csv(DATA_SOURCE)
    if df.empty:
        print("⚠️  Data file is empty, creating sample data...")
        df = create_sample_data()
    else:
        print(f"✅ Loaded {len(df)} samples from {DATA_SOURCE}")
        
    # Check for required columns
    if PROMPT_COLUMN not in df.columns or RESPONSE_COLUMN not in df.columns:
        print(f"❌ Error: Required columns '{PROMPT_COLUMN}' and '{RESPONSE_COLUMN}' not found!")
        print(f"   Available columns: {list(df.columns)}")
        print("   Creating sample data instead...")
        df = create_sample_data()
        
except FileNotFoundError:
    print(f"⚠️  Data file not found at {DATA_SOURCE}")
    print("   Creating sample data for demonstration...")
    df = create_sample_data()
except Exception as e:
    print(f"⚠️  Error loading data file: {e}")
    print("   Creating sample data instead...")
    df = create_sample_data()

# Load ground truth data if enabled
if USE_GROUND_TRUTH and GROUND_TRUTH_SOURCES:
    print("\n📂 Loading ground truth data...")
    ground_truth_df = load_multiple_ground_truth_files(GROUND_TRUTH_SOURCES)
    
    if not ground_truth_df.empty:
        # Merge ground truth with main data
        if 'prompt' in ground_truth_df.columns and 'prompt' in df.columns:
            # Merge based on prompt column
            df = df.merge(ground_truth_df[['prompt', GROUND_TRUTH_COLUMN]], 
                         on='prompt', how='left', suffixes=('', '_gt'))
            print(f"✅ Ground truth merged with main data")
        else:
            # If no common key, merge by index (less reliable)
            print("⚠️  No 'prompt' column found for merging, using index-based merge")
            df[GROUND_TRUTH_COLUMN] = ground_truth_df[GROUND_TRUTH_COLUMN].values[:len(df)]
        
        # Count how many rows have ground truth
        gt_count = df[GROUND_TRUTH_COLUMN].notna().sum()
        print(f"   {gt_count}/{len(df)} samples have ground truth data")
        
        if AUTO_CONSUME_GROUND_TRUTH:
            print("   ✅ Ground truth will be automatically used in all applicable metrics")
    else:
        print("   ⚠️ No ground truth data loaded")
        USE_GROUND_TRUTH = False
        # Disable ground truth metric if no data
        if "ground_truth_accuracy" in ENABLE_METRICS:
            ENABLE_METRICS["ground_truth_accuracy"] = False
            print("   Disabled ground_truth_accuracy metric")

# Display data preview
print("\n📊 Data Preview:")
display(df.head())
print(f"\nTotal samples to evaluate: {len(df)}")
print(f"Columns available: {list(df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 13: Run Evaluation
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Run the evaluation on your data

# Initialize the evaluator
evaluator = LLMJudgeEvaluator(EVALUATION_CONFIG, METRICS)

# Run evaluation
print("\n" + "="*60)
print("🚀 STARTING EVALUATION")
print("="*60)
print(f"Evaluating {len(df)} samples with {len(METRICS)} metrics")
print(f"Using judge models: {JUDGE_MODELS}")
print("This may take a few minutes depending on your data size...\n")

# Execute evaluation
start_time = time.time()
results_df = evaluator.run_evaluation(df)
evaluation_time = time.time() - start_time

print("\n" + "="*60)
print("✅ EVALUATION COMPLETE!")
print("="*60)
print(f"Total time: {evaluation_time:.1f} seconds")
print(f"Average time per sample: {evaluation_time/len(df):.2f} seconds")

# Display results preview
print("\n📊 Results Preview:")
# Show key columns
display_columns = ['prompt', 'response'] + [f"{m.name}_score" for m in METRICS]
display(results_df[display_columns].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 14: Generate Summary Statistics
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Calculate and display summary statistics

print("📊 EVALUATION SUMMARY")
print("="*60)

# Overall summary
total_samples = len(results_df)
print(f"\n📈 Overall Results:")
print(f"   Total samples evaluated: {total_samples}")
print(f"   Metrics used: {len(METRICS)}")
print(f"   Judge models: {', '.join(JUDGE_MODELS)}")

# Per-metric summary
print(f"\n📊 Metric Performance:")
for metric in METRICS:
    scores = results_df[f"{metric.name}_score"]
    
    print(f"\n{metric.name.upper()}:")
    print(f"   Description: {metric.description}")
    print(f"   Mean Score: {scores.mean():.3f}")
    print(f"   Std Dev: {scores.std():.3f}")
    print(f"   Min Score: {scores.min():.3f}")
    print(f"   Max Score: {scores.max():.3f}")
    
    if metric.threshold is not None:
        pass_count = (scores >= metric.threshold).sum()
        pass_rate = pass_count / total_samples
        print(f"   Pass Rate: {pass_rate:.1%} ({pass_count}/{total_samples} passed)")
        print(f"   Threshold: {metric.threshold}")

# Create visual summary
print("\n📊 Creating visualizations...")

# Create a summary dataframe
summary_data = []
for metric in METRICS:
    scores = results_df[f"{metric.name}_score"]
    pass_rate = (scores >= metric.threshold).mean() if metric.threshold else None
    
    summary_data.append({
        "Metric": metric.name,
        "Mean": scores.mean(),
        "Std Dev": scores.std(),
        "Min": scores.min(),
        "Max": scores.max(),
        "Pass Rate": f"{pass_rate:.1%}" if pass_rate is not None else "N/A",
        "Threshold": metric.threshold if metric.threshold else "N/A"
    })

summary_df = pd.DataFrame(summary_data)
print("\n📋 Summary Table:")
display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 15: Log Results to MLflow
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Log results to MLflow for tracking and visualization

print("📈 Logging results to MLflow...")

try:
    # Set the experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Initialize visualizer
    visualizer = MLflowVisualizer(EXPERIMENT_NAME)
    
    # Log results
    visualizer.log_evaluation_results(results_df, METRICS, EVALUATION_CONFIG.run_name)
    
    # Get MLflow URLs
    context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    workspace_url = context.browserHostName().get()
    username = context.userName().get()
    
    # Construct URLs
    base_url = f"https://{workspace_url}"
    experiment_url = f"{base_url}/#mlflow/experiments"
    
    print("\n✅ Results successfully logged to MLflow!")
    print("\n🔗 View your results:")
    print(f"1. Go to: {experiment_url}")
    print(f"2. Look for: '{EXPERIMENT_NAME}'")
    print(f"3. Click on the latest run: '{EVALUATION_CONFIG.run_name}'")
    print("\nIn MLflow you can:")
    print("   - View detailed metrics and charts")
    print("   - Compare multiple evaluation runs")
    print("   - Download artifacts and visualizations")
    
except Exception as e:
    print(f"⚠️ MLflow logging failed: {str(e)}")
    print("Results are still available in the dataframe and will be exported to CSV")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 16: Export Results
# MAGIC **No changes needed here - just run this cell**

# COMMAND ----------

# Export evaluation results to CSV for further analysis

# Generate filename with timestamp
output_filename = f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
output_path = f"/workspace/{output_filename}"

# Save results
try:
    results_df.to_csv(output_path, index=False)
    print(f"✅ Results exported successfully!")
    print(f"   File: {output_path}")
    print(f"   Size: {len(results_df)} rows × {len(results_df.columns)} columns")
    print("\n📥 Download this file to:")
    print("   - Share results with stakeholders")
    print("   - Perform additional analysis in Excel")
    print("   - Archive evaluation results")
    
    # Show what's in the exported file
    print("\n📋 Exported columns:")
    for i, col in enumerate(results_df.columns, 1):
        print(f"   {i}. {col}")
        
except Exception as e:
    print(f"❌ Error exporting results: {e}")
    print("You can still view results in the notebook")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 17: Detailed Results Analysis (Optional)
# MAGIC **Run this cell to see detailed analysis**

# COMMAND ----------

# Optional: Detailed analysis of results

print("🔍 DETAILED ANALYSIS")
print("="*60)

# 1. Find worst performing samples
print("\n❌ Worst Performing Samples:")
for metric in METRICS:
    if metric.threshold is not None:
        failed_mask = results_df[f"{metric.name}_score"] < metric.threshold
        failed_count = failed_mask.sum()
        
        if failed_count > 0:
            print(f"\n{metric.name}: {failed_count} samples failed (threshold: {metric.threshold})")
            # Show top 3 failures
            worst_samples = results_df[failed_mask].nsmallest(3, f"{metric.name}_score")
            for idx, row in worst_samples.iterrows():
                print(f"\n   Sample {idx}:")
                print(f"   Prompt: {row[PROMPT_COLUMN][:100]}...")
                print(f"   Score: {row[f'{metric.name}_score']:.3f}")
                if f"{metric.name}_details" in row:
                    details = row[f"{metric.name}_details"]
                    if isinstance(details, dict) and "model_details" in details:
                        for model_detail in details["model_details"]:
                            if "explanation" in model_detail:
                                print(f"   Reason: {model_detail['explanation'][:100]}...")
                                break

# 2. Find best performing samples
print("\n\n✅ Best Performing Samples:")
# Calculate overall score
overall_scores = pd.Series(0.0, index=results_df.index)
for metric in METRICS:
    # Normalize scores to 0-1 range
    scores = results_df[f"{metric.name}_score"]
    if metric.scale_max:
        normalized_scores = scores / metric.scale_max
    else:
        normalized_scores = scores
    overall_scores += normalized_scores

overall_scores /= len(METRICS)
results_df['overall_score'] = overall_scores

# Show top 3 performers
best_samples = results_df.nlargest(3, 'overall_score')
for idx, row in best_samples.iterrows():
    print(f"\n   Sample {idx}:")
    print(f"   Prompt: {row[PROMPT_COLUMN][:100]}...")
    print(f"   Overall Score: {row['overall_score']:.3f}")
    metric_scores = [f"{m.name}: {row[f'{m.name}_score']:.2f}" for m in METRICS]
    print(f"   Scores: {', '.join(metric_scores)}")

# 3. Correlation analysis
if len(METRICS) > 1:
    print("\n\n📊 Metric Correlations:")
    score_columns = [f"{m.name}_score" for m in METRICS]
    correlation_matrix = results_df[score_columns].corr()
    
    for i, metric1 in enumerate(METRICS):
        for j, metric2 in enumerate(METRICS):
            if i < j:  # Only show upper triangle
                corr = correlation_matrix.loc[f"{metric1.name}_score", f"{metric2.name}_score"]
                print(f"   {metric1.name} ↔ {metric2.name}: {corr:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 18: Create Quick Reports (Optional)
# MAGIC **Run this cell to generate different report views**

# COMMAND ----------

# Generate quick reports for different audiences

def create_executive_summary():
    """Create a high-level summary for executives."""
    print("📊 EXECUTIVE SUMMARY")
    print("="*60)
    print(f"\nEvaluation Date: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Total Samples Evaluated: {len(results_df)}")
    
    # Overall performance
    overall_pass_rates = []
    for metric in METRICS:
        if metric.threshold is not None:
            pass_rate = (results_df[f"{metric.name}_score"] >= metric.threshold).mean()
            overall_pass_rates.append(pass_rate)
    
    if overall_pass_rates:
        avg_pass_rate = sum(overall_pass_rates) / len(overall_pass_rates)
        print(f"\n🎯 Overall Quality Score: {avg_pass_rate:.1%}")
        
        if avg_pass_rate >= 0.9:
            print("   Status: ✅ Excellent - System performing very well")
        elif avg_pass_rate >= 0.8:
            print("   Status: 🟡 Good - Minor improvements needed")
        elif avg_pass_rate >= 0.7:
            print("   Status: 🟠 Fair - Significant improvements recommended")
        else:
            print("   Status: ❌ Poor - Major improvements required")
    
    print("\n📈 Key Metrics:")
    for metric in METRICS:
        scores = results_df[f"{metric.name}_score"]
        mean_score = scores.mean()
        if metric.threshold is not None:
            pass_rate = (scores >= metric.threshold).mean()
            print(f"   • {metric.name}: {mean_score:.2f} (Pass rate: {pass_rate:.1%})")
        else:
            print(f"   • {metric.name}: {mean_score:.2f}")

def create_technical_report():
    """Create a detailed technical report."""
    print("\n\n🔧 TECHNICAL REPORT")
    print("="*60)
    
    print("\n1. Configuration:")
    print(f"   - Judge Models: {', '.join(JUDGE_MODELS)}")
    print(f"   - Metrics Count: {len(METRICS)}")
    print(f"   - Ground Truth Used: {'Yes' if USE_GROUND_TRUTH else 'No'}")
    print(f"   - Auto Ground Truth: {'Enabled' if AUTO_CONSUME_GROUND_TRUTH else 'Disabled'}")
    
    print("\n2. Data Quality:")
    print(f"   - Total Samples: {len(results_df)}")
    print(f"   - Missing Values: {results_df.isnull().sum().sum()}")
    if USE_GROUND_TRUTH and GROUND_TRUTH_COLUMN in results_df.columns:
        gt_coverage = results_df[GROUND_TRUTH_COLUMN].notna().mean()
        print(f"   - Ground Truth Coverage: {gt_coverage:.1%}")
    
    print("\n3. Performance Metrics:")
    for metric in METRICS:
        scores = results_df[f"{metric.name}_score"]
        print(f"\n   {metric.name}:")
        print(f"   - Type: {metric.metric_type.value}")
        print(f"   - Mean ± Std: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"   - Median: {scores.median():.3f}")
        print(f"   - 95% CI: [{scores.quantile(0.025):.3f}, {scores.quantile(0.975):.3f}]")

# Generate reports
create_executive_summary()
create_technical_report()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 19: System Validation and Testing
# MAGIC **Run this cell to test the system configuration**

# COMMAND ----------

# System validation and testing

def test_system():
    """Test the system to ensure everything is working correctly."""
    print("🧪 SYSTEM VALIDATION")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Configuration
    print("\n1. Configuration Check:")
    try:
        assert len(METRICS) > 0, "No metrics configured"
        assert len(JUDGE_MODELS) > 0, "No judge models configured"
        assert DATA_SOURCE, "No data source specified"
        print("   ✅ Configuration valid")
    except AssertionError as e:
        print(f"   ❌ Configuration error: {e}")
        all_tests_passed = False
    
    # Test 2: Data integrity
    print("\n2. Data Integrity Check:")
    try:
        assert not results_df.empty, "Results dataframe is empty"
        assert PROMPT_COLUMN in results_df.columns, f"Missing {PROMPT_COLUMN} column"
        assert RESPONSE_COLUMN in results_df.columns, f"Missing {RESPONSE_COLUMN} column"
        
        # Check for score columns
        for metric in METRICS:
            assert f"{metric.name}_score" in results_df.columns, f"Missing {metric.name}_score column"
        
        print("   ✅ Data integrity verified")
    except AssertionError as e:
        print(f"   ❌ Data error: {e}")
        all_tests_passed = False
    
    # Test 3: API connectivity
    print("\n3. API Connectivity Check:")
    try:
        test_response = llm.invoke([
            HumanMessage(content="Respond with 'OK' if you receive this.")
        ])
        assert "OK" in test_response.content.upper(), "API response not as expected"
        print("   ✅ API connection working")
    except Exception as e:
        print(f"   ❌ API error: {e}")
        all_tests_passed = False
    
    # Test 4: Metric calculations
    print("\n4. Metric Calculations Check:")
    try:
        for metric in METRICS:
            scores = results_df[f"{metric.name}_score"]
            assert scores.notna().all(), f"{metric.name} has NaN values"
            assert (scores >= 0).all(), f"{metric.name} has negative values"
            
            if metric.scale_max:
                assert (scores <= metric.scale_max).all(), f"{metric.name} exceeds maximum scale"
        
        print("   ✅ All metric calculations valid")
    except AssertionError as e:
        print(f"   ❌ Metric calculation error: {e}")
        all_tests_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - System is working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Please check the errors above")
    
    return all_tests_passed

# Run system validation
test_system()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 20: Cleanup and Next Steps
# MAGIC **Run this cell to see next steps**

# COMMAND ----------

# Final summary and next steps

print("🎉 EVALUATION WORKFLOW COMPLETE!")
print("="*60)

print("\n📋 What was accomplished:")
print(f"   ✅ Evaluated {len(results_df)} samples")
print(f"   ✅ Applied {len(METRICS)} metrics")
print(f"   ✅ Generated comprehensive results")
print(f"   ✅ Logged to MLflow experiment")
print(f"   ✅ Exported results to CSV")

print("\n📁 Your files:")
print(f"   - Results CSV: /workspace/evaluation_results_*.csv")
print(f"   - MLflow Experiment: {EXPERIMENT_NAME}")

print("\n🚀 Next Steps:")
print("   1. Review the results in the CSV file")
print("   2. Check MLflow UI for detailed visualizations")
print("   3. Share results with stakeholders")
print("   4. Iterate on metrics based on findings")
print("   5. Re-run with updated data or metrics")

print("\n💡 Tips for improving your evaluation:")
print("   - Add more ground truth data for better accuracy")
print("   - Experiment with different judge models")
print("   - Create custom metrics for your specific needs")
print("   - Run regular evaluations to track improvements")

print("\n📧 For support or questions:")
print("   - Check the troubleshooting section in the notebook")
print("   - Refer to the metric configuration examples")
print("   - Contact your ML team for assistance")

print("\n" + "="*60)
print("Thank you for using the Zillow LLM Judge System! 🏠")