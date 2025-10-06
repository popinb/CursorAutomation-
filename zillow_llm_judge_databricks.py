# Databricks notebook source
# MAGIC %md
# MAGIC # Zillow LLM Judge - Generalized Metrics Evaluation System
# MAGIC 
# MAGIC This notebook provides a flexible framework for Product Managers to evaluate AI responses using custom LLM-based metrics. The system is designed to be easily configurable and provides comprehensive MLflow 3.0 visualizations.
# MAGIC 
# MAGIC ## Features
# MAGIC - **Easy Configuration**: Add/remove metrics through simple configuration
# MAGIC - **Multiple Metric Types**: Support for binary, categorical, and continuous metrics
# MAGIC - **Ensemble Evaluation**: Use multiple LLM judges for more reliable scoring
# MAGIC - **MLflow Integration**: Automatic logging and visualization of results
# MAGIC - **Deterministic Results**: Consistent evaluation across runs
# MAGIC 
# MAGIC ## Quick Start
# MAGIC 1. Configure your metrics in the **Metrics Configuration** section
# MAGIC 2. Set your data source and model settings
# MAGIC 3. Run the evaluation pipeline
# MAGIC 4. View results in MLflow UI

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Installation and Setup

# COMMAND ----------

!pip install mlflow>=3.0 --upgrade --quiet
!pip install databricks-agents --quiet
!pip install langchain_openai langchain_core --quiet
!pip install plotly --quiet
!pip install python-docx --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

# Standard library imports
import argparse
import asyncio
import time
import os
import sys
import json
from typing import Any, Dict, List, Optional, Union
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

# MLflow imports
import mlflow
import mlflow.metrics
import mlflow.metrics.genai
from mlflow.genai.scorers import scorer

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration Classes

# COMMAND ----------

class MetricType(Enum):
    """Types of metrics supported by the system."""
    BINARY = "binary"  # 0/1 or True/False
    CATEGORICAL = "categorical"  # 1-5 scale or custom categories
    CONTINUOUS = "continuous"  # Any numeric value

@dataclass
class MetricConfig:
    """Configuration for a single metric."""
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
        if self.judge_models is None:
            self.judge_models = ["gpt-4o"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. PM Configuration - Simple Settings for Non-Technical Users
# MAGIC 
# MAGIC ### Instructions: Change the values below to configure your evaluation

# COMMAND ----------

# =============================================================================
# BASIC SETTINGS - Change these to match your project
# =============================================================================

# Your experiment name (will appear in MLflow UI)
# This helps you identify your experiments in the MLflow dashboard
EXPERIMENT_NAME = "my_zillow_evaluation"  # Change to your project name

# Your data file path (CSV file with prompts and responses)
# This should contain the data you want to evaluate
# Required columns: prompt, response
# Optional columns: user_profile, context
DATA_SOURCE = "/workspace/my_data.csv"  # Change to your CSV file path

# Ground truth file path (optional - CSV or DOCX file with known good responses)
# Ground truth provides reference answers for comparison
# This is optional but recommended for better evaluation quality
GROUND_TRUTH_SOURCE = "/workspace/ground_truth.csv"  # Change to your ground truth file path (.csv or .docx)
GROUND_TRUTH_FORMAT = "csv"  # File format: "csv" or "docx"
USE_GROUND_TRUTH = False  # Set to True if you want to verify against ground truth
AUTO_CONSUME_GROUND_TRUTH = True  # Automatically use ground truth in all applicable metrics (recommended: True)

# =============================================================================
# METRICS SETTINGS - Turn metrics ON/OFF and set pass/fail thresholds
# =============================================================================

# Turn metrics ON (True) or OFF (False)
# You can enable/disable any combination of these metrics
# Start with 1-2 metrics and add more as needed
ENABLE_METRICS = {
    "response_quality": True,           # Is the response helpful and relevant? (Binary: True/False)
    "personalization_accuracy": True,   # Does it use user info correctly? (Binary: True/False)
    "helpfulness": True,                # How helpful is the response? (Scale: 1-5)
    "ground_truth_accuracy": False,     # How close is response to ground truth? (Scale: 0-1)
}

# Set what score counts as "PASS" for each metric
# These thresholds determine pass/fail status in the results
# Adjust these based on your quality requirements
METRIC_THRESHOLDS = {
    "response_quality": 1.0,           # 1 = pass, 0 = fail (binary metric)
    "personalization_accuracy": 1.0,   # 1 = pass, 0 = fail (binary metric)
    "helpfulness": 3.0,                # 3+ = pass, 1-2 = fail (1-5 scale metric)
    "ground_truth_accuracy": 0.8,      # 0.8+ = pass, 0.0-0.79 = fail (0-1 scale metric)
}

# =============================================================================
# JUDGE MODEL SETTINGS - Choose which AI model to use for evaluation
# =============================================================================

# Choose 1 or 2 models (more models = more reliable but slower)
# Standard options: ["gpt-4o"], ["gpt-4o-mini"], ["gpt-4o", "gpt-4o-mini"]
# Note: ["databricks-llm"] is available but experimental and not recommended for production
# Single model: Faster and cheaper
# Multiple models: More reliable through ensemble voting
JUDGE_MODELS = ["gpt-4o"]  # Change to your preferred model(s)

# How many evaluations to run at once (concurrency control)
# Higher values = faster but may hit rate limits
# Lower values = slower but more reliable
# Recommended: 2 for most cases, 1 if you get rate limit errors
MAX_CONCURRENCY = 2  # Keep as 2 (or change to 1 if you get errors)

# =============================================================================
# ADVANCED SETTINGS - Usually don't need to change these
# =============================================================================

# Column names in your CSV file (only change if your CSV has different column names)
# These should match the column names in your data files
PROMPT_COLUMN = "prompt"          # Column with user questions (required)
RESPONSE_COLUMN = "response"      # Column with AI responses (required)
USER_PROFILE_COLUMN = "user_profile"  # Column with user info (optional, for personalization metrics)
GROUND_TRUTH_COLUMN = "ground_truth"  # Column with ground truth responses (optional, for ground truth metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Helper Functions

# COMMAND ----------

def get_enabled_metrics():
    """
    Get list of enabled metrics.
    
    Returns:
        List of metric names that are currently enabled
    """
    return [name for name, enabled in ENABLE_METRICS.items() if enabled]

def get_metric_threshold(metric_name):
    """
    Get threshold for a specific metric.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Threshold value for the metric (default: 1.0)
    """
    return METRIC_THRESHOLDS.get(metric_name, 1.0)

def print_config():
    """
    Print current configuration.
    
    This helps you verify your settings before running evaluation.
    """
    print("Current Configuration:")
    print("=" * 40)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Data Source: {DATA_SOURCE}")
    print(f"Judge Models: {JUDGE_MODELS}")
    print(f"Enabled Metrics: {get_enabled_metrics()}")
    print(f"Max Concurrency: {MAX_CONCURRENCY}")
    print(f"Use Ground Truth: {USE_GROUND_TRUTH}")
    if USE_GROUND_TRUTH:
        print(f"Ground Truth Source: {GROUND_TRUTH_SOURCE}")
        print(f"Ground Truth Format: {GROUND_TRUTH_FORMAT}")
        print(f"Auto Consume Ground Truth: {AUTO_CONSUME_GROUND_TRUTH}")

# Quick setup functions for common configurations
# These functions help you quickly configure the system for common use cases

def setup_basic_evaluation():
    """
    Set up basic evaluation with common metrics.
    
    Enables: response_quality, helpfulness
    Good for: Quick testing and basic quality assessment
    """
    global ENABLE_METRICS
    ENABLE_METRICS = {
        "response_quality": True,
        "helpfulness": True,
    }
    print("✅ Basic evaluation setup complete!")

def setup_full_evaluation():
    """
    Set up full evaluation with all metrics.
    
    Enables: response_quality, personalization_accuracy, helpfulness
    Good for: Comprehensive evaluation when you have user profile data
    """
    global ENABLE_METRICS
    ENABLE_METRICS = {
        "response_quality": True,
        "personalization_accuracy": True,
        "helpfulness": True,
    }
    print("✅ Full evaluation setup complete!")

def setup_reliable_evaluation():
    """
    Set up evaluation with multiple judge models for reliability.
    
    Uses: gpt-4o and gpt-4o-mini for ensemble evaluation
    Good for: Important evaluations where reliability is critical
    Note: Slower and more expensive but more reliable
    """
    global JUDGE_MODELS
    JUDGE_MODELS = ["gpt-4o", "gpt-4o-mini"]
    global MAX_CONCURRENCY
    MAX_CONCURRENCY = 1
    print("✅ Reliable evaluation setup complete!")

def setup_databricks_llm_evaluation():
    """
    Set up evaluation using Databricks default LLM (experimental).
    
    Uses: databricks-llm model
    Good for: Testing or when you want to use Databricks native LLM
    Note: Experimental feature, not recommended for production
    """
    global JUDGE_MODELS
    JUDGE_MODELS = ["databricks-llm"]
    print("⚠️  Databricks LLM evaluation setup complete! (Experimental - not recommended for production)")

def setup_ground_truth_evaluation():
    """
    Set up evaluation with ground truth verification.
    
    Enables: ground_truth_accuracy metric and ground truth auto-consumption
    Good for: When you have reference answers to compare against
    Note: Requires ground truth file to be uploaded
    """
    global USE_GROUND_TRUTH
    global ENABLE_METRICS
    USE_GROUND_TRUTH = True
    ENABLE_METRICS["ground_truth_accuracy"] = True
    print("✅ Ground truth evaluation setup complete!")

# Display current configuration
print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Predefined Metrics Library

# COMMAND ----------

# Example 1: Response Quality (Binary)
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
```json
{{
  "response_quality_score": 1,
  "explanation": "Brief explanation of your decision"
}}
```
""",
    threshold=1.0,
    required_variables=["prompt", "response"]
)

# Example 2: Personalization Accuracy (Binary)
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
```json
{{
  "personalization_accuracy_score": 1,
  "explanation": "Brief explanation of your decision"
}}
```
""",
    threshold=1.0,
    required_variables=["prompt", "response", "user_profile"]
)

# Example 3: Helpfulness Scale (Categorical 1-5)
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
```json
{{
  "helpfulness_score": 4,
  "explanation": "Brief explanation of your rating"
}}
```
""",
    threshold=3.0,
    scale_min=1.0,
    scale_max=5.0,
    required_variables=["prompt", "response"]
)

# Example 4: Ground Truth Accuracy (Continuous 0-1)
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
```json
{{
  "ground_truth_accuracy_score": 0.85,
  "explanation": "Brief explanation of your scoring decision"
}}
```
""",
    threshold=0.8,
    scale_min=0.0,
    scale_max=1.0,
    required_variables=["prompt", "response", "ground_truth"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Metrics Configuration

# COMMAND ----------

# Get enabled metrics from configuration
enabled_metrics = get_enabled_metrics()

# Create metrics dictionary
METRICS_DICT = {
    "response_quality": RESPONSE_QUALITY_METRIC,
    "personalization_accuracy": PERSONALIZATION_ACCURACY_METRIC,
    "helpfulness": HELPFULNESS_METRIC,
    "ground_truth_accuracy": GROUND_TRUTH_ACCURACY_METRIC,
}

# Filter to only enabled metrics
METRICS = [METRICS_DICT[name] for name in enabled_metrics if name in METRICS_DICT]

# Update thresholds from configuration
for metric in METRICS:
    metric.threshold = get_metric_threshold(metric.name)

# Display configured metrics
print("Configured Metrics:")
for i, metric in enumerate(METRICS, 1):
    print(f"{i}. {metric.name} - {metric.description}")
    print(f"   Type: {metric.metric_type.value}, Threshold: {metric.threshold}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. System Configuration

# COMMAND ----------

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="hungc-secure-scope", key="openai_key")

# Configure your evaluation using PM config
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

print(f"Experiment: {EVALUATION_CONFIG.experiment_name}")
print(f"Run: {EVALUATION_CONFIG.run_name}")
print(f"Data Source: {EVALUATION_CONFIG.data_source}")
print(f"Judge Models: {EVALUATION_CONFIG.judge_models}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Core Evaluation Framework

# COMMAND ----------

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
                    # Use OpenAI-compatible models
                    self.models[model_name] = ChatOpenAI(
                        model_name=model_name,
                        temperature=0,
                        model_kwargs={"response_format": {"type": "json_object"}}
                    )
            except Exception as e:
                print(f"Warning: Could not initialize model {model_name}: {e}")
    
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
                    # Add ground truth section to prompt dynamically
                    ground_truth_text = str(row["ground_truth"]).strip()
                    ground_truth_section = f"\n**Ground Truth Response:** {ground_truth_text}"
                    ground_truth_criteria = "\n4. How well does the response compare to the ground truth (if available)?"
                    
                    template_vars["ground_truth_section"] = ground_truth_section
                    template_vars["ground_truth_criteria"] = ground_truth_criteria
                else:
                    # No ground truth available or auto-consume disabled
                    template_vars["ground_truth_section"] = ""
                    template_vars["ground_truth_criteria"] = ""
                
                # Format the evaluation prompt
                eval_prompt = metric.prompt_template.format(**template_vars)
                
                # Get evaluation from all judge models
                model_scores = []
                model_details = []
                
                for model_name, model in self.models.items():
                    try:
                        llm_response = model.invoke(eval_prompt)
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
                        print(f"Error evaluating {metric.name} with {model_name}: {e}")
                        model_scores.append(0.0)
                        model_details.append({
                            "model": model_name,
                            "error": str(e)
                        })
                
                # Use majority vote or average for final score
                if len(model_scores) > 1:
                    # Majority vote for binary, average for others
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
        """
        Run evaluation on the provided dataframe.
        
        Args:
            df: DataFrame with prompts, responses, and optional ground truth
            
        Returns:
            DataFrame with evaluation results added
            
        Raises:
            ValueError: If required columns are missing
            Exception: If evaluation fails
        """
        print(f"Running evaluation with {len(self.metrics)} metrics...")
        
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
        
        # Run each metric with error handling
        for metric in self.metrics:
            try:
                print(f"Evaluating {metric.name}...")
                evaluator_func = self._create_evaluator_function(metric)
                results = evaluator_func(eval_data)
                
                # Validate results
                if f"{metric.name}/scores" not in results:
                    raise ValueError(f"No scores returned for metric {metric.name}")
                
                scores = results[f"{metric.name}/scores"]
                if len(scores) != len(df):
                    raise ValueError(f"Score count mismatch for metric {metric.name}: expected {len(df)}, got {len(scores)}")
                
                # Add results to dataframe
                df[f"{metric.name}_score"] = scores
                df[f"{metric.name}_details"] = results.get(f"{metric.name}/details", [])
                
                # Add status based on threshold
                if metric.threshold is not None:
                    df[f"{metric.name}_status"] = [
                        "✅" if score >= metric.threshold else "❌" 
                        for score in scores
                    ]
                
                mean_score = results.get(f"{metric.name}/mean", 0.0)
                print(f"✅ {metric.name}: {mean_score:.3f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {metric.name}: {str(e)}")
                # Add default values to prevent dataframe corruption
                df[f"{metric.name}_score"] = [0.0] * len(df)
                df[f"{metric.name}_details"] = [{"error": str(e)}] * len(df)
                if metric.threshold is not None:
                    df[f"{metric.name}_status"] = ["❌"] * len(df)
                print(f"   Added default values for {metric.name}")
        
        return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. MLflow Integration and Visualization

# COMMAND ----------

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
            
            # Time series plot (if index represents time)
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Sample Data Generation

# COMMAND ----------

def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing the evaluation system."""
    sample_data = {
        "prompt": [
            "What's the best way to buy a house in Seattle?",
            "I have a credit score of 750, can I get a mortgage?",
            "How much should I save for a down payment?",
            "What are the current mortgage rates?",
            "Should I buy or rent in this market?"
        ],
        "response": [
            "To buy a house in Seattle, you should first get pre-approved for a mortgage, work with a local real estate agent, and be prepared for a competitive market. Consider your budget, location preferences, and timeline.",
            "With a credit score of 750, you're in excellent position to qualify for a mortgage. You'll likely get the best interest rates available. I recommend getting pre-approved to see your exact loan options.",
            "Aim to save 20% of the home's purchase price for a down payment to avoid PMI. However, many programs allow as little as 3-5% down. Consider your monthly payment comfort level when deciding.",
            "Current mortgage rates vary by loan type and your credit profile. As of today, 30-year fixed rates are around 6.5-7%, but rates change daily. Check with lenders for current rates.",
            "The buy vs rent decision depends on your financial situation, timeline, and local market conditions. Generally, if you plan to stay 5+ years and can afford the monthly payment, buying often makes sense."
        ],
        "user_profile": [
            "Location: Seattle, WA; Income: $120k; Credit Score: 720; Down Payment: $50k",
            "Location: Seattle, WA; Income: $120k; Credit Score: 750; Down Payment: $50k",
            "Location: Seattle, WA; Income: $120k; Credit Score: 720; Down Payment: $50k",
            "Location: Seattle, WA; Income: $120k; Credit Score: 720; Down Payment: $50k",
            "Location: Seattle, WA; Income: $120k; Credit Score: 720; Down Payment: $50k"
        ]
    }
    return pd.DataFrame(sample_data)

def load_ground_truth_file(file_path: str, file_format: str) -> pd.DataFrame:
    """
    Load ground truth data from CSV or DOCX file.
    
    Args:
        file_path: Path to the ground truth file
        file_format: File format ('csv' or 'docx')
        
    Returns:
        pandas.DataFrame: Ground truth data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
        Exception: If file cannot be parsed
    """
    try:
        if file_format.lower() == "csv":
            # Load CSV file with error handling
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("CSV file is empty")
            return df
            
        elif file_format.lower() == "docx":
            # Load DOCX file and convert to DataFrame
            doc = Document(file_path)
            data = []
            
            # Extract data from tables in the DOCX
            if doc.tables:
                for table in doc.tables:
                    if len(table.rows) < 2:  # Need at least header + 1 data row
                        continue
                        
                    headers = [cell.text.strip() for cell in table.rows[0].cells]
                    if not headers or not any(headers):  # Skip empty headers
                        continue
                        
                    for row in table.rows[1:]:  # Skip header row
                        row_data = [cell.text.strip() for cell in row.cells]
                        # Pad row_data if it's shorter than headers
                        while len(row_data) < len(headers):
                            row_data.append("")
                        # Truncate if longer
                        row_data = row_data[:len(headers)]
                        data.append(dict(zip(headers, row_data)))
            
            if not data:
                # If no tables found, try to extract from paragraphs
                paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                if len(paragraphs) >= 2:
                    # Try different delimiters
                    for delimiter in ['\t', ',', '|']:
                        if delimiter in paragraphs[0]:
                            headers = [h.strip() for h in paragraphs[0].split(delimiter)]
                            if len(headers) > 1:  # Valid headers found
                                for para in paragraphs[1:]:
                                    if delimiter in para:
                                        row_data = [d.strip() for d in para.split(delimiter)]
                                        # Pad or truncate to match headers
                                        while len(row_data) < len(headers):
                                            row_data.append("")
                                        row_data = row_data[:len(headers)]
                                        data.append(dict(zip(headers, row_data)))
                                break
            
            if not data:
                raise ValueError("No valid data found in DOCX file. Ensure it contains a table or properly formatted text.")
            
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("DOCX file could not be parsed into valid data")
            return df
            
        else:
            raise ValueError(f"Unsupported file format: {file_format}. Supported formats: csv, docx")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading ground truth file {file_path}: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Load Data

# COMMAND ----------

# Load main data (or create sample data if file doesn't exist)
try:
    df = pd.read_csv(DATA_SOURCE)
    if df.empty:
        print("⚠️  Data file is empty, creating sample data...")
        df = create_sample_data()
        print(f"✅ Created {len(df)} sample records")
    else:
        print(f"✅ Loaded {len(df)} samples from {DATA_SOURCE}")
except FileNotFoundError:
    print("⚠️  Data file not found, creating sample data...")
    df = create_sample_data()
    print(f"✅ Created {len(df)} sample records")
except Exception as e:
    print(f"⚠️  Error loading data file: {e}")
    print("   Creating sample data instead...")
    df = create_sample_data()
    print(f"✅ Created {len(df)} sample records")

# Load ground truth data if enabled
if USE_GROUND_TRUTH:
    try:
        ground_truth_df = load_ground_truth_file(GROUND_TRUTH_SOURCE, GROUND_TRUTH_FORMAT)
        print(f"✅ Loaded {len(ground_truth_df)} ground truth samples from {GROUND_TRUTH_SOURCE} ({GROUND_TRUTH_FORMAT.upper()})")
        
        # Merge ground truth with main data
        # Assuming both have a common key (like prompt or index)
        if 'prompt' in ground_truth_df.columns and 'prompt' in df.columns:
            df = df.merge(ground_truth_df[['prompt', GROUND_TRUTH_COLUMN]], on='prompt', how='left')
        else:
            # If no common key, merge by index
            df[GROUND_TRUTH_COLUMN] = ground_truth_df[GROUND_TRUTH_COLUMN].values[:len(df)]
        
        print(f"✅ Ground truth data merged with main data")
        
        # If auto-consume is enabled, add ground truth to all applicable metrics
        if AUTO_CONSUME_GROUND_TRUTH:
            print("✅ Ground truth will be automatically included in all applicable metrics")
        
    except FileNotFoundError:
        print(f"⚠️  Ground truth file not found at {GROUND_TRUTH_SOURCE}")
        print("   Ground truth evaluation will be skipped")
        USE_GROUND_TRUTH = False
        # Remove ground truth metrics if no ground truth data
        if "ground_truth_accuracy" in ENABLE_METRICS:
            ENABLE_METRICS["ground_truth_accuracy"] = False
            print("   Disabled ground_truth_accuracy metric")
    except Exception as e:
        print(f"⚠️  Error loading ground truth file: {e}")
        print("   Ground truth evaluation will be skipped")
        USE_GROUND_TRUTH = False

# Display the data
print(f"\nData Preview:")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Run Evaluation

# COMMAND ----------

# Initialize the evaluator with your configuration and enabled metrics
# This creates the evaluation engine that will run all your enabled metrics
evaluator = LLMJudgeEvaluator(EVALUATION_CONFIG, METRICS)

# Run the evaluation
# This will evaluate each row in your data against all enabled metrics
print("🚀 Starting evaluation...")
print("=" * 50)

# Execute the evaluation - this may take a few minutes depending on your data size
results_df = evaluator.run_evaluation(df)

print("\n✅ Evaluation complete!")
print("=" * 50)

# Display results
# Shows the original prompts/responses plus the scores for each metric
display(results_df[['prompt', 'response'] + [f"{m.name}_score" for m in METRICS]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Log Results to MLflow

# COMMAND ----------

# Initialize MLflow visualizer
# This will create visualizations and log results to MLflow for analysis
visualizer = MLflowVisualizer(EVALUATION_CONFIG.experiment_name)

# Log results to MLflow
# This creates charts, tables, and detailed metrics in the MLflow UI
print("📈 Logging results to MLflow...")
visualizer.log_evaluation_results(results_df, METRICS, EVALUATION_CONFIG.run_name)

print("✅ Results logged to MLflow!")
print(f"Check MLflow UI for experiment: {EVALUATION_CONFIG.experiment_name}")
print("   You can find detailed charts, metrics, and analysis in the MLflow dashboard")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Display Summary Statistics

# COMMAND ----------

# Display summary statistics
# This shows you the overall performance across all your metrics
print("📊 Evaluation Summary:")
print("=" * 50)

# Calculate and display statistics for each metric
for metric in METRICS:
    scores = results_df[f"{metric.name}_score"]
    print(f"\n{metric.name.upper()}:")
    print(f"  Mean: {scores.mean():.3f}")  # Average score
    print(f"  Std:  {scores.std():.3f}")   # Standard deviation (consistency)
    print(f"  Min:  {scores.min():.3f}")   # Lowest score
    print(f"  Max:  {scores.max():.3f}")   # Highest score
    
    # Show pass rate if threshold is defined
    if metric.threshold is not None:
        pass_rate = (scores >= metric.threshold).mean()
        print(f"  Pass Rate: {pass_rate:.1%}")  # Percentage that passed the threshold

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Export Results

# COMMAND ----------

# Export results to CSV
# This saves all your evaluation results to a CSV file for further analysis
output_path = f"/workspace/evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_path, index=False)
print(f"✅ Results exported to: {output_path}")
print("   You can download this file for further analysis or sharing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Quick Setup Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use these functions to quickly configure your evaluation
# MAGIC 
# MAGIC **Basic Evaluation** (Response Quality + Helpfulness):
# MAGIC ```python
# MAGIC setup_basic_evaluation()
# MAGIC ```
# MAGIC 
# MAGIC **Full Evaluation** (All metrics):
# MAGIC ```python
# MAGIC setup_full_evaluation()
# MAGIC ```
# MAGIC 
# MAGIC **Reliable Evaluation** (Multiple judge models):
# MAGIC ```python
# MAGIC setup_reliable_evaluation()
# MAGIC ```
# MAGIC 
# MAGIC **Databricks LLM Evaluation** (Experimental - Use Databricks default LLM):
# MAGIC ```python
# MAGIC setup_databricks_llm_evaluation()  # ⚠️ Experimental - not recommended for production
# MAGIC ```
# MAGIC 
# MAGIC **Ground Truth Evaluation** (Verify against ground truth):
# MAGIC ```python
# MAGIC setup_ground_truth_evaluation()
# MAGIC ```

# COMMAND ----------

# Example: Set up basic evaluation
# setup_basic_evaluation()

# Example: Set up full evaluation
# setup_full_evaluation()

# Example: Set up reliable evaluation
# setup_reliable_evaluation()

# Example: Set up Databricks LLM evaluation
# setup_databricks_llm_evaluation()

# Example: Set up ground truth evaluation
# setup_ground_truth_evaluation()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16.5. System Test and Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the system to ensure everything is working correctly
# MAGIC 
# MAGIC This section runs a quick test to verify the system is properly configured and working.

# COMMAND ----------

def test_system():
    """
    Test the system to ensure all components are working correctly.
    
    This function validates:
    - Configuration is properly set
    - Required dependencies are available
    - Data loading works
    - Metrics are properly configured
    """
    print("🧪 Testing system configuration...")
    
    # Test 1: Configuration validation
    try:
        print("✅ Configuration loaded successfully")
        print(f"   - Experiment: {EXPERIMENT_NAME}")
        print(f"   - Data source: {DATA_SOURCE}")
        print(f"   - Judge models: {JUDGE_MODELS}")
        print(f"   - Enabled metrics: {get_enabled_metrics()}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Test 2: Dependencies check
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from langchain_openai import ChatOpenAI
        from docx import Document
        print("✅ All required dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    # Test 3: Data loading test
    try:
        test_df = create_sample_data()
        if test_df.empty:
            print("❌ Sample data creation failed")
            return False
        print(f"✅ Sample data created successfully ({len(test_df)} rows)")
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False
    
    # Test 4: Metrics configuration test
    try:
        if not METRICS:
            print("❌ No metrics configured")
            return False
        print(f"✅ {len(METRICS)} metrics configured")
        for metric in METRICS:
            print(f"   - {metric.name}: {metric.metric_type.value}")
    except Exception as e:
        print(f"❌ Metrics configuration error: {e}")
        return False
    
    # Test 5: Ground truth loading test (if enabled)
    if USE_GROUND_TRUTH:
        try:
            test_gt = load_ground_truth_file(GROUND_TRUTH_SOURCE, GROUND_TRUTH_FORMAT)
            print(f"✅ Ground truth loading test passed ({len(test_gt)} rows)")
        except Exception as e:
            print(f"⚠️  Ground truth loading test failed: {e}")
            print("   This is expected if ground truth file doesn't exist yet")
    
    print("\n🎉 System test completed successfully!")
    print("   The system is ready for evaluation.")
    return True

# Run the system test
test_system()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Ground Truth File Format

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ground Truth File Requirements
# MAGIC 
# MAGIC Your ground truth file (CSV or DOCX) should have these columns:
# MAGIC 
# MAGIC **Required Columns:**
# MAGIC - `prompt`: User questions (must match your main data)
# MAGIC - `ground_truth`: Known good responses for comparison
# MAGIC 
# MAGIC **Optional Columns:**
# MAGIC - `user_profile`: User information (if using personalization metrics)
# MAGIC - `context`: Additional context (if needed)
# MAGIC 
# MAGIC **Supported Formats:**
# MAGIC - **CSV**: Standard comma-separated values file
# MAGIC - **DOCX**: Microsoft Word document with tables
# MAGIC 
# MAGIC **Example Ground Truth File (CSV):**
# MAGIC ```csv
# MAGIC prompt,ground_truth,user_profile
# MAGIC "What's the best way to buy a house in Seattle?","To buy a house in Seattle, you should first get pre-approved for a mortgage, work with a local real estate agent, and be prepared for a competitive market. Consider your budget, location preferences, and timeline.","Location: Seattle, WA; Income: $120k; Credit Score: 720"
# MAGIC "I have a credit score of 750, can I get a mortgage?","With a credit score of 750, you're in excellent position to qualify for a mortgage. You'll likely get the best interest rates available. I recommend getting pre-approved to see your exact loan options.","Location: Seattle, WA; Income: $120k; Credit Score: 750"
# MAGIC ```
# MAGIC 
# MAGIC **Example Ground Truth File (DOCX):**
# MAGIC Create a Word document with a table containing the same columns as the CSV example above.
# MAGIC 
# MAGIC **Upload Instructions:**
# MAGIC 1. Upload your ground truth file (CSV or DOCX) to `/workspace/` folder in Databricks
# MAGIC 2. Update `GROUND_TRUTH_SOURCE` and `GROUND_TRUTH_FORMAT` in the configuration
# MAGIC 3. Set `USE_GROUND_TRUTH = True`
# MAGIC 4. Enable `ground_truth_accuracy` metric (optional - ground truth will be auto-consumed by all metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Tips for PMs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating Effective Metrics
# MAGIC 
# MAGIC 1. **Start Simple**: Begin with 1-2 metrics and expand gradually
# MAGIC 2. **Be Specific**: Define clear evaluation criteria
# MAGIC 3. **Test Prompts**: Validate your prompt templates with sample data
# MAGIC 4. **Set Thresholds**: Define what constitutes pass/fail
# MAGIC 5. **Use Context**: Leverage user_profile and context when relevant
# MAGIC 
# MAGIC ### Common Patterns
# MAGIC 
# MAGIC **Binary Metrics** (Pass/Fail):
# MAGIC - Safety compliance
# MAGIC - Accuracy checks
# MAGIC - Policy adherence
# MAGIC 
# MAGIC **Categorical Metrics** (1-5 Scale):
# MAGIC - Quality ratings
# MAGIC - Helpfulness scores
# MAGIC - Accuracy levels
# MAGIC 
# MAGIC **Continuous Metrics**:
# MAGIC - Ground truth accuracy (0-1 scale)
# MAGIC - Confidence scores
# MAGIC - Detailed ratings
# MAGIC - Performance metrics
# MAGIC 
# MAGIC ### Best Practices
# MAGIC 
# MAGIC 1. **Iterate on Prompts**: Refine your evaluation prompts based on results
# MAGIC 2. **Use Ensemble**: Multiple judge models provide more reliable results
# MAGIC 3. **Monitor Costs**: LLM evaluation can be expensive with large datasets
# MAGIC 4. **Validate Results**: Spot-check evaluations to ensure quality
# MAGIC 5. **Document Changes**: Keep track of metric modifications
# MAGIC 6. **Use Ground Truth**: When available, ground truth provides objective evaluation
# MAGIC 7. **Leverage Databricks LLM**: Use Databricks default LLM for cost-effective evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Troubleshooting

# COMMAND ----------

# MAGIC %md
# MAGIC ### Common Issues and Solutions
# MAGIC 
# MAGIC **1. API Key Issues**
# MAGIC - Ensure your OpenAI API key is correctly set in the secrets
# MAGIC - Check that the key has sufficient credits
# MAGIC - For Databricks LLM, ensure you have access to the model
# MAGIC - Run `test_system()` to verify API connectivity
# MAGIC 
# MAGIC **2. Data Format Issues**
# MAGIC - Ensure your CSV has the required columns (prompt, response)
# MAGIC - Check that user_profile column exists if using personalization metrics
# MAGIC - For ground truth, ensure prompt column matches between files
# MAGIC - Use sample data first to test the system
# MAGIC 
# MAGIC **3. Evaluation Errors**
# MAGIC - Check that your prompt templates use correct variable names
# MAGIC - Ensure JSON output format is properly specified
# MAGIC - Verify that required_variables match your template
# MAGIC - For ground truth metrics, ensure ground_truth column exists
# MAGIC - Check the error details in the results dataframe
# MAGIC 
# MAGIC **4. Performance Issues**
# MAGIC - Reduce max_concurrency if hitting rate limits
# MAGIC - Use fewer judge models for faster evaluation
# MAGIC - Consider using smaller models for initial testing
# MAGIC - Use Databricks LLM for cost-effective evaluation
# MAGIC - Start with small datasets for testing
# MAGIC 
# MAGIC **5. Configuration Issues**
# MAGIC - Use print_config() to check your settings
# MAGIC - Verify that enabled metrics exist in the metrics dictionary
# MAGIC - Check that thresholds are appropriate for your metric types
# MAGIC - Ensure ground truth file path is correct if using ground truth
# MAGIC - Run test_system() to validate configuration
# MAGIC 
# MAGIC **6. Ground Truth Issues**
# MAGIC - Ensure ground truth file is uploaded to correct folder
# MAGIC - Check that prompt column matches between main data and ground truth
# MAGIC - Verify ground truth column name matches configuration
# MAGIC - Use USE_GROUND_TRUTH flag to enable/disable ground truth evaluation
# MAGIC - Test with CSV format first before trying DOCX
# MAGIC 
# MAGIC **7. MLflow Issues**
# MAGIC - Ensure MLflow is properly installed and configured
# MAGIC - Check that experiment name doesn't contain special characters
# MAGIC - Verify you have write permissions to the MLflow tracking server
# MAGIC - Check MLflow UI for detailed error messages

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Conclusion
# MAGIC 
# MAGIC This generalized LLM judge system provides a flexible framework for Product Managers to evaluate AI responses using custom metrics. The system is designed to be:
# MAGIC 
# MAGIC - **Easy to Use**: Simple configuration for non-technical PMs
# MAGIC - **Flexible**: Easy to add/remove metrics and support for CSV/DOCX ground truth
# MAGIC - **Reliable**: Deterministic results with ensemble evaluation and comprehensive error handling
# MAGIC - **Visual**: Rich MLflow 3.0 visualizations and detailed reporting
# MAGIC - **Scalable**: Handles both small and large datasets with proper concurrency control
# MAGIC - **Robust**: Comprehensive testing and validation built-in
# MAGIC 
# MAGIC **Key Features:**
# MAGIC - ✅ **4 Pre-built Metrics**: Response quality, personalization accuracy, helpfulness, ground truth accuracy
# MAGIC - ✅ **Multiple File Formats**: CSV and DOCX support for ground truth
# MAGIC - ✅ **Auto Ground Truth**: Automatically includes ground truth in all applicable metrics
# MAGIC - ✅ **Ensemble Evaluation**: Multiple judge models for reliability
# MAGIC - ✅ **Comprehensive Testing**: Built-in system validation
# MAGIC - ✅ **Rich Visualizations**: MLflow 3.0 integration with charts and tables
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC 1. **Test the system**: Run `test_system()` to verify everything works
# MAGIC 2. **Customize metrics**: Modify the configuration for your specific use case
# MAGIC 3. **Upload your data**: Replace sample data with your actual prompts and responses
# MAGIC 4. **Add ground truth**: Upload reference answers for better evaluation quality
# MAGIC 5. **Iterate and improve**: Refine prompt templates based on results
# MAGIC 6. **Set up automation**: Create scheduled evaluation pipelines
# MAGIC 7. **Monitor results**: Use MLflow UI for ongoing analysis
# MAGIC 
# MAGIC **Support:** For questions or issues, refer to the troubleshooting section or contact the ML team.

# COMMAND ----------