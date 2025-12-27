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

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

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

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_openai import ChatOpenAI
import mlflow
import mlflow.metrics
import mlflow.metrics.genai
import collections

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
# MAGIC ## 3. Load PM Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Your Configuration
# MAGIC 
# MAGIC The configuration is loaded from `pm_config.py`. Make sure to update that file with your settings before running this notebook.

# COMMAND ----------

# Import PM configuration
from pm_config import *

# Display current configuration
print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Predefined Metrics Library

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example Metrics for Zillow PMs
# MAGIC 
# MAGIC Below are the predefined metrics that can be enabled/disabled in the configuration.

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

**Evaluation Criteria:**
1. Is the response directly relevant to the user's question?
2. Does it provide useful information or actionable advice?
3. Is the response clear and well-structured?

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metrics Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Your Metrics Here
# MAGIC 
# MAGIC **Instructions for PMs:**
# MAGIC 1. **Enable/disable metrics**: Use the ENABLE_METRICS setting in pm_config.py
# MAGIC 2. **Set thresholds**: Adjust METRIC_THRESHOLDS in pm_config.py
# MAGIC 3. **Available variables**: Use {prompt}, {response}, {user_profile}, {context} in your templates

# COMMAND ----------

# Get enabled metrics from configuration
enabled_metrics = get_enabled_metrics()

# Create metrics dictionary
METRICS_DICT = {
    "response_quality": RESPONSE_QUALITY_METRIC,
    "personalization_accuracy": PERSONALIZATION_ACCURACY_METRIC,
    "helpfulness": HELPFULNESS_METRIC,
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
# MAGIC ## 6. System Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Your Evaluation Parameters

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