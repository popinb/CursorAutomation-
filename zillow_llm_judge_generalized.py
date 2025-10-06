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
# MAGIC ## 3. Predefined Metrics Library

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example Metrics for Zillow PMs
# MAGIC 
# MAGIC Below are 3 example metrics that demonstrate different evaluation patterns. PMs can use these as templates and modify them for their specific needs.

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
# MAGIC ## 4. Metrics Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Your Metrics Here
# MAGIC 
# MAGIC **Instructions for PMs:**
# MAGIC 1. **Add metrics**: Add new MetricConfig objects to the METRICS list below
# MAGIC 2. **Remove metrics**: Comment out or delete unwanted metrics
# MAGIC 3. **Modify existing**: Edit the prompt templates and thresholds as needed
# MAGIC 4. **Available variables**: Use {prompt}, {response}, {user_profile}, {context} in your templates

# COMMAND ----------

# Configure your metrics here - add, remove, or modify as needed
METRICS = [
    RESPONSE_QUALITY_METRIC,
    PERSONALIZATION_ACCURACY_METRIC,
    HELPFULNESS_METRIC,
]

# Display configured metrics
print("Configured Metrics:")
for i, metric in enumerate(METRICS, 1):
    print(f"{i}. {metric.name} - {metric.description}")
    print(f"   Type: {metric.metric_type.value}, Threshold: {metric.threshold}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. System Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Your Evaluation Parameters

# COMMAND ----------

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="hungc-secure-scope", key="openai_key")

# Configure your evaluation
EVALUATION_CONFIG = EvaluationConfig(
    experiment_name="zillow_llm_judge_demo",
    run_name=f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}",
    data_source="/workspace/sample_data.csv",  # Update this path
    prompt_column="prompt",
    response_column="response",
    judge_models=["gpt-4o"],  # Add more models for ensemble evaluation
    response_model="GOLDEN_RESPONSE",  # or "FIRST_CALL" or specific LLM model
    max_concurrency=2
)

print(f"Experiment: {EVALUATION_CONFIG.experiment_name}")
print(f"Run: {EVALUATION_CONFIG.run_name}")
print(f"Data Source: {EVALUATION_CONFIG.data_source}")
print(f"Judge Models: {EVALUATION_CONFIG.judge_models}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Core Evaluation Framework

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
                    else:
                        template_vars[var] = row.get(var, "")
                
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
        """Run evaluation on the provided dataframe."""
        print(f"Running evaluation with {len(self.metrics)} metrics...")
        
        # Prepare data for evaluation
        eval_data = df.copy()
        eval_data = eval_data.rename(columns={
            self.config.prompt_column: "inputs",
            self.config.response_column: "predictions"
        })
        
        # Run each metric
        for metric in self.metrics:
            print(f"Evaluating {metric.name}...")
            evaluator_func = self._create_evaluator_function(metric)
            results = evaluator_func(eval_data)
            
            # Add results to dataframe
            df[f"{metric.name}_score"] = results[f"{metric.name}/scores"]
            df[f"{metric.name}_details"] = results[f"{metric.name}/details"]
            
            # Add status based on threshold
            if metric.threshold is not None:
                df[f"{metric.name}_status"] = [
                    "✅" if score >= metric.threshold else "❌" 
                    for score in results[f"{metric.name}/scores"]
                ]
            
            print(f"✅ {metric.name}: {results[f'{metric.name}/mean']:.3f}")
        
        return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. MLflow Integration and Visualization

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
# MAGIC ## 8. Sample Data Generation

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Main Evaluation Pipeline

# COMMAND ----------

def run_complete_evaluation():
    """Run the complete evaluation pipeline."""
    print("🚀 Starting Zillow LLM Judge Evaluation")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    if os.path.exists(EVALUATION_CONFIG.data_source):
        df = pd.read_csv(EVALUATION_CONFIG.data_source)
    else:
        print("⚠️  Data file not found, using sample data")
        df = create_sample_data()
    
    print(f"Loaded {len(df)} samples")
    
    # Initialize evaluator
    print("🔧 Initializing evaluator...")
    evaluator = LLMJudgeEvaluator(EVALUATION_CONFIG, METRICS)
    
    # Run evaluation
    print("⚡ Running evaluation...")
    results_df = evaluator.run_evaluation(df)
    
    # Log to MLflow
    print("📈 Logging results to MLflow...")
    visualizer = MLflowVisualizer(EVALUATION_CONFIG.experiment_name)
    visualizer.log_evaluation_results(results_df, METRICS, EVALUATION_CONFIG.run_name)
    
    # Display results
    print("📋 Evaluation Results Summary:")
    print("=" * 50)
    for metric in METRICS:
        scores = results_df[f"{metric.name}_score"]
        print(f"{metric.name}: {scores.mean():.3f} ± {scores.std():.3f}")
        if metric.threshold is not None:
            pass_rate = (scores >= metric.threshold).mean()
            print(f"  Pass Rate: {pass_rate:.1%}")
    
    print(f"\n✅ Evaluation complete! Check MLflow UI for detailed results.")
    print(f"Experiment: {EVALUATION_CONFIG.experiment_name}")
    print(f"Run: {EVALUATION_CONFIG.run_name}")
    
    return results_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Run Evaluation

# COMMAND ----------

# Run the complete evaluation
results = run_complete_evaluation()

# Display the results
display(results[['prompt', 'response'] + [f"{m.name}_score" for m in METRICS]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Adding New Metrics - Quick Guide

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Add a New Metric
# MAGIC 
# MAGIC **Step 1:** Create a new MetricConfig object
# MAGIC ```python
# MAGIC NEW_METRIC = MetricConfig(
# MAGIC     name="your_metric_name",
# MAGIC     description="What this metric measures",
# MAGIC     metric_type=MetricType.BINARY,  # or CATEGORICAL, CONTINUOUS
# MAGIC     prompt_template="""
# MAGIC Your evaluation prompt here...
# MAGIC Use {prompt}, {response}, {user_profile}, {context} as needed
# MAGIC """,
# MAGIC     threshold=1.0,  # Optional threshold for pass/fail
# MAGIC     required_variables=["prompt", "response"]  # Variables needed
# MAGIC )
# MAGIC ```
# MAGIC 
# MAGIC **Step 2:** Add it to the METRICS list in section 4
# MAGIC ```python
# MAGIC METRICS = [
# MAGIC     RESPONSE_QUALITY_METRIC,
# MAGIC     PERSONALIZATION_ACCURACY_METRIC,
# MAGIC     HELPFULNESS_METRIC,
# MAGIC     NEW_METRIC,  # Add your new metric here
# MAGIC ]
# MAGIC ```
# MAGIC 
# MAGIC **Step 3:** Re-run the evaluation pipeline
# MAGIC 
# MAGIC ### Available Variables for Prompts:
# MAGIC - `{prompt}`: User's question/query
# MAGIC - `{response}`: AI's response
# MAGIC - `{user_profile}`: User's personal information
# MAGIC - `{context}`: Additional context (if available)
# MAGIC 
# MAGIC ### Metric Types:
# MAGIC - **BINARY**: 0/1 or True/False responses
# MAGIC - **CATEGORICAL**: 1-5 scale or custom categories
# MAGIC - **CONTINUOUS**: Any numeric value

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Troubleshooting and Tips

# COMMAND ----------

# MAGIC %md
# MAGIC ### Common Issues and Solutions
# MAGIC 
# MAGIC **1. API Key Issues**
# MAGIC - Ensure your OpenAI API key is correctly set in the secrets
# MAGIC - Check that the key has sufficient credits
# MAGIC 
# MAGIC **2. Data Format Issues**
# MAGIC - Ensure your CSV has the required columns (prompt, response)
# MAGIC - Check that user_profile column exists if using personalization metrics
# MAGIC 
# MAGIC **3. Evaluation Errors**
# MAGIC - Check that your prompt templates use correct variable names
# MAGIC - Ensure JSON output format is properly specified
# MAGIC - Verify that required_variables match your template
# MAGIC 
# MAGIC **4. Performance Issues**
# MAGIC - Reduce max_concurrency if hitting rate limits
# MAGIC - Use fewer judge models for faster evaluation
# MAGIC - Consider using smaller models for initial testing
# MAGIC 
# MAGIC ### Best Practices
# MAGIC 1. **Start Simple**: Begin with 1-2 metrics and expand gradually
# MAGIC 2. **Test Prompts**: Validate your prompt templates with a few examples first
# MAGIC 3. **Use Ensemble**: Multiple judge models provide more reliable results
# MAGIC 4. **Set Thresholds**: Define clear pass/fail criteria for your metrics
# MAGIC 5. **Monitor Costs**: LLM evaluation can be expensive with large datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Advanced Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Custom Model Configuration
# MAGIC 
# MAGIC You can customize the LLM models used for evaluation:

# COMMAND ----------

# Example: Using different models for different metrics
CUSTOM_EVALUATION_CONFIG = EvaluationConfig(
    experiment_name="zillow_llm_judge_custom",
    run_name=f"custom_evaluation_{time.strftime('%Y%m%d_%H%M%S')}",
    data_source="/workspace/your_data.csv",
    judge_models=["gpt-4o", "gpt-4o-mini"],  # Ensemble evaluation
    max_concurrency=3  # Adjust based on your rate limits
)

# Example: Custom metric with specific requirements
CUSTOM_METRIC = MetricConfig(
    name="zillow_specific_accuracy",
    description="Evaluates Zillow-specific knowledge accuracy",
    metric_type=MetricType.CATEGORICAL,
    prompt_template="""
You are evaluating a real estate AI assistant's response for Zillow-specific accuracy.

**User Query:** {prompt}
**AI Response:** {response}
**User Context:** {user_profile}

**Evaluation Criteria:**
1. Accuracy of real estate market information
2. Correctness of Zillow-specific features mentioned
3. Appropriateness of advice for the user's situation

**Rate 1-5:**
- 1: Contains significant errors or misinformation
- 2: Some inaccuracies but mostly correct
- 3: Generally accurate with minor issues
- 4: Highly accurate with good insights
- 5: Perfect accuracy with exceptional value

**Output Format:**
```json
{{
  "zillow_specific_accuracy_score": 4,
  "explanation": "Your detailed explanation"
}}
```
""",
    threshold=3.0,
    scale_min=1.0,
    scale_max=5.0,
    required_variables=["prompt", "response", "user_profile"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch Processing for Large Datasets
# MAGIC 
# MAGIC For large datasets, you can process them in batches:

# COMMAND ----------

def run_batch_evaluation(data_path: str, batch_size: int = 100):
    """Run evaluation on large datasets in batches."""
    print(f"Processing {data_path} in batches of {batch_size}")
    
    # Read data in chunks
    chunk_results = []
    for chunk in pd.read_csv(data_path, chunksize=batch_size):
        print(f"Processing batch of {len(chunk)} samples...")
        
        # Run evaluation on this chunk
        evaluator = LLMJudgeEvaluator(EVALUATION_CONFIG, METRICS)
        chunk_result = evaluator.run_evaluation(chunk)
        chunk_results.append(chunk_result)
    
    # Combine all results
    final_results = pd.concat(chunk_results, ignore_index=True)
    
    # Log to MLflow
    visualizer = MLflowVisualizer(EVALUATION_CONFIG.experiment_name)
    visualizer.log_evaluation_results(final_results, METRICS, f"{EVALUATION_CONFIG.run_name}_batch")
    
    return final_results

# Example usage (uncomment to use):
# results = run_batch_evaluation("/path/to/large_dataset.csv", batch_size=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Export and Integration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exporting Results
# MAGIC 
# MAGIC You can export results in various formats for further analysis:

# COMMAND ----------

def export_results(df: pd.DataFrame, export_path: str = None):
    """Export evaluation results to various formats."""
    if export_path is None:
        export_path = f"/workspace/evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Export to CSV
    csv_path = f"{export_path}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Results exported to {csv_path}")
    
    # Export summary statistics
    summary_data = []
    for metric in METRICS:
        scores = df[f"{metric.name}_score"]
        summary_data.append({
            "metric": metric.name,
            "mean": scores.mean(),
            "std": scores.std(),
            "min": scores.min(),
            "max": scores.max(),
            "pass_rate": (scores >= metric.threshold).mean() if metric.threshold else None
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"{export_path}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary exported to {summary_path}")
    
    return csv_path, summary_path

# Export current results
if 'results' in locals():
    csv_path, summary_path = export_results(results)
    print(f"Results available at: {csv_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Conclusion
# MAGIC 
# MAGIC This generalized LLM judge system provides a flexible framework for Product Managers to evaluate AI responses using custom metrics. The system is designed to be:
# MAGIC 
# MAGIC - **Easy to Use**: Simple configuration for adding/removing metrics
# MAGIC - **Deterministic**: Consistent results across runs
# MAGIC - **Scalable**: Handles both small and large datasets
# MAGIC - **Visual**: Rich MLflow 3.0 visualizations
# MAGIC - **Extensible**: Easy to add new metric types and evaluation patterns
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC 1. Customize the metrics for your specific use case
# MAGIC 2. Test with your own data
# MAGIC 3. Iterate on prompt templates for better accuracy
# MAGIC 4. Set up automated evaluation pipelines
# MAGIC 5. Monitor results in MLflow UI
# MAGIC 
# MAGIC **Support:** For questions or issues, refer to the troubleshooting section or contact the ML team.

# COMMAND ----------