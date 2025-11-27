# Databricks notebook source
# MAGIC %md
# MAGIC # 🏠 Zillow LLM-as-a-Judge Evaluation Framework
# MAGIC 
# MAGIC ## 🎯 Purpose
# MAGIC This notebook provides a flexible framework for Product Managers at Zillow to evaluate LLM responses using custom metrics. You can easily add, modify, or remove metrics to suit your evaluation needs.
# MAGIC 
# MAGIC ## 📋 Quick Start Guide
# MAGIC 1. **Clone this notebook** to create your own version
# MAGIC 2. **Define your metrics** in Section 4 (see examples provided)
# MAGIC 3. **Configure your experiment** in Section 3
# MAGIC 4. **Run the evaluation** and view results in MLflow
# MAGIC 
# MAGIC ## 📊 Metric Examples Included
# MAGIC - **Response Quality**: Evaluates accuracy, completeness, and relevance
# MAGIC - **Zillow Context Accuracy**: Checks if real estate data is used correctly
# MAGIC - **User Intent Alignment**: Measures how well the response addresses user needs
# MAGIC - **Personalization Coverage**: Evaluates use of user-specific information

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 📦 Installation & Imports

# COMMAND ----------

!pip install mlflow>=3.0 --upgrade --quiet
!pip install databricks-agents --quiet
!pip install langchain_openai langchain_core --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import asyncio
import collections
import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_openai import ChatOpenAI
import mlflow
import mlflow.metrics
import mlflow.metrics.genai
from mlflow.genai.scorers import scorer

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 🔐 Authentication Setup
# MAGIC 
# MAGIC Replace these with your own credentials or use Databricks secrets

# COMMAND ----------

# Authentication - Replace with your own credentials
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="your-scope", key="openai_key")
USER_ID = dbutils.secrets.get(scope="your-scope", key="zuid")
LOGIN_MEMENTO = dbutils.secrets.get(scope="your-scope", key="login_memento")
ZGS_BETH_COPILOT_KONG_KEY = dbutils.secrets.get(scope="your-scope", key="copilot_kong_key")
ZGAI_PERSISTENT_MEMORY_KONG_KEY = dbutils.secrets.get(scope="your-scope", key="memory_kong_key")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. ⚙️ Configuration
# MAGIC 
# MAGIC **🚨 IMPORTANT: Update these settings for your evaluation**

# COMMAND ----------

# ===== EXPERIMENT CONFIGURATION =====
# Change this for each new experiment
EXPERIMENT_NAME = "/Users/your_name/zillow_llm_evaluation_experiment"  # UPDATE THIS!
RUN_NAME = f"evaluation_run_{time.strftime('%Y%m%d_%H%M%S')}"

# ===== MODEL CONFIGURATION =====
# Models available for evaluation
AVAILABLE_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4.1": "gpt-4.1", 
    "claude-opus": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-3.5": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "o3": "o3",
    "o3-mini": "o3-mini"
}

# Select models for response generation and judging
RESPONSE_MODEL = "gpt-4o"  # Model to generate responses
JUDGE_MODELS = ["gpt-4o"]   # Models to evaluate responses (can use multiple for ensemble)

# ===== DATA CONFIGURATION =====
# Input data settings
INPUT_FILE_PATH = "/path/to/your/evaluation_data.csv"  # UPDATE THIS!
PROMPT_COLUMN = "user_query"  # Column containing user prompts
RESPONSE_COLUMN = "model_response"  # Column for model responses

# Output settings
OUTPUT_DIR = "/path/to/save/results"  # UPDATE THIS!
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== API CONFIGURATION =====
BASE_URL = "https://zgs-beth-copilot.stage.kong-nonprod.zg-int.net"
API_URL = BASE_URL + "/copilot_eval?execution=sync"
os.environ["OPENAI_API_BASE"] = "https://zgai-llm-api.int.stage-k8s.zg-aip.net/openai/v1"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 📏 Metric Definition
# MAGIC 
# MAGIC ### 🎯 How to Add Your Own Metrics:
# MAGIC 1. Copy one of the example templates below
# MAGIC 2. Modify the prompt template to evaluate what you care about
# MAGIC 3. Set the scoring scale (e.g., 0-1 for binary, 1-5 for scale)
# MAGIC 4. Add your metric to `EVALUATION_METRICS` dictionary
# MAGIC 
# MAGIC ### 📝 Available Variables in Templates:
# MAGIC - `{prompt}`: The user's original question
# MAGIC - `{response}`: The model's response
# MAGIC - `{user_context}`: Any user-specific context (optional)
# MAGIC - `{expected_answer}`: Ground truth answer (if available)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example Metric 1: Response Quality

# COMMAND ----------

RESPONSE_QUALITY_PROMPT = """You are an expert evaluator assessing the quality of AI responses for Zillow users.

### Task
Evaluate the overall quality of the AI assistant's response based on accuracy, completeness, and usefulness.

### Materials
**User Query:**
```
{prompt}
```

**AI Response:**
```
{response}
```

### Evaluation Criteria
1. **Accuracy (40%)**: Is the information factually correct and reliable?
2. **Completeness (30%)**: Does the response fully address all aspects of the user's question?
3. **Usefulness (30%)**: Is the response practical and actionable for the user?

### Scoring Scale
- 5: Exceptional - Exceeds expectations in all criteria
- 4: Good - Meets all criteria with minor room for improvement
- 3: Adequate - Meets most criteria but has notable gaps
- 2: Poor - Fails to meet several criteria
- 1: Unacceptable - Significantly flawed or unhelpful

### Output Format
Return ONLY this JSON:
```json
{{
  "response_quality_score": <1-5>,
  "explanation": "<Brief explanation of score>",
  "strengths": ["<strength1>", "<strength2>"],
  "improvements": ["<improvement1>", "<improvement2>"]
}}
```
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example Metric 2: Zillow Context Accuracy

# COMMAND ----------

ZILLOW_CONTEXT_ACCURACY_PROMPT = """You are a Zillow domain expert evaluating how accurately the AI uses real estate and Zillow-specific information.

### Task
Assess whether the AI response correctly uses Zillow data, real estate terminology, and market information.

### Materials
**User Query:**
```
{prompt}
```

**AI Response:**
```
{response}
```

### Evaluation Guidelines
1. **Real Estate Data Accuracy**: Check if home prices, market trends, and property details are reasonable
2. **Zillow Feature References**: Verify correct mention of Zillow tools (Zestimate, mortgage calculator, etc.)
3. **Market Knowledge**: Assess if local market conditions and trends are accurately represented
4. **Terminology**: Ensure real estate terms are used correctly

### Scoring
- 1: Accurate - All Zillow/real estate information is correct
- 0: Inaccurate - Contains errors in Zillow/real estate information

### Output Format
Return ONLY this JSON:
```json
{{
  "zillow_context_accuracy_score": <0 or 1>,
  "explanation": "<What was accurate or inaccurate>",
  "errors_found": ["<error1>", "<error2>"] 
}}
```
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example Metric 3: User Intent Alignment

# COMMAND ----------

USER_INTENT_ALIGNMENT_PROMPT = """You are evaluating how well the AI response aligns with the user's intent and needs.

### Task
Determine if the AI response addresses what the user is actually trying to accomplish.

### Materials
**User Query:**
```
{prompt}
```

**AI Response:**
```
{response}
```

### Evaluation Process
1. **Identify User Intent**: What is the user trying to achieve?
2. **Assess Alignment**: Does the response help achieve that goal?
3. **Check for Misunderstandings**: Did the AI misinterpret the request?
4. **Evaluate Completeness**: Are all aspects of the intent addressed?

### Scoring Scale
- 5: Perfect alignment - Fully addresses user intent with relevant extras
- 4: Strong alignment - Addresses main intent well
- 3: Moderate alignment - Addresses intent but missing key elements  
- 2: Weak alignment - Partially addresses intent
- 1: Misaligned - Fails to address user intent

### Output Format
Return ONLY this JSON:
```json
{{
  "user_intent_alignment_score": <1-5>,
  "identified_intent": "<What user is trying to achieve>",
  "alignment_explanation": "<How well response aligns>",
  "missing_elements": ["<element1>", "<element2>"]
}}
```
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example Metric 4: Property Information Accuracy

# COMMAND ----------

PROPERTY_INFO_ACCURACY_PROMPT = """You are a real estate data expert evaluating the accuracy of property-related information.

### Task
Verify that any property details mentioned (square footage, bedrooms, bathrooms, lot size, year built, etc.) are reasonable and consistent.

### Materials
**User Query:**
```
{prompt}
```

**AI Response:**
```
{response}
```

### Evaluation Guidelines
1. **Data Reasonableness**: Check if property specifications are within realistic ranges
   - Square footage aligns with bedroom/bathroom count
   - Year built is plausible for the location mentioned
   - Price estimates align with property characteristics
   
2. **Internal Consistency**: Ensure all property details mentioned are consistent with each other
3. **Market Alignment**: Property values should align with mentioned location/market

### Scoring
- 1: All property information is reasonable and consistent
- 0: Contains unrealistic or inconsistent property information

### Output Format
Return ONLY this JSON:
```json
{{
  "property_info_accuracy_score": <0 or 1>,
  "explanation": "<What was evaluated>",
  "issues_found": ["<issue1>", "<issue2>"]
}}
```
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example Metric 5: Actionability Score

# COMMAND ----------

ACTIONABILITY_SCORE_PROMPT = """You are evaluating how actionable and practical the AI response is for a Zillow user.

### Task
Assess whether the response provides clear, actionable next steps that the user can actually implement.

### Materials
**User Query:**
```
{prompt}
```

**AI Response:**
```
{response}
```

### Evaluation Criteria
1. **Clear Next Steps**: Does the response provide specific actions the user can take?
2. **Practical Guidance**: Are the suggestions realistic and implementable?
3. **Tool References**: Does it mention relevant Zillow tools/features the user can use?
4. **Specificity**: Avoids vague advice in favor of concrete recommendations

### Scoring Scale
- 5: Highly actionable with clear steps and tools
- 4: Good actionability with some concrete guidance
- 3: Moderate - some actionable elements but could be clearer
- 2: Limited actionability, mostly general advice
- 1: Not actionable, vague or theoretical only

### Output Format
Return ONLY this JSON:
```json
{{
  "actionability_score": <1-5>,
  "explanation": "<Why this score>",
  "action_items": ["<action1>", "<action2>"],
  "zillow_tools_mentioned": ["<tool1>", "<tool2>"]
}}
```
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔧 Configure Your Metrics Here

# COMMAND ----------

# ===== ADD YOUR METRICS TO THIS DICTIONARY =====
EVALUATION_METRICS = {
    "response_quality": {
        "prompt_template": RESPONSE_QUALITY_PROMPT,
        "threshold": 3,  # Scores >= 3 are considered "passing"
        "scale": "1-5",
        "description": "Overall quality of the response"
    },
    "zillow_context_accuracy": {
        "prompt_template": ZILLOW_CONTEXT_ACCURACY_PROMPT,
        "threshold": 1,  # Binary: 1 = accurate, 0 = inaccurate
        "scale": "0-1",
        "description": "Accuracy of Zillow-specific information"
    },
    "user_intent_alignment": {
        "prompt_template": USER_INTENT_ALIGNMENT_PROMPT,
        "threshold": 3,
        "scale": "1-5", 
        "description": "How well the response addresses user intent"
    },
    "property_info_accuracy": {
        "prompt_template": PROPERTY_INFO_ACCURACY_PROMPT,
        "threshold": 1,
        "scale": "0-1",
        "description": "Accuracy and consistency of property details"
    },
    "actionability": {
        "prompt_template": ACTIONABILITY_SCORE_PROMPT,
        "threshold": 3,
        "scale": "1-5",
        "description": "How actionable the response is for users"
    },
    # ===== ADD NEW METRICS HERE =====
    # "your_metric_name": {
    #     "prompt_template": YOUR_METRIC_PROMPT,
    #     "threshold": 3,
    #     "scale": "1-5",
    #     "description": "What this metric measures"
    # },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 How to Add Your Own Metrics - Step by Step Guide
# MAGIC 
# MAGIC ### Step 1: Define What You Want to Measure
# MAGIC Think about what aspect of the response you want to evaluate. Common categories:
# MAGIC - **Accuracy**: Is the information correct?
# MAGIC - **Completeness**: Does it answer all parts of the question?
# MAGIC - **Relevance**: Is it on-topic and useful?
# MAGIC - **Style**: Is it well-written and appropriate?
# MAGIC - **Safety**: Is it appropriate and unbiased?
# MAGIC 
# MAGIC ### Step 2: Choose Your Scoring Scale
# MAGIC - **Binary (0-1)**: For yes/no evaluations (accurate/inaccurate)
# MAGIC - **Scale (1-5)**: For graduated quality assessments
# MAGIC - **Percentage (0-100)**: For coverage or completion metrics
# MAGIC 
# MAGIC ### Step 3: Write Your Evaluation Prompt
# MAGIC 1. Copy the template below
# MAGIC 2. Replace placeholder text with your criteria
# MAGIC 3. Be specific about what constitutes each score level
# MAGIC 4. Include examples if helpful
# MAGIC 
# MAGIC ### Step 4: Add to EVALUATION_METRICS
# MAGIC 1. Give your metric a descriptive name (use underscores, no spaces)
# MAGIC 2. Set the threshold for "passing"
# MAGIC 3. Add a clear description
# MAGIC 
# MAGIC ### Step 5: Test and Iterate
# MAGIC 1. Run on a small sample first
# MAGIC 2. Check if scores align with your expectations
# MAGIC 3. Refine the prompt if needed

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📝 Metric Template (Copy this to create new metrics)

# COMMAND ----------

# TEMPLATE FOR NEW METRICS - COPY AND MODIFY THIS
METRIC_TEMPLATE = """You are an expert evaluator for [DOMAIN/PURPOSE].

### Task
[Describe what you want to evaluate]

### Materials
**User Query:**
```
{prompt}
```

**AI Response:**
```
{response}
```

[Add other materials if needed, like {user_context} or {expected_answer}]

### Evaluation Criteria
1. **[Criterion 1]**: [Description]
2. **[Criterion 2]**: [Description]

### Scoring Scale
[Define your scale, e.g.:]
- 5: Excellent
- 4: Good
- 3: Adequate
- 2: Poor
- 1: Unacceptable

### Output Format
Return ONLY this JSON:
```json
{{
  "[your_metric_name]_score": <score>,
  "explanation": "<explanation>",
  "[other_fields]": "..."
}}
```
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Common Zillow PM Metric Use Cases
# MAGIC 
# MAGIC ### For Search & Discovery PMs:
# MAGIC - **Search Relevance**: How well do results match user intent?
# MAGIC - **Listing Quality**: Are property details accurate and complete?
# MAGIC - **Recommendation Accuracy**: Do suggested homes match user preferences?
# MAGIC 
# MAGIC ### For Consumer Experience PMs:
# MAGIC - **Response Helpfulness**: Does the answer solve the user's problem?
# MAGIC - **Feature Discovery**: Does it guide users to relevant Zillow features?
# MAGIC - **User Education**: Does it help users understand the home buying/selling process?
# MAGIC 
# MAGIC ### For Mortgage/Finance PMs:
# MAGIC - **Calculator Accuracy**: Are financial calculations correct?
# MAGIC - **Affordability Guidance**: Is financial advice appropriate for the user?
# MAGIC - **Regulatory Compliance**: Does it avoid giving specific financial advice?
# MAGIC 
# MAGIC ### For Agent/Premier Agent PMs:
# MAGIC - **Lead Quality**: Would this interaction generate a qualified lead?
# MAGIC - **Professional Tone**: Is the response appropriate for agent interactions?
# MAGIC - **Contact Facilitation**: Does it appropriately connect users with agents?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 🛠️ Helper Functions

# COMMAND ----------

def create_openai_model(model_name: str = "gpt-4o") -> ChatOpenAI:
    """Create an OpenAI-compatible chat model."""
    return ChatOpenAI(
        model_name=model_name,
        default_headers={"apikey": os.environ["OPENAI_API_KEY"]},
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

def create_ensemble_evaluator(metric_name: str, metric_config: dict, judge_models: List[str]) -> callable:
    """Create an evaluator that uses multiple judge models and takes majority vote."""
    prompt_template = metric_config["prompt_template"]
    models = [create_openai_model(model_name=m) for m in judge_models]
    
    def evaluator(eval_df, builtin_metrics=None):
        results = []
        details = []
        
        for _, row in eval_df.iterrows():
            # Get required fields
            prompt = row.get("inputs", row.get(PROMPT_COLUMN, ""))
            response = row.get("predictions", row.get(RESPONSE_COLUMN, ""))
            user_context = row.get("user_context", "")
            expected_answer = row.get("expected_answer", "")
            
            # Format evaluation prompt
            eval_prompt = prompt_template.format(
                prompt=prompt,
                response=response,
                user_context=user_context,
                expected_answer=expected_answer
            )
            
            # Get scores from all judge models
            model_outputs = []
            scores = []
            score_key = f"{metric_name}_score"
            
            for model in models:
                try:
                    llm_response = model.invoke(eval_prompt)
                    result_json = json.loads(llm_response.content)
                    model_outputs.append({
                        "model": model.model_name,
                        "result": result_json
                    })
                    
                    # Extract score
                    score = float(result_json.get(score_key, 0))
                    scores.append(score)
                    
                except Exception as e:
                    print(f"Error with {model.model_name}: {str(e)}")
                    model_outputs.append({
                        "model": model.model_name,
                        "error": str(e)
                    })
                    scores.append(0)
            
            # Take majority vote or average
            if len(set(scores)) == 1:  # All judges agree
                final_score = scores[0]
            else:  # Take average for continuous scores
                final_score = sum(scores) / len(scores)
            
            results.append(final_score)
            details.append({
                "judges": model_outputs,
                "individual_scores": scores,
                "final_score": final_score
            })
        
        # Calculate metrics
        mean_score = sum(results) / len(results) if results else 0
        threshold = metric_config["threshold"]
        pass_rate = sum(1 for r in results if r >= threshold) / len(results) if results else 0
        
        return {
            f"{metric_name}/mean": mean_score,
            f"{metric_name}/pass_rate": pass_rate,
            f"{metric_name}/scores": results,
            f"{metric_name}/details": details,
        }
    
    return evaluator

# COMMAND ----------

# Create evaluator functions for all configured metrics
def create_all_evaluators(metrics_config: dict, judge_models: List[str]) -> dict:
    """Create evaluator functions for all metrics."""
    evaluators = {}
    for metric_name, metric_config in metrics_config.items():
        evaluators[metric_name] = create_ensemble_evaluator(
            metric_name, metric_config, judge_models
        )
    return evaluators

# COMMAND ----------

# MLflow scorer functions for metrics
def create_mlflow_scorers(metrics_config: dict) -> List[callable]:
    """Create MLflow scorer functions for tracking."""
    scorers = []
    
    for metric_name, config in metrics_config.items():
        # Create scorer for raw score
        @scorer(name=f"{metric_name}_score")
        def score_fn(outputs, metric=metric_name):
            return outputs.get(f"{metric}_score", 0)
        
        scorers.append(score_fn)
        
        # Create scorer for pass/fail status
        @scorer(name=f"{metric_name}_pass")
        def pass_fn(outputs, metric=metric_name, threshold=config["threshold"]):
            score = outputs.get(f"{metric}_score", 0)
            return score >= threshold
        
        scorers.append(pass_fn)
    
    return scorers

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 📊 Visualization Functions

# COMMAND ----------

def create_metric_visualizations(df: pd.DataFrame, metrics_config: dict) -> None:
    """Create comprehensive visualizations for evaluation metrics."""
    # Create multiple visualization types
    create_distribution_plots(df, metrics_config)
    create_pass_rate_chart(df, metrics_config)
    create_correlation_heatmap(df, metrics_config)
    create_metric_trends(df, metrics_config)

def create_distribution_plots(df: pd.DataFrame, metrics_config: dict) -> None:
    """Create distribution histograms for each metric."""
    num_metrics = len(metrics_config)
    cols = 3
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_metrics > 1 else [axes]
    
    for idx, (metric_name, config) in enumerate(metrics_config.items()):
        ax = axes[idx]
        scores = df[metric_name].values
        
        # Create histogram
        ax.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(config["threshold"], color='red', linestyle='--', 
                  label=f'Threshold: {config["threshold"]}', linewidth=2)
        ax.axvline(scores.mean(), color='green', linestyle='-', 
                  label=f'Mean: {scores.mean():.2f}', linewidth=2)
        
        ax.set_title(f'{metric_name}\n{config["description"]}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Score ({config["scale"]})')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_pass_rate_chart(df: pd.DataFrame, metrics_config: dict) -> None:
    """Create a bar chart showing pass rates for each metric."""
    pass_rates = []
    metric_names = []
    
    for metric_name, config in metrics_config.items():
        pass_rate = (df[f"{metric_name}_pass"] == "✅").mean() * 100
        pass_rates.append(pass_rate)
        metric_names.append(metric_name)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, pass_rates, color='lightgreen', edgecolor='darkgreen')
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, pass_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.axhline(y=80, color='red', linestyle='--', label='80% Target', alpha=0.7)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Pass Rate (%)', fontsize=12)
    plt.title('Metric Pass Rates', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pass_rates.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_correlation_heatmap(df: pd.DataFrame, metrics_config: dict) -> None:
    """Create a correlation heatmap between metrics."""
    metric_cols = list(metrics_config.keys())
    if len(metric_cols) < 2:
        return
    
    correlation_matrix = df[metric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": .8})
    plt.title('Metric Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/metric_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_metric_trends(df: pd.DataFrame, metrics_config: dict) -> None:
    """Create a line plot showing metric scores across samples."""
    if len(df) < 5:  # Skip if too few samples
        return
        
    plt.figure(figsize=(12, 6))
    
    for metric_name, config in metrics_config.items():
        scores = df[metric_name].values
        # Create rolling average for smoother trends
        window_size = min(5, len(scores) // 3)
        rolling_avg = pd.Series(scores).rolling(window=window_size, center=True).mean()
        
        plt.plot(range(len(scores)), scores, alpha=0.3, label=f'{metric_name} (raw)')
        plt.plot(range(len(rolling_avg)), rolling_avg, linewidth=2, label=f'{metric_name} (avg)')
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metric Trends Across Samples', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/metric_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_dashboard(df: pd.DataFrame, metrics_config: dict) -> pd.DataFrame:
    """Create a summary dashboard of all metrics."""
    summary_data = []
    
    for metric_name, config in metrics_config.items():
        scores = df[metric_name].values
        threshold = config["threshold"]
        
        summary_data.append({
            'Metric': metric_name,
            'Description': config['description'],
            'Mean Score': scores.mean(),
            'Std Dev': scores.std(),
            'Min Score': scores.min(),
            'Max Score': scores.max(),
            'Pass Rate': (scores >= threshold).mean() * 100,
            'Threshold': threshold,
            'Scale': config['scale']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create a nice formatted display
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY DASHBOARD")
    print("="*80)
    
    for _, row in summary_df.iterrows():
        print(f"\n📏 {row['Metric'].upper()}")
        print(f"   Description: {row['Description']}")
        print(f"   Scale: {row['Scale']} | Threshold: {row['Threshold']}")
        print(f"   Mean: {row['Mean Score']:.2f} | Std: {row['Std Dev']:.2f}")
        print(f"   Range: [{row['Min Score']:.2f}, {row['Max Score']:.2f}]")
        print(f"   ✅ Pass Rate: {row['Pass Rate']:.1f}%")
    
    print("\n" + "="*80)
    
    return summary_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. 🚀 Main Evaluation Pipeline

# COMMAND ----------

async def generate_responses(prompts: List[str], model_name: str) -> List[str]:
    """Generate responses using specified model."""
    print(f"🤖 Generating responses using {model_name}...")
    
    model = create_openai_model(model_name)
    responses = []
    
    for i, prompt in enumerate(prompts):
        try:
            response = model.invoke(prompt)
            responses.append(response.content)
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{len(prompts)} responses...")
        except Exception as e:
            print(f"   Error generating response {i + 1}: {str(e)}")
            responses.append(f"Error: {str(e)}")
    
    return responses

def run_evaluation_pipeline(
    df: pd.DataFrame,
    metrics_config: dict,
    judge_models: List[str],
    experiment_name: str,
    run_name: str
) -> pd.DataFrame:
    """Run the complete evaluation pipeline."""
    print("\n🔍 Starting Evaluation Pipeline...")
    
    # Create evaluators
    evaluators = create_all_evaluators(metrics_config, judge_models)
    
    # Prepare data for evaluation
    eval_data = df.copy()
    if "inputs" not in eval_data.columns:
        eval_data["inputs"] = eval_data[PROMPT_COLUMN]
    if "predictions" not in eval_data.columns:
        eval_data["predictions"] = eval_data[RESPONSE_COLUMN]
    
    # Run evaluations
    print("\n📏 Running metric evaluations...")
    all_scores = {}
    all_details = {}
    
    for metric_name, evaluator_func in evaluators.items():
        print(f"   Evaluating {metric_name}...")
        results = evaluator_func(eval_data)
        
        # Extract scores and details
        scores = results.get(f"{metric_name}/scores", [])
        details = results.get(f"{metric_name}/details", [])
        mean_score = results.get(f"{metric_name}/mean", 0)
        pass_rate = results.get(f"{metric_name}/pass_rate", 0)
        
        all_scores[metric_name] = scores
        all_details[f"{metric_name}_details"] = details
        
        print(f"      ✅ {metric_name}: Mean={mean_score:.2f}, Pass Rate={pass_rate*100:.1f}%")
    
    # Add scores to dataframe
    for metric_name, scores in all_scores.items():
        df[metric_name] = scores
        threshold = metrics_config[metric_name]["threshold"]
        df[f"{metric_name}_pass"] = ["✅" if s >= threshold else "❌" for s in scores]
    
    for detail_name, details in all_details.items():
        df[detail_name] = details
    
    # Log to MLflow
    print("\n📊 Logging to MLflow...")
    mlflow.set_experiment(experiment_name)
    
    # Prepare data for MLflow
    mlflow_data = []
    for _, row in df.iterrows():
        item = {
            "inputs": {
                "question": row[PROMPT_COLUMN],
                "answer": row[RESPONSE_COLUMN]
            },
            "outputs": {}
        }
        
        for metric_name in metrics_config:
            item["outputs"][f"{metric_name}_score"] = row[metric_name]
            item["outputs"][f"{metric_name}_details"] = row.get(f"{metric_name}_details", {})
        
        mlflow_data.append(item)
    
    # Create MLflow scorers
    scorers = create_mlflow_scorers(metrics_config)
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("judge_models", judge_models)
        mlflow.log_param("response_model", RESPONSE_MODEL)
        mlflow.log_param("num_samples", len(df))
        mlflow.log_param("metrics", list(metrics_config.keys()))
        
        # Run MLflow evaluation
        eval_results = mlflow.genai.evaluate(
            data=mlflow_data,
            scorers=scorers
        )
        
        # Log summary metrics
        for metric_name in metrics_config:
            mean_score = df[metric_name].mean()
            pass_rate = (df[f"{metric_name}_pass"] == "✅").mean()
            mlflow.log_metric(f"{metric_name}_mean", mean_score)
            mlflow.log_metric(f"{metric_name}_pass_rate", pass_rate)
    
    print("✅ Evaluation complete!")
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. 🎯 Run Evaluation

# COMMAND ----------

# Load your data
print("📂 Loading evaluation data...")
# For demo purposes, create sample data
sample_data = {
    PROMPT_COLUMN: [
        "What's the average home price in Seattle?",
        "How do I calculate my mortgage payment?",
        "What factors affect my home's Zestimate?",
        "Should I buy or rent in the current market?",
        "What are the best neighborhoods for families in Austin?"
    ],
    RESPONSE_COLUMN: [
        "The average home price in Seattle is approximately $850,000 as of late 2024. This represents a 5% increase from last year. Popular neighborhoods like Capitol Hill and Queen Anne tend to be above this average, while areas further from downtown like Rainier Valley offer more affordable options.",
        "To calculate your mortgage payment, use this formula: M = P[r(1+r)^n]/[(1+r)^n-1], where M is monthly payment, P is principal, r is monthly interest rate, and n is number of payments. For a $400,000 loan at 7% for 30 years, your payment would be about $2,661/month. Don't forget to add property taxes, insurance, and HOA fees.",
        "Your Zestimate is influenced by: 1) Recent sales of comparable homes in your area, 2) Your home's features (bedrooms, bathrooms, square footage), 3) Local market trends, 4) Property tax assessments, 5) Any updates or renovations you've reported. The algorithm updates daily based on new market data.",
        "In the current market with interest rates around 7%, buying makes sense if you plan to stay 5+ years and have 20% down payment. Renting offers flexibility but you miss out on building equity. Consider your job stability, local market appreciation rates, and the rent vs. buy calculator on Zillow to make an informed decision.",
        "Top family-friendly neighborhoods in Austin include: 1) Circle C Ranch - excellent schools and parks, 2) Avery Ranch - affordable with good amenities, 3) Steiner Ranch - lakefront community with top schools, 4) Mueller - walkable with modern amenities. Consider proximity to schools, parks, and commute times when choosing."
    ]
}

df = pd.DataFrame(sample_data)

# If you have a CSV file, uncomment this:
# df = pd.read_csv(INPUT_FILE_PATH)

print(f"✅ Loaded {len(df)} samples for evaluation")
df.head()

# COMMAND ----------

# Generate responses if needed (skip if you already have responses)
if RESPONSE_COLUMN not in df.columns or df[RESPONSE_COLUMN].isna().any():
    print("\n🤖 Generating model responses...")
    responses = await generate_responses(
        df[PROMPT_COLUMN].tolist(), 
        RESPONSE_MODEL
    )
    df[RESPONSE_COLUMN] = responses

# COMMAND ----------

# Run the evaluation
print("\n🚀 Starting evaluation process...")
results_df = run_evaluation_pipeline(
    df=df,
    metrics_config=EVALUATION_METRICS,
    judge_models=JUDGE_MODELS,
    experiment_name=EXPERIMENT_NAME,
    run_name=RUN_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. 📊 View Results & Visualizations

# COMMAND ----------

# Create visualizations
print("\n📈 Creating visualizations...")
create_metric_visualizations(results_df, EVALUATION_METRICS)

# COMMAND ----------

# Create summary dashboard
summary_df = create_summary_dashboard(results_df, EVALUATION_METRICS)

# COMMAND ----------

# Display detailed results
print("\n📋 Detailed Results Sample:")
display_columns = [PROMPT_COLUMN, RESPONSE_COLUMN] + list(EVALUATION_METRICS.keys())
results_df[display_columns].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. 💾 Save Results

# COMMAND ----------

# Save detailed results
output_file = f"{OUTPUT_DIR}/evaluation_results_{RUN_NAME}.csv"
results_df.to_csv(output_file, index=False)
print(f"\n💾 Detailed results saved to: {output_file}")

# Save summary
summary_file = f"{OUTPUT_DIR}/evaluation_summary_{RUN_NAME}.csv"
summary_df.to_csv(summary_file, index=False)
print(f"💾 Summary saved to: {summary_file}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. 🔍 Drill-Down Analysis

# COMMAND ----------

# Analyze failures for each metric
print("\n🔍 Analyzing Failed Cases:")
for metric_name in EVALUATION_METRICS:
    failed_mask = results_df[f"{metric_name}_pass"] == "❌"
    num_failed = failed_mask.sum()
    
    if num_failed > 0:
        print(f"\n❌ {metric_name}: {num_failed} failed cases")
        print("Sample failed case:")
        failed_sample = results_df[failed_mask].iloc[0]
        print(f"  Query: {failed_sample[PROMPT_COLUMN][:100]}...")
        print(f"  Score: {failed_sample[metric_name]}")
        if f"{metric_name}_details" in results_df.columns:
            details = failed_sample[f"{metric_name}_details"]
            if isinstance(details, dict) and "judges" in details:
                print(f"  Explanation: {details['judges'][0]['result'].get('explanation', 'N/A')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. 🎉 Next Steps
# MAGIC 
# MAGIC ### To add new metrics:
# MAGIC 1. Copy the metric template from Section 4
# MAGIC 2. Define your evaluation criteria and scoring
# MAGIC 3. Add to `EVALUATION_METRICS` dictionary
# MAGIC 4. Re-run the notebook
# MAGIC 
# MAGIC ### To view results in MLflow:
# MAGIC 1. Navigate to the Experiments tab
# MAGIC 2. Find your experiment: `{EXPERIMENT_NAME}`
# MAGIC 3. Click on the run: `{RUN_NAME}`
# MAGIC 4. View metrics, parameters, and artifacts
# MAGIC 
# MAGIC ### To share results:
# MAGIC 1. Export the summary CSV
# MAGIC 2. Share the MLflow experiment link
# MAGIC 3. Include the visualization PNG files

# COMMAND ----------

print("\n🎉 Evaluation Complete!")
print(f"📊 View results in MLflow: {EXPERIMENT_NAME}")
print(f"💾 Results saved to: {OUTPUT_DIR}")