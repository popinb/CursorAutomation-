# Databricks notebook source
# MAGIC %md
# MAGIC # Zillow LLM Judge - Simplified Evaluation System
# MAGIC
# MAGIC This notebook provides a streamlined framework for PMs to evaluate AI responses using LLM judges.
# MAGIC
# MAGIC ## Quick Start:
# MAGIC 1. **Run Cell 1** - Install packages
# MAGIC 2. **Run Cell 3** - Upload your files using the UI
# MAGIC 3. **Run Cell 4** - Select your judge model and settings
# MAGIC 4. **Run Cell 5** - Define your custom metrics (Binary, 1-5, or Percentage)
# MAGIC 5. **Run All Remaining Cells** - Execute evaluation
# MAGIC
# MAGIC ## Features:
# MAGIC - ✅ Direct file upload UI
# MAGIC - ✅ Simple model selection dropdown
# MAGIC - ✅ Three metric types: Binary, 1-5 Scale, Percentage
# MAGIC - ✅ Automatic ground truth matching
# MAGIC - ✅ MLflow tracking and CSV export

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Installation and Setup
# MAGIC **Just run this cell - no changes needed**

# COMMAND ----------

# Install required packages using %pip (Databricks best practice)
# %pip automatically handles environment changes - no manual restart needed

%pip install mlflow>=3.0 --quiet
%pip install langchain-community langchain-openai --quiet
%pip install openai --quiet
%pip install pandas --quiet
%pip install plotly --quiet
%pip install python-docx --quiet

print("✅ Packages installed successfully!")
print("\n📌 Note: Dependency warnings about protobuf are expected and can be ignored.")
print("   Databricks manages these dependencies internally.")
print("\n➡️ Continue directly to Cell 2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Import Libraries
# MAGIC **Just run this cell - no changes needed**

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

# MAGIC %md
# MAGIC ## Cell 3: 📤 Upload Your Files
# MAGIC **Use the widgets below to specify your file paths**

# COMMAND ----------

# =============================================================================
# FILE UPLOAD CONFIGURATION
# =============================================================================

# Create file upload widgets
dbutils.widgets.text(
    "evaluation_data_path", 
    "/workspace/evaluation_data.csv", 
    "📁 Evaluation Data (CSV with 'prompt' and 'response' columns)"
)

dbutils.widgets.text(
    "ground_truth_paths",
    "/workspace/ground_truth.csv",
    "📚 Ground Truth Files (comma-separated paths, optional)"
)

dbutils.widgets.dropdown(
    "ground_truth_format",
    "csv",
    ["csv", "docx"],
    "📄 Ground Truth Format"
)

# Get file paths
EVAL_DATA_PATH = dbutils.widgets.get("evaluation_data_path")
GROUND_TRUTH_PATHS = dbutils.widgets.get("ground_truth_paths")
GROUND_TRUTH_FORMAT = dbutils.widgets.get("ground_truth_format")

print("📁 FILE CONFIGURATION")
print("="*60)
print(f"Evaluation Data: {EVAL_DATA_PATH}")
print(f"Ground Truth Files: {GROUND_TRUTH_PATHS}")
print(f"Ground Truth Format: {GROUND_TRUTH_FORMAT}")
print("="*60)

# Validate and load evaluation data
try:
    if os.path.exists(EVAL_DATA_PATH):
        eval_df = pd.read_csv(EVAL_DATA_PATH)
        print(f"\n✅ Loaded evaluation data: {len(eval_df)} rows")
        
        # Check required columns
        required_cols = ['prompt', 'response']
        missing_cols = [col for col in required_cols if col not in eval_df.columns]
        if missing_cols:
            print(f"❌ ERROR: Missing required columns: {missing_cols}")
            print(f"   Your file has columns: {list(eval_df.columns)}")
        else:
            print("✅ All required columns present")
            print("\nData Preview:")
            display(eval_df.head(3))
    else:
        print(f"❌ ERROR: File not found at {EVAL_DATA_PATH}")
        print("Creating sample data for demonstration...")
        
        # Create sample data
        eval_df = pd.DataFrame({
            'prompt': [
                "What's the best mortgage for first-time buyers?",
                "How do I improve my credit score?",
                "What are current interest rates?"
            ],
            'response': [
                "FHA loans are popular for first-time buyers with 3.5% down payment.",
                "Pay bills on time, reduce credit utilization, and check for errors.",
                "Current rates vary by lender, typically 6-7% for 30-year fixed."
            ]
        })
        print("✅ Created sample data")
        display(eval_df)
        
except Exception as e:
    print(f"❌ ERROR loading data: {e}")
    eval_df = pd.DataFrame()

# Load ground truth if provided
ground_truth_df = pd.DataFrame()
if GROUND_TRUTH_PATHS and GROUND_TRUTH_PATHS.strip():
    print("\n📚 Loading ground truth files...")
    
    ground_truth_files = [f.strip() for f in GROUND_TRUTH_PATHS.split(",")]
    all_ground_truth = []
    
    for file_path in ground_truth_files:
        if os.path.exists(file_path):
            try:
                if GROUND_TRUTH_FORMAT == "csv":
                    gt_df = pd.read_csv(file_path)
                else:  # docx
                    # Simple DOCX parsing
                    doc = Document(file_path)
                    data = []
                    for table in doc.tables:
                        headers = [cell.text.strip() for cell in table.rows[0].cells]
                        for row in table.rows[1:]:
                            row_data = [cell.text.strip() for cell in row.cells]
                            data.append(dict(zip(headers, row_data)))
                    gt_df = pd.DataFrame(data)
                
                all_ground_truth.append(gt_df)
                print(f"   ✅ Loaded {len(gt_df)} entries from {file_path}")
            except Exception as e:
                print(f"   ❌ Error loading {file_path}: {e}")
        else:
            print(f"   ⚠️ File not found: {file_path}")
    
    if all_ground_truth:
        ground_truth_df = pd.concat(all_ground_truth, ignore_index=True)
        ground_truth_df = ground_truth_df.drop_duplicates(subset=['prompt'], keep='last')
        print(f"\n✅ Total ground truth entries: {len(ground_truth_df)}")
        
        # Merge with evaluation data
        if 'prompt' in ground_truth_df.columns:
            eval_df = eval_df.merge(
                ground_truth_df[['prompt', 'ground_truth']], 
                on='prompt', 
                how='left'
            )
            gt_coverage = eval_df['ground_truth'].notna().sum()
            print(f"✅ Ground truth matched for {gt_coverage}/{len(eval_df)} samples")

# Store data globally
EVALUATION_DATA = eval_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: 🤖 Configure Judge Model and Settings
# MAGIC **Select your LLM judge and evaluation settings**

# COMMAND ----------

# =============================================================================
# MODEL AND EVALUATION SETTINGS
# =============================================================================

# Model selection
dbutils.widgets.dropdown(
    "judge_model",
    "gpt-4o-mini",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-sonnet", "databricks-llm"],
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

dbutils.widgets.dropdown(
    "parallel_evaluations",
    "2",
    ["1", "2", "3", "4"],
    "⚡ Parallel Evaluations"
)

# Get settings
JUDGE_MODEL = dbutils.widgets.get("judge_model")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
INCLUDE_GROUND_TRUTH = dbutils.widgets.get("include_ground_truth") == "Yes"
MAX_CONCURRENCY = int(dbutils.widgets.get("parallel_evaluations"))

print("🤖 EVALUATION SETTINGS")
print("="*60)
print(f"Judge Model: {JUDGE_MODEL}")
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

# Test connection
try:
    test_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'OK'"}],
        max_tokens=10
    )
    print("✅ LLM connection successful!")
except Exception as e:
    print(f"❌ LLM connection failed: {e}")

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
    # EXAMPLE 1: BINARY METRIC (Pass/Fail)
    # ========================================
    # {
    #     "name": "accuracy_check",
    #     "type": "binary",
    #     "description": "Checks if the response contains accurate information",
    #     "evaluation_prompt": """
    # Evaluate if the response contains accurate information.
    # 
    # User Query: {prompt}
    # AI Response: {response}
    # Ground Truth (if available): {ground_truth}
    # 
    # Accuracy Criteria:
    # - All facts must be correct
    # - No misleading information
    # - Numbers and statistics must be accurate
    # - Procedures described correctly
    # 
    # Scoring:
    # - 1 (PASS): All information is accurate
    # - 0 (FAIL): Contains any inaccurate information
    # 
    # Return JSON:
    # {{
    #     "accuracy_check_score": 1,
    #     "explanation": "All facts verified as accurate. The response correctly states..."
    # }}
    # """
    # },
    
    # ========================================
    # EXAMPLE 2: 1-5 SCALE METRIC
    # ========================================
    # {
    #     "name": "helpfulness_rating",
    #     "type": "scale_1_5",
    #     "description": "Rates how helpful the response is on a 1-5 scale",
    #     "evaluation_prompt": """
    # Rate the helpfulness of this response from 1 to 5.
    # 
    # User Query: {prompt}
    # AI Response: {response}
    # Ground Truth (if available): {ground_truth}
    # 
    # Helpfulness Scale:
    # 5 = Extremely helpful - Comprehensive answer with actionable steps
    # 4 = Very helpful - Good answer with useful information
    # 3 = Moderately helpful - Adequate but could be better
    # 2 = Slightly helpful - Limited value, missing key information
    # 1 = Not helpful - Fails to address the question
    # 
    # Consider:
    # - Does it answer the user's question?
    # - Is the information actionable?
    # - Are next steps clear?
    # 
    # Return JSON:
    # {{
    #     "helpfulness_rating_score": 4,
    #     "explanation": "Very helpful response that answers the main question and provides clear next steps..."
    # }}
    # """
    # },
    
    # ========================================
    # EXAMPLE 3: PERCENTAGE METRIC (0-100%)
    # ========================================
    # {
    #     "name": "completeness_percentage",
    #     "type": "percentage",
    #     "description": "Measures what percentage of the question was addressed",
    #     "evaluation_prompt": """
    # Evaluate what percentage (0-100%) of the user's question was addressed.
    # 
    # User Query: {prompt}
    # AI Response: {response}
    # Ground Truth (if available): {ground_truth}
    # 
    # Assessment Process:
    # 1. Identify all components of the user's question
    # 2. Check which components were addressed
    # 3. Calculate percentage of coverage
    # 
    # Examples:
    # - 90-100%: Fully addresses all aspects
    # - 70-89%: Most important parts covered
    # - 50-69%: About half addressed
    # - 30-49%: Some parts addressed
    # - 0-29%: Minimal coverage
    # 
    # Return JSON (use decimal, e.g., 0.85 for 85%):
    # {{
    #     "completeness_percentage_score": 0.85,
    #     "explanation": "The response addresses 85% of the question. It covers the main topic well but misses..."
    # }}
    # """
    # },
    
    # ========================================
    # ADD YOUR METRICS BELOW
    # ========================================
    
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

# MAGIC %md
# MAGIC ## Cell 6: System Classes and Functions
# MAGIC **Just run this cell - no changes needed**

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
                
                resp = client.chat.completions.create(
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
            result_json = json.loads(result.content)
            
            # Extract score
            score_key = f"{metric.name}_score"
            score = result_json.get(score_key, 0)
            explanation = result_json.get("explanation", "No explanation provided")
            
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
            continue  # Skip invalid types
        
        config = MetricConfig(
            name=metric['name'],
            description=metric['description'],
            metric_type=metric_type,
            prompt_template=metric['evaluation_prompt'],
            threshold=METRIC_THRESHOLDS.get(metric['name'], 1.0)
        )
        configs.append(config)
    
    return configs

print("✅ System classes loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Run Evaluation
# MAGIC **Just run this cell to execute the evaluation**

# COMMAND ----------

# Process metrics and create evaluator
metric_configs = process_custom_metrics(CUSTOM_METRICS)

if not metric_configs:
    print("❌ No valid metrics to evaluate!")
    print("Please define metrics in Cell 5")
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Generate Summary Report
# MAGIC **Just run this cell to see evaluation summary**

# COMMAND ----------

if 'results_df' in globals() and not results_df.empty:
    print("📊 EVALUATION SUMMARY REPORT")
    print("="*60)
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"   Total Samples: {len(results_df)}")
    print(f"   Metrics Evaluated: {len(metric_configs)}")
    
    # Per-metric summary
    summary_data = []
    
    for metric in metric_configs:
        scores = results_df[f"{metric.name}_score"]
        
        # Calculate statistics
        mean_score = scores.mean()
        std_score = scores.std()
        min_score = scores.min()
        max_score = scores.max()
        pass_count = (scores >= metric.threshold).sum()
        pass_rate = pass_count / len(scores)
        
        summary_data.append({
            'Metric': metric.name,
            'Type': metric.metric_type.value,
            'Mean': f"{mean_score:.3f}",
            'Std Dev': f"{std_score:.3f}",
            'Min': f"{min_score:.3f}",
            'Max': f"{max_score:.3f}",
            'Pass Rate': f"{pass_rate:.1%}",
            'Passed': f"{pass_count}/{len(scores)}"
        })
        
        print(f"\n📊 {metric.name.upper()}:")
        print(f"   Type: {metric.metric_type.value}")
        print(f"   Mean Score: {mean_score:.3f}")
        print(f"   Pass Rate: {pass_rate:.1%} ({pass_count}/{len(scores)})")
        print(f"   Threshold: {metric.threshold}")
    
    # Display summary table
    summary_df = pd.DataFrame(summary_data)
    print("\n📋 Summary Table:")
    display(summary_df)
    
    # Create visualizations
    if len(metric_configs) > 0:
        fig = make_subplots(
            rows=1, 
            cols=len(metric_configs),
            subplot_titles=[m.name for m in metric_configs]
        )
        
        for i, metric in enumerate(metric_configs, 1):
            scores = results_df[f"{metric.name}_score"]
            
            # Create histogram
            fig.add_trace(
                go.Histogram(x=scores, name=metric.name, nbinsx=20),
                row=1, col=i
            )
        
        fig.update_layout(
            height=400, 
            showlegend=False,
            title_text="Score Distributions"
        )
        fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Export Results
# MAGIC **Just run this cell to save results**

# COMMAND ----------

if 'results_df' in globals() and not results_df.empty:
    # Generate filename with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_filename = f"llm_judge_results_{timestamp}.csv"
    output_path = f"/workspace/{output_filename}"
    
    # Save results
    try:
        results_df.to_csv(output_path, index=False)
        print(f"✅ Results exported to: {output_path}")
        print(f"   File size: {len(results_df)} rows × {len(results_df.columns)} columns")
        
        # Also save summary
        summary_filename = f"llm_judge_summary_{timestamp}.csv"
        summary_path = f"/workspace/{summary_filename}"
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Summary exported to: {summary_path}")
        
        print("\n📥 You can download these files for:")
        print("   - Sharing with stakeholders")
        print("   - Further analysis in Excel")
        print("   - Historical comparison")
        
    except Exception as e:
        print(f"❌ Error exporting results: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: MLflow Logging (Optional)
# MAGIC **Run this cell to log results to MLflow**

# COMMAND ----------

if 'results_df' in globals() and not results_df.empty:
    try:
        # Set experiment
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"llm_judge_{timestamp}"):
            # Log parameters
            mlflow.log_param("judge_model", JUDGE_MODEL)
            mlflow.log_param("num_samples", len(results_df))
            mlflow.log_param("num_metrics", len(metric_configs))
            
            # Log metrics
            for metric in metric_configs:
                scores = results_df[f"{metric.name}_score"]
                mlflow.log_metric(f"{metric.name}_mean", scores.mean())
                mlflow.log_metric(f"{metric.name}_pass_rate", (scores >= metric.threshold).mean())
            
            # Log artifacts
            mlflow.log_artifact(output_path)
            mlflow.log_artifact(summary_path)
            
            print(f"✅ Results logged to MLflow experiment: {EXPERIMENT_NAME}")
            print(f"   View in MLflow UI to compare runs and track progress")
            
    except Exception as e:
        print(f"⚠️  MLflow logging failed: {e}")
        print("   Results are still saved in CSV files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 Evaluation Complete!
# MAGIC
# MAGIC ### Next Steps:
# MAGIC 1. **Review the summary** in Cell 8
# MAGIC 2. **Download the CSV files** from Cell 9
# MAGIC 3. **Check MLflow** for tracking (if enabled)
# MAGIC 4. **Modify metrics** in Cell 5 and re-run for different evaluations
# MAGIC
# MAGIC ### Tips:
# MAGIC - Use Binary metrics for clear pass/fail criteria
# MAGIC - Use 1-5 Scale for quality assessments
# MAGIC - Use Percentage for completeness or coverage metrics
# MAGIC - Ground truth improves evaluation quality when available