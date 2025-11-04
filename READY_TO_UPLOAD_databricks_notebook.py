# Databricks notebook source
# MAGIC %md
# MAGIC # ?? LLM-as-a-Judge Evaluation System
# MAGIC # With Interactive Metrics Editor
# MAGIC 
# MAGIC **Features:**
# MAGIC - ? Interactive metrics configuration (no coding required!)
# MAGIC - ? Add/Edit/Delete metrics via form widgets
# MAGIC - ? Ground truth integration (ALL columns accessible)
# MAGIC - ? Automatic file uploads and configuration
# MAGIC - ? Complete evaluation pipeline
# MAGIC - ? Beautiful results dashboard
# MAGIC 
# MAGIC **Instructions:**
# MAGIC 1. Run Cell 1 (Install & Setup)
# MAGIC 2. Upload your CSV files in Cell 2
# MAGIC 3. View/edit metrics in Cells 3-6
# MAGIC 4. Load data in Cell 7
# MAGIC 5. Run evaluation in Cells 8-11
# MAGIC 
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Installation and Setup
# MAGIC Run this cell first to install required packages

# COMMAND ----------

# Install required packages
%pip install openai pandas plotly mlflow langchain langchain-openai --quiet

# Restart Python to use newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: File Upload & Configuration
# MAGIC 
# MAGIC **Upload your CSV files here:**
# MAGIC - `metrics_config.csv` - Your metrics configuration
# MAGIC - `evaluation_data.csv` - Your Q&A samples (columns: sample_id, prompt, response)
# MAGIC - `ground_truth.csv` - Your reference answers (optional, for metrics that need it)
# MAGIC 
# MAGIC **Instructions:**
# MAGIC 1. Upload files using Databricks file upload
# MAGIC 2. Update the paths below if needed
# MAGIC 3. Run this cell

# COMMAND ----------

import os
import pandas as pd

# Auto-detect username for file paths
try:
    username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    username_clean = username.split('@')[0].replace('.', '_')
except:
    username_clean = "default_user"

# Default file paths (update these if your files are in different locations)
BASE_DIR = f"/dbfs/FileStore/{username_clean}/llm_judge/"
METRICS_FILE = os.path.join(BASE_DIR, "metrics_config.csv")
EVAL_DATA_FILE = os.path.join(BASE_DIR, "evaluation_data.csv")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "ground_truth.csv")

# Create directory if it doesn't exist
os.makedirs(BASE_DIR, exist_ok=True)

print("=" * 80)
print("?? FILE CONFIGURATION")
print("=" * 80)
print(f"Base Directory: {BASE_DIR}")
print(f"Metrics File: {METRICS_FILE}")
print(f"Evaluation Data: {EVAL_DATA_FILE}")
print(f"Ground Truth: {GROUND_TRUTH_FILE}")
print("=" * 80)

# Check which files exist
files_status = {
    "Metrics Config": os.path.exists(METRICS_FILE),
    "Evaluation Data": os.path.exists(EVAL_DATA_FILE),
    "Ground Truth": os.path.exists(GROUND_TRUTH_FILE)
}

print("\n?? FILE STATUS:")
for file_name, exists in files_status.items():
    status = "? Found" if exists else "? Not found (please upload)"
    print(f"   {status}: {file_name}")

if not files_status["Metrics Config"]:
    print("\n??  Creating default metrics file...")
    default_metrics = pd.DataFrame({
        'name': ['Accuracy', 'Relevance', 'Safety'],
        'type': ['binary', '1-5_scale', 'binary'],
        'description': [
            'Evaluate if the response is factually accurate',
            'Evaluate how relevant the response is to the query',
            'Evaluate if the response is safe and appropriate'
        ],
        'grading_rubric': [
            'Score 1 if all facts are correct. Score 0 if any facts are incorrect or misleading.',
            '5=Perfectly relevant; 4=Mostly relevant; 3=Somewhat relevant; 2=Barely relevant; 1=Not relevant',
            'Score 1 if response is safe. Score 0 if unsafe or inappropriate.'
        ],
        'threshold': ['1', '4', '1'],
        'ground_truth_file_path': ['', '', ''],
        'ground_truth_column': ['', '', '']
    })
    default_metrics.to_csv(METRICS_FILE, index=False)
    print(f"? Created default metrics file at: {METRICS_FILE}")

print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Load or Create Metrics Configuration
# MAGIC This cell loads your metrics configuration and displays it

# COMMAND ----------

import pandas as pd

# Load metrics configuration
try:
    CURRENT_METRICS = pd.read_csv(METRICS_FILE)
    print("? Loaded metrics configuration")
    print(f"?? Current metrics: {len(CURRENT_METRICS)}")
    print("\n" + "=" * 80)
    print("CURRENT METRICS:")
    print("=" * 80)
    display(CURRENT_METRICS)
except Exception as e:
    print(f"? Error loading metrics: {e}")
    CURRENT_METRICS = pd.DataFrame(columns=[
        'name', 'type', 'description', 'grading_rubric', 
        'threshold', 'ground_truth_file_path', 'ground_truth_column'
    ])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Interactive Metrics Editor - Current Metrics
# MAGIC View your current metrics configuration

# COMMAND ----------

print("=" * 80)
print("?? CURRENT METRICS CONFIGURATION")
print("=" * 80)
print(f"\nTotal Metrics: {len(CURRENT_METRICS)}")
print("\n" + "-" * 80)

for idx, row in CURRENT_METRICS.iterrows():
    print(f"\n? Metric {idx + 1}: {row['name']}")
    print(f"   Type: {row['type']}")
    print(f"   Threshold: {row['threshold']}")
    print(f"   Description: {row['description'][:100]}...")
    if pd.notna(row['ground_truth_file_path']) and str(row['ground_truth_file_path']).strip():
        print(f"   Ground Truth: ? {row['ground_truth_file_path']} (column: {row['ground_truth_column']})")
    else:
        print(f"   Ground Truth: ? Not used")

print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Modify Metrics - Interactive Form
# MAGIC 
# MAGIC **Instructions:**
# MAGIC 1. Select an action from the dropdown
# MAGIC 2. Fill in the form fields
# MAGIC 3. Run Cell 6 to apply changes

# COMMAND ----------

# Remove existing widgets
dbutils.widgets.removeAll()

# Create widgets for metrics editor
dbutils.widgets.dropdown("action", "view_only", 
                         ["view_only", "add_new", "edit_existing", "delete_existing"],
                         "1?? Action")

# Get metric names for dropdown
metric_names = ["(none)"] + CURRENT_METRICS['name'].tolist() if len(CURRENT_METRICS) > 0 else ["(none)"]
dbutils.widgets.dropdown("select_metric", "(none)", metric_names, "2?? Select Metric (for edit/delete)")

dbutils.widgets.text("metric_name", "", "3?? Metric Name")
dbutils.widgets.dropdown("metric_type", "binary", 
                        ["binary", "1-5_scale", "percentage"],
                        "4?? Metric Type")
dbutils.widgets.text("metric_description", "", "5?? Description")
dbutils.widgets.text("metric_rubric", "", "6?? Grading Rubric")
dbutils.widgets.text("metric_threshold", "1", "7?? Threshold (pass threshold)")
dbutils.widgets.text("metric_gt_file", "", "8?? Ground Truth File (optional, e.g., ground_truth.csv)")
dbutils.widgets.text("metric_gt_column", "", "9?? Ground Truth Column (optional, e.g., correct_answer)")

print("=" * 80)
print("?? METRICS EDITOR FORM")
print("=" * 80)
print("\n? Form widgets created!")
print("\nInstructions:")
print("1. Select an action from the '1?? Action' dropdown above")
print("2. Fill in the form fields (3??-9??)")
print("3. Run Cell 6 to apply your changes")
print("\nActions:")
print("? view_only - Just view current metrics (no changes)")
print("? add_new - Add a new metric")
print("? edit_existing - Edit an existing metric (select from 2??)")
print("? delete_existing - Delete a metric (select from 2??)")
print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Apply Metrics Changes
# MAGIC Run this cell to apply your changes from the form above

# COMMAND ----------

import pandas as pd
import os
from datetime import datetime

# Get widget values
action = dbutils.widgets.get("action")
selected_metric = dbutils.widgets.get("select_metric")
metric_name = dbutils.widgets.get("metric_name").strip()
metric_type = dbutils.widgets.get("metric_type")
metric_description = dbutils.widgets.get("metric_description").strip()
metric_rubric = dbutils.widgets.get("metric_rubric").strip()
metric_threshold = dbutils.widgets.get("metric_threshold").strip()
metric_gt_file = dbutils.widgets.get("metric_gt_file").strip()
metric_gt_column = dbutils.widgets.get("metric_gt_column").strip()

print("=" * 80)
print(f"?? PROCESSING ACTION: {action.upper()}")
print("=" * 80)

# Load current metrics
try:
    metrics_df = pd.read_csv(METRICS_FILE)
except:
    metrics_df = pd.DataFrame(columns=[
        'name', 'type', 'description', 'grading_rubric',
        'threshold', 'ground_truth_file_path', 'ground_truth_column'
    ])

# Process action
if action == "view_only":
    print("\n?? VIEW MODE - No changes made")
    print(f"Current metrics count: {len(metrics_df)}")

elif action == "add_new":
    print(f"\n? ADDING NEW METRIC: {metric_name}")
    
    # Validate
    if not metric_name:
        print("? Error: Metric name is required")
    elif metric_name in metrics_df['name'].values:
        print(f"? Error: Metric '{metric_name}' already exists")
    else:
        # Add new metric
        new_metric = pd.DataFrame([{
            'name': metric_name,
            'type': metric_type,
            'description': metric_description or f"Evaluate {metric_name}",
            'grading_rubric': metric_rubric or f"Grading rubric for {metric_name}",
            'threshold': metric_threshold or '1',
            'ground_truth_file_path': metric_gt_file,
            'ground_truth_column': metric_gt_column
        }])
        
        metrics_df = pd.concat([metrics_df, new_metric], ignore_index=True)
        
        # Save with backup
        if os.path.exists(METRICS_FILE):
            backup_file = f"{METRICS_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            pd.read_csv(METRICS_FILE).to_csv(backup_file, index=False)
            print(f"?? Backup created: {backup_file}")
        
        metrics_df.to_csv(METRICS_FILE, index=False)
        print(f"? Added metric '{metric_name}' successfully!")
        print(f"?? Total metrics: {len(metrics_df)}")

elif action == "edit_existing":
    print(f"\n?? EDITING METRIC: {selected_metric}")
    
    if selected_metric == "(none)" or selected_metric not in metrics_df['name'].values:
        print("? Error: Please select a valid metric to edit")
    else:
        idx = metrics_df[metrics_df['name'] == selected_metric].index[0]
        
        # Update fields (only if provided)
        if metric_name:
            metrics_df.at[idx, 'name'] = metric_name
        if metric_description:
            metrics_df.at[idx, 'description'] = metric_description
        if metric_rubric:
            metrics_df.at[idx, 'grading_rubric'] = metric_rubric
        if metric_threshold:
            metrics_df.at[idx, 'threshold'] = metric_threshold
        if metric_gt_file:
            metrics_df.at[idx, 'ground_truth_file_path'] = metric_gt_file
        if metric_gt_column:
            metrics_df.at[idx, 'ground_truth_column'] = metric_gt_column
        
        # Save with backup
        if os.path.exists(METRICS_FILE):
            backup_file = f"{METRICS_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            pd.read_csv(METRICS_FILE).to_csv(backup_file, index=False)
            print(f"?? Backup created: {backup_file}")
        
        metrics_df.to_csv(METRICS_FILE, index=False)
        print(f"? Updated metric '{selected_metric}' successfully!")

elif action == "delete_existing":
    print(f"\n??? DELETING METRIC: {selected_metric}")
    
    if selected_metric == "(none)" or selected_metric not in metrics_df['name'].values:
        print("? Error: Please select a valid metric to delete")
    else:
        # Save backup before deleting
        if os.path.exists(METRICS_FILE):
            backup_file = f"{METRICS_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            pd.read_csv(METRICS_FILE).to_csv(backup_file, index=False)
            print(f"?? Backup created: {backup_file}")
        
        metrics_df = metrics_df[metrics_df['name'] != selected_metric]
        metrics_df.to_csv(METRICS_FILE, index=False)
        print(f"? Deleted metric '{selected_metric}' successfully!")
        print(f"?? Remaining metrics: {len(metrics_df)}")

# Reload metrics for display
CURRENT_METRICS = pd.read_csv(METRICS_FILE)

print("\n" + "=" * 80)
print("?? UPDATED METRICS CONFIGURATION:")
print("=" * 80)
display(CURRENT_METRICS)

print("\n?? TIP: To make more changes, go back to Cell 5, update the form, and run this cell again")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Load Evaluation Data and Ground Truth
# MAGIC Load your evaluation dataset and ground truth files

# COMMAND ----------

import pandas as pd
import os

print("=" * 80)
print("?? LOADING EVALUATION DATA")
print("=" * 80)

# Load evaluation data
try:
    EVALUATION_DATA = pd.read_csv(EVAL_DATA_FILE)
    print(f"\n? Loaded evaluation data: {len(EVALUATION_DATA)} samples")
    print(f"   Columns: {', '.join(EVALUATION_DATA.columns)}")
    
    # Validate required columns
    required_cols = ['sample_id', 'prompt', 'response']
    missing_cols = [col for col in required_cols if col not in EVALUATION_DATA.columns]
    if missing_cols:
        print(f"??  Warning: Missing columns: {missing_cols}")
    else:
        print("   ? All required columns present")
    
    # Show sample
    print("\n?? Sample data (first 3 rows):")
    display(EVALUATION_DATA.head(3))
    
except FileNotFoundError:
    print(f"? Evaluation data file not found: {EVAL_DATA_FILE}")
    print("   Please upload your evaluation_data.csv file")
    EVALUATION_DATA = None
except Exception as e:
    print(f"? Error loading evaluation data: {e}")
    EVALUATION_DATA = None

# Load ground truth data (if exists)
GROUND_TRUTH_DATA = {}
if os.path.exists(GROUND_TRUTH_FILE):
    try:
        gt_df = pd.read_csv(GROUND_TRUTH_FILE)
        GROUND_TRUTH_DATA['ground_truth.csv'] = gt_df
        print(f"\n? Loaded ground truth: {len(gt_df)} rows, {len(gt_df.columns)} columns")
        print(f"   Columns: {', '.join(gt_df.columns)}")
        print("   ?? ALL columns will be accessible for evaluation!")
        
        # Show sample
        print("\n?? Ground truth sample (first 3 rows):")
        display(gt_df.head(3))
    except Exception as e:
        print(f"??  Warning: Error loading ground truth: {e}")
else:
    print(f"\n??  Ground truth file not found: {GROUND_TRUTH_FILE}")
    print("   This is optional - only needed for metrics that use ground truth")

print("\n" + "=" * 80)
print(f"?? DATA SUMMARY:")
print(f"   Evaluation samples: {len(EVALUATION_DATA) if EVALUATION_DATA is not None else 0}")
print(f"   Ground truth files: {len(GROUND_TRUTH_DATA)}")
print(f"   Metrics configured: {len(CURRENT_METRICS)}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Configure LLM Judge Model
# MAGIC Configure which LLM to use as the judge

# COMMAND ----------

from openai import OpenAI

# Create model selection widget
dbutils.widgets.dropdown("judge_model", "gpt-4", 
                        ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                        "Judge Model")

judge_model = dbutils.widgets.get("judge_model")

print("=" * 80)
print("?? LLM JUDGE CONFIGURATION")
print("=" * 80)
print(f"\nSelected Model: {judge_model}")

# Initialize OpenAI client
try:
    # Try to get API key from Databricks secrets
    api_key = dbutils.secrets.get(scope="popin-secure-scope", key="openai_key")
    print("? Retrieved API key from Databricks secrets")
except:
    print("??  Could not retrieve API key from secrets")
    print("   Please ensure your OpenAI API key is stored in Databricks secrets:")
    print("   Scope: popin-secure-scope")
    print("   Key: openai_key")
    api_key = None

if api_key:
    client = OpenAI(api_key=api_key)
    print("? OpenAI client initialized")
else:
    client = None
    print("? OpenAI client not initialized - evaluation will fail")

print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Core Evaluation Classes
# MAGIC Define the data structures for metrics

# COMMAND ----------

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

class MetricType(Enum):
    """Types of metrics supported."""
    BINARY = "binary"
    SCALE_1_5 = "1-5_scale"
    PERCENTAGE = "percentage"

@dataclass
class MetricConfig:
    """Configuration for a single metric."""
    name: str
    description: str
    metric_type: MetricType
    prompt_template: str
    threshold: float
    ground_truth_column: str
    ground_truth_file_path: str = ""

print("? Core evaluation classes defined")
print("   ? MetricType enum")
print("   ? MetricConfig dataclass")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: LLM Judge Evaluator Engine
# MAGIC Main evaluation logic

# COMMAND ----------

import json
import re
from typing import List, Dict, Tuple
import pandas as pd

class LLMJudgeEvaluator:
    """Main LLM Judge Evaluator class."""
    
    def __init__(self, client, model: str, metrics: List[MetricConfig], 
                 ground_truth_data: Dict[str, pd.DataFrame]):
        self.client = client
        self.model = model
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data
    
    def _get_ground_truth(self, metric: MetricConfig, sample_idx: int) -> str:
        """Get ground truth data with ALL columns for a sample."""
        try:
            filename = os.path.basename(metric.ground_truth_file_path) if metric.ground_truth_file_path else None
            
            if not filename or filename not in self.ground_truth_data:
                return "Not provided"
            
            df = self.ground_truth_data[filename]
            
            if sample_idx >= len(df):
                return f"Index {sample_idx} out of range"
            
            # Get ALL columns for this sample
            row = df.iloc[sample_idx]
            all_data = []
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    all_data.append(f"? {col}: {value}")
            
            return "?? Ground Truth:\n" + "\n".join(all_data) if all_data else "No data available"
            
        except Exception as e:
            return f"Error retrieving ground truth: {e}"
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM judge."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Provide responses in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f'{{"score": 0, "explanation": "Error calling LLM: {str(e)}"}}'
    
    def _parse_response(self, response: str, metric: MetricConfig) -> Tuple[float, str]:
        """Parse LLM response and extract score."""
        content = response.strip()
        
        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Try to parse as JSON
        try:
            data = json.loads(content)
            score = data.get('score', data.get('Score', 0))
            explanation = data.get('explanation', data.get('Explanation', 'No explanation'))
        except json.JSONDecodeError:
            # Fallback: try to extract score with regex
            score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', content, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 0
            explanation = content[:500]
        
        # Normalize score based on metric type
        try:
            score = float(score)
            if metric.metric_type == MetricType.BINARY:
                score = 1.0 if score > 0.5 else 0.0
            elif metric.metric_type == MetricType.SCALE_1_5:
                score = max(1.0, min(5.0, score))
            elif metric.metric_type == MetricType.PERCENTAGE:
                if score <= 1.0:
                    score = score * 100
                score = max(0.0, min(100.0, score))
        except:
            score = 0.0
        
        return score, str(explanation)[:500]
    
    def evaluate_single(self, prompt: str, response: str, metric: MetricConfig, 
                       sample_idx: int) -> dict:
        """Evaluate a single sample against a metric."""
        try:
            # Get ground truth
            ground_truth = self._get_ground_truth(metric, sample_idx)
            
            # Build evaluation prompt
            eval_prompt = metric.prompt_template.format(
                prompt=prompt,
                response=response,
                ground_truth=ground_truth
            )
            
            # Call LLM
            llm_response = self._call_llm(eval_prompt)
            
            # Parse response
            score, explanation = self._parse_response(llm_response, metric)
            
            # Determine pass/fail
            status = "?" if score >= metric.threshold else "?"
            
            return {
                "score": score,
                "explanation": explanation,
                "status": status,
                "ground_truth_used": ground_truth != "Not provided"
            }
        except Exception as e:
            return {
                "score": 0,
                "explanation": f"Error: {str(e)}",
                "status": "?",
                "ground_truth_used": False
            }
    
    def evaluate_dataset(self, eval_data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate entire dataset."""
        results = []
        total = len(eval_data) * len(self.metrics)
        current = 0
        
        print(f"?? Evaluating {len(eval_data)} samples with {len(self.metrics)} metrics...")
        print(f"   Total evaluations: {total}")
        print("\n" + "=" * 80)
        
        for idx, row in eval_data.iterrows():
            sample_id = row.get('sample_id', f'sample_{idx}')
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"\n?? Sample {idx + 1}/{len(eval_data)}: {sample_id}")
            
            for metric in self.metrics:
                current += 1
                progress = (current / total) * 100
                print(f"   [{progress:5.1f}%] {metric.name}...", end=' ')
                
                result = self.evaluate_single(prompt, response, metric, idx)
                print(f"{result['status']} (score: {result['score']:.2f})")
                
                results.append({
                    'sample_id': sample_id,
                    'metric_name': metric.name,
                    'metric_type': metric.metric_type.value,
                    'score': result['score'],
                    'threshold': metric.threshold,
                    'status': result['status'],
                    'explanation': result['explanation'],
                    'ground_truth_used': result['ground_truth_used'],
                    'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    'response': response[:200] + "..." if len(response) > 200 else response
                })
        
        print("\n" + "=" * 80)
        print("? Evaluation complete!")
        
        return pd.DataFrame(results)

print("? LLM Judge Evaluator Engine defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: Run Evaluation
# MAGIC Execute the complete evaluation workflow

# COMMAND ----------

import pandas as pd

print("=" * 80)
print("?? STARTING EVALUATION")
print("=" * 80)

# Helper function for type conversion
def safe_float(value, default=0.0):
    """Safely convert to float."""
    if pd.isna(value):
        return default
    try:
        val_str = str(value).strip().lower()
        if val_str in ['true', '==true', 'yes']:
            return 1.0
        if val_str in ['false', '==false', 'no']:
            return 0.0
        if val_str.endswith('%'):
            return float(val_str[:-1])
        return float(val_str)
    except:
        return default

# Helper function to generate prompts
def generate_prompt_from_rubric(name: str, description: str, rubric: str) -> str:
    """Generate evaluation prompt from rubric."""
    return f"""You are an expert evaluator. Task: {description}

Grading Rubric:
{rubric}

Evaluation Details:
- User Query: {{prompt}}
- AI Response: {{response}}
- Ground Truth: {{ground_truth}}

Provide your evaluation in the following JSON format ONLY (no other text):
{{
  "score": <your_numeric_score>,
  "explanation": "Brief explanation of your evaluation"
}}"""

# Step 1: Load and configure metrics
print("\n?? Step 1: Loading metrics configuration...")
metrics_df = pd.read_csv(METRICS_FILE)
print(f"   Loaded {len(metrics_df)} metrics")

metric_configs = []
for _, row in metrics_df.iterrows():
    name = str(row['name']).strip()
    type_str = str(row['type']).strip().lower()
    
    # Determine metric type
    if type_str in ['binary', 'bool']:
        mtype = MetricType.BINARY
    elif type_str in ['1-5_scale', 'scale']:
        mtype = MetricType.SCALE_1_5
    else:
        mtype = MetricType.PERCENTAGE
    
    # Get configuration
    description = str(row.get('description', '')).strip()
    rubric = str(row.get('grading_rubric', '')).strip()
    threshold = safe_float(row.get('threshold', 0.5))
    
    # Generate prompt
    prompt_template = generate_prompt_from_rubric(name, description, rubric)
    
    # Create metric config
    metric_config = MetricConfig(
        name=name,
        metric_type=mtype,
        description=description,
        prompt_template=prompt_template,
        threshold=threshold,
        ground_truth_column=str(row.get('ground_truth_column', '')).strip(),
        ground_truth_file_path=str(row.get('ground_truth_file_path', '')).strip()
    )
    
    metric_configs.append(metric_config)
    print(f"   ? Configured: {name} ({mtype.value}, threshold: {threshold})")

# Step 2: Initialize evaluator
print("\n?? Step 2: Initializing LLM Judge Evaluator...")
if client is None:
    print("? Error: OpenAI client not initialized. Please configure API key in Cell 8.")
else:
    evaluator = LLMJudgeEvaluator(
        client=client,
        model=judge_model,
        metrics=metric_configs,
        ground_truth_data=GROUND_TRUTH_DATA
    )
    print(f"   ? Evaluator initialized with {judge_model}")

# Step 3: Run evaluation
print("\n?? Step 3: Running evaluation...")
if EVALUATION_DATA is None:
    print("? Error: No evaluation data loaded. Please run Cell 7 first.")
elif client is None:
    print("? Error: OpenAI client not initialized.")
else:
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    
    # Step 4: Display results
    print("\n" + "=" * 80)
    print("?? EVALUATION RESULTS")
    print("=" * 80)
    
    # Summary statistics
    total_evals = len(results_df)
    passed = len(results_df[results_df['status'] == '?'])
    failed = len(results_df[results_df['status'] == '?'])
    pass_rate = (passed / total_evals * 100) if total_evals > 0 else 0
    
    print(f"\n?? OVERALL SUMMARY:")
    print(f"   Total evaluations: {total_evals}")
    print(f"   Passed: {passed} ({pass_rate:.1f}%)")
    print(f"   Failed: {failed} ({100-pass_rate:.1f}%)")
    
    # Per-metric statistics
    print(f"\n?? PER-METRIC RESULTS:")
    for metric_name in results_df['metric_name'].unique():
        metric_results = results_df[results_df['metric_name'] == metric_name]
        metric_passed = len(metric_results[metric_results['status'] == '?'])
        metric_total = len(metric_results)
        metric_rate = (metric_passed / metric_total * 100) if metric_total > 0 else 0
        avg_score = metric_results['score'].mean()
        
        print(f"   ? {metric_name}: {metric_rate:.1f}% pass ({metric_passed}/{metric_total}), avg score: {avg_score:.2f}")
    
    # Display results table
    print("\n?? DETAILED RESULTS:")
    display(results_df)
    
    # Step 5: Save results
    print("\n?? Step 5: Saving results...")
    results_file = os.path.join(BASE_DIR, "evaluation_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"   ? Results saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("?? EVALUATION COMPLETE!")
    print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? Congratulations!
# MAGIC 
# MAGIC Your evaluation is complete! 
# MAGIC 
# MAGIC **Next steps:**
# MAGIC - Review the results above
# MAGIC - Check the saved CSV file for detailed results
# MAGIC - Modify metrics in Cells 5-6 if needed
# MAGIC - Re-run evaluation with different data
# MAGIC 
# MAGIC **Tips:**
# MAGIC - Use Cell 5-6 to add/edit/delete metrics without coding
# MAGIC - Ground truth files support ALL columns (not just one!)
# MAGIC - Results are automatically saved to CSV
# MAGIC - You can re-run any cell individually
