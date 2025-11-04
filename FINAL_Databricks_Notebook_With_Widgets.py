# Databricks notebook source
# MAGIC %md
# MAGIC # ?? LLM-as-a-Judge: Interactive Evaluation System
# MAGIC # **With Interactive Widgets for Metrics & Evaluations**
# MAGIC 
# MAGIC **Features:**
# MAGIC - ? **Interactive widgets to add/edit/delete metrics**
# MAGIC - ? **Widget to select judge model** (databricks-llm or OpenAI)
# MAGIC - ? **Databricks LLM = Claude Sonnet via Serving Endpoints**
# MAGIC - ? Enhanced ground truth with ALL columns
# MAGIC - ? Robust JSON parsing and error handling
# MAGIC - ? File upload support for evaluation data
# MAGIC 
# MAGIC **Instructions:**
# MAGIC 1. Run Cell 1 (Install packages)
# MAGIC 2. Run Cell 2 (Upload evaluation data & ground truth)
# MAGIC 3. Run Cell 3 (Interactive Metrics Editor - Add/Edit/Delete metrics)
# MAGIC 4. Run Cell 4 (Configure LLM Judge)
# MAGIC 5. Run Cell 5-6 (Setup & Run Evaluation)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Installation and Setup

# COMMAND ----------

%pip install openai pandas requests --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: File Upload & Data Loading
# MAGIC 
# MAGIC **Upload your files:**
# MAGIC - Evaluation data CSV (prompt, response columns)
# MAGIC - Ground truth CSV files (optional)
# MAGIC - Metrics configuration CSV (optional - or use interactive editor in Cell 3)

# COMMAND ----------

import pandas as pd
import os

# Create widgets for file paths
dbutils.widgets.text("eval_data_path", "", "?? Evaluation Data CSV")
dbutils.widgets.text("ground_truth_paths", "", "?? Ground Truth CSV (semicolon separated)")
dbutils.widgets.text("metrics_csv_path", "", "?? Metrics CSV (optional)")

# Get paths
EVAL_DATA_PATH = dbutils.widgets.get("eval_data_path")
GT_PATHS = dbutils.widgets.get("ground_truth_paths")
METRICS_CSV_PATH = dbutils.widgets.get("metrics_csv_path")

print("="*80)
print("?? FILE LOADING")
print("="*80)

# Load evaluation data
EVALUATION_DATA = None
if EVAL_DATA_PATH and os.path.exists(EVAL_DATA_PATH):
    try:
        EVALUATION_DATA = pd.read_csv(EVAL_DATA_PATH)
        print(f"? Loaded evaluation data: {len(EVALUATION_DATA)} rows")
        display(EVALUATION_DATA.head(3))
    except Exception as e:
        print(f"? Error loading evaluation data: {e}")
else:
    print("??  No evaluation data loaded - provide path in widget above")

# Load ground truth files
ground_truth_data = {}
if GT_PATHS:
    for gt_path in GT_PATHS.split(';'):
        gt_path = gt_path.strip()
        if gt_path and os.path.exists(gt_path):
            try:
                filename = os.path.basename(gt_path)
                df = pd.read_csv(gt_path)
                ground_truth_data[filename] = df
                print(f"? Loaded ground truth: {filename} ({len(df)} rows, {len(df.columns)} columns)")
            except Exception as e:
                print(f"? Error loading {gt_path}: {e}")

# Load metrics CSV if provided
METRICS_CONFIG_DATA = None
if METRICS_CSV_PATH and os.path.exists(METRICS_CSV_PATH):
    try:
        METRICS_CONFIG_DATA = pd.read_csv(METRICS_CSV_PATH)
        print(f"? Loaded metrics CSV: {len(METRICS_CONFIG_DATA)} metrics")
    except Exception as e:
        print(f"? Error loading metrics CSV: {e}")
else:
    # Create empty metrics DataFrame
    METRICS_CONFIG_DATA = pd.DataFrame(columns=[
        'name', 'type', 'description', 'grading_rubric', 'threshold',
        'ground_truth_file_path', 'ground_truth_column'
    ])
    print("?? No metrics CSV - use interactive editor in Cell 3")

print("\n? Data loading complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Interactive Metrics Editor
# MAGIC 
# MAGIC **Use widgets below to manage metrics:**
# MAGIC - Add new metrics
# MAGIC - Edit existing metrics
# MAGIC - Delete metrics
# MAGIC - Save to CSV

# COMMAND ----------

import pandas as pd

# Initialize metrics DataFrame if not exists
if 'METRICS_CONFIG_DATA' not in globals() or METRICS_CONFIG_DATA is None:
    METRICS_CONFIG_DATA = pd.DataFrame(columns=[
        'name', 'type', 'description', 'grading_rubric', 'threshold',
        'ground_truth_file_path', 'ground_truth_column'
    ])

print("="*80)
print("?? INTERACTIVE METRICS EDITOR")
print("="*80)

# Display current metrics
if len(METRICS_CONFIG_DATA) > 0:
    print(f"\n?? Current Metrics ({len(METRICS_CONFIG_DATA)}):")
    display(METRICS_CONFIG_DATA[['name', 'type', 'threshold']])
else:
    print("\n?? No metrics defined yet")

# Create widgets for adding/editing metrics
dbutils.widgets.text("metric_name", "", "Metric Name")
dbutils.widgets.dropdown("metric_type", "binary", ["binary", "1-5_scale", "percentage"], "Metric Type")
dbutils.widgets.text("metric_description", "", "Description")
dbutils.widgets.text("metric_rubric", "", "Grading Rubric")
dbutils.widgets.text("metric_threshold", "1", "Threshold")
dbutils.widgets.text("metric_gt_file", "", "Ground Truth File (optional)")
dbutils.widgets.text("metric_gt_column", "", "GT Column (optional)")

# Action buttons
dbutils.widgets.dropdown("action", "none", ["none", "add", "delete", "save"], "Action")

action = dbutils.widgets.get("action")

if action == "add":
    # Add new metric
    new_metric = {
        'name': dbutils.widgets.get("metric_name"),
        'type': dbutils.widgets.get("metric_type"),
        'description': dbutils.widgets.get("metric_description"),
        'grading_rubric': dbutils.widgets.get("metric_rubric"),
        'threshold': dbutils.widgets.get("metric_threshold"),
        'ground_truth_file_path': dbutils.widgets.get("metric_gt_file"),
        'ground_truth_column': dbutils.widgets.get("metric_gt_column")
    }
    
    if new_metric['name']:
        METRICS_CONFIG_DATA = pd.concat([METRICS_CONFIG_DATA, pd.DataFrame([new_metric])], ignore_index=True)
        print(f"\n? Added metric: {new_metric['name']}")
        display(METRICS_CONFIG_DATA)
    else:
        print("\n? Metric name is required")

elif action == "delete":
    # Delete metric by name
    name_to_delete = dbutils.widgets.get("metric_name")
    if name_to_delete:
        METRICS_CONFIG_DATA = METRICS_CONFIG_DATA[METRICS_CONFIG_DATA['name'] != name_to_delete]
        print(f"\n? Deleted metric: {name_to_delete}")
        display(METRICS_CONFIG_DATA)
    else:
        print("\n? Provide metric name to delete")

elif action == "save":
    # Save to CSV
    save_path = "/tmp/metrics_config.csv"
    METRICS_CONFIG_DATA.to_csv(save_path, index=False)
    print(f"\n? Saved metrics to: {save_path}")
    print(f"?? {len(METRICS_CONFIG_DATA)} metrics saved")

print("\n?? Instructions:")
print("   1. Fill in the widgets above with metric details")
print("   2. Select 'add' from Action dropdown")
print("   3. Run this cell again to add the metric")
print("   4. Repeat for more metrics")
print("   5. Select 'save' to save to CSV")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Configure LLM Judge Model
# MAGIC 
# MAGIC **Select your judge model from dropdown**

# COMMAND ----------

from openai import OpenAI
import os
import requests

# Widget for model selection
dbutils.widgets.dropdown(
    "judge_model",
    "databricks-llm",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "databricks-llm"],
    "?? Judge Model"
)

JUDGE_MODEL = dbutils.widgets.get("judge_model")

print("?? MODEL CONFIGURATION")
print("="*60)
print(f"Selected Model: {JUDGE_MODEL}")
print("="*60)

client = None
client_type = None

if JUDGE_MODEL == "databricks-llm":
    print("\n?? Databricks LLM (Claude Sonnet via Serving Endpoints)")
    
    try:
        # Get workspace credentials
        databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
        
        headers = {
            "Authorization": f"Bearer {databricks_token}",
            "Content-Type": "application/json"
        }
        
        # Query serving endpoints
        print("   ?? Discovering serving endpoints...")
        url = f"https://{workspace_url}/api/2.0/serving-endpoints"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            endpoints = response.json().get('endpoints', [])
            print(f"   ? Found {len(endpoints)} endpoints")
            
            # Find Claude Sonnet
            databricks_endpoint = None
            for ep in endpoints:
                if 'claude-sonnet' in ep['name'].lower():
                    databricks_endpoint = ep['name']
                    print(f"   ? Using: {databricks_endpoint}")
                    break
            
            if not databricks_endpoint and endpoints:
                databricks_endpoint = endpoints[0]['name']
                print(f"   ??  No Claude Sonnet, using: {databricks_endpoint}")
            
            if databricks_endpoint:
                client = {
                    'type': 'databricks',
                    'workspace_url': workspace_url,
                    'token': databricks_token,
                    'headers': headers,
                    'endpoint': databricks_endpoint
                }
                client_type = "databricks"
                print(f"   ? Ready!")
            else:
                print("   ? No endpoints available")
        else:
            print(f"   ? Failed to query endpoints: {response.status_code}")
            
    except Exception as e:
        print(f"   ? Error: {e}")

else:
    # OpenAI models
    print("\n?? OpenAI via Zillow Labs Proxy")
    
    try:
        OPENAI_KEY = dbutils.secrets.get("popin-secure-scope", "openai_key")
        client = OpenAI(
            base_url="https://api.zillowlabs.com/openai/v1",
            api_key=OPENAI_KEY
        )
        client_type = "openai"
        print("   ? OpenAI client initialized")
    except Exception as e:
        print(f"   ? Error: {e}")

print("\n" + "="*60)
if client:
    print("? READY TO EVALUATE!")
else:
    print("? CLIENT NOT INITIALIZED")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Create LLM Judge Evaluator (Enhanced)

# COMMAND ----------

import json
import re
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
import time

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

class LLMJudgeEvaluator:
    """Enhanced LLM Judge Evaluator with robust JSON parsing"""
    
    def __init__(self, judge_model: str, metrics: List[MetricConfig], ground_truth_data: Dict[str, pd.DataFrame]):
        self.judge_model = judge_model
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data or {}
        self.is_databricks_llm = judge_model == "databricks-llm"
        
        if self.is_databricks_llm:
            self._init_databricks_llm()
        else:
            self._init_openai_llm()
    
    def _init_databricks_llm(self):
        """Initialize Databricks serving endpoint"""
        if client and client.get('type') == 'databricks':
            self.databricks_config = client
            print(f"? Databricks evaluator initialized: {client['endpoint']}")
        else:
            raise ValueError("Databricks client not configured")
    
    def _init_openai_llm(self):
        """Initialize OpenAI client"""
        if client and client_type == "openai":
            self.llm_client = client
            print(f"? OpenAI evaluator initialized: {self.judge_model}")
        else:
            raise ValueError("OpenAI client not configured")
    
    def _get_ground_truth(self, metric: MetricConfig, sample_idx: int) -> str:
        """Get ALL columns from ground truth"""
        try:
            if not metric.ground_truth_file_path:
                return "Not provided"
            
            filename = os.path.basename(metric.ground_truth_file_path)
            if filename not in self.ground_truth_data:
                return "Not provided"
            
            df = self.ground_truth_data[filename]
            if sample_idx >= len(df):
                return f"Index {sample_idx} out of range"
            
            # Get ALL columns
            row = df.iloc[sample_idx]
            all_data = []
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    all_data.append(f"? {col}: {value}")
            
            return "?? Ground Truth:\n" + "\n".join(all_data) if all_data else "No data"
        except Exception as e:
            return f"Error: {e}"
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with proper error handling"""
        try:
            if self.is_databricks_llm:
                # Call Databricks serving endpoint
                url = f"https://{self.databricks_config['workspace_url']}/serving-endpoints/{self.databricks_config['endpoint']}/invocations"
                
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.1
                }
                
                response = requests.post(
                    url, 
                    headers=self.databricks_config['headers'], 
                    json=payload, 
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        return content
                    else:
                        return '{"score": 0, "explanation": "Empty response from endpoint"}'
                else:
                    return f'{{"score": 0, "explanation": "HTTP {response.status_code}: {response.text[:100]}"}}'
            else:
                # Call OpenAI
                response = self.llm_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Provide responses in JSON format only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                return response.choices[0].message.content
                
        except Exception as e:
            return f'{{"score": 0, "explanation": "LLM call error: {str(e)[:200]}"}}'
    
    def _parse_response(self, response: str, metric: MetricConfig) -> tuple:
        """Robust JSON parsing with multiple fallbacks"""
        content = response.strip()
        
        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Remove leading/trailing whitespace and newlines
        content = content.strip()
        
        # Try direct JSON parse
        try:
            data = json.loads(content)
            score = data.get('score', data.get('Score', 0))
            explanation = data.get('explanation', data.get('Explanation', 'No explanation'))
        except json.JSONDecodeError as e:
            # Fallback: try to extract with regex
            try:
                score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', content, re.IGNORECASE)
                exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', content, re.IGNORECASE | re.DOTALL)
                
                score = float(score_match.group(1)) if score_match else 0
                explanation = exp_match.group(1) if exp_match else f"Parse error: {str(e)[:100]}"
            except Exception as e2:
                score = 0
                explanation = f"Failed to parse: {content[:200]}"
        
        # Normalize score
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
    
    def evaluate_single(self, prompt: str, response: str, metric: MetricConfig, sample_idx: int) -> dict:
        """Evaluate single sample"""
        try:
            ground_truth = self._get_ground_truth(metric, sample_idx)
            
            eval_prompt = metric.prompt_template.format(
                prompt=prompt,
                response=response,
                ground_truth=ground_truth
            )
            
            llm_response = self._call_llm(eval_prompt)
            score, explanation = self._parse_response(llm_response, metric)
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
                "explanation": f"Evaluation error: {str(e)[:200]}",
                "status": "?",
                "ground_truth_used": False
            }
    
    def evaluate_dataset(self, eval_data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate entire dataset"""
        results = []
        total = len(eval_data) * len(self.metrics)
        current = 0
        
        print(f"?? Starting evaluation: {len(eval_data)} samples ? {len(self.metrics)} metrics = {total} evaluations")
        print("="*80)
        
        for idx, row in eval_data.iterrows():
            sample_id = row.get('sample_id', idx)
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"\n?? Sample {idx+1}/{len(eval_data)}: ID {sample_id}")
            
            for metric in self.metrics:
                current += 1
                progress = (current / total) * 100
                print(f"   [{progress:5.1f}%] {metric.name}...", end=' ', flush=True)
                
                result = self.evaluate_single(prompt, response, metric, idx)
                print(f"{result['status']} ({result['score']:.2f})")
                
                results.append({
                    'sample_id': sample_id,
                    'metric_name': metric.name,
                    'metric_type': metric.metric_type.value,
                    'score': result['score'],
                    'threshold': metric.threshold,
                    'status': result['status'],
                    'explanation': result['explanation'],
                    'ground_truth_used': result['ground_truth_used']
                })
        
        print("\n" + "="*80)
        print("? Evaluation complete!")
        return pd.DataFrame(results)

print("? LLM Judge Evaluator class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: RUN EVALUATION! ??

# COMMAND ----------

print("="*80)
print("?? STARTING LLM EVALUATION")
print("="*80)

# Helper functions
def safe_float(value, default=0.0):
    if pd.isna(value):
        return default
    try:
        val_str = str(value).strip().lower()
        if val_str in ['true', 'yes', '1']:
            return 1.0
        if val_str in ['false', 'no', '0']:
            return 0.0
        if val_str.endswith('%'):
            return float(val_str[:-1])
        return float(val_str)
    except:
        return default

def generate_prompt(name, description, rubric):
    return f"""You are an expert evaluator. Task: {description}

Grading Rubric:
{rubric}

Evaluation Details:
- User Query: {{prompt}}
- AI Response: {{response}}
- Ground Truth: {{ground_truth}}

Provide evaluation in JSON format ONLY:
{{
  "score": <numeric_score>,
  "explanation": "Brief explanation"
}}"""

# Configure metrics
print("\n?? Configuring metrics...")
metric_configs = []

for _, row in METRICS_CONFIG_DATA.iterrows():
    name = str(row['name']).strip()
    type_str = str(row['type']).strip().lower()
    
    if type_str in ['binary', 'bool']:
        mtype = MetricType.BINARY
    elif type_str in ['1-5_scale', 'scale']:
        mtype = MetricType.SCALE_1_5
    else:
        mtype = MetricType.PERCENTAGE
    
    description = str(row.get('description', '')).strip()
    rubric = str(row.get('grading_rubric', '')).strip()
    threshold = safe_float(row.get('threshold', 0.5))
    
    prompt_template = generate_prompt(name, description, rubric)
    
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
    print(f"   ? {name} ({mtype.value}, threshold: {threshold})")

# Initialize evaluator
print("\n?? Initializing evaluator...")
if client is None:
    print("? Error: LLM client not initialized")
elif EVALUATION_DATA is None:
    print("? Error: No evaluation data loaded")
elif len(metric_configs) == 0:
    print("? Error: No metrics configured")
else:
    evaluator = LLMJudgeEvaluator(
        judge_model=JUDGE_MODEL,
        metrics=metric_configs,
        ground_truth_data=ground_truth_data
    )
    
    # Run evaluation
    print("\n?? Running evaluation...")
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    # Display results
    print("\n" + "="*80)
    print("?? EVALUATION RESULTS")
    print("="*80)
    
    total = len(results_df)
    passed = len(results_df[results_df['status'] == '?'])
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n?? OVERALL:")
    print(f"   Total: {total}")
    print(f"   Passed: {passed} ({pass_rate:.1f}%)")
    print(f"   Failed: {total - passed}")
    print(f"   Time: {eval_time:.1f}s")
    
    print(f"\n?? PER-METRIC:")
    for metric_name in results_df['metric_name'].unique():
        metric_results = results_df[results_df['metric_name'] == metric_name]
        metric_passed = len(metric_results[metric_results['status'] == '?'])
        metric_total = len(metric_results)
        metric_rate = (metric_passed / metric_total * 100) if metric_total > 0 else 0
        avg_score = metric_results['score'].mean()
        
        print(f"   ? {metric_name}: {metric_rate:.1f}% ({metric_passed}/{metric_total}), avg: {avg_score:.2f}")
    
    print("\n?? DETAILED RESULTS:")
    display(results_df[['sample_id', 'metric_name', 'score', 'threshold', 'status', 'explanation']])
    
    print("\n" + "="*80)
    print("?? EVALUATION COMPLETE!")
    print("="*80)
