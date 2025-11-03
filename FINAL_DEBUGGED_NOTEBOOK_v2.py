# Databricks notebook source
# MAGIC %md
# MAGIC # LLM-as-a-Judge: DEBUGGED VERSION v2
# MAGIC 
# MAGIC **This version has extensive debugging to show exactly what's happening**
# MAGIC 
# MAGIC Changes from previous:
# MAGIC - Added DEBUG output at every step
# MAGIC - Shows LLM calls in real-time
# MAGIC - Shows errors with full tracebacks
# MAGIC - Uses flush=True for immediate output
# MAGIC - Simple PASS/FAIL (no emojis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Install Packages

# COMMAND ----------

%pip install openai pandas --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Load Cinderella Data

# COMMAND ----------

import pandas as pd

print("="*80)
print("CELL 2: LOADING DATA")
print("="*80)

# Metrics Configuration
METRICS_CONFIG_JSON = [
    {
        "name": "Story_Accuracy",
        "type": "binary",
        "description": "Evaluate if the response is factually accurate about the Cinderella story",
        "grading_rubric": "Score 1 if all story facts are correct and align with the classic Cinderella tale. Score 0 if any facts are incorrect, made up, or contradict the original story.",
        "threshold": "1",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    },
    {
        "name": "Response_Completeness",
        "type": "1-5_scale",
        "description": "Evaluate how complete and thorough the response is",
        "grading_rubric": "5=Fully complete, addresses all aspects comprehensively; 4=Mostly complete with minor gaps; 3=Partially complete, missing some details; 2=Barely complete, many gaps; 1=Incomplete or inadequate",
        "threshold": "4",
        "ground_truth_file_path": "",
        "ground_truth_column": ""
    },
    {
        "name": "Child_Friendliness",
        "type": "percentage",
        "description": "Evaluate what percentage of the response is appropriate and understandable for children",
        "grading_rubric": "100%=Perfectly child-friendly language and content; 75%=Mostly appropriate with minor complex words; 50%=Somewhat child-friendly; 25%=Barely appropriate for children; 0%=Not suitable for children",
        "threshold": "75",
        "ground_truth_file_path": "",
        "ground_truth_column": ""
    }
]

# Evaluation Data
EVALUATION_DATA_JSON = [
    {"sample_id": 1, "prompt": "Who is Cinderella?", "response": "Cinderella is a young girl who lives with her mean stepmother and two stepsisters. With help from her Fairy Godmother, she goes to the prince's ball and loses her glass slipper at midnight."},
    {"sample_id": 2, "prompt": "What did the Fairy Godmother turn into a carriage?", "response": "The Fairy Godmother used her magic wand to turn a big orange pumpkin into a beautiful golden carriage."},
    {"sample_id": 3, "prompt": "What happened at midnight?", "response": "When the clock struck midnight, Cinderella had to run away from the ball. In her hurry, she lost one of her glass slippers on the palace steps."}
]

# Ground Truth
GROUND_TRUTH_JSON = [
    {"sample_id": 1, "correct_answer": "Cinderella is a kind young girl mistreated by her stepmother and stepsisters. With help from her Fairy Godmother, she attends a royal ball, loses her glass slipper at midnight, and is found by the prince.", "story_element": "Main plot"},
    {"sample_id": 2, "correct_answer": "A pumpkin", "story_element": "Magic transformation"},
    {"sample_id": 3, "correct_answer": "Cinderella had to leave the ball because the magic spell would break at midnight. She ran away and lost her glass slipper.", "story_element": "Midnight deadline"}
]

# Convert to DataFrames
METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
EVALUATION_DATA = pd.DataFrame(EVALUATION_DATA_JSON)
GROUND_TRUTH_DATA_DF = pd.DataFrame(GROUND_TRUTH_JSON)
GROUND_TRUTH_DATA = {'ground_truth.csv': GROUND_TRUTH_DATA_DF}

print(f"\nData loaded:")
print(f"  Metrics: {len(METRICS_CONFIG_DATA)}")
print(f"  Samples: {len(EVALUATION_DATA)}")
print(f"  Ground truth: {len(GROUND_TRUTH_DATA_DF)}")
print(f"\n{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Configure OpenAI Client

# COMMAND ----------

from openai import OpenAI
import os

print("="*80)
print("CELL 3: CONFIGURE LLM CLIENT")
print("="*80)

# Get API key
try:
    OPENAI_KEY = dbutils.secrets.get("popin-secure-scope", "openai_key")
    print("  API key retrieved from popin-secure-scope")
except Exception as e:
    print(f"  ERROR: Could not get API key: {e}")
    raise

# Initialize client
client = OpenAI(
    base_url="https://api.zillowlabs.com/openai/v1",
    api_key=OPENAI_KEY
)

JUDGE_MODEL = "gpt-4o-mini"
client_type = "openai"

print(f"  Client initialized")
print(f"  Base URL: https://api.zillowlabs.com/openai/v1")
print(f"  Model: {JUDGE_MODEL}")
print(f"  Client type: {client_type}")

# Test connection
try:
    print("\n  Testing connection...")
    test_response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5
    )
    print(f"  Connection test: SUCCESS")
except Exception as e:
    print(f"  Connection test: FAILED - {e}")
    raise

print(f"\n{'='*80}")
print("READY TO EVALUATE")
print(f"{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Define Classes

# COMMAND ----------

from enum import Enum
from dataclasses import dataclass

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

print("Classes defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Create Evaluator (WITH EXTENSIVE DEBUG)

# COMMAND ----------

import json
import re
import os
import pandas as pd

class LLMJudgeEvaluator:
    """LLM Judge Evaluator with extensive debugging."""
    
    def __init__(self, client, model, metrics, ground_truth_data, client_type):
        self.client = client
        self.model = model
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data
        self.client_type = client_type
        print(f"[INIT] Evaluator created: {client_type}, {len(metrics)} metrics")
    
    def _get_ground_truth(self, metric, sample_idx):
        """Get ground truth for a sample."""
        try:
            filename = os.path.basename(metric.ground_truth_file_path) if metric.ground_truth_file_path else None
            if not filename or filename not in self.ground_truth_data:
                return "Not provided"
            
            df = self.ground_truth_data[filename]
            if sample_idx >= len(df):
                return f"Index {sample_idx} out of range"
            
            row = df.iloc[sample_idx]
            all_data = []
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    all_data.append(f"- {col}: {value}")
            
            return "Ground Truth:\n" + "\n".join(all_data) if all_data else "No data"
        except Exception as e:
            print(f"[ERROR] Ground truth error: {e}")
            return f"Error: {e}"
    
    def _call_llm(self, prompt):
        """Call LLM with extensive debugging."""
        print(f"\n    [LLM CALL START]", flush=True)
        print(f"      Client type: {self.client_type}", flush=True)
        print(f"      Model: {self.model}", flush=True)
        print(f"      Prompt length: {len(prompt)} chars", flush=True)
        
        try:
            print(f"      Making API call...", flush=True)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond with ONLY valid JSON: {\"score\": <number>, \"explanation\": \"<text>\"}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            print(f"      API call SUCCESS!", flush=True)
            print(f"      Response length: {len(content)} chars", flush=True)
            print(f"      Response preview: {content[:100]}...", flush=True)
            print(f"    [LLM CALL END]", flush=True)
            
            return content
            
        except Exception as e:
            print(f"\n    [LLM CALL FAILED]", flush=True)
            print(f"      Error: {str(e)}", flush=True)
            import traceback
            print(f"      Traceback:", flush=True)
            print(traceback.format_exc(), flush=True)
            return f'{{"score": 0, "explanation": "API call failed: {str(e)}"}}'
    
    def _parse_response(self, response, metric):
        """Parse LLM response."""
        content = response.strip()
        
        # Remove markdown if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        try:
            data = json.loads(content)
            score = data.get('score', data.get('Score', 0))
            explanation = data.get('explanation', data.get('Explanation', 'No explanation'))
        except Exception as e:
            print(f"\n      [PARSE ERROR]", flush=True)
            print(f"        Error: {str(e)}", flush=True)
            print(f"        Content: {content[:200]}", flush=True)
            
            # Try regex fallback
            try:
                score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', content, re.IGNORECASE)
                score = float(score_match.group(1)) if score_match else 0
                explanation = f"Parse error (using regex): {str(e)[:100]}"
                print(f"        Regex fallback: score={score}", flush=True)
            except Exception as e2:
                score = 0
                explanation = f"Complete parse failure: {str(e)[:100]}"
                print(f"        Regex also failed: {e2}", flush=True)
        
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
        except Exception as e:
            print(f"      [NORMALIZE ERROR] {e}", flush=True)
            score = 0.0
        
        return score, str(explanation)[:500]
    
    def evaluate_single(self, prompt, response, metric, sample_idx):
        """Evaluate single sample."""
        try:
            # Get ground truth
            ground_truth = self._get_ground_truth(metric, sample_idx)
            
            # Format prompt
            eval_prompt = metric.prompt_template.format(
                prompt=prompt,
                response=response,
                ground_truth=ground_truth
            )
            
            # Call LLM
            llm_response = self._call_llm(eval_prompt)
            
            # Parse response
            score, explanation = self._parse_response(llm_response, metric)
            
            # Determine status
            status = "PASS" if score >= metric.threshold else "FAIL"
            
            return {
                "score": score,
                "explanation": explanation,
                "status": status,
                "ground_truth_used": ground_truth != "Not provided"
            }
        except Exception as e:
            print(f"\n    [CRITICAL ERROR in evaluate_single]", flush=True)
            print(f"      Metric: {metric.name}", flush=True)
            print(f"      Error: {str(e)}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            
            return {
                "score": 0,
                "explanation": f"Error: {str(e)}",
                "status": "FAIL",
                "ground_truth_used": False
            }
    
    def evaluate_dataset(self, eval_data):
        """Evaluate entire dataset."""
        results = []
        total = len(eval_data) * len(self.metrics)
        current = 0
        
        print(f"\n{'='*80}")
        print(f"STARTING EVALUATION")
        print(f"{'='*80}")
        print(f"Samples: {len(eval_data)}")
        print(f"Metrics: {len(self.metrics)}")
        print(f"Total evaluations: {total}")
        print(f"{'='*80}\n")
        
        for idx, row in eval_data.iterrows():
            sample_id = row.get('sample_id', idx)
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"Sample {idx+1}/{len(eval_data)}: ID {sample_id}")
            
            for metric in self.metrics:
                current += 1
                progress = (current / total) * 100
                print(f"  [{progress:5.1f}%] {metric.name}...", flush=True)
                
                result = self.evaluate_single(prompt, response, metric, idx)
                
                print(f"    Result: {result['status']} (score: {result['score']:.2f})", flush=True)
                
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
        
        print(f"\n{'='*80}")
        print("EVALUATION DATASET COMPLETE")
        print(f"{'='*80}")
        return pd.DataFrame(results)

print("Evaluator class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: RUN EVALUATION

# COMMAND ----------

import time

print("="*80)
print("CELL 6: RUN EVALUATION")
print("="*80)

# Check prerequisites
print("\nChecking prerequisites...")
print(f"  client is None: {client is None}")
if client is not None:
    print(f"  client_type: {client_type}")
    print(f"  model: {JUDGE_MODEL}")
print(f"  Samples: {len(EVALUATION_DATA)}")
print(f"  Metrics: {len(METRICS_CONFIG_DATA)}")

# Helper function
def safe_float(value, default=0.0):
    """Safely convert to float."""
    if pd.isna(value) or value is None:
        return default
    try:
        val_str = str(value).strip().lower()
        if val_str in ['true', '==true', 'yes', '1']:
            return 1.0
        if val_str in ['false', '==false', 'no', '0']:
            return 0.0
        if val_str.endswith('%'):
            return float(val_str[:-1])
        return float(val_str)
    except:
        return default

def generate_prompt(name, description, rubric):
    """Generate evaluation prompt."""
    return f"""You are an expert evaluator. Task: {description}

Grading Rubric:
{rubric}

Evaluation Details:
- User Query: {{{{prompt}}}}
- AI Response: {{{{response}}}}
- Ground Truth: {{{{ground_truth}}}}

IMPORTANT: Respond with ONLY valid JSON (no other text):
{{{{
  "score": <your_numeric_score>,
  "explanation": "Brief explanation"
}}}}"""

# Load metrics
print("\nLoading metrics...")
metric_configs = []

for idx, row in METRICS_CONFIG_DATA.iterrows():
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
    print(f"  Loaded: {name} ({mtype.value}, threshold={threshold})")

print(f"\nTotal metrics loaded: {len(metric_configs)}")

# Initialize evaluator
if client is None:
    print("\nERROR: Client not initialized!")
else:
    print("\nInitializing evaluator...")
    evaluator = LLMJudgeEvaluator(
        client=client,
        model=JUDGE_MODEL,
        metrics=metric_configs,
        ground_truth_data=GROUND_TRUTH_DATA,
        client_type=client_type
    )
    
    print(f"Evaluator ready")
    print(f"\n{'='*80}")
    print("ABOUT TO START EVALUATION")
    print(f"{'='*80}")
    print("Watch for [LLM CALL START] messages below...")
    print("If you don't see them, the LLM is not being called!")
    print(f"{'='*80}\n")
    
    # Force flush
    import sys
    sys.stdout.flush()
    
    # Run evaluation
    start_time = time.time()
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    eval_time = time.time() - start_time
    
    # Results
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE in {eval_time:.1f}s")
    print(f"{'='*80}")
    
    if eval_time < 5.0:
        print(f"\nWARNING: Evaluation was very fast ({eval_time:.1f}s)")
        print(f"Expected time for {len(results_df)} LLM calls: 20-60 seconds")
        print(f"This suggests LLM calls are failing silently!")
    
    # Analyze results
    total = len(results_df)
    passed = len(results_df[results_df['status'] == 'PASS'])
    failed = len(results_df[results_df['status'] == 'FAIL'])
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\nRESULTS SUMMARY:")
    print(f"  Total: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Pass rate: {pass_rate:.1f}%")
    print(f"  Time: {eval_time:.1f}s")
    
    # Check for all zeros
    if results_df['score'].sum() == 0:
        print(f"\nWARNING: All scores are 0!")
        print(f"This means:")
        print(f"  1. LLM calls are failing, OR")
        print(f"  2. LLM responses cannot be parsed, OR")
        print(f"  3. There's an error in evaluate_single")
        print(f"\nCheck the debug output above for [LLM CALL START] messages")
    
    # Show first few results
    print(f"\nFirst 3 results:")
    for i in range(min(3, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n  {i+1}. Sample {row['sample_id']} - {row['metric_name']}:")
        print(f"     Score: {row['score']:.2f}")
        print(f"     Status: {row['status']}")
        print(f"     Explanation: {row['explanation'][:150]}")
    
    # Per-metric
    print(f"\nPer-metric results:")
    for metric_name in results_df['metric_name'].unique():
        metric_results = results_df[results_df['metric_name'] == metric_name]
        m_passed = len(metric_results[metric_results['status'] == 'PASS'])
        m_total = len(metric_results)
        m_rate = (m_passed / m_total * 100) if m_total > 0 else 0
        m_avg = metric_results['score'].mean()
        
        print(f"  {metric_name}: {m_rate:.1f}% pass ({m_passed}/{m_total}), avg score: {m_avg:.2f}")
    
    # Display full results
    print(f"\nDetailed results table:")
    display(results_df[['sample_id', 'metric_name', 'score', 'threshold', 'status', 'explanation']])
    
    print(f"\n{'='*80}")
    print("END OF CELL 6")
    print(f"{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Debug Guide
# MAGIC 
# MAGIC **What to look for in Cell 6 output:**
# MAGIC 
# MAGIC 1. **[LLM CALL START]** - Should appear for each evaluation (9 times for 3 samples x 3 metrics)
# MAGIC    - If missing: LLM is NOT being called
# MAGIC 
# MAGIC 2. **API call SUCCESS!** - Should appear after each LLM call
# MAGIC    - If missing: API calls are failing
# MAGIC 
# MAGIC 3. **Evaluation time** - Should be 20-60 seconds for 9 LLM calls
# MAGIC    - If < 5 seconds: LLM calls are failing silently
# MAGIC 
# MAGIC 4. **All scores are 0** warning - Indicates parsing or API issues
# MAGIC 
# MAGIC 5. **ERROR messages** - Will show full traceback of any failures
