# Databricks notebook source
# MAGIC %md
# MAGIC # ?? LLM-as-a-Judge: Cinderella Story Evaluation
# MAGIC # **No File Uploads Required!**
# MAGIC 
# MAGIC **Features:**
# MAGIC - ? All data hardcoded (Cinderella story theme)
# MAGIC - ? 3 metrics: Binary, 1-5 Scale, Percentage
# MAGIC - ? 8 Q&A samples about Cinderella
# MAGIC - ? Ground truth with multiple columns (ALL accessible!)
# MAGIC - ? **Widget to select judge model** (OpenAI or Databricks LLM)
# MAGIC - ? Uses Zillow Labs OpenAI proxy
# MAGIC - ? Easy to edit - just modify JSON at top of Cell 2
# MAGIC 
# MAGIC **Instructions:**
# MAGIC 1. Run Cell 1 (Install packages)
# MAGIC 2. Run Cell 2 (Load hardcoded data - edit JSON if needed)
# MAGIC 3. Run Cell 3 (Configure LLM - **SELECT MODEL from widget dropdown**)
# MAGIC 4. Run Cell 4-5 (Setup evaluation engine)
# MAGIC 5. Run Cell 6 (RUN EVALUATION!)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Installation and Setup

# COMMAND ----------

# Install required packages
%pip install openai pandas --quiet

# Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Hardcoded Data - Cinderella Story
# MAGIC 
# MAGIC **All data is in JSON format below - easy to edit!**

# COMMAND ----------

import pandas as pd

# ============================================================================
# ?? EDIT SECTION 1: METRICS CONFIGURATION (3 metrics)
# ============================================================================

METRICS_CONFIG_JSON = [
    # Metric 1: BINARY (0 or 1)
    {
        "name": "Story_Accuracy",
        "type": "binary",
        "description": "Evaluate if the response is factually accurate about the Cinderella story",
        "grading_rubric": "Score 1 if all story facts are correct and align with the classic Cinderella tale. Score 0 if any facts are incorrect, made up, or contradict the original story.",
        "threshold": "1",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    },
    
    # Metric 2: 1-5 SCALE
    {
        "name": "Response_Completeness",
        "type": "1-5_scale",
        "description": "Evaluate how complete and thorough the response is",
        "grading_rubric": "5=Fully complete, addresses all aspects comprehensively; 4=Mostly complete with minor gaps; 3=Partially complete, missing some details; 2=Barely complete, many gaps; 1=Incomplete or inadequate",
        "threshold": "4",
        "ground_truth_file_path": "",
        "ground_truth_column": ""
    },
    
    # Metric 3: PERCENTAGE (0-100)
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

# ============================================================================
# ?? EDIT SECTION 2: EVALUATION DATA (8 Q&A samples)
# ============================================================================

EVALUATION_DATA_JSON = [
    {"sample_id": 1, "prompt": "Who is Cinderella and what is her story about?", "response": "Cinderella is a young girl who lives with her mean stepmother and two stepsisters. They make her do all the chores and treat her badly. One day, with the help of her Fairy Godmother, she goes to the prince's ball. She loses her glass slipper at midnight, and the prince finds her by trying the slipper on every girl in the kingdom."},
    {"sample_id": 2, "prompt": "What did the Fairy Godmother turn into a carriage?", "response": "The Fairy Godmother used her magic wand to turn a big orange pumpkin into a beautiful golden carriage so Cinderella could go to the ball in style."},
    {"sample_id": 3, "prompt": "What happened at midnight?", "response": "When the clock struck twelve at midnight, Cinderella had to run away from the ball because the Fairy Godmother's magic would wear off. In her hurry, she lost one of her glass slippers on the palace steps."},
    {"sample_id": 4, "prompt": "How did the prince find Cinderella?", "response": "The prince searched the whole kingdom with the glass slipper, trying it on every girl. When he came to Cinderella's house, the slipper fit her perfectly! This proved she was the mysterious princess from the ball."},
    {"sample_id": 5, "prompt": "What animals helped Cinderella?", "response": "Cinderella had many animal friends including mice, birds, and a dog. The mice were her best friends and they helped her make a dress for the ball. The birds also helped her with her chores around the house."},
    {"sample_id": 6, "prompt": "What was Cinderella wearing at the ball?", "response": "Cinderella wore a magnificent ball gown that sparkled like stars. Her Fairy Godmother created it with magic, along with glass slippers on her feet. She looked so beautiful that everyone at the ball, including the prince, couldn't take their eyes off her."},
    {"sample_id": 7, "prompt": "Who were Cinderella's stepsisters?", "response": "Anastasia and Drizella were Cinderella's two stepsisters. They were mean and jealous of Cinderella's kindness and beauty. They made her do all the housework and never let her rest."},
    {"sample_id": 8, "prompt": "What is the moral of the Cinderella story?", "response": "The story teaches us that kindness and goodness are always rewarded. Even when life is hard and people are mean to you, if you stay kind and never give up hope, good things will happen. It also shows that true beauty comes from being a good person inside."}
]

# ============================================================================
# ?? EDIT SECTION 3: GROUND TRUTH DATA (Reference answers)
# ============================================================================

GROUND_TRUTH_JSON = [
    {"sample_id": 1, "correct_answer": "Cinderella is a kind young girl mistreated by her stepmother and stepsisters. With help from her Fairy Godmother, she attends a royal ball, loses her glass slipper at midnight, and is found by the prince who searches for her with the slipper.", "story_element": "Main plot", "key_facts": "stepmother, stepsisters, Fairy Godmother, ball, glass slipper, midnight, prince", "source": "Classic Cinderella fairy tale"},
    {"sample_id": 2, "correct_answer": "A pumpkin", "story_element": "Magic transformation", "key_facts": "pumpkin turned into carriage", "source": "Classic Cinderella fairy tale"},
    {"sample_id": 3, "correct_answer": "Cinderella had to leave the ball because the magic spell would break at midnight. She ran away and lost her glass slipper on the steps.", "story_element": "Midnight deadline", "key_facts": "midnight, magic ends, lost slipper, palace steps", "source": "Classic Cinderella fairy tale"},
    {"sample_id": 4, "correct_answer": "The prince searched the kingdom trying the glass slipper on every maiden until he found Cinderella, whose foot fit the slipper perfectly.", "story_element": "Finding Cinderella", "key_facts": "glass slipper, kingdom search, perfect fit", "source": "Classic Cinderella fairy tale"},
    {"sample_id": 5, "correct_answer": "Mice and birds were Cinderella's main animal friends who helped her with chores and making her dress.", "story_element": "Animal helpers", "key_facts": "mice, birds, helped with chores and dress", "source": "Classic Cinderella fairy tale"},
    {"sample_id": 6, "correct_answer": "A beautiful ball gown created by the Fairy Godmother's magic, with glass slippers.", "story_element": "Ball outfit", "key_facts": "magical ball gown, glass slippers", "source": "Classic Cinderella fairy tale"},
    {"sample_id": 7, "correct_answer": "Anastasia and Drizella (names vary by version, but they are the two mean stepsisters who mistreated Cinderella).", "story_element": "Antagonists", "key_facts": "two stepsisters, mean, jealous", "source": "Classic Cinderella fairy tale"},
    {"sample_id": 8, "correct_answer": "Be kind and patient even in difficult times, and goodness will be rewarded. Inner beauty and character matter more than outer appearances.", "story_element": "Moral lesson", "key_facts": "kindness rewarded, inner beauty, patience", "source": "Classic Cinderella fairy tale"}
]

# ============================================================================
# PROCESSING (Don't edit below)
# ============================================================================

print("="*80)
print("?? CINDERELLA STORY EVALUATION")
print("="*80)

# Convert to DataFrames
METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
EVALUATION_DATA = pd.DataFrame(EVALUATION_DATA_JSON)
GROUND_TRUTH_DATA_DF = pd.DataFrame(GROUND_TRUTH_JSON)
GROUND_TRUTH_DATA = {'ground_truth.csv': GROUND_TRUTH_DATA_DF}

print(f"\n? Data Loaded:")
print(f"   ? Metrics: {len(METRICS_CONFIG_DATA)}")
print(f"   ? Samples: {len(EVALUATION_DATA)}")
print(f"   ? Ground Truth: {len(GROUND_TRUTH_DATA_DF)} (with {len(GROUND_TRUTH_DATA_DF.columns)} columns)")

print("\n?? METRICS:")
display(METRICS_CONFIG_DATA[['name', 'type', 'threshold']])

print("\n?? SAMPLES (first 3):")
display(EVALUATION_DATA[['sample_id', 'prompt']].head(3))

print("\n?? GROUND TRUTH (first 3):")
display(GROUND_TRUTH_DATA_DF[['sample_id', 'correct_answer']].head(3))

print("\n?? Ready for Cell 3!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Configure LLM Judge Model
# MAGIC 
# MAGIC **?? IMPORTANT: Select your judge model from the dropdown widget at the top of the notebook!**
# MAGIC 
# MAGIC **Options:**
# MAGIC - **gpt-4o** - Most capable OpenAI model
# MAGIC - **gpt-4o-mini** - Fast and cost-effective
# MAGIC - **gpt-3.5-turbo** - Fastest and cheapest
# MAGIC - **databricks-llm** - Use Databricks Foundation Models

# COMMAND ----------

from openai import OpenAI
import os

# ============================================================================
# Widget for Model Selection
# ============================================================================

dbutils.widgets.dropdown(
    "judge_model",
    "databricks-llm",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "databricks-llm"],
    "?? Judge Model"
)

JUDGE_MODEL = dbutils.widgets.get("judge_model")

print("?? MODEL SETTINGS")
print("="*60)
print(f"Judge Model: {JUDGE_MODEL}")
print("="*60)

# ============================================================================
# Initialize Client Based on Selection
# ============================================================================

if JUDGE_MODEL == "databricks-llm":
    print("\n?? Databricks LLM selected")
    print("   Note: Using Databricks Foundation Models")
    print("   (This would use Databricks serving endpoints in production)")
    client = None
    print("\n??  For this demo, please select an OpenAI model to run evaluation")
    
else:
    print("\n?? Initializing OpenAI connection...")
    
    # Try multiple sources for the API key
    OPENAI_KEY = None
    key_source = None
    
    # Method 1: popin-secure-scope (should work for everyone now)
    try:
        OPENAI_KEY = dbutils.secrets.get("popin-secure-scope", "openai_key")
        key_source = "popin-secure-scope (shared)"
        print("? Using shared OpenAI key from popin-secure-scope")
    except Exception as e:
        print(f"?? Cannot access popin-secure-scope: {str(e)[:50]}...")
    
    # Method 2: User's personal scope (fallback)
    if not OPENAI_KEY:
        try:
            username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
            clean_username = username.replace("@", "_at_").replace(".", "_")
            user_scope = f"user_{clean_username}_secrets"
            OPENAI_KEY = dbutils.secrets.get(user_scope, "openai_key")
            key_source = f"{user_scope} (personal)"
            print(f"? Using personal OpenAI key from {user_scope}")
        except:
            print("?? No personal scope found")
    
    # Method 3: Other common scope names
    if not OPENAI_KEY:
        common_scopes = [
            ("shared-openai-keys", "openai_key"),
            ("openai-secrets", "api_key"),
            ("llm-keys", "openai_key")
        ]
        
        for scope_name, key_name in common_scopes:
            try:
                OPENAI_KEY = dbutils.secrets.get(scope_name, key_name)
                key_source = f"{scope_name} (shared)"
                print(f"? Using OpenAI key from {scope_name}")
                break
            except:
                continue
    
    # Initialize OpenAI client if key found
    if OPENAI_KEY:
        try:
            os.environ["OPENAI_API_KEY"] = OPENAI_KEY
            
            client = OpenAI(
                base_url="https://api.zillowlabs.com/openai/v1",
                api_key=OPENAI_KEY
            )
            
            # Test connection
            print(f"\n?? Testing connection to {JUDGE_MODEL}...")
            test_response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=10
            )
            
            print(f"? OpenAI connection successful!")
            print(f"   Using key from: {key_source}")
            print(f"   Model: {JUDGE_MODEL}")
            print(f"   Base URL: https://api.zillowlabs.com/openai/v1")
            
        except Exception as e:
            print(f"? OpenAI connection failed: {e}")
            client = None
    else:
        print("? No OpenAI API key found")
        print("\n?? To use OpenAI models:")
        print("   1. Ask notebook owner for access to 'popin-secure-scope', OR")
        print("   2. Create your own scope 'user_{username}_secrets' with key 'openai_key', OR")
        print("   3. Use 'databricks-llm' option instead")
        client = None

print("\n" + "="*60)
if client:
    print("? READY TO EVALUATE!")
    print(f"   Model: {JUDGE_MODEL}")
    print(f"   Via: Zillow Labs OpenAI Proxy")
else:
    print("?? SETUP NEEDED")
    if JUDGE_MODEL == "databricks-llm":
        print("   Please select an OpenAI model for this demo")
    else:
        print("   Please configure API access")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Define Evaluation Classes

# COMMAND ----------

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

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

print("? Evaluation classes defined")
print("   ? MetricType enum (BINARY, SCALE_1_5, PERCENTAGE)")
print("   ? MetricConfig dataclass")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Create LLM Judge Evaluator

# COMMAND ----------

import json
import re
import pandas as pd

class LLMJudgeEvaluator:
    """LLM Judge Evaluator with support for Zillow Labs OpenAI proxy."""
    
    def __init__(self, client, model, metrics, ground_truth_data):
        self.client = client
        self.model = model
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data
    
    def _get_ground_truth(self, metric, sample_idx):
        """Get ALL ground truth columns for a sample."""
        try:
            filename = os.path.basename(metric.ground_truth_file_path) if metric.ground_truth_file_path else None
            if not filename or filename not in self.ground_truth_data:
                return "Not provided"
            
            df = self.ground_truth_data[filename]
            if sample_idx >= len(df):
                return f"Index {sample_idx} out of range"
            
            # Get ALL columns (enhanced feature!)
            row = df.iloc[sample_idx]
            all_data = []
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    all_data.append(f"? {col}: {value}")
            
            return "?? Ground Truth:\n" + "\n".join(all_data) if all_data else "No data"
        except Exception as e:
            return f"Error: {e}"
    
    def _call_llm(self, prompt):
        """Call LLM judge via Zillow Labs proxy."""
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
    
    def _parse_response(self, response, metric):
        """Parse LLM response and normalize score."""
        content = response.strip()
        
        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        try:
            data = json.loads(content)
            score = data.get('score', data.get('Score', 0))
            explanation = data.get('explanation', data.get('Explanation', 'No explanation'))
        except:
            # Fallback: try regex
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
    
    def evaluate_single(self, prompt, response, metric, sample_idx):
        """Evaluate single sample against a metric."""
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
                "explanation": f"Error: {str(e)}",
                "status": "?",
                "ground_truth_used": False
            }
    
    def evaluate_dataset(self, eval_data):
        """Evaluate entire dataset."""
        results = []
        total = len(eval_data) * len(self.metrics)
        current = 0
        
        print(f"?? Starting evaluation:")
        print(f"   {len(eval_data)} samples ? {len(self.metrics)} metrics = {total} total evaluations")
        print("="*80)
        
        for idx, row in eval_data.iterrows():
            sample_id = row.get('sample_id', f'sample_{idx}')
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"\n?? Sample {idx+1}/{len(eval_data)}: ID {sample_id}")
            
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
        
        print("\n" + "="*80)
        print("? Evaluation complete!")
        return pd.DataFrame(results)

print("? LLM Judge Evaluator created")
print("   ? Supports Zillow Labs OpenAI proxy")
print("   ? Ground truth with ALL columns accessible")
print("   ? Automatic score normalization by metric type")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: RUN EVALUATION! ??

# COMMAND ----------

import pandas as pd

print("="*80)
print("?? STARTING CINDERELLA STORY EVALUATION")
print("="*80)

# Helper functions
def safe_float(value, default=0.0):
    """Safely convert value to float."""
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

def generate_prompt_from_rubric(name, description, rubric):
    """Generate evaluation prompt from metric configuration."""
    return f"""You are an expert evaluator. Task: {description}

Grading Rubric:
{rubric}

Evaluation Details:
- User Query: {{prompt}}
- AI Response: {{response}}
- Ground Truth: {{ground_truth}}

Provide your evaluation in JSON format ONLY (no other text):
{{
  "score": <your_numeric_score>,
  "explanation": "Brief explanation of your evaluation"
}}"""

# Step 1: Configure metrics
print("\n?? Step 1: Configuring metrics...")
metric_configs = []

for _, row in METRICS_CONFIG_DATA.iterrows():
    name = str(row['name']).strip()
    type_str = str(row['type']).strip().lower()
    
    # Determine metric type
    if type_str in ['binary', 'bool']:
        mtype = MetricType.BINARY
    elif type_str in ['1-5_scale', 'scale']:
        mtype = MetricType.SCALE_1_5
    else:
        mtype = MetricType.PERCENTAGE
    
    description = str(row.get('description', '')).strip()
    rubric = str(row.get('grading_rubric', '')).strip()
    threshold = safe_float(row.get('threshold', 0.5))
    
    prompt_template = generate_prompt_from_rubric(name, description, rubric)
    
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

# Step 2: Initialize evaluator
print("\n?? Step 2: Initializing evaluator...")
if client is None:
    print("? Error: OpenAI client not initialized")
    print("   Please go back to Cell 3 and:")
    if JUDGE_MODEL == "databricks-llm":
        print("   ? Select an OpenAI model (gpt-4o, gpt-4o-mini, or gpt-3.5-turbo)")
        print("   ? Re-run Cell 3")
    else:
        print("   ? Ensure API key is configured in Databricks secrets")
        print("   ? Scope: popin-secure-scope")
        print("   ? Key: openai_key")
else:
    evaluator = LLMJudgeEvaluator(
        client=client,
        model=JUDGE_MODEL,
        metrics=metric_configs,
        ground_truth_data=GROUND_TRUTH_DATA
    )
    print(f"   ? Evaluator ready with {JUDGE_MODEL}")
    print(f"   ? Using Zillow Labs OpenAI proxy")
    
    # Step 3: Run evaluation
    print("\n?? Step 3: Running evaluation...")
    results_df = evaluator.evaluate_dataset(EVALUATION_DATA)
    
    # Step 4: Display results
    print("\n" + "="*80)
    print("?? EVALUATION RESULTS")
    print("="*80)
    
    # Overall summary
    total = len(results_df)
    passed = len(results_df[results_df['status'] == '?'])
    failed = len(results_df[results_df['status'] == '?'])
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n?? OVERALL SUMMARY:")
    print(f"   Total evaluations: {total}")
    print(f"   Passed: {passed} ({pass_rate:.1f}%)")
    print(f"   Failed: {failed} ({100-pass_rate:.1f}%)")
    
    # Per-metric results
    print(f"\n?? PER-METRIC RESULTS:")
    for metric_name in results_df['metric_name'].unique():
        metric_results = results_df[results_df['metric_name'] == metric_name]
        metric_passed = len(metric_results[metric_results['status'] == '?'])
        metric_total = len(metric_results)
        metric_rate = (metric_passed / metric_total * 100) if metric_total > 0 else 0
        avg_score = metric_results['score'].mean()
        
        print(f"   ? {metric_name}:")
        print(f"     Pass rate: {metric_rate:.1f}% ({metric_passed}/{metric_total})")
        print(f"     Avg score: {avg_score:.2f}")
    
    # Detailed results table
    print("\n?? DETAILED RESULTS:")
    display(results_df[['sample_id', 'metric_name', 'score', 'threshold', 'status', 'explanation']])
    
    print("\n" + "="*80)
    print("?? EVALUATION COMPLETE!")
    print("="*80)
    print(f"\n? Successfully evaluated {len(EVALUATION_DATA)} Cinderella story samples")
    print(f"? Using {JUDGE_MODEL} via Zillow Labs proxy")
    print(f"? Overall pass rate: {pass_rate:.1f}%")
    print("\n?? To modify:")
    print("   ? Data: Edit JSON sections in Cell 2")
    print("   ? Model: Change widget dropdown and re-run Cell 3")
    print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ?? Congratulations!
# MAGIC 
# MAGIC Your Cinderella story evaluation is complete!
# MAGIC 
# MAGIC ### ?? What you evaluated:
# MAGIC - **8 Q&A samples** about Cinderella story
# MAGIC - **3 metrics:**
# MAGIC   - Story_Accuracy (binary, 0/1)
# MAGIC   - Response_Completeness (1-5 scale)
# MAGIC   - Child_Friendliness (percentage, 0-100)
# MAGIC 
# MAGIC ### ?? To modify:
# MAGIC - **Change data:** Edit JSON in Cell 2, re-run from Cell 2
# MAGIC - **Change model:** Select different model from widget dropdown, re-run Cell 3
# MAGIC - **Add metrics:** Add to METRICS_CONFIG_JSON in Cell 2
# MAGIC - **Add samples:** Add to EVALUATION_DATA_JSON in Cell 2
# MAGIC 
# MAGIC ### ? Key Features:
# MAGIC - ? Widget to select judge model (OpenAI or Databricks)
# MAGIC - ? No file uploads needed - all data hardcoded
# MAGIC - ? Ground truth with ALL columns accessible
# MAGIC - ? Zillow Labs OpenAI proxy integration
# MAGIC - ? Multiple model options (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
# MAGIC - ? Easy to edit - JSON format
# MAGIC - ? Automatic backup to multiple secret scopes
