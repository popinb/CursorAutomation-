# Databricks notebook source
# MAGIC %md
# MAGIC # LLM-as-a-Judge: Simple Working Version
# MAGIC 
# MAGIC **This version focuses on working evaluation first**
# MAGIC - No fancy features, just solid evaluation
# MAGIC - Tested for actual Databricks environment
# MAGIC - Widget comes later after this works

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Install Packages

# COMMAND ----------

%pip install openai pandas --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Load Data

# COMMAND ----------

import pandas as pd
import json

# Simple test data
METRICS_CONFIG_JSON = [
    {
        "name": "Accuracy",
        "type": "binary",
        "description": "Is the response accurate?",
        "grading_rubric": "Score 1 if accurate, 0 if not",
        "threshold": "1",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    },
    {
        "name": "Completeness",
        "type": "1-5_scale",
        "description": "How complete is the response?",
        "grading_rubric": "5=Complete, 1=Incomplete",
        "threshold": "3",
        "ground_truth_file_path": "",
        "ground_truth_column": ""
    }
]

EVALUATION_DATA_JSON = [
    {"sample_id": 1, "prompt": "What is 2+2?", "response": "2+2 equals 4"},
    {"sample_id": 2, "prompt": "What is the capital of France?", "response": "The capital of France is Paris"}
]

GROUND_TRUTH_JSON = [
    {"sample_id": 1, "correct_answer": "4", "explanation": "Basic addition"},
    {"sample_id": 2, "correct_answer": "Paris", "explanation": "Capital city"}
]

# Convert to DataFrames
METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
EVALUATION_DATA = pd.DataFrame(EVALUATION_DATA_JSON)
GROUND_TRUTH_DATA_DF = pd.DataFrame(GROUND_TRUTH_JSON)
GROUND_TRUTH_DATA = {'ground_truth.csv': GROUND_TRUTH_DATA_DF}

print("Data loaded:")
print(f"  Metrics: {len(METRICS_CONFIG_DATA)}")
print(f"  Samples: {len(EVALUATION_DATA)}")
print(f"  Ground truth: {len(GROUND_TRUTH_DATA_DF)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Configure OpenAI Client (Simple Version)

# COMMAND ----------

from openai import OpenAI
import os

# Get API key
try:
    OPENAI_KEY = dbutils.secrets.get("popin-secure-scope", "openai_key")
    print("API key retrieved successfully")
except Exception as e:
    print(f"Error getting API key: {e}")
    print("Please configure your OpenAI API key in secrets")
    raise

# Initialize client
client = OpenAI(
    base_url="https://api.zillowlabs.com/openai/v1",
    api_key=OPENAI_KEY
)

JUDGE_MODEL = "gpt-4o-mini"

# Test connection
try:
    test_response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5
    )
    print(f"Connection test successful!")
    print(f"Model: {JUDGE_MODEL}")
except Exception as e:
    print(f"Connection test failed: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Simple Evaluator Class

# COMMAND ----------

import json
import re

class SimpleEvaluator:
    def __init__(self, client, model, ground_truth_data):
        self.client = client
        self.model = model
        self.ground_truth_data = ground_truth_data
    
    def get_ground_truth(self, metric, sample_idx):
        """Get ground truth for a metric"""
        try:
            # Check if metric uses ground truth
            if not metric.get('ground_truth_file_path'):
                return None
            
            # Get the ground truth file
            filename = metric['ground_truth_file_path']
            if filename not in self.ground_truth_data:
                return None
            
            df = self.ground_truth_data[filename]
            if sample_idx >= len(df):
                return None
            
            # Get all columns as context
            row = df.iloc[sample_idx]
            gt_text = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            return gt_text
            
        except Exception as e:
            print(f"Error getting ground truth: {e}")
            return None
    
    def call_llm(self, prompt):
        """Call the LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return json.dumps({"score": 0, "explanation": f"Error: {str(e)}"})
    
    def parse_response(self, response_text, metric_type):
        """Parse LLM response"""
        try:
            # Clean response
            text = response_text.strip()
            
            # Remove markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            data = json.loads(text)
            score = float(data.get('score', 0))
            explanation = data.get('explanation', 'No explanation')
            
            # Normalize score based on metric type
            if metric_type == 'binary':
                score = 1.0 if score >= 0.5 else 0.0
            elif metric_type == '1-5_scale':
                score = max(1.0, min(5.0, score))
            elif metric_type == 'percentage':
                if score <= 1.0:
                    score = score * 100
                score = max(0.0, min(100.0, score))
            
            return score, explanation
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response was: {response_text[:200]}")
            return 0.0, f"Parse error: {str(e)}"
    
    def evaluate_single(self, sample, metric, sample_idx):
        """Evaluate a single sample with a metric"""
        try:
            # Get ground truth if available
            gt = self.get_ground_truth(metric, sample_idx)
            gt_text = f"\n\nGround Truth:\n{gt}" if gt else "\n\nGround Truth: Not provided"
            
            # Build prompt
            prompt = f"""Task: {metric['description']}

Grading Rubric:
{metric['grading_rubric']}

User Query: {sample['prompt']}
AI Response: {sample['response']}{gt_text}

Provide your evaluation as JSON:
{{"score": <number>, "explanation": "<brief explanation>"}}"""
            
            # Call LLM
            llm_response = self.call_llm(prompt)
            
            # Parse response
            score, explanation = self.parse_response(llm_response, metric['type'])
            
            # Determine pass/fail
            threshold = float(metric['threshold'])
            passed = score >= threshold
            
            return {
                'score': score,
                'explanation': explanation,
                'passed': passed,
                'threshold': threshold
            }
            
        except Exception as e:
            print(f"Error in evaluate_single: {e}")
            return {
                'score': 0.0,
                'explanation': f"Error: {str(e)}",
                'passed': False,
                'threshold': float(metric['threshold'])
            }
    
    def evaluate_dataset(self, eval_data, metrics):
        """Evaluate entire dataset"""
        results = []
        
        print(f"Starting evaluation: {len(eval_data)} samples x {len(metrics)} metrics")
        
        for idx, sample in eval_data.iterrows():
            sample_id = sample.get('sample_id', idx)
            print(f"\nSample {idx+1}/{len(eval_data)}: ID {sample_id}")
            
            for metric in metrics:
                metric_name = metric['name']
                print(f"  Evaluating {metric_name}...", end=' ')
                
                result = self.evaluate_single(sample.to_dict(), metric, idx)
                
                status_str = "PASS" if result['passed'] else "FAIL"
                print(f"{status_str} (score: {result['score']:.2f})")
                
                results.append({
                    'sample_id': sample_id,
                    'metric_name': metric_name,
                    'metric_type': metric['type'],
                    'score': result['score'],
                    'threshold': result['threshold'],
                    'passed': result['passed'],
                    'explanation': result['explanation'],
                    'prompt': sample['prompt'],
                    'response': sample['response']
                })
        
        return pd.DataFrame(results)

print("Evaluator class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Run Evaluation

# COMMAND ----------

print("="*80)
print("STARTING EVALUATION")
print("="*80)

# Initialize evaluator
evaluator = SimpleEvaluator(
    client=client,
    model=JUDGE_MODEL,
    ground_truth_data=GROUND_TRUTH_DATA
)

# Convert metrics to list of dicts
metrics_list = METRICS_CONFIG_DATA.to_dict('records')

# Run evaluation
results_df = evaluator.evaluate_dataset(EVALUATION_DATA, metrics_list)

# Display results
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

total = len(results_df)
passed = len(results_df[results_df['passed'] == True])
failed = total - passed
pass_rate = (passed / total * 100) if total > 0 else 0

print(f"\nOverall:")
print(f"  Total evaluations: {total}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Pass rate: {pass_rate:.1f}%")

print(f"\nPer-metric:")
for metric_name in results_df['metric_name'].unique():
    metric_results = results_df[results_df['metric_name'] == metric_name]
    metric_passed = len(metric_results[metric_results['passed'] == True])
    metric_total = len(metric_results)
    metric_rate = (metric_passed / metric_total * 100) if metric_total > 0 else 0
    avg_score = metric_results['score'].mean()
    
    print(f"  {metric_name}:")
    print(f"    Pass rate: {metric_rate:.1f}% ({metric_passed}/{metric_total})")
    print(f"    Avg score: {avg_score:.2f}")

print("\nDetailed results:")
display(results_df[['sample_id', 'metric_name', 'score', 'threshold', 'passed', 'explanation']])

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC 
# MAGIC This is a simplified version that focuses on:
# MAGIC - Working evaluation logic
# MAGIC - Proper error handling
# MAGIC - Clear pass/fail determination
# MAGIC - Simple output
# MAGIC 
# MAGIC Once this works, we can add:
# MAGIC - Widget for model selection
# MAGIC - Databricks Serving Endpoints support
# MAGIC - More sophisticated features
