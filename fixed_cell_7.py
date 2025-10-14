# COMMAND ----------

# =============================================================================
# LLM JUDGE EVALUATOR CLASS - FIXED FOR OPENAI 1.0+
# =============================================================================

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

class MetricType(Enum):
    BINARY = "binary"
    SCALE_1_5 = "1-5_scale"
    PERCENTAGE = "percentage"

@dataclass
class MetricConfig:
    name: str
    metric_type: MetricType
    description: str
    prompt_template: str
    threshold: float
    ground_truth_column: str
    ground_truth_file_path: str

class LLMJudgeEvaluator:
    def __init__(self, judge_model: str, metrics: List[MetricConfig]):
        self.judge_model = judge_model
        self.metrics = metrics
        self.is_databricks_llm = self._is_databricks_model(judge_model)
        
        # Initialize LLM client based on model type
        if self.is_databricks_llm:
            self._init_databricks_llm()
        else:
            self._init_openai_llm()
    
    def _is_databricks_model(self, model: str) -> bool:
        """Check if the model is a Databricks LLM endpoint."""
        databricks_models = [
            "databricks-llama-2-70b-chat",
            "databricks-llama-2-13b-chat", 
            "databricks-mpt-30b-instruct",
            "databricks-mpt-7b-instruct",
            "databricks-dolly-v2-12b",
            "databricks-dolly-v2-7b",
            "databricks-dolly-v2-3b"
        ]
        return model in databricks_models
    
    def _init_databricks_llm(self):
        """Initialize Databricks LLM client."""
        try:
            from databricks_genai_inference import ChatCompletion
            self.llm_client = ChatCompletion
            print(f"✅ Databricks LLM client initialized for {self.judge_model}")
        except ImportError:
            raise ImportError("Databricks GenAI library not available. Please install: pip install databricks-genai-inference")
    
    def _init_openai_llm(self):
        """Initialize OpenAI LLM client with new API."""
        try:
            import openai
            self.llm_client = openai.OpenAI()
            print(f"✅ OpenAI client initialized for {self.judge_model}")
        except ImportError:
            raise ImportError("OpenAI library not available. Please install: pip install openai")
    
    def _call_databricks_llm(self, prompt: str) -> str:
        """Call Databricks LLM endpoint."""
        try:
            response = self.llm_client.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Databricks LLM: {e}")
            return ""
    
    def _call_openai_llm(self, prompt: str) -> str:
        """Call OpenAI LLM endpoint with new API."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI LLM: {e}")
            return ""
    
    def evaluate_single(self, prompt: str, response: str, ground_truth_data: dict, metric: MetricConfig) -> dict:
        """Evaluate a single sample with one metric."""
        print(f"   🚀 DEBUG: Starting evaluation for metric: {metric.name}")
        
        try:
            # Get the specific ground truth for this metric
            ground_truth = ground_truth_data.get(metric.ground_truth_column, "Not provided")
            print(f"   🚀 DEBUG: Ground truth column: {metric.ground_truth_column}")
            print(f"   🚀 DEBUG: Ground truth value: {ground_truth}")
            
            # FIXED: Handle format string issues in prompt template
            # Replace any {metric_name_score} placeholders with literal text
            safe_prompt_template = metric.prompt_template
            if f"{{{metric.name}_score}}" in safe_prompt_template:
                safe_prompt_template = safe_prompt_template.replace(f"{{{metric.name}_score}}", f"{{{{{metric.name}_score}}}}")
                print(f"   🔧 DEBUG: Fixed format string issue in prompt template")
            
            # Format the evaluation prompt with actual values
            eval_prompt = safe_prompt_template.format(
                prompt=prompt,
                response=response,
                ground_truth=ground_truth if ground_truth else "Not provided"
            )
            print(f"   🚀 DEBUG: Evaluation prompt created successfully")
            print(f"   🔍 DEBUG: Evaluation prompt: {eval_prompt[:200]}...")
            
            # Route to appropriate LLM based on model type
            if self.is_databricks_llm:
                print(f"   🏢 Calling Databricks LLM for {metric.name}")
                llm_response = self._call_databricks_llm(eval_prompt)
            else:
                print(f"   🤖 Calling OpenAI ({self.judge_model}) for {metric.name}")
                llm_response = self._call_openai_llm(eval_prompt)
            
            print(f"   🚀 DEBUG: LLM response received: {type(llm_response)}")
            print(f"   🔍 DEBUG: LLM response content: {str(llm_response)[:200]}...")
            
            # Validate response
            if not llm_response or str(llm_response).strip() == "":
                print(f"   Warning: Empty response for {metric.name}")
                return {
                    "score": 0,
                    "explanation": "Empty response from LLM",
                    "status": "❌"
                }
            
            # Clean response content
            content = str(llm_response).strip()
            print(f"   🔍 DEBUG: Raw LLM response: {content[:200]}...")
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            print(f"   🔍 DEBUG: Cleaned content: {content[:200]}...")
            
            # Try to parse as JSON first
            try:
                result_json = json.loads(content)
                print(f"   ✅ Successfully parsed JSON for {metric.name}")
                print(f"   🔍 DEBUG: JSON object: {result_json}")
                print(f"   🔍 DEBUG: JSON keys: {list(result_json.keys())}")
                
                # SAFE JSON PARSING - Try multiple key variations
                score = 0
                explanation = "No explanation provided"
                
                # Try multiple variations of the score key
                score_key_variations = [
                    f"{metric.name}_score",           # accuracy_check_score
                    f'"{metric.name}_score"',         # "accuracy_check_score"
                    f"{metric.name}Score",            # accuracyCheckScore
                    f"{metric.name}score",            # accuracycheckscore
                    "score",                          # just "score"
                    "Score",                          # just "Score"
                    "value",                          # just "value"
                    "Value"                           # just "Value"
                ]
                
                print(f"   🔍 DEBUG: Trying score key variations: {score_key_variations}")
                
                # Try to find score
                for key in score_key_variations:
                    if key in result_json:
                        score = result_json[key]
                        print(f"   ✅ Found score using key: '{key}' = {score}")
                        break
                    else:
                        print(f"   ❌ Key '{key}' not found")
                
                # Try multiple variations of the explanation key
                explanation_key_variations = [
                    "explanation",
                    "Explanation", 
                    "reason",
                    "Reason",
                    "comment",
                    "Comment",
                    "rationale",
                    "Rationale"
                ]
                
                # Try to find explanation
                for key in explanation_key_variations:
                    if key in result_json:
                        explanation = result_json[key]
                        print(f"   ✅ Found explanation using key: '{key}'")
                        break
                
            except json.JSONDecodeError as e:
                print(f"   Warning: Response is not JSON for {metric.name}: {e}")
                print(f"   Raw response: {content[:100]}...")
                
                # Fallback: Extract score from non-JSON text
                import re
                score = 0
                explanation = content
                
                # Extract scores based on metric type
                if metric.metric_type == MetricType.BINARY:
                    # Look for pass/fail indicators
                    if any(word in content.lower() for word in ['pass', 'correct', 'accurate', 'yes', 'true', '1']):
                        score = 1
                    elif any(word in content.lower() for word in ['fail', 'incorrect', 'inaccurate', 'no', 'false', '0']):
                        score = 0
                
                elif metric.metric_type == MetricType.SCALE_1_5:
                    # Look for numbers 1-5
                    numbers = re.findall(r'\b[1-5]\b', content)
                    if numbers:
                        score = int(numbers[0])
                
                elif metric.metric_type == MetricType.PERCENTAGE:
                    # Look for percentages or decimals
                    percentages = re.findall(r'(\d+(?:\.\d+)?)[%]?', content)
                    if percentages:
                        score = float(percentages[0])
                        if score > 1:  # Convert percentage to decimal
                            score = score / 100
            
            # Ensure score is numeric
            if not isinstance(score, (int, float)):
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    print(f"   Warning: Could not convert score to number: {score}")
                    score = 0
            
            # Ensure score is within valid range
            if metric.metric_type == MetricType.BINARY:
                score = 1 if score > 0.5 else 0
            elif metric.metric_type == MetricType.SCALE_1_5:
                score = max(1, min(5, score))
            elif metric.metric_type == MetricType.PERCENTAGE:
                score = max(0, min(1, score))
            
            print(f"   Final score: {score} (threshold: {metric.threshold})")
            
            return {
                "score": score,
                "explanation": str(explanation)[:500],  # Truncate long explanations
                "status": "✅" if score >= metric.threshold else "❌"
            }
            
        except Exception as e:
            print(f"Error evaluating {metric.name}: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {
                "score": 0,
                "explanation": f"Evaluation error: {str(e)}",
                "status": "❌"
            }
    
    def evaluate_dataset(self, evaluation_data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate entire dataset with all metrics."""
        print(f"\n🔍 Evaluating {len(evaluation_data)} samples with {len(self.metrics)} metrics...")
        
        # Prepare results structure
        results = []
        
        # Process each sample
        for idx, row in evaluation_data.iterrows():
            sample_id = row.get('sample_id', f'sample_{idx}')
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"\n📝 Processing sample {idx + 1}/{len(evaluation_data)}: {sample_id}")
            
            # Evaluate with each metric
            for metric in self.metrics:
                print(f"   📊 Evaluating {metric.name}...")
                
                # Get ground truth data for this sample
                ground_truth_data = {}
                for col in evaluation_data.columns:
                    if col not in ['sample_id', 'prompt', 'response']:
                        ground_truth_data[col] = row.get(col, '')
                
                # Evaluate single metric
                result = self.evaluate_single(prompt, response, ground_truth_data, metric)
                
                # Store result
                results.append({
                    'sample_id': sample_id,
                    'metric_name': metric.name,
                    'metric_type': metric.metric_type.value,
                    'score': result['score'],
                    'threshold': metric.threshold,
                    'status': result['status'],
                    'explanation': result['explanation'],
                    'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    'response': response[:100] + "..." if len(response) > 100 else response
                })
        
        return pd.DataFrame(results)

def display_evaluation_results(results_df: pd.DataFrame):
    """Display evaluation results in a formatted way."""
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    # Overall statistics
    total_evaluations = len(results_df)
    passed_evaluations = len(results_df[results_df['status'] == '✅'])
    pass_rate = (passed_evaluations / total_evaluations) * 100 if total_evaluations > 0 else 0
    
    print(f"📈 Total Evaluations: {total_evaluations}")
    print(f"✅ Passed: {passed_evaluations}")
    print(f"❌ Failed: {total_evaluations - passed_evaluations}")
    print(f"📊 Pass Rate: {pass_rate:.1f}%")
    
    # Per-metric statistics
    print(f"\n📋 Per-Metric Results:")
    print("-" * 60)
    
    for metric_name in results_df['metric_name'].unique():
        metric_results = results_df[results_df['metric_name'] == metric_name]
        metric_passed = len(metric_results[metric_results['status'] == '✅'])
        metric_total = len(metric_results)
        metric_pass_rate = (metric_passed / metric_total) * 100 if metric_total > 0 else 0
        
        avg_score = metric_results['score'].mean()
        
        print(f"🔹 {metric_name}:")
        print(f"   Pass Rate: {metric_pass_rate:.1f}% ({metric_passed}/{metric_total})")
        print(f"   Average Score: {avg_score:.2f}")
        print(f"   Threshold: {metric_results['threshold'].iloc[0]}")
        print()
    
    # Sample results
    print(f"\n📝 Sample Results (first 5):")
    print("-" * 80)
    
    sample_results = results_df.head(5)
    for _, row in sample_results.iterrows():
        print(f"Sample: {row['sample_id']} | Metric: {row['metric_name']} | Score: {row['score']} | Status: {row['status']}")
        print(f"Explanation: {row['explanation'][:100]}...")
        print()
    
    print("="*80)
