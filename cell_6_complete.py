# MAGIC %md
# MAGIC ## Cell 6: LLM Judge Evaluator (Main Logic)
# MAGIC
# MAGIC **Purpose**: This cell contains the core evaluation engine that uses a language model to judge AI responses against defined metrics.
# MAGIC
# MAGIC **What it does**:
# MAGIC - Implements the LLMJudgeEvaluator class that handles all evaluation logic
# MAGIC - **ENHANCED**: Now provides ALL columns from ground truth files to metrics for richer context
# MAGIC - Provides robust JSON parsing that works with any custom metric naming convention
# MAGIC - Handles both OpenAI and Databricks LLM endpoints for evaluation
# MAGIC - Implements smart score extraction that can find scores regardless of how the LLM formats its response
# MAGIC - Includes fallback parsing methods for when JSON parsing fails
# MAGIC - Manages ground truth data integration for reference-based evaluations with full column access
# MAGIC - Provides comprehensive error handling and status reporting
# MAGIC
# MAGIC **When to run**: Run this cell after Cell 5 to initialize the evaluation engine
# MAGIC
# MAGIC **Expected output**: Confirmation message that the LLM Judge Evaluator has been defined with enhanced ground truth access

# COMMAND ----------

import json
import time
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import os

class LLMJudgeEvaluator:
    """
    Main evaluator class with bulletproof JSON parsing for ANY custom metric name.
    
    Key Innovation: Smart key detection that works with any metric naming convention.
    ENHANCED: Now provides ALL columns from ground truth files for richer evaluation context.
    """
    
    def __init__(self, judge_model: str, metrics: List[MetricConfig], ground_truth_data: Dict[str, pd.DataFrame] = None):
        self.judge_model = judge_model
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data or {}
        self.is_databricks_llm = judge_model == "databricks-llm"
        
        if self.is_databricks_llm:
            self._init_databricks_llm()
        else:
            self._init_openai_llm()
    
    def _init_databricks_llm(self):
        """Initialize Databricks LLM client."""
        try:
            self.databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            self.workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
            
            self.databricks_headers = {
                "Authorization": f"Bearer {self.databricks_token}",
                "Content-Type": "application/json"
            }
            
            import requests
            url = f"https://{self.workspace_url}/api/2.0/serving-endpoints"
            response = requests.get(url, headers=self.databricks_headers, timeout=10)
            
            if response.status_code == 200:
                endpoints = response.json().get('endpoints', [])
                available_models = [ep['name'] for ep in endpoints]
                
                # Find Claude Sonnet (preferred)
                for endpoint_name in available_models:
                    if 'claude-sonnet' in endpoint_name.lower():
                        self.databricks_endpoint = endpoint_name
                        self.response_format = 'openai'
                        print(f"✅ Using Databricks endpoint: {endpoint_name}")
                        return
                
                # Fallback to first available
                if available_models:
                    self.databricks_endpoint = available_models[0]
                    self.response_format = 'openai'
                    print(f"✅ Using Databricks endpoint: {available_models[0]}")
                else:
                    raise Exception("No endpoints found")
            else:
                raise Exception(f"Failed to list endpoints: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Databricks init failed: {e}")
            raise
    
    def _init_openai_llm(self):
        """Initialize OpenAI LLM client."""
        if 'client' in globals() and client is not None:
            self.llm_client = client
            print(f"✅ OpenAI client initialized")
        else:
            raise ValueError("OpenAI client not found")
    
    def _call_databricks_llm(self, prompt: str) -> str:
        """Call Databricks LLM endpoint."""
        try:
            import requests
            url = f"https://{self.workspace_url}/serving-endpoints/{self.databricks_endpoint}/invocations"
            
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            response = requests.post(url, headers=self.databricks_headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
            
            return ""
                
        except Exception as e:
            print(f"Error calling Databricks LLM: {e}")
            return ""
    
    def _call_openai_llm(self, prompt: str) -> str:
        """Call OpenAI LLM endpoint."""
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
    
    def _get_ground_truth_for_metric(self, metric: MetricConfig, sample_idx: int) -> str:
        """
        ENHANCED: Get ALL columns from ground truth files for richer evaluation context.
        
        How it works:
        1. PM specifies just the FILENAME in CSV (e.g., 'ground_truth_accuracy.csv')
        2. Full paths come from UI widget (e.g., '/Workspace/Users/email/ground_truth_accuracy.csv')
        3. Code matches filename from CSV to actual uploaded files
        4. **NEW**: Returns ALL columns from the ground truth file, not just one specific column
        
        This gives the LLM judge much richer context for evaluation!
        """
        try:
            # Parse file paths (handle semicolon or comma separated)
            files = []
            if ';' in metric.ground_truth_file_path:
                files = [f.strip() for f in metric.ground_truth_file_path.split(';') if f.strip()]
            elif ',' in metric.ground_truth_file_path:
                files = [f.strip() for f in metric.ground_truth_file_path.split(',') if f.strip()]
            else:
                files = [metric.ground_truth_file_path.strip()]
            
            if not files or not files[0]:
                return "Not provided"
            
            # Match by filename (PM provides filename, widget provides full paths)
            for file_path in files:
                filename = os.path.basename(file_path)  # Extract just the filename
                if filename in self.ground_truth_data:
                    df = self.ground_truth_data[filename]
                    
                    if sample_idx >= len(df):
                        return f"Sample index {sample_idx} out of range (file has {len(df)} rows)"
                    
                    # 🎯 ENHANCED: Return ALL columns for richer context
                    row_data = df.iloc[sample_idx]
                    
                    # Format all columns as structured key-value pairs
                    all_data = []
                    for col, value in row_data.items():
                        if pd.notna(value) and str(value).strip():
                            # Clean up the column name and value
                            clean_col = str(col).strip()
                            clean_value = str(value).strip()
                            all_data.append(f"• {clean_col}: {clean_value}")
                    
                    if all_data:
                        # Return formatted ground truth with all columns
                        result = "📚 Ground Truth Reference (All Available Data):\n" + "\n".join(all_data)
                        return result
                    else:
                        return "No ground truth data available for this sample (all values are empty/null)"
            
            return f"Ground truth file not found: {os.path.basename(files[0]) if files else 'No file specified'}"
            
        except Exception as e:
            return f"Error accessing ground truth: {str(e)}"
    
    def _escape_prompt_template(self, template: str) -> str:
        """
        Properly escape prompt template to handle JSON examples.
        
        Protects {prompt}, {response}, {ground_truth} while escaping other braces.
        """
        actual_placeholders = {
            '{prompt}': '<<<PROMPT_PLACEHOLDER>>>',
            '{response}': '<<<RESPONSE_PLACEHOLDER>>>',
            '{ground_truth}': '<<<GROUND_TRUTH_PLACEHOLDER>>>'
        }
        
        escaped_template = template
        for placeholder, marker in actual_placeholders.items():
            escaped_template = escaped_template.replace(placeholder, marker)
        
        # Escape all remaining curly braces
        escaped_template = escaped_template.replace('{', '{{').replace('}', '}}')
        
        # Restore actual placeholders
        for placeholder, marker in actual_placeholders.items():
            escaped_template = escaped_template.replace(marker, placeholder)
        
        return escaped_template
    
    def evaluate_single(self, prompt: str, response: str, ground_truth_data: dict, metric: MetricConfig, sample_idx: int = 0) -> dict:
        """Evaluate a single sample with one metric."""
        try:
            # 🎯 ENHANCED: Get ALL columns from ground truth
            ground_truth = self._get_ground_truth_for_metric(metric, sample_idx)
            
            # Escape prompt template
            safe_template = self._escape_prompt_template(metric.prompt_template)
            
            # Format prompt with enhanced ground truth
            eval_prompt = safe_template.format(
                prompt=prompt,
                response=response,
                ground_truth=ground_truth if ground_truth else "Not provided"
            )
            
            # Call LLM
            if self.is_databricks_llm:
                llm_response = self._call_databricks_llm(eval_prompt)
            else:
                llm_response = self._call_openai_llm(eval_prompt)
            
            if not llm_response or str(llm_response).strip() == "":
                return {
                    "score": 0,
                    "explanation": "Empty response from LLM",
                    "status": "❌"
                }
            
            # Parse response
            score, explanation = self._parse_llm_response(llm_response, metric)
            
            status = "✅" if score >= metric.threshold else "❌"
            
            return {
                "score": score,
                "explanation": explanation,
                "status": status
            }
            
        except Exception as e:
            return {
                "score": 0,
                "explanation": f"Evaluation error: {str(e)}",
                "status": "❌"
            }
    
    def _parse_llm_response(self, llm_response: str, metric: MetricConfig) -> tuple:
        """
        BULLETPROOF JSON parsing that works with ANY custom metric name.
        
        Strategy:
        1. Try to parse as JSON
        2. Look for score using intelligent key detection
        3. Fall back to text parsing if JSON fails
        """
        content = str(llm_response).strip()
        
        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        # Try JSON parsing
        try:
            result_json = json.loads(content)
            
            # BULLETPROOF: Smart score extraction
            score = self._smart_extract_score(result_json, metric)
            
            # BULLETPROOF: Smart explanation extraction
            explanation = self._smart_extract_explanation(result_json)
            
        except json.JSONDecodeError:
            # Fallback to text parsing
            score, explanation = self._fallback_parse(content, metric)
        
        # Normalize score
        score = self._normalize_score(score, metric.metric_type)
        
        return score, str(explanation)[:500]
    
    def _smart_extract_score(self, json_obj: dict, metric: MetricConfig) -> Any:
        """
        BULLETPROOF score extraction that handles ANY custom metric name.
        
        Strategy:
        1. Try standard keys first (score, value, rating)
        2. Try the exact metric name
        3. Try metric name variations (with/without common suffixes)
        4. Try case-insensitive search through all keys
        5. Try finding any numeric value in the JSON
        """
        # Step 1: Try standard keys first (most common)
        standard_keys = ["score", "Score", "value", "Value", "rating", "Rating", "result", "Result"]
        for key in standard_keys:
            if key in json_obj:
                return json_obj[key]
        
        # Step 2: Try exact metric name
        if metric.name in json_obj:
            return json_obj[metric.name]
        
        # Step 3: Try metric name with common suffix variations
        # Remove common suffixes if they exist
        base_name = metric.name
        common_suffixes = ['_score', '_rating', '_check', '_value', '_result', 'Score', 'Rating', 'Check', 'Value', 'Result']
        
        for suffix in common_suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        # Try base name without suffix
        if base_name in json_obj:
            return json_obj[base_name]
        
        # Try base name with different suffixes
        for suffix in ['_score', '_rating', '_value', 'Score', 'Rating', 'Value']:
            key = f"{base_name}{suffix}"
            if key in json_obj:
                return json_obj[key]
        
        # Step 4: Case-insensitive search
        metric_name_lower = metric.name.lower()
        for key, value in json_obj.items():
            if key.lower() == metric_name_lower:
                return value
        
        # Step 5: Look for keys containing the metric name or common score words
        score_keywords = ['score', 'rating', 'value', 'result', metric.name.lower()]
        for key, value in json_obj.items():
            key_lower = key.lower()
            for keyword in score_keywords:
                if keyword in key_lower and isinstance(value, (int, float, str)):
                    try:
                        # Try to convert to number
                        return float(value) if '.' in str(value) else int(value)
                    except:
                        pass
        
        # Step 6: Last resort - find ANY numeric value in the JSON
        for key, value in json_obj.items():
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                try:
                    return float(value) if '.' in value else int(value)
                except:
                    pass
        
        # If nothing found, return 0
        return 0
    
    def _smart_extract_explanation(self, json_obj: dict) -> str:
        """
        BULLETPROOF explanation extraction.
        
        Tries multiple common keys for explanations.
        """
        explanation_keys = [
            "explanation", "Explanation", 
            "reason", "Reason", 
            "comment", "Comment", 
            "rationale", "Rationale", 
            "justification", "Justification",
            "reasoning", "Reasoning",
            "details", "Details",
            "description", "Description"
        ]
        
        # Try standard keys
        for key in explanation_keys:
            if key in json_obj:
                return json_obj[key]
        
        # Case-insensitive search
        for key, value in json_obj.items():
            key_lower = key.lower()
            if any(exp_key.lower() in key_lower for exp_key in explanation_keys):
                if isinstance(value, str):
                    return value
        
        # Look for any string value that's not too short
        for key, value in json_obj.items():
            if isinstance(value, str) and len(value) > 10:
                return value
        
        return "No explanation provided"
    
    def _fallback_parse(self, content: str, metric: MetricConfig) -> tuple:
        """Fallback parsing when JSON parsing fails."""
        score = 0
        explanation = content
        
        if metric.metric_type == MetricType.BINARY:
            if any(word in content.lower() for word in ['pass', 'correct', 'accurate', 'yes', 'true', '1']):
                score = 1
        
        elif metric.metric_type == MetricType.SCALE_1_5:
            numbers = re.findall(r'\b[1-5]\b', content)
            if numbers:
                score = int(numbers[0])
        
        elif metric.metric_type == MetricType.PERCENTAGE:
            percentages = re.findall(r'(\d+(?:\.\d+)?)[%]?', content)
            if percentages:
                score = float(percentages[0])
                # Keep raw score - _normalize_score will handle range conversion
        
        return score, explanation
    
    def _normalize_score(self, score: Any, metric_type: MetricType) -> float:
        """Normalize score to valid range."""
        try:
            score = float(score)
        except (ValueError, TypeError):
            return 0.0
        
        if metric_type == MetricType.BINARY:
            return 1.0 if score > 0.5 else 0.0
        elif metric_type == MetricType.SCALE_1_5:
            return max(1.0, min(5.0, score))
        elif metric_type == MetricType.PERCENTAGE:
            # Keep percentage in 0-100 range to match thresholds in CSV
            # If LLM returns 0-1 range, convert to 0-100
            if score <= 1.0:
                score = score * 100
            return max(0.0, min(100.0, score))
        
        return score
    
    def evaluate_dataset(self, evaluation_data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate entire dataset with all metrics."""
        print(f"\n🔍 Evaluating {len(evaluation_data)} samples with {len(self.metrics)} metrics...")
        print("🎯 Using ENHANCED ground truth access - all columns available to metrics!")
        
        results = []
        
        for idx, row in evaluation_data.iterrows():
            sample_id = row.get('sample_id', f'sample_{idx}')
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"\n📝 Sample {idx + 1}/{len(evaluation_data)}: {sample_id}")
            
            for metric in self.metrics:
                print(f"   📊 {metric.name}...", end=' ')
                
                ground_truth_data = {}
                for col in evaluation_data.columns:
                    if col not in ['sample_id', 'prompt', 'response']:
                        ground_truth_data[col] = row.get(col, '')
                
                result = self.evaluate_single(prompt, response, ground_truth_data, metric, idx)
                
                print(f"{result['status']}")
                
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
        
        print(f"\n✅ Evaluation complete with enhanced ground truth context!")
        return pd.DataFrame(results)

print("✅ LLM Judge Evaluator defined with ENHANCED ground truth access")
print("🎯 All metrics now receive ALL columns from ground truth files for richer evaluation context!")