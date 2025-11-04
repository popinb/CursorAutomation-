"""
Test the actual evaluation logic with mock LLM responses
"""
import sys
import json
import re
import pandas as pd
from enum import Enum
from dataclasses import dataclass

# Mock the OpenAI client
class MockOpenAIResponse:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockOpenAIResponse(content)

class MockChatCompletion:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

class MockOpenAIClient:
    def __init__(self):
        self.chat = self
        self.completions = self
        self.call_count = 0
    
    def create(self, model, messages, temperature, max_tokens):
        self.call_count += 1
        print(f"\n  MOCK LLM CALL #{self.call_count}")
        print(f"    Model: {model}")
        print(f"    Messages: {len(messages)} messages")
        
        # Return a valid JSON response
        response = {
            "score": 1,
            "explanation": "This is a test evaluation"
        }
        return MockChatCompletion(json.dumps(response))

# Define the same classes as in notebook
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

# Copy the actual LLMJudgeEvaluator class from the notebook
class LLMJudgeEvaluator:
    """LLM Judge Evaluator supporting both OpenAI and Databricks Serving Endpoints."""
    
    def __init__(self, client, model, metrics, ground_truth_data, client_type):
        self.client = client
        self.model = model
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data
        self.client_type = client_type
        print(f"Evaluator initialized: {client_type}, {len(metrics)} metrics")
    
    def _get_ground_truth(self, metric, sample_idx):
        """Get ALL ground truth columns for a sample."""
        try:
            import os
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
            return f"Error: {e}"
    
    def _call_llm(self, prompt):
        """Call LLM judge."""
        print(f"\n  DEBUG _call_llm: Attempting LLM call", flush=True)
        print(f"    Client type: {self.client_type}", flush=True)
        print(f"    Prompt length: {len(prompt)} chars", flush=True)
        
        try:
            if self.client_type == "openai":
                print(f"    Using OpenAI (model: {self.model})", flush=True)
                print(f"    Calling OpenAI API...", flush=True)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Provide responses in JSON format only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                content = response.choices[0].message.content
                print(f"    OpenAI success! Response length: {len(content)} chars", flush=True)
                return content
            else:
                return '{"score": 0, "explanation": "Databricks not implemented in test"}'
        except Exception as e:
            error_msg = f"Error calling LLM: {str(e)}"
            print(f"\nERROR in _call_llm: {error_msg}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            return f'{{"score": 0, "explanation": "{error_msg}"}}'
    
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
        except Exception as e:
            print(f"\nERROR parsing JSON response:")
            print(f"  Error: {str(e)}")
            print(f"  Response (first 300 chars): {content[:300]}")
            
            try:
                score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', content, re.IGNORECASE)
                score = float(score_match.group(1)) if score_match else 0
                explanation = f"Parse error: {str(e)[:200]}"
            except Exception as e2:
                print(f"  Regex fallback also failed: {str(e2)}")
                score = 0
                explanation = f"Complete parse failure: {str(e)[:200]}"
        
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
            status = "PASS" if score >= metric.threshold else "FAIL"
            
            return {
                "score": score,
                "explanation": explanation,
                "status": status,
                "ground_truth_used": ground_truth != "Not provided"
            }
        except Exception as e:
            print(f"\nERROR in evaluate_single: {str(e)}")
            import traceback
            print(traceback.format_exc())
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
        
        print(f"\nStarting evaluation:")
        print(f"  {len(eval_data)} samples x {len(self.metrics)} metrics = {total} evaluations")
        
        for idx, row in eval_data.iterrows():
            sample_id = row.get('sample_id', idx)
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"\nSample {idx+1}/{len(eval_data)}: ID {sample_id}")
            
            for metric in self.metrics:
                current += 1
                progress = (current / total) * 100
                print(f"  [{progress:5.1f}%] Evaluating {metric.name}...", end=' ', flush=True)
                
                result = self.evaluate_single(prompt, response, metric, idx)
                print(f"{result['status']} (score: {result['score']:.2f})", flush=True)
                
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
        
        return pd.DataFrame(results)

# Run the test
print("="*80)
print("TESTING EVALUATION LOGIC")
print("="*80)

# Create test data
eval_data = pd.DataFrame([
    {"sample_id": 1, "prompt": "What is 2+2?", "response": "2+2 equals 4"},
    {"sample_id": 2, "prompt": "What is the capital of France?", "response": "Paris"}
])

ground_truth = pd.DataFrame([
    {"sample_id": 1, "correct_answer": "4"},
    {"sample_id": 2, "correct_answer": "Paris"}
])

ground_truth_data = {'ground_truth.csv': ground_truth}

# Create metrics
metrics = [
    MetricConfig(
        name="Accuracy",
        description="Is it accurate?",
        metric_type=MetricType.BINARY,
        prompt_template="Evaluate: {prompt}\nResponse: {response}\nGround Truth: {ground_truth}\n\nReturn JSON: {{\"score\": <0 or 1>, \"explanation\": \"...\"}}",
        threshold=1.0,
        ground_truth_column="correct_answer",
        ground_truth_file_path="ground_truth.csv"
    )
]

# Create mock client
mock_client = MockOpenAIClient()

# Create evaluator
evaluator = LLMJudgeEvaluator(
    client=mock_client,
    model="gpt-4o",
    metrics=metrics,
    ground_truth_data=ground_truth_data,
    client_type="openai"
)

# Run evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION")
print("="*80)

results_df = evaluator.evaluate_dataset(eval_data)

# Show results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nResults shape: {results_df.shape}")
print(f"Mock LLM was called: {mock_client.call_count} times")
print(f"\nResults:")
print(results_df[['sample_id', 'metric_name', 'score', 'status', 'explanation']])

# Check if it worked
total = len(results_df)
passed = len(results_df[results_df['status'] == 'PASS'])
print(f"\n{'='*80}")
print(f"TEST RESULT:")
print(f"  Total: {total}")
print(f"  Passed: {passed}")
print(f"  LLM calls: {mock_client.call_count}")
print(f"\n  Expected LLM calls: {len(eval_data) * len(metrics)} = {len(eval_data) * len(metrics)}")
print(f"  Actual LLM calls: {mock_client.call_count}")

if mock_client.call_count == len(eval_data) * len(metrics):
    print(f"\n? SUCCESS: LLM was called the correct number of times!")
else:
    print(f"\n? FAIL: LLM call count mismatch!")

if passed > 0:
    print(f"? SUCCESS: At least one evaluation passed!")
else:
    print(f"? FAIL: No evaluations passed!")

print(f"{'='*80}")
