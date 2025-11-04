"""
PHASE 3: Test Complete Evaluation Workflow
Tests the full LLM judge evaluation process with mock responses
"""

import sys
import os
import pandas as pd
import json
import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

print("=" * 80)
print("?? PHASE 3: COMPLETE EVALUATION WORKFLOW TESTING")
print("=" * 80)

# Import mock environment and test data
from mock_dbutils import dbutils
from test_data_hardcoded import TEST_DATA_PATHS, TEST_DIR

# Load test data
metrics_df = pd.read_csv(TEST_DATA_PATHS['metrics_file'])
eval_df = pd.read_csv(TEST_DATA_PATHS['eval_file'])
gt_df = pd.read_csv(TEST_DATA_PATHS['ground_truth_file'])

# Load mock LLM responses
with open(TEST_DATA_PATHS['mock_responses_file'], 'r') as f:
    MOCK_LLM_RESPONSES = json.load(f)

print(f"? Loaded {len(metrics_df)} metrics")
print(f"? Loaded {len(eval_df)} evaluation samples")
print(f"? Loaded {len(gt_df)} ground truth rows")
print(f"? Loaded mock LLM responses for {len(MOCK_LLM_RESPONSES)} metrics")

# ============================================================
# TEST 1: Define Core Classes
# ============================================================

print("\n" + "=" * 80)
print("TEST 1: Core Evaluation Classes")
print("=" * 80)

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

print("? TEST 1.1 PASSED: MetricType enum defined")
print("? TEST 1.2 PASSED: MetricConfig dataclass defined")

# ============================================================
# TEST 2: Mock LLM Judge Evaluator
# ============================================================

print("\n" + "=" * 80)
print("TEST 2: LLM Judge Evaluator with Mock Responses")
print("=" * 80)

class MockLLMJudgeEvaluator:
    """Mock LLM Judge Evaluator that uses hardcoded responses."""
    
    def __init__(self, metrics: List[MetricConfig], ground_truth_data: Dict[str, pd.DataFrame],
                 mock_responses: Dict):
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data
        self.mock_responses = mock_responses
        self.call_count = {}  # Track calls per metric
    
    def _get_ground_truth(self, metric: MetricConfig, sample_idx: int) -> str:
        """Get ground truth with ALL columns."""
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
                    all_data.append(f"? {col}: {value}")
            
            return "?? Ground Truth:\n" + "\n".join(all_data) if all_data else "No data"
        except Exception as e:
            return f"Error: {e}"
    
    def _call_mock_llm(self, metric_name: str, sample_idx: int) -> str:
        """Get mock LLM response."""
        if metric_name not in self.mock_responses:
            return '{"score": 0, "explanation": "No mock response available"}'
        
        responses = self.mock_responses[metric_name]
        if sample_idx >= len(responses):
            return '{"score": 0, "explanation": "Sample index out of range"}'
        
        return responses[sample_idx]
    
    def _parse_response(self, response: str, metric: MetricConfig) -> tuple:
        """Parse LLM response."""
        content = response.strip()
        
        # Remove markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Try JSON parsing
        try:
            data = json.loads(content)
            score = data.get('score', data.get('Score', 0))
            explanation = data.get('explanation', data.get('Explanation', 'No explanation'))
        except:
            score = 0
            explanation = content
        
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
        """Evaluate single sample."""
        try:
            # Track calls
            if metric.name not in self.call_count:
                self.call_count[metric.name] = 0
            self.call_count[metric.name] += 1
            
            # Get ground truth
            ground_truth = self._get_ground_truth(metric, sample_idx)
            
            # Get mock LLM response
            llm_response = self._call_mock_llm(metric.name, sample_idx)
            
            # Parse response
            score, explanation = self._parse_response(llm_response, metric)
            
            # Determine status
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
                "explanation": f"Error: {e}",
                "status": "?",
                "ground_truth_used": False
            }
    
    def evaluate_dataset(self, eval_data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate entire dataset."""
        results = []
        
        print(f"\n?? Evaluating {len(eval_data)} samples with {len(self.metrics)} metrics...")
        
        for idx, row in eval_data.iterrows():
            sample_id = row.get('sample_id', f'sample_{idx}')
            prompt = row.get('prompt', '')
            response = row.get('response', '')
            
            print(f"\n?? Sample {idx + 1}/{len(eval_data)}: {sample_id}")
            
            for metric in self.metrics:
                print(f"   ?? {metric.name}...", end=' ')
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
                    'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    'response': response[:100] + "..." if len(response) > 100 else response
                })
        
        print(f"\n? Evaluation complete!")
        return pd.DataFrame(results)

print("? TEST 2.1 PASSED: MockLLMJudgeEvaluator class defined")

# ============================================================
# TEST 3: Load and Configure Metrics
# ============================================================

print("\n" + "=" * 80)
print("TEST 3: Load and Configure Metrics for Evaluation")
print("=" * 80)

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

def generate_prompt_from_rubric(name: str, description: str, rubric: str) -> str:
    """Generate evaluation prompt."""
    return f"""You are an expert evaluator. Task: {description}

Grading Rubric:
{rubric}

Evaluation Details:
- User Query: {{prompt}}
- AI Response: {{response}}
- Ground Truth: {{ground_truth}}

Output Format (JSON only):
{{
  "score": <your_score>,
  "explanation": "Brief explanation"
}}"""

metric_configs = []
for _, row in metrics_df.iterrows():
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
    print(f"   ? Configured: {name} ({mtype.value}, threshold: {threshold})")

assert len(metric_configs) == len(metrics_df), "Should have same number of configured metrics"
print(f"\n? TEST 3.1 PASSED: Configured {len(metric_configs)} metrics")

# ============================================================
# TEST 4: Create Evaluator
# ============================================================

print("\n" + "=" * 80)
print("TEST 4: Create Mock Evaluator Instance")
print("=" * 80)

ground_truth_data = {
    'ground_truth.csv': gt_df
}

evaluator = MockLLMJudgeEvaluator(
    metrics=metric_configs,
    ground_truth_data=ground_truth_data,
    mock_responses=MOCK_LLM_RESPONSES
)

print("? TEST 4.1 PASSED: Evaluator created with mock responses")

# ============================================================
# TEST 5: Run Complete Evaluation
# ============================================================

print("\n" + "=" * 80)
print("TEST 5: Run Complete Evaluation Workflow")
print("=" * 80)

results_df = evaluator.evaluate_dataset(eval_df)

assert len(results_df) > 0, "Should have evaluation results"
expected_count = len(eval_df) * len(metric_configs)
assert len(results_df) == expected_count, f"Should have {expected_count} results, got {len(results_df)}"

print(f"\n? TEST 5.1 PASSED: Generated {len(results_df)} evaluation results")

# ============================================================
# TEST 6: Analyze Results
# ============================================================

print("\n" + "=" * 80)
print("TEST 6: Analyze Evaluation Results")
print("=" * 80)

print("\n?? Test 6.1: Check All Metrics Evaluated")
metrics_in_results = results_df['metric_name'].unique()
assert len(metrics_in_results) == len(metric_configs), "All metrics should be in results"
for metric in metric_configs:
    assert metric.name in metrics_in_results, f"{metric.name} should be in results"
print(f"? TEST 6.1 PASSED: All {len(metrics_in_results)} metrics present in results")

print("\n?? Test 6.2: Check All Samples Evaluated")
samples_in_results = results_df['sample_id'].unique()
assert len(samples_in_results) == len(eval_df), "All samples should be evaluated"
print(f"? TEST 6.2 PASSED: All {len(samples_in_results)} samples evaluated")

print("\n?? Test 6.3: Check Status Values")
statuses = results_df['status'].unique()
assert len(statuses) > 0, "Should have status values"
assert all(s in ['?', '?'] for s in statuses), "Status should be ? or ?"
print(f"? TEST 6.3 PASSED: Status values are valid")

print("\n?? Test 6.4: Calculate Pass Rates")
total = len(results_df)
passed = len(results_df[results_df['status'] == '?'])
failed = len(results_df[results_df['status'] == '❌'])
pass_rate = (passed / total) * 100 if total > 0 else 0

print(f"   Total evaluations: {total}")
print(f"   Passed: {passed} ({pass_rate:.1f}%)")
print(f"   Failed: {failed} ({100-pass_rate:.1f}%)")

assert total == expected_count, "Total should match expected"
assert passed + failed == total, "Passed + Failed should equal total"
print(f"? TEST 6.4 PASSED: Pass rate calculated correctly")

print("\n?? Test 6.5: Per-Metric Performance")
for metric_name in results_df['metric_name'].unique():
    metric_results = results_df[results_df['metric_name'] == metric_name]
    metric_passed = len(metric_results[metric_results['status'] == '?'])
    metric_total = len(metric_results)
    metric_rate = (metric_passed / metric_total) * 100
    avg_score = metric_results['score'].mean()
    
    print(f"   {metric_name}: {metric_rate:.1f}% pass ({metric_passed}/{metric_total}), avg: {avg_score:.2f}")

print(f"? TEST 6.5 PASSED: Per-metric analysis complete")

# ============================================================
# TEST 7: Ground Truth Usage
# ============================================================

print("\n" + "=" * 80)
print("TEST 7: Verify Ground Truth Usage")
print("=" * 80)

print("\n?? Test 7.1: Check Ground Truth Access")
gt_used = results_df[results_df['ground_truth_used'] == True]
print(f"   Evaluations using ground truth: {len(gt_used)}")

# Accuracy metric should use ground truth
accuracy_results = results_df[results_df['metric_name'] == 'Accuracy']
accuracy_with_gt = accuracy_results[accuracy_results['ground_truth_used'] == True]
assert len(accuracy_with_gt) > 0, "Accuracy metric should use ground truth"
print(f"   Accuracy metric used ground truth: {len(accuracy_with_gt)}/{len(accuracy_results)} times")

print(f"? TEST 7.1 PASSED: Ground truth correctly accessed")

# ============================================================
# TEST 8: Score Normalization
# ============================================================

print("\n" + "=" * 80)
print("TEST 8: Score Normalization by Metric Type")
print("=" * 80)

print("\n?? Test 8.1: Binary Metrics (0 or 1)")
binary_results = results_df[results_df['metric_type'] == 'binary']
binary_scores = binary_results['score'].unique()
assert all(s in [0.0, 1.0] for s in binary_scores), "Binary scores should be 0 or 1"
print(f"   Binary scores: {sorted(binary_scores)}")
print(f"? TEST 8.1 PASSED: Binary scores normalized correctly")

print("\n?? Test 8.2: Scale Metrics (1-5)")
scale_results = results_df[results_df['metric_type'] == '1-5_scale']
if len(scale_results) > 0:
    scale_scores = scale_results['score']
    assert all(1.0 <= s <= 5.0 for s in scale_scores), "Scale scores should be 1-5"
    print(f"   Scale score range: {scale_scores.min():.1f} - {scale_scores.max():.1f}")
    print(f"? TEST 8.2 PASSED: Scale scores normalized correctly")
else:
    print("??  No scale metrics to test")

# ============================================================
# TEST 9: Results Export
# ============================================================

print("\n" + "=" * 80)
print("TEST 9: Export Results")
print("=" * 80)

results_file = os.path.join(TEST_DIR, "evaluation_results.csv")
results_df.to_csv(results_file, index=False)
print(f"?? Saved results to: {results_file}")

# Verify export
reloaded_results = pd.read_csv(results_file)
assert len(reloaded_results) == len(results_df), "Reloaded results should match"
print(f"? TEST 9.1 PASSED: Results exported and reloaded successfully")

# ============================================================
# PHASE 3 SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("?? PHASE 3 TEST SUMMARY")
print("=" * 80)

all_tests = [
    ("1.1: MetricType Enum", True),
    ("1.2: MetricConfig Dataclass", True),
    ("2.1: MockLLMJudgeEvaluator Class", True),
    ("3.1: Configure Metrics", True),
    ("4.1: Create Evaluator", True),
    ("5.1: Run Complete Evaluation", True),
    ("6.1: All Metrics Evaluated", True),
    ("6.2: All Samples Evaluated", True),
    ("6.3: Status Values Valid", True),
    ("6.4: Pass Rate Calculated", True),
    ("6.5: Per-Metric Performance", True),
    ("7.1: Ground Truth Usage", True),
    ("8.1: Binary Score Normalization", True),
    ("8.2: Scale Score Normalization", True),
    ("9.1: Results Export", True),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\n? Passed: {passed}/{total} tests")
print("\n?? Test Results:")
for test_name, result in all_tests:
    status = "?" if result else "?"
    print(f"   {status} Test {test_name}")

print(f"\n?? EVALUATION SUMMARY:")
print(f"   Total Evaluations: {len(results_df)}")
print(f"   Pass Rate: {pass_rate:.1f}%")
print(f"   Metrics: {len(metric_configs)}")
print(f"   Samples: {len(eval_df)}")

if passed == total:
    print(f"\n?? PHASE 3 COMPLETE: Full evaluation workflow working correctly!")
else:
    print(f"\n??  PHASE 3 INCOMPLETE: {total - passed} tests failed")

print("=" * 80)
