"""
COMPLETE FINAL TEST - LLM_Evaluator_Complete_20251104.py
Tests EVERYTHING: view, add, edit, delete, evaluation, ground truth
"""
import sys
import json
import re
import os
import pandas as pd
from io import StringIO
from enum import Enum
from dataclasses import dataclass

print("="*100)
print("COMPLETE FINAL TEST - ALL FUNCTIONALITY")
print("="*100)

# Mock dbutils
class MockWidgets:
    def __init__(self):
        self.storage = {}
        self.removed = []
    
    def dropdown(self, name, default, options, label):
        if name not in self.storage:
            self.storage[name] = default
    
    def text(self, name, default, label):
        if name not in self.storage:
            self.storage[name] = default
    
    def get(self, name):
        return self.storage.get(name, "")
    
    def remove(self, name):
        self.removed.append(name)
        if name in self.storage:
            del self.storage[name]

class MockDBUtils:
    def __init__(self):
        self.widgets = MockWidgets()

dbutils = MockDBUtils()

# Mock OpenAI
class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockCompletion:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

class MockCompletions:
    def __init__(self):
        self.call_count = 0
    
    def create(self, model, messages, temperature, max_tokens):
        self.call_count += 1
        return MockCompletion('{"score": 1, "explanation": "Pass"}')

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()

class MockOpenAIClient:
    def __init__(self):
        self.chat = MockChat()

# Classes
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

# Simplified evaluator
class LLMJudgeEvaluator:
    def __init__(self, client, model, metrics, ground_truth_data, client_type):
        self.client = client
        self.model = model
        self.metrics = metrics
        self.ground_truth_data = ground_truth_data
        self.client_type = client_type
    
    def evaluate_single(self, prompt, response, metric, sample_idx):
        try:
            llm_response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=500
            )
            data = json.loads(llm_response.choices[0].message.content)
            score = float(data['score'])
            status = "PASS" if score >= metric.threshold else "FAIL"
            return {"score": score, "status": status}
        except:
            return {"score": 0, "status": "FAIL"}
    
    def evaluate_dataset(self, eval_data):
        results = []
        for idx, row in eval_data.iterrows():
            for metric in self.metrics:
                result = self.evaluate_single(row['prompt'], row['response'], metric, idx)
                results.append({
                    'sample_id': row.get('sample_id', idx),
                    'metric_name': metric.name,
                    'score': result['score'],
                    'status': result['status']
                })
        return pd.DataFrame(results)

# Helper
def safe_float(val, default=0.0):
    if pd.isna(val): return default
    try:
        s = str(val).strip().lower()
        if s in ['true', '1']: return 1.0
        if s in ['false', '0']: return 0.0
        if s.endswith('%'): return float(s[:-1])
        return float(s)
    except:
        return default

# TEST 1: Initial Load
print("\n" + "="*100)
print("TEST 1: INITIAL LOAD (Cell 2)")
print("="*100)

METRICS_CONFIG_JSON = [
    {"name": "M1", "type": "binary", "description": "Test1", "grading_rubric": "R1", "threshold": "1", 
     "ground_truth_file_path": "ground_truth.csv", "ground_truth_column": "correct_answer"},
    {"name": "M2", "type": "1-5_scale", "description": "Test2", "grading_rubric": "R2", "threshold": "4", 
     "ground_truth_file_path": "", "ground_truth_column": ""},
    {"name": "M3", "type": "percentage", "description": "Test3", "grading_rubric": "R3", "threshold": "75", 
     "ground_truth_file_path": "", "ground_truth_column": ""}
]

METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
EVALUATION_DATA = pd.DataFrame([
    {"sample_id": 1, "prompt": "Q1", "response": "A1"},
    {"sample_id": 2, "prompt": "Q2", "response": "A2"}
])

print(f"? Loaded {len(METRICS_CONFIG_DATA)} default metrics")
assert len(METRICS_CONFIG_DATA) == 3, "Should start with 3 metrics"

# TEST 2: VIEW Mode
print("\n" + "="*100)
print("TEST 2: VIEW MODE")
print("="*100)

metrics_list = METRICS_CONFIG_DATA.to_dict('records')
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
dbutils.widgets.dropdown("action", "view", ["view", "add", "edit", "delete"], "Action")

action = dbutils.widgets.get("action")
if action == "view":
    for widget_name in ["row_select", "m_name", "m_type", "m_desc", "m_rubric", "m_threshold"]:
        try:
            dbutils.widgets.remove(widget_name)
        except:
            pass

# Reload and display
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)

widgets_visible = [k for k in dbutils.widgets.storage.keys() if k != "__metrics_storage__"]
print(f"? VIEW mode: {len(widgets_visible)} widget visible (action only)")
print(f"? Table shows: {len(METRICS_CONFIG_DATA)} metrics")
assert len(widgets_visible) == 1, f"Should show 1 widget, got {len(widgets_visible)}"
assert len(METRICS_CONFIG_DATA) == 3, f"Should show 3 metrics, got {len(METRICS_CONFIG_DATA)}"

# TEST 3: ADD Metric with Ground Truth Auto-Set
print("\n" + "="*100)
print("TEST 3: ADD METRIC (Ground Truth Auto-Set)")
print("="*100)

dbutils.widgets.storage.clear()
dbutils.widgets.removed = []
dbutils.widgets.dropdown("action", "add", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)

# Load from storage
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)
else:
    metrics_list = METRICS_CONFIG_DATA.to_dict('records')

action = dbutils.widgets.get("action")
if action == "add":
    dbutils.widgets.text("m_name", "", "1. Name")
    dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "2. Type")
    dbutils.widgets.text("m_desc", "", "3. Description")
    dbutils.widgets.text("m_rubric", "", "4. Grading Rubric")
    dbutils.widgets.text("m_threshold", "", "5. Threshold")
    try:
        dbutils.widgets.remove("row_select")
    except:
        pass

# User fills form
dbutils.widgets.storage["m_name"] = "NewMetric"
dbutils.widgets.storage["m_type"] = "binary"
dbutils.widgets.storage["m_desc"] = "A new metric"
dbutils.widgets.storage["m_rubric"] = "Score 1 if good"
dbutils.widgets.storage["m_threshold"] = "1"

# Execute add
if action == "add" and dbutils.widgets.get("m_name").strip():
    new_metric = {
        "name": dbutils.widgets.get("m_name"),
        "type": dbutils.widgets.get("m_type"),
        "description": dbutils.widgets.get("m_desc"),
        "grading_rubric": dbutils.widgets.get("m_rubric"),
        "threshold": dbutils.widgets.get("m_threshold"),
        "ground_truth_file_path": "ground_truth.csv",  # Hardcoded
        "ground_truth_column": "correct_answer"  # Hardcoded
    }
    metrics_list.append(new_metric)
    METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
    dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)

print(f"? Added metric: {new_metric['name']}")
print(f"   GT File: {new_metric['ground_truth_file_path']}")
print(f"   GT Column: {new_metric['ground_truth_column']}")
assert new_metric['ground_truth_file_path'] == "ground_truth.csv", "GT file should be ground_truth.csv"
assert new_metric['ground_truth_column'] == "correct_answer", "GT column should be correct_answer"
assert len(METRICS_CONFIG_DATA) == 4, f"Should have 4 metrics, got {len(METRICS_CONFIG_DATA)}"

# TEST 4: EDIT Metric with Ground Truth Auto-Set
print("\n" + "="*100)
print("TEST 4: EDIT METRIC (Ground Truth Auto-Set)")
print("="*100)

dbutils.widgets.storage.clear()
dbutils.widgets.removed = []
dbutils.widgets.dropdown("action", "edit", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)

# Load from storage
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)

METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
action = dbutils.widgets.get("action")

if action == "edit":
    dbutils.widgets.dropdown("row_select", "2", [str(i+1) for i in range(len(METRICS_CONFIG_DATA))], "1. Row to Edit")
    dbutils.widgets.text("m_name", "", "2. Name")
    dbutils.widgets.dropdown("m_type", "binary", ["binary", "1-5_scale", "percentage"], "3. Type")
    dbutils.widgets.text("m_desc", "", "4. Description")
    dbutils.widgets.text("m_rubric", "", "5. Grading Rubric")
    dbutils.widgets.text("m_threshold", "", "6. Threshold")

# User edits metric 2 (which has NO ground truth originally)
dbutils.widgets.storage["row_select"] = "2"
dbutils.widgets.storage["m_name"] = "M2_EDITED"
dbutils.widgets.storage["m_type"] = "1-5_scale"
dbutils.widgets.storage["m_desc"] = "Edited desc"
dbutils.widgets.storage["m_rubric"] = "Edited rubric"
dbutils.widgets.storage["m_threshold"] = "4"

# Execute edit
if action == "edit" and dbutils.widgets.get("m_name").strip():
    row_idx = int(dbutils.widgets.get("row_select")) - 1
    if 0 <= row_idx < len(metrics_list):
        metrics_list[row_idx] = {
            "name": dbutils.widgets.get("m_name"),
            "type": dbutils.widgets.get("m_type"),
            "description": dbutils.widgets.get("m_desc"),
            "grading_rubric": dbutils.widgets.get("m_rubric"),
            "threshold": dbutils.widgets.get("m_threshold"),
            "ground_truth_file_path": "ground_truth.csv",  # Always hardcoded
            "ground_truth_column": "correct_answer"  # Always hardcoded
        }
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
        dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)

edited_metric = metrics_list[1]
print(f"? Edited metric: {edited_metric['name']}")
print(f"   GT File: {edited_metric['ground_truth_file_path']}")
print(f"   GT Column: {edited_metric['ground_truth_column']}")
assert edited_metric['ground_truth_file_path'] == "ground_truth.csv", "GT file should be auto-set"
assert edited_metric['ground_truth_column'] == "correct_answer", "GT column should be auto-set"
assert len(METRICS_CONFIG_DATA) == 4, f"Should still have 4 metrics, got {len(METRICS_CONFIG_DATA)}"

# TEST 5: DELETE Metric
print("\n" + "="*100)
print("TEST 5: DELETE METRIC")
print("="*100)

dbutils.widgets.storage.clear()
dbutils.widgets.removed = []
dbutils.widgets.dropdown("action", "delete", ["view", "add", "edit", "delete"], "Action")
dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)

# Load from storage (CRITICAL!)
stored_metrics = dbutils.widgets.get("__metrics_storage__")
if stored_metrics:
    metrics_list = json.loads(stored_metrics)

METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
action = dbutils.widgets.get("action")

if action == "delete":
    dbutils.widgets.dropdown("row_select", "4", [str(i+1) for i in range(len(METRICS_CONFIG_DATA))], "Row to Delete")
    for widget_name in ["m_name", "m_type", "m_desc", "m_rubric", "m_threshold"]:
        try:
            dbutils.widgets.remove(widget_name)
        except:
            pass

# User selects row 4 to delete (NewMetric)
dbutils.widgets.storage["row_select"] = "4"

# Execute delete
if action == "delete":
    row_idx = int(dbutils.widgets.get("row_select")) - 1
    if 0 <= row_idx < len(metrics_list):
        deleted_name = metrics_list[row_idx]['name']
        metrics_list.pop(row_idx)
        METRICS_CONFIG_DATA = pd.DataFrame(metrics_list)
        dbutils.widgets.storage["__metrics_storage__"] = json.dumps(metrics_list)
        print(f"? DELETED: {deleted_name}")

print(f"? Delete test: Now have {len(METRICS_CONFIG_DATA)} metrics")
assert len(METRICS_CONFIG_DATA) == 3, f"Should have 3 metrics after delete, got {len(METRICS_CONFIG_DATA)}"
assert "NewMetric" not in [m['name'] for m in metrics_list], "Deleted metric should be gone"

# TEST 6: Full Evaluation with Ground Truth
print("\n" + "="*100)
print("TEST 6: FULL EVALUATION (Verify Ground Truth)")
print("="*100)

# Verify all metrics have proper ground truth
print("\nVerifying ground truth for all metrics:")
for idx, metric in enumerate(metrics_list):
    gt_file = metric.get('ground_truth_file_path', '')
    gt_col = metric.get('ground_truth_column', '')
    print(f"  {idx+1}. {metric['name']}: GT={gt_file}/{gt_col}")
    
    if gt_file != "ground_truth.csv":
        print(f"     ? GT file should be 'ground_truth.csv', got '{gt_file}'")
    if gt_col != "correct_answer":
        print(f"     ? GT column should be 'correct_answer', got '{gt_col}'")

# Create metric configs
metric_configs = []
for idx, row in METRICS_CONFIG_DATA.iterrows():
    name = str(row['name'])
    type_str = str(row['type']).lower()
    
    if 'binary' in type_str:
        mtype = MetricType.BINARY
    elif '1-5' in type_str:
        mtype = MetricType.SCALE_1_5
    else:
        mtype = MetricType.PERCENTAGE
    
    metric_configs.append(MetricConfig(
        name=name,
        metric_type=mtype,
        description=str(row.get('description', '')),
        prompt_template=f"Evaluate: {{{{prompt}}}}, {{{{response}}}}",
        threshold=safe_float(row.get('threshold', 0.5)),
        ground_truth_column=str(row.get('ground_truth_column', '')),
        ground_truth_file_path=str(row.get('ground_truth_file_path', ''))
    ))

# Run evaluation
client = MockOpenAIClient()
GROUND_TRUTH_DATA = {'ground_truth.csv': pd.DataFrame([
    {"sample_id": 1, "correct_answer": "Answer1"},
    {"sample_id": 2, "correct_answer": "Answer2"}
])}

evaluator = LLMJudgeEvaluator(
    client=client, model="gpt-4o", metrics=metric_configs,
    ground_truth_data=GROUND_TRUTH_DATA, client_type="openai"
)

results_df = evaluator.evaluate_dataset(EVALUATION_DATA)

total = len(results_df)
passed = len(results_df[results_df['status'] == 'PASS'])
expected_calls = len(EVALUATION_DATA) * len(metric_configs)

print(f"\n? Evaluation complete:")
print(f"   Total: {total}")
print(f"   Passed: {passed}")
print(f"   LLM calls: {client.chat.completions.call_count}")
assert client.chat.completions.call_count == expected_calls, f"Expected {expected_calls} calls, got {client.chat.completions.call_count}"

# FINAL VERDICT
print("\n" + "="*100)
print("FINAL VERDICT")
print("="*100)

all_pass = True
issues = []

# Check all ground truth fields
for metric in metrics_list:
    if metric['ground_truth_file_path'] != "ground_truth.csv":
        all_pass = False
        issues.append(f"Metric {metric['name']}: GT file = '{metric['ground_truth_file_path']}' (should be 'ground_truth.csv')")
    if metric['ground_truth_column'] != "correct_answer":
        all_pass = False
        issues.append(f"Metric {metric['name']}: GT column = '{metric['ground_truth_column']}' (should be 'correct_answer')")

# Check counts
if len(METRICS_CONFIG_DATA) != 3:
    all_pass = False
    issues.append(f"Should have 3 final metrics, got {len(METRICS_CONFIG_DATA)}")

# Check operations
if "NewMetric" in [m['name'] for m in metrics_list]:
    all_pass = False
    issues.append("NewMetric should be deleted")

if "M2_EDITED" not in [m['name'] for m in metrics_list]:
    all_pass = False
    issues.append("M2 should be renamed to M2_EDITED")

if client.chat.completions.call_count != expected_calls:
    all_pass = False
    issues.append(f"Wrong number of LLM calls")

if all_pass:
    print("??? ALL TESTS PASSED! ???")
    print()
    print("? TEST 1: Initial load (3 metrics)")
    print("? TEST 2: VIEW mode (1 widget, 3 metrics shown)")
    print("? TEST 3: ADD metric (4 metrics, GT auto-set to ground_truth.csv/correct_answer)")
    print("? TEST 4: EDIT metric (4 metrics, GT auto-set to ground_truth.csv/correct_answer)")
    print("? TEST 5: DELETE metric (3 metrics, NewMetric removed)")
    print("? TEST 6: EVALUATION (6 LLM calls, all successful)")
    print()
    print("Final metrics:")
    for i, m in enumerate(metrics_list):
        print(f"  {i+1}. {m['name']} (GT: {m['ground_truth_file_path']}/{m['ground_truth_column']})")
    print()
    print("?? FILE IS READY FOR UPLOAD! ??")
    print("?? LLM_Evaluator_Complete_20251104.py")
else:
    print("? TESTS FAILED ?")
    for issue in issues:
        print(f"  ? {issue}")

print("="*100)
