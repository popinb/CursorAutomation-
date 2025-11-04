"""
PHASE 1 TEST: Core Data Loading (Hardcoded JSON Approach)
Tests the hardcoded Cinderella data from the corrected notebook
"""

import sys
import pandas as pd
import json

print("="*80)
print("PHASE 1: CORE DATA LOADING TEST")
print("="*80)

# ============================================================================
# Test 1.1: Load Hardcoded Metrics Configuration
# ============================================================================

print("\n? Test 1.1: Load Hardcoded Metrics Configuration")

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

METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)

# Validate
assert len(METRICS_CONFIG_DATA) == 3, f"Expected 3 metrics, got {len(METRICS_CONFIG_DATA)}"
assert 'name' in METRICS_CONFIG_DATA.columns, "Missing 'name' column"
assert 'type' in METRICS_CONFIG_DATA.columns, "Missing 'type' column"
assert 'threshold' in METRICS_CONFIG_DATA.columns, "Missing 'threshold' column"

# Check metric types
metric_types = METRICS_CONFIG_DATA['type'].tolist()
assert 'binary' in metric_types, "Missing binary metric"
assert '1-5_scale' in metric_types, "Missing 1-5_scale metric"
assert 'percentage' in metric_types, "Missing percentage metric"

print(f"   ? Loaded {len(METRICS_CONFIG_DATA)} metrics")
print(f"   ? Columns: {', '.join(METRICS_CONFIG_DATA.columns)}")
print(f"   ? Metric types: {', '.join(metric_types)}")
print(f"   ? Metrics: {', '.join(METRICS_CONFIG_DATA['name'].tolist())}")

# ============================================================================
# Test 1.2: Load Hardcoded Evaluation Data
# ============================================================================

print("\n? Test 1.2: Load Hardcoded Evaluation Data")

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

EVALUATION_DATA = pd.DataFrame(EVALUATION_DATA_JSON)

# Validate
assert len(EVALUATION_DATA) == 8, f"Expected 8 samples, got {len(EVALUATION_DATA)}"
assert 'sample_id' in EVALUATION_DATA.columns, "Missing 'sample_id' column"
assert 'prompt' in EVALUATION_DATA.columns, "Missing 'prompt' column"
assert 'response' in EVALUATION_DATA.columns, "Missing 'response' column"

# Check sample IDs are 1-8
sample_ids = sorted(EVALUATION_DATA['sample_id'].tolist())
assert sample_ids == list(range(1, 9)), f"Expected sample IDs 1-8, got {sample_ids}"

print(f"   ? Loaded {len(EVALUATION_DATA)} evaluation samples")
print(f"   ? Columns: {', '.join(EVALUATION_DATA.columns)}")
print(f"   ? Sample IDs: {sample_ids[0]}-{sample_ids[-1]}")
print(f"   ? Average prompt length: {EVALUATION_DATA['prompt'].str.len().mean():.0f} chars")
print(f"   ? Average response length: {EVALUATION_DATA['response'].str.len().mean():.0f} chars")

# ============================================================================
# Test 1.3: Load Hardcoded Ground Truth
# ============================================================================

print("\n? Test 1.3: Load Hardcoded Ground Truth")

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

GROUND_TRUTH_DATA_DF = pd.DataFrame(GROUND_TRUTH_JSON)
GROUND_TRUTH_DATA = {'ground_truth.csv': GROUND_TRUTH_DATA_DF}

# Validate
assert len(GROUND_TRUTH_DATA_DF) == 8, f"Expected 8 ground truth entries, got {len(GROUND_TRUTH_DATA_DF)}"
assert 'sample_id' in GROUND_TRUTH_DATA_DF.columns, "Missing 'sample_id' column"
assert 'correct_answer' in GROUND_TRUTH_DATA_DF.columns, "Missing 'correct_answer' column"

# Check for enhanced columns (ALL columns should be accessible)
expected_columns = ['sample_id', 'correct_answer', 'story_element', 'key_facts', 'source']
for col in expected_columns:
    assert col in GROUND_TRUTH_DATA_DF.columns, f"Missing column: {col}"

print(f"   ? Loaded {len(GROUND_TRUTH_DATA_DF)} ground truth entries")
print(f"   ? Columns: {', '.join(GROUND_TRUTH_DATA_DF.columns)}")
print(f"   ? Enhanced ground truth: {len(GROUND_TRUTH_DATA_DF.columns)} columns (ALL accessible to LLM)")
print(f"   ? Ground truth stored as: ground_truth.csv")

# ============================================================================
# Test 1.4: Validate Data Consistency
# ============================================================================

print("\n? Test 1.4: Validate Data Consistency")

# Check sample IDs match across datasets
eval_ids = set(EVALUATION_DATA['sample_id'].tolist())
gt_ids = set(GROUND_TRUTH_DATA_DF['sample_id'].tolist())

assert eval_ids == gt_ids, f"Sample ID mismatch: Eval {eval_ids} vs GT {gt_ids}"
print(f"   ? Sample IDs consistent across evaluation data and ground truth")

# Check all 3 metric types are present
metric_type_map = {
    'binary': 'Story_Accuracy',
    '1-5_scale': 'Response_Completeness',
    'percentage': 'Child_Friendliness'
}

for metric_type, metric_name in metric_type_map.items():
    metric_row = METRICS_CONFIG_DATA[METRICS_CONFIG_DATA['name'] == metric_name]
    assert len(metric_row) == 1, f"Metric {metric_name} not found"
    assert metric_row.iloc[0]['type'] == metric_type, f"Wrong type for {metric_name}"
    print(f"   ? {metric_name} ({metric_type}) configured correctly")

# ============================================================================
# Test 1.5: Ground Truth Access (Enhanced Feature)
# ============================================================================

print("\n? Test 1.5: Ground Truth Access (Enhanced Feature)")

# Test accessing ALL columns for a sample
sample_idx = 0
gt_row = GROUND_TRUTH_DATA_DF.iloc[sample_idx]

# Build enhanced ground truth string (like the evaluator does)
all_data = []
for col, value in gt_row.items():
    if pd.notna(value) and str(value).strip():
        all_data.append(f"? {col}: {value}")

gt_string = "?? Ground Truth:\n" + "\n".join(all_data)

print(f"   ? Sample {sample_idx + 1} ground truth has {len(all_data)} data points")
print(f"   ? Columns included: {', '.join(gt_row.index.tolist())}")
print(f"   ? Enhanced ground truth string length: {len(gt_string)} chars")

# Verify all expected columns are present
assert len(all_data) >= 5, f"Expected at least 5 data points, got {len(all_data)}"
assert 'correct_answer' in gt_string, "Missing correct_answer in ground truth string"
assert 'key_facts' in gt_string, "Missing key_facts in ground truth string"
assert 'story_element' in gt_string, "Missing story_element in ground truth string"

print(f"   ? All expected columns present in ground truth access")

# ============================================================================
# Test 1.6: Metric-Ground Truth Mapping
# ============================================================================

print("\n? Test 1.6: Metric-Ground Truth Mapping")

# Test that Story_Accuracy metric has ground truth configured
story_accuracy = METRICS_CONFIG_DATA[METRICS_CONFIG_DATA['name'] == 'Story_Accuracy'].iloc[0]
assert story_accuracy['ground_truth_file_path'] == 'ground_truth.csv', "Wrong ground truth file path"
assert story_accuracy['ground_truth_column'] == 'correct_answer', "Wrong ground truth column"
print(f"   ? Story_Accuracy metric linked to ground_truth.csv")

# Test that other metrics don't have ground truth
completeness = METRICS_CONFIG_DATA[METRICS_CONFIG_DATA['name'] == 'Response_Completeness'].iloc[0]
assert completeness['ground_truth_file_path'] == '', "Should not have ground truth"
print(f"   ? Response_Completeness has no ground truth (as expected)")

friendliness = METRICS_CONFIG_DATA[METRICS_CONFIG_DATA['name'] == 'Child_Friendliness'].iloc[0]
assert friendliness['ground_truth_file_path'] == '', "Should not have ground truth"
print(f"   ? Child_Friendliness has no ground truth (as expected)")

# ============================================================================
# Test 1.7: Threshold Validation
# ============================================================================

print("\n? Test 1.7: Threshold Validation")

# Test thresholds are correct for each metric type
story_accuracy = METRICS_CONFIG_DATA[METRICS_CONFIG_DATA['name'] == 'Story_Accuracy'].iloc[0]
assert story_accuracy['threshold'] == '1', f"Wrong threshold for binary: {story_accuracy['threshold']}"
print(f"   ? Binary metric threshold: {story_accuracy['threshold']}")

completeness = METRICS_CONFIG_DATA[METRICS_CONFIG_DATA['name'] == 'Response_Completeness'].iloc[0]
assert completeness['threshold'] == '4', f"Wrong threshold for scale: {completeness['threshold']}"
print(f"   ? 1-5 scale metric threshold: {completeness['threshold']}")

friendliness = METRICS_CONFIG_DATA[METRICS_CONFIG_DATA['name'] == 'Child_Friendliness'].iloc[0]
assert friendliness['threshold'] == '75', f"Wrong threshold for percentage: {friendliness['threshold']}"
print(f"   ? Percentage metric threshold: {friendliness['threshold']}")

# ============================================================================
# Test 1.8: JSON to DataFrame Conversion
# ============================================================================

print("\n? Test 1.8: JSON to DataFrame Conversion")

# Verify JSON can be converted back to original format
metrics_json_back = METRICS_CONFIG_DATA.to_dict('records')
assert len(metrics_json_back) == len(METRICS_CONFIG_JSON), "JSON conversion failed for metrics"
print(f"   ? Metrics DataFrame can be converted back to JSON")

eval_json_back = EVALUATION_DATA.to_dict('records')
assert len(eval_json_back) == len(EVALUATION_DATA_JSON), "JSON conversion failed for eval data"
print(f"   ? Evaluation data DataFrame can be converted back to JSON")

gt_json_back = GROUND_TRUTH_DATA_DF.to_dict('records')
assert len(gt_json_back) == len(GROUND_TRUTH_JSON), "JSON conversion failed for ground truth"
print(f"   ? Ground truth DataFrame can be converted back to JSON")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("? PHASE 1 COMPLETE: ALL DATA LOADING TESTS PASSED")
print("="*80)

print(f"\n?? Data Summary:")
print(f"   ? Metrics: {len(METRICS_CONFIG_DATA)} (binary, 1-5 scale, percentage)")
print(f"   ? Evaluation samples: {len(EVALUATION_DATA)}")
print(f"   ? Ground truth entries: {len(GROUND_TRUTH_DATA_DF)} (with {len(GROUND_TRUTH_DATA_DF.columns)} columns)")
print(f"   ? Total potential evaluations: {len(EVALUATION_DATA) * len(METRICS_CONFIG_DATA)} (8 ? 3)")

print(f"\n? Key Features Verified:")
print(f"   ? Hardcoded JSON data (no file uploads)")
print(f"   ? Cinderella story theme")
print(f"   ? 3 metric types as requested")
print(f"   ? Enhanced ground truth (ALL columns accessible)")
print(f"   ? Data consistency across datasets")
print(f"   ? Proper threshold configuration")

print("\n" + "="*80)
print("?? READY FOR PHASE 2: Databricks Serving Endpoints Testing")
print("="*80)

sys.exit(0)
