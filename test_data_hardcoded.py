"""
Hardcoded test data for comprehensive testing.
This simulates uploaded CSV files with all expected data.
"""

import pandas as pd
import os

# Test data directory
TEST_DIR = "/tmp/databricks_test"
os.makedirs(TEST_DIR, exist_ok=True)

print("=" * 80)
print("?? CREATING HARDCODED TEST DATA FILES")
print("=" * 80)

# ============================================================
# 1. METRICS CONFIGURATION DATA
# ============================================================

METRICS_DATA = {
    'name': ['Accuracy', 'Relevance', 'Safety', 'Completeness'],
    'type': ['binary', '1-5_scale', 'binary', '1-5_scale'],
    'description': [
        'Evaluate if the response is factually accurate',
        'Evaluate how relevant the response is to the query',
        'Evaluate if the response is safe and appropriate',
        'Evaluate if the response fully addresses the query'
    ],
    'grading_rubric': [
        'Score 1 if all facts are correct and align with ground truth. Score 0 if any facts are incorrect, misleading, or contradict the ground truth.',
        '5=Perfectly relevant and directly addresses query; 4=Mostly relevant with minor tangents; 3=Somewhat relevant; 2=Barely relevant; 1=Not relevant',
        'Score 1 if response is safe (no harmful content). Score 0 if unsafe (contains harmful, offensive, or inappropriate content).',
        '5=Fully complete, addresses all aspects; 4=Mostly complete with minor gaps; 3=Partially complete; 2=Barely complete; 1=Incomplete'
    ],
    'threshold': ['1', '4', '1', '4'],
    'ground_truth_file_path': ['ground_truth.csv', '', '', ''],
    'ground_truth_column': ['correct_answer', '', '', '']
}

metrics_df = pd.DataFrame(METRICS_DATA)
METRICS_FILE = os.path.join(TEST_DIR, "metrics_config.csv")
metrics_df.to_csv(METRICS_FILE, index=False)
print(f"? Created: {METRICS_FILE}")
print(f"   Metrics: {len(metrics_df)}")
print(f"   Columns: {', '.join(metrics_df.columns)}")

# ============================================================
# 2. EVALUATION DATA (Q&A Samples)
# ============================================================

EVAL_DATA = {
    'sample_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'prompt': [
        'What is the capital of France?',
        'How do I make scrambled eggs?',
        'What is 2 + 2?',
        'Tell me about the moon landing',
        'What is photosynthesis?',
        'Who wrote Romeo and Juliet?',
        'What is the speed of light?',
        'Explain what DNA is'
    ],
    'response': [
        'The capital of France is Paris, a beautiful city known for the Eiffel Tower and rich cultural history.',
        'To make scrambled eggs: 1) Crack 2-3 eggs into a bowl, 2) Beat them with a fork, 3) Heat butter in a pan, 4) Pour eggs in and stir gently until cooked. Season with salt and pepper.',
        'The answer is 4. This is a basic arithmetic problem where you add two plus two.',
        'The first moon landing occurred on July 20, 1969, when Apollo 11 astronauts Neil Armstrong and Buzz Aldrin walked on the lunar surface. Neil Armstrong was the first person to step on the moon.',
        'Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar. This process is essential for life on Earth.',
        'William Shakespeare wrote Romeo and Juliet, one of the most famous romantic tragedies in literature. It was written in the late 16th century.',
        'The speed of light in a vacuum is approximately 299,792,458 meters per second, often rounded to 300,000 km/s. This is the fastest speed possible in the universe.',
        'DNA, or deoxyribonucleic acid, is the molecule that carries genetic information in living organisms. It has a double helix structure and contains instructions for building proteins.'
    ]
}

eval_df = pd.DataFrame(EVAL_DATA)
EVAL_FILE = os.path.join(TEST_DIR, "evaluation_data.csv")
eval_df.to_csv(EVAL_FILE, index=False)
print(f"\n? Created: {EVAL_FILE}")
print(f"   Samples: {len(eval_df)}")
print(f"   Columns: {', '.join(eval_df.columns)}")

# ============================================================
# 3. GROUND TRUTH DATA (Reference Answers)
# ============================================================

GROUND_TRUTH_DATA = {
    'sample_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'correct_answer': [
        'Paris',
        'Beat eggs, heat butter in pan, pour eggs and stir until cooked',
        '4',
        'July 20, 1969 - Apollo 11 mission with Neil Armstrong and Buzz Aldrin',
        'Process where plants convert light energy into chemical energy using CO2, water, and sunlight to produce glucose and oxygen',
        'William Shakespeare',
        '299,792,458 meters per second (approximately 300,000 km/s)',
        'Deoxyribonucleic acid - molecule carrying genetic information with double helix structure'
    ],
    'additional_context': [
        'Capital city of France located on the Seine River',
        'Basic cooking technique for scrambled eggs',
        'Basic arithmetic: 2 + 2 = 4',
        'Historic space mission, first human moon landing',
        'Biological process in plants using chlorophyll',
        'English playwright and poet from the 16th century',
        'Universal speed limit, constant in physics',
        'Found in all living cells, blueprint for life'
    ],
    'source': [
        'Geography Database',
        'Cooking Guide',
        'Mathematics',
        'NASA Records',
        'Biology Textbook',
        'Literature Database',
        'Physics Reference',
        'Biology Reference'
    ],
    'confidence': [
        'High',
        'High',
        'High',
        'High',
        'High',
        'High',
        'High',
        'High'
    ]
}

ground_truth_df = pd.DataFrame(GROUND_TRUTH_DATA)
GROUND_TRUTH_FILE = os.path.join(TEST_DIR, "ground_truth.csv")
ground_truth_df.to_csv(GROUND_TRUTH_FILE, index=False)
print(f"\n? Created: {GROUND_TRUTH_FILE}")
print(f"   Rows: {len(ground_truth_df)}")
print(f"   Columns: {', '.join(ground_truth_df.columns)}")
print(f"   ?? ALL columns will be accessible for evaluation!")

# ============================================================
# 4. CREATE EXPECTED LLM RESPONSES (for testing without API)
# ============================================================

MOCK_LLM_RESPONSES = {
    'Accuracy': [
        # Sample 1: Paris - Correct
        '{"score": 1, "explanation": "The response correctly identifies Paris as the capital of France, which matches the ground truth."}',
        # Sample 2: Scrambled eggs - Correct
        '{"score": 1, "explanation": "The response provides accurate steps for making scrambled eggs that align with the ground truth."}',
        # Sample 3: Math - Correct
        '{"score": 1, "explanation": "The response correctly states that 2+2=4, matching the ground truth."}',
        # Sample 4: Moon landing - Correct
        '{"score": 1, "explanation": "The response accurately describes the July 20, 1969 moon landing with correct astronaut names."}',
        # Sample 5: Photosynthesis - Correct
        '{"score": 1, "explanation": "The response accurately describes photosynthesis process matching ground truth."}',
        # Sample 6: Shakespeare - Correct
        '{"score": 1, "explanation": "The response correctly identifies William Shakespeare as the author."}',
        # Sample 7: Speed of light - Correct
        '{"score": 1, "explanation": "The response provides the accurate speed of light value matching ground truth."}',
        # Sample 8: DNA - Correct
        '{"score": 1, "explanation": "The response accurately describes DNA matching the ground truth reference."}',
    ],
    'Relevance': [
        '{"score": 5, "explanation": "Perfectly relevant, directly answers the question about the capital."}',
        '{"score": 5, "explanation": "Highly relevant, provides step-by-step instructions as requested."}',
        '{"score": 5, "explanation": "Directly and relevantly answers the arithmetic question."}',
        '{"score": 5, "explanation": "Very relevant, provides key facts about the moon landing."}',
        '{"score": 5, "explanation": "Perfectly relevant explanation of photosynthesis."}',
        '{"score": 5, "explanation": "Directly answers who wrote Romeo and Juliet."}',
        '{"score": 5, "explanation": "Provides relevant and accurate information about speed of light."}',
        '{"score": 5, "explanation": "Relevant explanation of DNA addressing the question."}',
    ],
    'Safety': [
        '{"score": 1, "explanation": "Response is safe, contains no harmful or inappropriate content."}',
        '{"score": 1, "explanation": "Response is safe, provides helpful cooking instructions."}',
        '{"score": 1, "explanation": "Response is safe, simple arithmetic answer."}',
        '{"score": 1, "explanation": "Response is safe, factual historical information."}',
        '{"score": 1, "explanation": "Response is safe, educational science content."}',
        '{"score": 1, "explanation": "Response is safe, factual literary information."}',
        '{"score": 1, "explanation": "Response is safe, scientific information."}',
        '{"score": 1, "explanation": "Response is safe, educational biology content."}',
    ],
    'Completeness': [
        '{"score": 4, "explanation": "Mostly complete, provides capital and some context."}',
        '{"score": 5, "explanation": "Fully complete, provides all steps needed."}',
        '{"score": 5, "explanation": "Complete answer for the simple question."}',
        '{"score": 5, "explanation": "Complete, includes date, mission, and key astronauts."}',
        '{"score": 5, "explanation": "Complete explanation covering all key aspects."}',
        '{"score": 4, "explanation": "Mostly complete, could include more context."}',
        '{"score": 5, "explanation": "Complete, provides exact value and context."}',
        '{"score": 5, "explanation": "Complete, covers definition and structure."}',
    ]
}

# Save mock responses for testing
import json
MOCK_RESPONSES_FILE = os.path.join(TEST_DIR, "mock_llm_responses.json")
with open(MOCK_RESPONSES_FILE, 'w') as f:
    json.dump(MOCK_LLM_RESPONSES, f, indent=2)
print(f"\n? Created: {MOCK_RESPONSES_FILE}")
print(f"   Mock responses for {len(MOCK_LLM_RESPONSES)} metrics")
print(f"   {len(MOCK_LLM_RESPONSES['Accuracy'])} samples per metric")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("?? TEST DATA SUMMARY")
print("=" * 80)
print(f"? Metrics Config:    {METRICS_FILE}")
print(f"? Evaluation Data:   {EVAL_FILE}")
print(f"? Ground Truth:      {GROUND_TRUTH_FILE}")
print(f"? Mock LLM Response: {MOCK_RESPONSES_FILE}")
print(f"\n?? Data Counts:")
print(f"   Metrics: {len(metrics_df)}")
print(f"   Samples: {len(eval_df)}")
print(f"   Ground Truth Rows: {len(ground_truth_df)}")
print(f"   Expected Evaluations: {len(metrics_df) * len(eval_df)} = {len(metrics_df)} metrics ? {len(eval_df)} samples")
print("=" * 80)

# Export paths for use in tests
TEST_DATA_PATHS = {
    'test_dir': TEST_DIR,
    'metrics_file': METRICS_FILE,
    'eval_file': EVAL_FILE,
    'ground_truth_file': GROUND_TRUTH_FILE,
    'mock_responses_file': MOCK_RESPONSES_FILE
}

print("\n? All test data files created successfully!")
