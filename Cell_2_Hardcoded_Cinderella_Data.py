# Databricks notebook source
# MAGIC %md
# MAGIC ## Cell 2: Hardcoded Data Configuration (Cinderella Story Theme)
# MAGIC 
# MAGIC **All data is hardcoded below - no file uploads needed!**
# MAGIC 
# MAGIC You can modify the JSON structures below to change:
# MAGIC - Evaluation samples (Q&A about Cinderella)
# MAGIC - Ground truth references
# MAGIC - Metrics configuration (3 metrics: Binary, 1-5 Scale, Percentage)

# COMMAND ----------

import pandas as pd
import os

print("=" * 80)
print("?? HARDCODED DATA - CINDERELLA STORY THEME")
print("=" * 80)
print("\n? No file uploads needed! All data is configured below.\n")

# ============================================================================
# HARDCODED METRICS CONFIGURATION (3 metrics as requested)
# ============================================================================

METRICS_CONFIG_JSON = [
    {
        "name": "Story_Accuracy",
        "type": "binary",  # Binary: 0 or 1
        "description": "Evaluate if the response is factually accurate about the Cinderella story",
        "grading_rubric": "Score 1 if all story facts are correct and align with the classic Cinderella tale. Score 0 if any facts are incorrect, made up, or contradict the original story.",
        "threshold": "1",
        "ground_truth_file_path": "ground_truth.csv",
        "ground_truth_column": "correct_answer"
    },
    {
        "name": "Response_Completeness",
        "type": "1-5_scale",  # 1-5 Scale
        "description": "Evaluate how complete and thorough the response is",
        "grading_rubric": "5=Fully complete, addresses all aspects comprehensively; 4=Mostly complete with minor gaps; 3=Partially complete, missing some details; 2=Barely complete, many gaps; 1=Incomplete or inadequate",
        "threshold": "4",
        "ground_truth_file_path": "",
        "ground_truth_column": ""
    },
    {
        "name": "Child_Friendliness",
        "type": "percentage",  # Percentage: 0-100
        "description": "Evaluate what percentage of the response is appropriate and understandable for children",
        "grading_rubric": "100%=Perfectly child-friendly language and content; 75%=Mostly appropriate with minor complex words; 50%=Somewhat child-friendly; 25%=Barely appropriate for children; 0%=Not suitable for children",
        "threshold": "75",
        "ground_truth_file_path": "",
        "ground_truth_column": ""
    }
]

print("?? METRICS CONFIGURATION:")
print("-" * 80)
for i, metric in enumerate(METRICS_CONFIG_JSON, 1):
    print(f"\n{i}. {metric['name']} ({metric['type']})")
    print(f"   Threshold: {metric['threshold']}")
    print(f"   Ground Truth: {'? Yes' if metric['ground_truth_file_path'] else '? No'}")
print("\n" + "-" * 80)

# ============================================================================
# HARDCODED EVALUATION DATA (Cinderella Story Q&A)
# ============================================================================

EVALUATION_DATA_JSON = [
    {
        "sample_id": 1,
        "prompt": "Who is Cinderella and what is her story about?",
        "response": "Cinderella is a young girl who lives with her mean stepmother and two stepsisters. They make her do all the chores and treat her badly. One day, with the help of her Fairy Godmother, she goes to the prince's ball. She loses her glass slipper at midnight, and the prince finds her by trying the slipper on every girl in the kingdom."
    },
    {
        "sample_id": 2,
        "prompt": "What did the Fairy Godmother turn into a carriage?",
        "response": "The Fairy Godmother used her magic wand to turn a big orange pumpkin into a beautiful golden carriage so Cinderella could go to the ball in style."
    },
    {
        "sample_id": 3,
        "prompt": "What happened at midnight?",
        "response": "When the clock struck twelve at midnight, Cinderella had to run away from the ball because the Fairy Godmother's magic would wear off. In her hurry, she lost one of her glass slippers on the palace steps."
    },
    {
        "sample_id": 4,
        "prompt": "How did the prince find Cinderella?",
        "response": "The prince searched the whole kingdom with the glass slipper, trying it on every girl. When he came to Cinderella's house, the slipper fit her perfectly! This proved she was the mysterious princess from the ball."
    },
    {
        "sample_id": 5,
        "prompt": "What animals helped Cinderella?",
        "response": "Cinderella had many animal friends including mice, birds, and a dog. The mice were her best friends and they helped her make a dress for the ball. The birds also helped her with her chores around the house."
    },
    {
        "sample_id": 6,
        "prompt": "What was Cinderella wearing at the ball?",
        "response": "Cinderella wore a magnificent ball gown that sparkled like stars. Her Fairy Godmother created it with magic, along with glass slippers on her feet. She looked so beautiful that everyone at the ball, including the prince, couldn't take their eyes off her."
    },
    {
        "sample_id": 7,
        "prompt": "Who were Cinderella's stepsisters?",
        "response": "Anastasia and Drizella were Cinderella's two stepsisters. They were mean and jealous of Cinderella's kindness and beauty. They made her do all the housework and never let her rest."
    },
    {
        "sample_id": 8,
        "prompt": "What is the moral of the Cinderella story?",
        "response": "The story teaches us that kindness and goodness are always rewarded. Even when life is hard and people are mean to you, if you stay kind and never give up hope, good things will happen. It also shows that true beauty comes from being a good person inside."
    }
]

print("\n?? EVALUATION DATA (Cinderella Q&A):")
print("-" * 80)
print(f"Total samples: {len(EVALUATION_DATA_JSON)}")
print("\nSample questions:")
for i, item in enumerate(EVALUATION_DATA_JSON[:3], 1):
    print(f"{i}. {item['prompt']}")
print("   ... and 5 more questions")
print("-" * 80)

# ============================================================================
# HARDCODED GROUND TRUTH DATA (Reference Answers for Story_Accuracy)
# ============================================================================

GROUND_TRUTH_JSON = [
    {
        "sample_id": 1,
        "correct_answer": "Cinderella is a kind young girl mistreated by her stepmother and stepsisters. With help from her Fairy Godmother, she attends a royal ball, loses her glass slipper at midnight, and is found by the prince who searches for her with the slipper.",
        "story_element": "Main plot summary",
        "key_facts": "stepmother, stepsisters, Fairy Godmother, ball, glass slipper, midnight, prince",
        "source": "Classic Cinderella fairy tale"
    },
    {
        "sample_id": 2,
        "correct_answer": "A pumpkin",
        "story_element": "Magic transformation",
        "key_facts": "pumpkin turned into carriage",
        "source": "Classic Cinderella fairy tale"
    },
    {
        "sample_id": 3,
        "correct_answer": "Cinderella had to leave the ball because the magic spell would break at midnight. She ran away and lost her glass slipper on the steps.",
        "story_element": "Midnight deadline",
        "key_facts": "midnight, magic ends, lost slipper, palace steps",
        "source": "Classic Cinderella fairy tale"
    },
    {
        "sample_id": 4,
        "correct_answer": "The prince searched the kingdom trying the glass slipper on every maiden until he found Cinderella, whose foot fit the slipper perfectly.",
        "story_element": "Finding Cinderella",
        "key_facts": "glass slipper, kingdom search, perfect fit",
        "source": "Classic Cinderella fairy tale"
    },
    {
        "sample_id": 5,
        "correct_answer": "Mice and birds were Cinderella's main animal friends who helped her with chores and making her dress.",
        "story_element": "Animal helpers",
        "key_facts": "mice, birds, helped with chores and dress",
        "source": "Classic Cinderella fairy tale"
    },
    {
        "sample_id": 6,
        "correct_answer": "A beautiful ball gown created by the Fairy Godmother's magic, with glass slippers.",
        "story_element": "Ball outfit",
        "key_facts": "magical ball gown, glass slippers",
        "source": "Classic Cinderella fairy tale"
    },
    {
        "sample_id": 7,
        "correct_answer": "Anastasia and Drizella (names vary by version, but they are the two mean stepsisters who mistreated Cinderella).",
        "story_element": "Antagonists",
        "key_facts": "two stepsisters, mean, jealous",
        "source": "Classic Cinderella fairy tale"
    },
    {
        "sample_id": 8,
        "correct_answer": "Be kind and patient even in difficult times, and goodness will be rewarded. Inner beauty and character matter more than outer appearances.",
        "story_element": "Moral lesson",
        "key_facts": "kindness rewarded, inner beauty, patience",
        "source": "Classic Cinderella fairy tale"
    }
]

print("\n?? GROUND TRUTH DATA:")
print("-" * 80)
print(f"Total ground truth entries: {len(GROUND_TRUTH_JSON)}")
print(f"Columns available: sample_id, correct_answer, story_element, key_facts, source")
print("?? ALL columns will be accessible during evaluation!")
print("-" * 80)

# ============================================================================
# Convert JSON to DataFrames
# ============================================================================

print("\n?? Converting JSON to DataFrames...")

# Convert to pandas DataFrames
METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
EVALUATION_DATA = pd.DataFrame(EVALUATION_DATA_JSON)
GROUND_TRUTH_DATA_DF = pd.DataFrame(GROUND_TRUTH_JSON)

# Store ground truth in dictionary format (as expected by evaluator)
GROUND_TRUTH_DATA = {
    'ground_truth.csv': GROUND_TRUTH_DATA_DF
}

print("? DataFrames created successfully!")

# ============================================================================
# Display Data Samples
# ============================================================================

print("\n" + "=" * 80)
print("?? METRICS CONFIGURATION PREVIEW:")
print("=" * 80)
display(METRICS_CONFIG_DATA)

print("\n" + "=" * 80)
print("?? EVALUATION DATA PREVIEW (First 3 samples):")
print("=" * 80)
display(EVALUATION_DATA.head(3))

print("\n" + "=" * 80)
print("?? GROUND TRUTH DATA PREVIEW (First 3 entries):")
print("=" * 80)
display(GROUND_TRUTH_DATA_DF.head(3))

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("? DATA LOADING COMPLETE!")
print("=" * 80)
print(f"\n?? Summary:")
print(f"   ? Metrics configured: {len(METRICS_CONFIG_DATA)}")
print(f"     - Binary (Story_Accuracy): threshold = {METRICS_CONFIG_DATA.iloc[0]['threshold']}")
print(f"     - 1-5 Scale (Response_Completeness): threshold = {METRICS_CONFIG_DATA.iloc[1]['threshold']}")
print(f"     - Percentage (Child_Friendliness): threshold = {METRICS_CONFIG_DATA.iloc[2]['threshold']}%")
print(f"\n   ? Evaluation samples: {len(EVALUATION_DATA)}")
print(f"   ? Ground truth entries: {len(GROUND_TRUTH_DATA_DF)}")
print(f"   ? Ground truth columns: {len(GROUND_TRUTH_DATA_DF.columns)}")
print(f"\n?? Theme: Cinderella Story")
print(f"?? All data is hardcoded - no file uploads needed!")
print(f"\n?? To modify data:")
print(f"   1. Edit the JSON variables above:")
print(f"      - METRICS_CONFIG_JSON (for metrics)")
print(f"      - EVALUATION_DATA_JSON (for Q&A samples)")
print(f"      - GROUND_TRUTH_JSON (for reference answers)")
print(f"   2. Re-run this cell")
print(f"   3. Continue with Cell 3 onwards")
print("\n" + "=" * 80)

# ============================================================================
# Global Variables Ready for Next Cells
# ============================================================================
# These variables are now available for Cell 3 onwards:
# - EVALUATION_DATA (DataFrame)
# - METRICS_CONFIG_DATA (DataFrame) 
# - GROUND_TRUTH_DATA (dict with 'ground_truth.csv' key)

print("\n? Variables ready for next cells:")
print("   ? EVALUATION_DATA")
print("   ? METRICS_CONFIG_DATA")
print("   ? GROUND_TRUTH_DATA")
print("\n?? You can now run Cell 3 onwards!")
