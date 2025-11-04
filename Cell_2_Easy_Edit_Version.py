# Databricks notebook source
# MAGIC %md
# MAGIC ## Cell 2: Data Configuration - Cinderella Story (No Uploads Needed!)
# MAGIC 
# MAGIC **? All data is configured below in JSON format - easy to edit!**
# MAGIC 
# MAGIC ### ?? How to modify:
# MAGIC 1. Scroll down to the three JSON sections
# MAGIC 2. Edit the data directly
# MAGIC 3. Re-run this cell
# MAGIC 4. Continue with rest of notebook

# COMMAND ----------

import pandas as pd

# ============================================================================
# ?? EDIT SECTION 1: METRICS CONFIGURATION
# ============================================================================
# Define your 3 metrics here (binary, 1-5 scale, percentage)

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
# ?? EDIT SECTION 2: EVALUATION DATA (Q&A Samples)
# ============================================================================
# Your test questions and responses about Cinderella

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

# ============================================================================
# ?? EDIT SECTION 3: GROUND TRUTH DATA (Reference Answers)
# ============================================================================
# Correct answers for Story_Accuracy metric
# Note: ALL columns below will be available to the LLM judge!

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

# ============================================================================
# ?? PROCESSING SECTION (Don't edit below unless you know what you're doing)
# ============================================================================

print("=" * 80)
print("?? CINDERELLA STORY EVALUATION - HARDCODED DATA")
print("=" * 80)
print("\n? No file uploads needed! All data is configured in this cell.\n")

# Convert JSON to pandas DataFrames
METRICS_CONFIG_DATA = pd.DataFrame(METRICS_CONFIG_JSON)
EVALUATION_DATA = pd.DataFrame(EVALUATION_DATA_JSON)
GROUND_TRUTH_DATA_DF = pd.DataFrame(GROUND_TRUTH_JSON)

# Store ground truth in dictionary format (as expected by evaluator)
GROUND_TRUTH_DATA = {
    'ground_truth.csv': GROUND_TRUTH_DATA_DF
}

# Display summaries
print("?? METRICS CONFIGURATION:")
print("-" * 80)
for i, row in METRICS_CONFIG_DATA.iterrows():
    gt_indicator = "? with ground truth" if row['ground_truth_file_path'] else "? no ground truth"
    print(f"{i+1}. {row['name']} ({row['type']}) - threshold: {row['threshold']} {gt_indicator}")
print("-" * 80)

print(f"\n?? EVALUATION DATA: {len(EVALUATION_DATA)} samples")
print(f"?? GROUND TRUTH: {len(GROUND_TRUTH_DATA_DF)} entries with {len(GROUND_TRUTH_DATA_DF.columns)} columns")
print(f"    Columns: {', '.join(GROUND_TRUTH_DATA_DF.columns.tolist())}")

print("\n" + "=" * 80)
print("?? METRICS TABLE:")
print("=" * 80)
display(METRICS_CONFIG_DATA[['name', 'type', 'threshold', 'description']])

print("\n" + "=" * 80)
print("?? EVALUATION SAMPLES (First 3):")
print("=" * 80)
display(EVALUATION_DATA[['sample_id', 'prompt', 'response']].head(3))

print("\n" + "=" * 80)
print("?? GROUND TRUTH (First 3):")
print("=" * 80)
display(GROUND_TRUTH_DATA_DF[['sample_id', 'correct_answer', 'story_element']].head(3))

print("\n" + "=" * 80)
print("? DATA READY!")
print("=" * 80)
print(f"\n?? Summary:")
print(f"   ? {len(METRICS_CONFIG_DATA)} metrics configured")
print(f"   ? {len(EVALUATION_DATA)} evaluation samples loaded")
print(f"   ? {len(GROUND_TRUTH_DATA_DF)} ground truth entries loaded")
print(f"   ? {len(GROUND_TRUTH_DATA_DF.columns)} columns in ground truth (ALL accessible!)")
print(f"\n?? Theme: Cinderella Story")
print(f"? All data is hardcoded - no uploads needed!")
print(f"\n?? To modify data:")
print(f"   1. Edit the JSON sections at the top of this cell")
print(f"   2. Re-run this cell")
print(f"   3. Continue with Cell 3 onwards")
print(f"\n?? Ready for Cell 3!")
print("=" * 80)

# Variables exported for next cells:
# - EVALUATION_DATA
# - METRICS_CONFIG_DATA
# - GROUND_TRUTH_DATA
