# PM Configuration File - Easy to Modify
# This file contains simple configuration options for Product Managers

# =============================================================================
# BASIC CONFIGURATION - Modify these settings
# =============================================================================

# Your experiment name (will appear in MLflow UI)
EXPERIMENT_NAME = "my_zillow_evaluation"

# Your data file path (CSV format)
DATA_SOURCE = "/workspace/my_data.csv"

# Column names in your CSV file
PROMPT_COLUMN = "prompt"          # Column with user questions
RESPONSE_COLUMN = "response"      # Column with AI responses
USER_PROFILE_COLUMN = "user_profile"  # Column with user info (optional)

# LLM models to use for evaluation (you can use multiple for more reliable results)
JUDGE_MODELS = ["gpt-4o"]  # Options: "gpt-4o", "gpt-4o-mini", "gpt-4", etc.

# How many evaluations to run at once (reduce if you hit rate limits)
MAX_CONCURRENCY = 2

# =============================================================================
# METRICS CONFIGURATION - Add/remove metrics here
# =============================================================================

# Set to True to enable each metric, False to disable
ENABLE_METRICS = {
    "response_quality": True,           # Is the response helpful and relevant?
    "personalization_accuracy": True,   # Does it use user info correctly?
    "helpfulness": True,                # How helpful is the response (1-5)?
    "safety_compliance": False,         # Does it follow safety guidelines?
    "real_estate_accuracy": False,      # Is real estate info accurate?
}

# Customize thresholds (what score counts as "pass")
METRIC_THRESHOLDS = {
    "response_quality": 1.0,           # 1 = pass, 0 = fail
    "personalization_accuracy": 1.0,   # 1 = pass, 0 = fail
    "helpfulness": 3.0,                # 3+ = pass (out of 5)
    "safety_compliance": 1.0,          # 1 = pass, 0 = fail
    "real_estate_accuracy": 3.0,       # 3+ = pass (out of 5)
}

# =============================================================================
# ADVANCED CONFIGURATION - Usually don't need to change these
# =============================================================================

# Response generation method
RESPONSE_MODEL = "GOLDEN_RESPONSE"  # Options: "GOLDEN_RESPONSE", "FIRST_CALL", or specific LLM model

# Output settings
SAVE_RESULTS = True
EXPORT_FORMAT = "csv"  # Options: "csv", "json"

# =============================================================================
# HELPER FUNCTIONS - Don't modify these
# =============================================================================

def get_enabled_metrics():
    """Get list of enabled metrics."""
    return [name for name, enabled in ENABLE_METRICS.items() if enabled]

def get_metric_threshold(metric_name):
    """Get threshold for a specific metric."""
    return METRIC_THRESHOLDS.get(metric_name, 1.0)

def print_config():
    """Print current configuration."""
    print("Current Configuration:")
    print("=" * 40)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Data Source: {DATA_SOURCE}")
    print(f"Judge Models: {JUDGE_MOD models}")
    print(f"Enabled Metrics: {get_enabled_metrics()}")
    print(f"Max Concurrency: {MAX_CONCURRENCY}")

# =============================================================================
# QUICK START EXAMPLES
# =============================================================================

# Example 1: Basic evaluation with default metrics
def setup_basic_evaluation():
    """Set up a basic evaluation with common metrics."""
    global ENABLE_METRICS
    ENABLE_METRICS = {
        "response_quality": True,
        "helpfulness": True,
        "safety_compliance": True,
    }
    print("Basic evaluation setup complete!")

# Example 2: Real estate focused evaluation
def setup_real_estate_evaluation():
    """Set up evaluation focused on real estate metrics."""
    global ENABLE_METRICS
    ENABLE_METRICS = {
        "response_quality": True,
        "personalization_accuracy": True,
        "real_estate_accuracy": True,
        "safety_compliance": True,
    }
    print("Real estate evaluation setup complete!")

# Example 3: High-quality evaluation with multiple judges
def setup_high_quality_evaluation():
    """Set up evaluation with multiple judge models for reliability."""
    global JUDGE_MODELS
    JUDGE_MODELS = ["gpt-4o", "gpt-4o-mini"]
    global MAX_CONCURRENCY
    MAX_CONCURRENCY = 1  # Slower but more reliable
    print("High-quality evaluation setup complete!")

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

"""
HOW TO USE THIS CONFIGURATION FILE:

1. MODIFY BASIC SETTINGS:
   - Change EXPERIMENT_NAME to your project name
   - Set DATA_SOURCE to your CSV file path
   - Adjust column names if needed

2. ENABLE/DISABLE METRICS:
   - Set ENABLE_METRICS values to True/False
   - Adjust METRIC_THRESHOLDS as needed

3. CHOOSE JUDGE MODELS:
   - Use single model: ["gpt-4o"]
   - Use multiple models: ["gpt-4o", "gpt-4o-mini"]
   - Adjust MAX_CONCURRENCY if needed

4. RUN QUICK SETUPS:
   - Call setup_basic_evaluation() for common metrics
   - Call setup_real_estate_evaluation() for real estate focus
   - Call setup_high_quality_evaluation() for reliability

5. USE IN NOTEBOOK:
   - Import: from pm_config import *
   - Call print_config() to see your settings
   - Use the variables in your evaluation code

EXAMPLE USAGE:
```python
from pm_config import *

# Print current config
print_config()

# Set up for real estate evaluation
setup_real_estate_evaluation()

# Use in evaluation
evaluator = LLMJudgeEvaluator(config, get_enabled_metrics())
```

TIPS:
- Start with basic evaluation and add more metrics gradually
- Test with small datasets first
- Use multiple judge models for important evaluations
- Adjust thresholds based on your quality requirements
"""