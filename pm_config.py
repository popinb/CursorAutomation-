# PM Configuration File - Simple Settings for Non-Technical Users
# =============================================================================
# INSTRUCTIONS: Change the values below to configure your evaluation
# =============================================================================

# =============================================================================
# BASIC SETTINGS - Change these to match your project
# =============================================================================

# Your experiment name (will appear in MLflow UI)
EXPERIMENT_NAME = "my_zillow_evaluation"  # Change to your project name

# Your data file path (CSV file with prompts and responses)
DATA_SOURCE = "/workspace/my_data.csv"  # Change to your CSV file path

# =============================================================================
# METRICS SETTINGS - Turn metrics ON/OFF and set pass/fail thresholds
# =============================================================================

# Turn metrics ON (True) or OFF (False)
ENABLE_METRICS = {
    "response_quality": True,           # Is the response helpful and relevant? (True/False)
    "personalization_accuracy": True,   # Does it use user info correctly? (True/False)
    "helpfulness": True,                # How helpful is the response? (True/False)
}

# Set what score counts as "PASS" for each metric
METRIC_THRESHOLDS = {
    "response_quality": 1.0,           # 1 = pass, 0 = fail (don't change)
    "personalization_accuracy": 1.0,   # 1 = pass, 0 = fail (don't change)
    "helpfulness": 3.0,                # 3+ = pass, 1-2 = fail (can change to 2.0, 3.0, 4.0)
}

# =============================================================================
# JUDGE MODEL SETTINGS - Choose which AI model to use for evaluation
# =============================================================================

# Choose 1 or 2 models (more models = more reliable but slower)
JUDGE_MODELS = ["gpt-4o"]  # Options: ["gpt-4o"] or ["gpt-4o", "gpt-4o-mini"]

# How many evaluations to run at once (don't change unless you have issues)
MAX_CONCURRENCY = 2  # Keep as 2 (or change to 1 if you get errors)

# =============================================================================
# ADVANCED SETTINGS - Usually don't need to change these
# =============================================================================

# Column names in your CSV file (only change if your CSV has different column names)
PROMPT_COLUMN = "prompt"          # Column with user questions
RESPONSE_COLUMN = "response"      # Column with AI responses
USER_PROFILE_COLUMN = "user_profile"  # Column with user info (optional)

# =============================================================================
# HELPER FUNCTIONS - Don't change these
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
    print(f"Judge Models: {JUDGE_MODELS}")
    print(f"Enabled Metrics: {get_enabled_metrics()}")
    print(f"Max Concurrency: {MAX_CONCURRENCY}")

# =============================================================================
# QUICK SETUP FUNCTIONS - Use these for common configurations
# =============================================================================

def setup_basic_evaluation():
    """Set up basic evaluation with common metrics."""
    global ENABLE_METRICS
    ENABLE_METRICS = {
        "response_quality": True,
        "helpfulness": True,
    }
    print("✅ Basic evaluation setup complete!")

def setup_full_evaluation():
    """Set up full evaluation with all metrics."""
    global ENABLE_METRICS
    ENABLE_METRICS = {
        "response_quality": True,
        "personalization_accuracy": True,
        "helpfulness": True,
    }
    print("✅ Full evaluation setup complete!")

def setup_reliable_evaluation():
    """Set up evaluation with multiple judge models for reliability."""
    global JUDGE_MODELS
    JUDGE_MODELS = ["gpt-4o", "gpt-4o-mini"]
    global MAX_CONCURRENCY
    MAX_CONCURRENCY = 1
    print("✅ Reliable evaluation setup complete!")

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

"""
HOW TO USE THIS FILE:

1. CHANGE BASIC SETTINGS:
   - EXPERIMENT_NAME: Your project name
   - DATA_SOURCE: Path to your CSV file

2. ENABLE/DISABLE METRICS:
   - Set to True to enable, False to disable
   - Adjust thresholds if needed

3. CHOOSE JUDGE MODELS:
   - Single model: ["gpt-4o"] (faster)
   - Multiple models: ["gpt-4o", "gpt-4o-mini"] (more reliable)

4. USE QUICK SETUPS:
   - setup_basic_evaluation() - Common metrics only
   - setup_full_evaluation() - All metrics
   - setup_reliable_evaluation() - Multiple judges

5. IN YOUR NOTEBOOK:
   - Import: from pm_config import *
   - Check settings: print_config()
   - Use the variables in your code

EXAMPLE:
```python
from pm_config import *
print_config()  # See your settings
setup_full_evaluation()  # Enable all metrics
```

TIPS:
- Start with basic evaluation
- Use single judge model for testing
- Use multiple judges for important evaluations
- Keep thresholds as they are unless you have specific needs
"""