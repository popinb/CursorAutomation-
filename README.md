# Zillow Judge Evaluator for OpenAI Evals

This project converts a Custom GPT–based "LLM as a Judge" evaluator into a Python implementation compatible with the OpenAI Evals framework. The evaluator assesses LLM responses across 10 metrics using ground truth documents and buyability profiles.

## Overview

The Zillow Judge Evaluator replicates the behavior of a Custom GPT that evaluates LLM responses about home-buying and financial guidance. It uses two knowledge sources:
1. **Golden Responses** (`godenresponsealpha.docx`) - Reference answers for various buyability questions
2. **Buyability Profiles** (`buyabilityprofile.rtf`) - User financial profiles for personalization

## Evaluation Metrics

The evaluator assesses responses across 10 metrics:

1. **Personalization Accuracy** (Accurate/Inaccurate) - Matches user-specific figures
2. **Context-Based Personalization** (1-5) - Percentage of relevant customizations included
3. **Next-Step Identification** (Present/Not Present) - Actionable guidance provided
4. **Assumption Listing** (True/False) - Explicit statement of assumptions
5. **Assumption Trust** (1-5) - Transparency about limitations and gaps
6. **Calculation Accuracy** (True/False) - Mathematical correctness verification
7. **Faithfulness to Ground Truth** (True/False) - Alignment with Zillow guidance
8. **Overall Accuracy** (True/False) - Holistic correctness assessment
9. **Structured Presentation** (1-5) - Quality of formatting and organization
10. **Coherence** (True/False) - Logical consistency and flow
11. **Completeness** (1-5) - Coverage of expected question elements

## File Structure

```
├── zillow_judge_evaluator.py     # Core evaluator implementation
├── evals_wrapper.py              # Standalone wrapper with examples
├── oai_evals_zillow_judge.py     # OpenAI Evals compatible interface
├── assets/
│   ├── golden_responses.json     # Processed ground truth responses
│   └── buyability_profiles.json  # Processed user profiles
└── README.md                     # This documentation
```

## Installation

1. **Setup Python Environment**
   ```bash
   python3 -m venv eval_env
   source eval_env/bin/activate  # On Windows: eval_env\Scripts\activate
   pip install openai
   ```

2. **Clone/Download Files**
   ```bash
   # Place all .py files and assets/ directory in your working directory
   ```

## Usage

### Standalone Usage

**Basic Evaluation:**
```python
from zillow_judge_evaluator import ZillowJudgeEvaluator

evaluator = ZillowJudgeEvaluator()

# Example evaluation
candidate_answer = """
Your personalized BuyAbility estimate is $318,431, based on your specific financial profile.
This calculation uses your $90,000 annual income, $200 monthly debts, $18,000 down payment...
"""

question = "What factors were considered to calculate my Buyability?"
user_profile = {
    "annual_income": 90000,
    "monthly_debts": 200,
    "down_payment": 18000,
    "credit_score": "660-719"
}

result = evaluator.evaluate(candidate_answer, question, user_profile)
print(result)
```

**Using the Wrapper:**
```python
from evals_wrapper import ZillowEvalsWrapper

wrapper = ZillowEvalsWrapper()
wrapper.run_example_evaluation()  # Runs built-in example
```

### OpenAI Evals Framework Compatible

```python
from oai_evals_zillow_judge import ZillowJudgeOAIEval

evaluator = ZillowJudgeOAIEval()

# Single sample evaluation
sample = {
    "input": {
        "candidate_answer": "Your response here...",
        "question": "What factors were considered?",
        "user_profile": {"annual_income": 90000, "monthly_debts": 200}
    }
}

result = evaluator.eval_sample(sample)
print(f"Overall Score: {result['score']:.3f}")

# Batch evaluation
samples = [sample1, sample2, sample3]
batch_results = evaluator.run_eval(samples)
```

## Running the Examples

**Test the Core Evaluator:**
```bash
source eval_env/bin/activate
python zillow_judge_evaluator.py
```

**Test the Wrapper:**
```bash
python evals_wrapper.py
```

**Test OpenAI Evals Compatibility:**
```bash
python oai_evals_zillow_judge.py
```

## Input Format

### Required Fields
- `candidate_answer` (string): The LLM response to evaluate
- `question` (string, optional): The original question
- `user_profile` (dict, optional): User's buyability profile

### User Profile Format
```json
{
    "annual_income": 90000,
    "monthly_debts": 200,
    "down_payment": 18000,
    "credit_score": "660-719",
    "preferred_monthly_payment": 2500,
    "comfortable_max_monthly_payment": 3000
}
```

## Output Format

### Evaluation Table
The evaluator returns a markdown table with scores and detailed justifications:

```
| Metric | Score | Justification |
|--------|-------|---------------|
| Personalization Accuracy | Accurate | The response demonstrates excellent personalization... |
| Context based Personalization | 4 | The response includes 6 out of 10 relevant elements... |
| ... | ... | ... |
```

### Structured Output (OAI Evals)
```json
{
    "score": 0.85,
    "evaluation_table": "| Metric | Score | Justification |...",
    "detailed_scores": {
        "personalization_accuracy": {
            "score": "Accurate",
            "justification": "...",
            "numeric_score": 1.0
        }
    },
    "metadata": {
        "question": "...",
        "user_profile_provided": true,
        "answer_length": 145
    }
}
```

## Key Features

### Deterministic Evaluation
- Uses regex patterns and mathematical verification
- No LLM calls for scoring (only for the candidate answers being evaluated)
- Reproducible results across runs

### Comprehensive Coverage
- **Personalization**: Matches user-specific financial data
- **Mathematical Accuracy**: Verifies DTI calculations and financial math
- **Content Quality**: Assesses structure, coherence, and completeness
- **Practical Value**: Checks for actionable guidance and transparency

### Flexible Integration
- Standalone Python classes
- OpenAI Evals framework compatible
- Batch processing capabilities
- Detailed logging and error handling

## Customization

### Adding New Profiles
Edit `assets/buyability_profiles.json`:
```json
[
    {
        "profile_id": "Profile2",
        "down_payment": 50000,
        "credit_score": "excellent",
        "annual_income": 120000,
        "monthly_debts": 300
    }
]
```

### Modifying Ground Truth
Edit `assets/golden_responses.json` to add new question types or update response templates.

### Adjusting Scoring
Modify the evaluation methods in `ZillowJudgeEvaluator` class:
- `_evaluate_personalization_accuracy()`
- `_evaluate_calculation_accuracy()`
- etc.

## Troubleshooting

**FileNotFoundError for assets:**
- Ensure `assets/` directory is in the same location as the Python files
- Check that JSON files are properly formatted

**Import Errors:**
- Activate the virtual environment: `source eval_env/bin/activate`
- Install required packages: `pip install openai`

**Low Scores:**
- Verify user profile data matches the response content
- Check that responses include specific financial figures
- Ensure responses address the original question

## Integration with CI/CD

The evaluator can be integrated into automated testing pipelines:

```python
# Example test script
def test_llm_responses():
    evaluator = ZillowJudgeOAIEval()
    test_cases = load_test_cases()
    
    results = evaluator.run_eval(test_cases)
    
    assert results['average_score'] >= 0.7, f"Average score too low: {results['average_score']}"
    assert results['failed_evaluations'] == 0, "Some evaluations failed"
```

## Contributing

To extend the evaluator:
1. Add new metrics in the `ZillowJudgeEvaluator` class
2. Update the `evaluate()` method to include new metrics
3. Modify the output table format in `_format_evaluation_table()`
4. Add corresponding tests and examples

## License

This implementation is provided as-is for educational and evaluation purposes. Ensure compliance with your organization's policies when using financial data and evaluation criteria.