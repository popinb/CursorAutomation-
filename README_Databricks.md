# Databricks LLM Response Evaluator

A comprehensive evaluation system for LLM responses using Databricks notebooks, MLflow, and custom evaluation logic. This system implements the 11-metric evaluation framework for Zillow LLM responses.

## Overview

This system evaluates LLM responses across 11 key metrics:

1. **Personalization Accuracy** - Evaluates if personalization helps answer questions better
2. **Context-Based Personalization** - Scores personalization on a 1-5 scale
3. **Next-Step Identification** - Checks for actionable guidance
4. **Assumption Listing** - Verifies transparency of assumptions
5. **Assumption Trust** - Scores assumption transparency on a 1-5 scale
6. **Calculation Accuracy** - Validates mathematical calculations
7. **Faithfulness to Ground Truth** - Checks alignment with reference documents
8. **Overall Accuracy** - Holistic assessment of the response
9. **Structured Presentation** - Evaluates organization and formatting (1-5 scale)
10. **Coherence** - Checks logical consistency and flow
11. **Completeness** - Assesses coverage of user questions (1-5 scale)
12. **Fair Housing Classifier** - Verifies compliance with fair housing regulations

## Files Structure

```
├── databricks_notebook_ready.py          # Main Databricks notebook
├── databricks_llm_evaluator.py          # Complete Python implementation
├── databricks_llm_evaluator_notebook.py # Full-featured notebook with UI
├── databricks_requirements.txt           # Dependencies for Databricks
├── assets/
│   ├── golden_responses.json            # Sample golden responses
│   ├── buyability_profiles.json         # Sample user profiles
│   └── fair_housing_rules.json         # Fair housing compliance rules
└── README_Databricks.md                 # This file
```

## Setup Instructions

### 1. Install Dependencies

In your Databricks notebook, run:

```python
%pip install mlflow pandas numpy python-docx striprtf
```

### 2. Upload Ground Truth Files

Upload your three ground truth files to Databricks:
- `godenresponsealpha.docx` (Golden responses)
- `buyabilityprofile.rtf` (Buyability profiles)  
- `ZIllow_Fair_Housing_Classifier.docx` (Fair housing rules)

### 3. Load the Notebook

Copy the contents of `databricks_notebook_ready.py` into a new Databricks notebook.

## Usage

### Basic Evaluation

```python
# Initialize evaluator with your data
evaluator = DatabricksLLMEvaluator(
    golden_responses_data=your_golden_responses,
    buyability_profiles_data=your_buyability_profiles,
    fair_housing_rules_data=your_fair_housing_rules
)

# Evaluate a response
results = evaluator.evaluate_response(
    candidate_answer="Your LLM response here",
    question="Original question",
    user_profile={"buyability_score": 750, "monthly_payment": 2500}
)

# Display results
table = evaluator.format_evaluation_table(results)
print(table)
```

### Using the Main Function

```python
# Use the simplified evaluation function
results = evaluate_llm_response(
    candidate_answer="Your LLM response here",
    question="Original question",
    user_profile={"buyability_score": 750, "monthly_payment": 2500},
    golden_responses_data=your_golden_responses,
    buyability_profiles_data=your_buyability_profiles,
    fair_housing_rules_data=your_fair_housing_rules
)

print(results["formatted_table"])
```

### Batch Evaluation

```python
# Evaluate multiple responses
batch_data = [
    {
        "question": "What's my monthly payment?",
        "candidate_answer": "Based on your profile, your monthly payment would be $2,500.",
        "user_profile": {"buyability_score": 750, "monthly_payment": 2500}
    },
    # ... more responses
]

results = []
for data in batch_data:
    result = evaluate_llm_response(
        candidate_answer=data["candidate_answer"],
        question=data["question"],
        user_profile=data["user_profile"],
        golden_responses_data=your_golden_responses,
        buyability_profiles_data=your_buyability_profiles,
        fair_housing_rules_data=your_fair_housing_rules
    )
    results.append(result)
```

## MLflow Integration

The system automatically logs evaluation results to MLflow for tracking and analysis:

```python
# Log results to MLflow
evaluator.log_to_mlflow(
    evaluation_results=results,
    candidate_answer="Your response",
    question="Your question",
    user_profile=user_profile
)
```

## Output Format

The system generates a comprehensive evaluation table:

| Metric | Score | Justification |
|--------|-------|---------------|
| Personalization Accuracy | Accurate | Detailed explanation... |
| Context-Based Personalization | 4 | Detailed explanation... |
| Next-Step Identification | Present | Detailed explanation... |
| ... | ... | ... |

**Alpha evaluation: 85.5/100**

## Customization

### Adding New Metrics

To add new evaluation metrics, extend the `DatabricksLLMEvaluator` class:

```python
def _evaluate_new_metric(self, candidate_answer: str, scratchpad: str) -> EvaluationResult:
    # Your evaluation logic here
    return EvaluationResult(
        metric="New Metric",
        score=score_value,
        justification="Your justification"
    )
```

### Modifying Scoring Logic

Adjust the scoring algorithms in the individual evaluation methods to match your specific requirements.

### Ground Truth Integration

Modify the `_find_matching_ground_truth` method to implement more sophisticated matching logic based on your document structure.

## File Format Support

The system currently supports:
- **JSON**: Full support for structured data
- **DOCX**: Placeholder support (implement based on your needs)
- **RTF**: Placeholder support (implement based on your needs)

## Example Ground Truth Structure

### Golden Responses
```json
{
  "mortgage_calculation": {
    "question": "What's my monthly payment?",
    "response": "Based on your BuyAbility score of {buyability_score}...",
    "variables": ["buyability_score", "monthly_payment", "interest_rate"],
    "next_steps": ["apply for pre-approval", "schedule consultation"],
    "assumptions": ["30-year fixed-rate mortgage", "20% down payment"]
  }
}
```

### Buyability Profiles
```json
{
  "profile_1": {
    "buyability_score": 750,
    "monthly_payment": 2500,
    "interest_rate": 6.5,
    "loan_amount": 400000
  }
}
```

### Fair Housing Rules
```json
{
  "protected_classes": ["race", "color", "national_origin"],
  "prohibited_language": ["family with children", "single person"],
  "safe_language": ["credit score requirements", "income requirements"]
}
```

## Troubleshooting

### Common Issues

1. **File Loading Errors**: Ensure your files are properly uploaded and accessible
2. **Import Errors**: Verify all dependencies are installed
3. **MLflow Connection Issues**: Check your MLflow tracking URI configuration

### Debug Mode

Enable detailed logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Large Files**: For large ground truth documents, consider chunking or indexing
- **Batch Processing**: Use batch evaluation for multiple responses
- **Caching**: Implement caching for frequently accessed ground truth data

## Security Notes

- Never expose sensitive user data in logs
- Validate all input data before processing
- Use secure file handling practices

## Support

For issues or questions:
1. Check the example implementations
2. Review the error logs
3. Verify your data format matches the expected structure

## License

This system is designed for internal use and evaluation purposes.