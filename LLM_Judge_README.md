# 🏠 Zillow LLM-as-a-Judge Evaluation Framework

## 🚀 Quick Start Guide for Product Managers

This framework allows you to evaluate LLM responses using customizable metrics. Perfect for evaluating chatbots, search results, recommendations, or any text-based AI features.

## 📋 Prerequisites

1. **Databricks Access**: You need access to a Databricks workspace
2. **API Keys**: Required keys for OpenAI and Zillow services (stored in Databricks secrets)
3. **Data**: CSV file with prompts and responses to evaluate

## 🎯 5-Minute Setup

### Step 1: Clone the Notebook
1. Open `generalized_llm_judge_notebook.py` in Databricks
2. Click "Clone" to create your own copy
3. Rename it to something descriptive (e.g., `mortgage_calculator_evaluation.py`)

### Step 2: Configure Your Credentials
Replace the placeholder scope/keys with your own:
```python
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="your-scope", key="openai_key")
```

### Step 3: Set Your Experiment Name
```python
EXPERIMENT_NAME = "/Users/your_name/your_experiment_name"
```

### Step 4: Point to Your Data
```python
INPUT_FILE_PATH = "/path/to/your/data.csv"
PROMPT_COLUMN = "user_query"      # Column with user questions
RESPONSE_COLUMN = "ai_response"   # Column with AI responses
```

### Step 5: Run!
Execute all cells in order. Results will appear in MLflow and as visualizations.

## 📏 Adding Custom Metrics

### Example: Adding a "Professionalism" Metric

1. **Define the evaluation prompt**:
```python
PROFESSIONALISM_PROMPT = """You are evaluating the professionalism of a response.

### Materials
**User Query:**
```
{prompt}
```

**AI Response:**
```
{response}
```

### Scoring Scale
- 5: Highly professional
- 4: Professional
- 3: Adequate
- 2: Unprofessional elements
- 1: Completely unprofessional

### Output Format
Return ONLY this JSON:
```json
{
  "professionalism_score": <1-5>,
  "explanation": "<why this score>"
}
```
"""
```

2. **Add to EVALUATION_METRICS**:
```python
EVALUATION_METRICS = {
    # ... existing metrics ...
    "professionalism": {
        "prompt_template": PROFESSIONALISM_PROMPT,
        "threshold": 4,  # Scores >= 4 pass
        "scale": "1-5",
        "description": "Professional tone and language"
    }
}
```

3. **Run the notebook** - your new metric will be automatically included!

## 📊 Understanding Results

### In MLflow UI:
- **Metrics Tab**: See mean scores and pass rates
- **Artifacts Tab**: Download detailed results CSV
- **Charts Tab**: View auto-generated visualizations

### Generated Visualizations:
1. **Distribution Plots**: How scores are distributed
2. **Pass Rate Chart**: Percentage passing each metric
3. **Correlation Heatmap**: How metrics relate to each other
4. **Trend Lines**: Scores across your dataset

### Output Files:
- `evaluation_results_[timestamp].csv`: Detailed scores for each sample
- `evaluation_summary_[timestamp].csv`: Aggregated statistics
- `*.png` files: All visualization charts

## 🎯 Common Use Cases

### 1. Evaluating Search Result Quality
```python
# Metrics: relevance, completeness, ranking_quality
```

### 2. Testing Mortgage Calculator Responses
```python
# Metrics: calculation_accuracy, explanation_clarity, disclaimer_presence
```

### 3. Assessing Agent Recommendation Quality
```python
# Metrics: agent_match_quality, area_expertise, response_time_mentioned
```

### 4. Checking Listing Description Quality
```python
# Metrics: detail_completeness, accuracy, appeal_score
```

## 🐛 Troubleshooting

### "Module not found" Error
- Run the installation cells first
- Restart the Python kernel if needed

### API Rate Limits
- Reduce batch size
- Add delays between API calls
- Use fewer judge models

### Memory Issues
- Process data in smaller chunks
- Reduce the number of metrics evaluated simultaneously

### Scores Don't Match Expectations
- Review your metric prompts - be more specific
- Check threshold settings
- Look at the detailed explanations in results

## 📚 Best Practices

1. **Start Small**: Test with 10-20 samples first
2. **Iterate on Prompts**: Refine based on initial results
3. **Use Multiple Judges**: Ensemble evaluations are more robust
4. **Document Changes**: Keep notes on metric modifications
5. **Version Control**: Save different versions of your metrics

## 🤝 Getting Help

1. **Slack**: #llm-evaluation-help
2. **Wiki**: [Internal Zillow LLM Evaluation Guide]
3. **Office Hours**: Thursdays 2-3pm PT

## 📝 Metric Examples Library

### Binary Metrics (Yes/No)
- Contains PII
- Includes disclaimer
- Matches expected format
- Contains profanity

### Scale Metrics (1-5)
- Helpfulness
- Clarity
- Completeness
- Accuracy

### Percentage Metrics (0-100%)
- Coverage of requirements
- Keyword inclusion rate
- Feature mention rate

## 🚀 Advanced Features

### Using Ground Truth Data
If you have expected answers:
```python
df["expected_answer"] = df["golden_response"]
# Then reference {expected_answer} in your metric prompts
```

### Comparing Multiple Models
```python
RESPONSE_MODELS = ["gpt-4o", "claude-3.5", "internal-model"]
# Run evaluation for each and compare
```

### Custom Visualizations
Add your own plots in the visualization section:
```python
# After the standard visualizations
plt.figure(figsize=(10, 6))
# Your custom visualization code
plt.savefig(f'{OUTPUT_DIR}/custom_chart.png')
```

---

**Happy Evaluating! 🎉**

For questions or feature requests, contact the AI Platform team.